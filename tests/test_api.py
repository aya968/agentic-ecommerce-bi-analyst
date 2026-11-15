import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from app.database.db_manager import DatabaseManager
from app.session_store import SessionStore


class FastAPITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.store = SessionStore(self.root / "data", ttl_seconds=3600)
        self.store_patcher = patch.object(api, "session_store", self.store)
        self.store_patcher.start()
        self.addCleanup(self.store_patcher.stop)

        self.client = TestClient(api.app)
        with self.client:
            response = self.client.post("/sessions")
            self.session_id = response.json()["session_id"]

    def headers(self, session_id=None):
        return {
            "X-Session-ID": session_id or self.session_id
        }

    def upload(self, session_id=None, filename="sales.csv", content=None,
               replace_existing=False):
        if content is None:
            content = (
                "product,order_date,quantity,stock,lead_time_days\n"
                "Widget,2025-01-06,2,12,5\n"
            ).encode()
        return self.client.post(
            "/upload-data",
            headers=self.headers(session_id),
            files={"file": (filename, content, "text/csv")},
            data={"replace_existing": str(replace_existing).lower()}
        )

    def test_create_and_upload_uses_per_session_database_and_upload_dir(self):
        response = self.upload()
        session = self.store.get_session(self.session_id, touch=False)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertIn("product", response.json()["schema_profile"])
        self.assertTrue(session["database_path"].is_file())
        self.assertIsNotNone(session["uploaded_at"])
        self.assertEqual(
            len(list(session["uploads_directory"].iterdir())),
            1
        )
        self.assertEqual(
            DatabaseManager(session["database_path"])
            .get_dataframe().shape[0],
            1
        )

    def test_two_sessions_have_distinct_rows_and_sql_databases(self):
        second_session = self.client.post("/sessions").json()["session_id"]
        first_upload = self.upload(content=(
            b"product,quantity\nWidget,2\n"
        ))
        second_upload = self.upload(
            session_id=second_session,
            content=b"product,quantity\nGadget,9\n"
        )

        self.assertEqual(first_upload.status_code, 200)
        self.assertEqual(second_upload.status_code, 200)
        first_dataset = self.client.get(
            "/dataset",
            headers=self.headers(self.session_id)
        ).json()
        second_dataset = self.client.get(
            "/dataset",
            headers=self.headers(second_session)
        ).json()
        self.assertEqual(first_dataset["row_count"], 1)
        self.assertEqual(second_dataset["row_count"], 1)
        self.assertEqual(first_dataset["rows"][0]["product"], "Widget")
        self.assertEqual(second_dataset["rows"][0]["product"], "Gadget")

    def test_agent_receives_session_specific_database_path(self):
        second_session = self.client.post("/sessions").json()["session_id"]
        self.upload(content=b"product,quantity\nWidget,2\n")
        self.upload(
            session_id=second_session,
            content=b"product,quantity\nGadget,9\n"
        )

        def execute_for_session(question, schema, db_path):
            records = DatabaseManager(db_path).get_dataframe()
            return {
                "question": question,
                "sql": "SELECT product, quantity FROM ecommerce_data",
                "data": records.to_dict("records"),
                "columns": list(records.columns),
                "insight": "Session-scoped result.",
                "error": None,
                "success": True
            }

        with patch("api.ask_agent", side_effect=execute_for_session) as ask:
            first = self.client.post(
                "/ask-agent",
                headers=self.headers(self.session_id),
                json={"question": "Which product sold?"}
            )
            second = self.client.post(
                "/ask-agent",
                headers=self.headers(second_session),
                json={"question": "Which product sold?"}
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["data"][0]["product"], "Widget")
        self.assertEqual(second.json()["data"][0]["product"], "Gadget")
        self.assertNotEqual(
            ask.call_args_list[0].kwargs["db_path"],
            ask.call_args_list[1].kwargs["db_path"]
        )

    def test_existing_dataset_requires_confirmation_before_replacement(self):
        self.assertEqual(
            self.upload(content=b"product,quantity\nWidget,2\n").status_code,
            200
        )
        rejected = self.upload(content=b"product,quantity\nGadget,9\n")
        self.assertEqual(rejected.status_code, 409)

        replaced = self.upload(
            content=b"product,quantity\nGadget,9\n",
            replace_existing=True
        )
        dataset = self.client.get(
            "/dataset",
            headers=self.headers()
        ).json()

        self.assertEqual(replaced.status_code, 200)
        self.assertEqual(dataset["row_count"], 1)
        self.assertEqual(dataset["rows"][0]["product"], "Gadget")
        self.assertEqual(dataset["rows"][0]["quantity"], 9)

    def test_failed_replacement_preserves_old_dataset_and_cleans_file(self):
        self.upload(content=b"product,quantity\nWidget,2\n")
        session = self.store.get_session(self.session_id, touch=False)
        before_upload_count = len(
            list(session["uploads_directory"].iterdir())
        )
        failed = self.upload(
            content=b"",
            replace_existing=True
        )
        dataset = self.client.get(
            "/dataset",
            headers=self.headers()
        ).json()

        self.assertEqual(failed.status_code, 400)
        self.assertEqual(dataset["rows"][0]["product"], "Widget")
        self.assertEqual(
            len(list(session["uploads_directory"].iterdir())),
            before_upload_count
        )

    def test_upload_extension_and_size_are_validated(self):
        unsupported = self.upload(filename="sales.json")
        self.assertEqual(unsupported.status_code, 400)

        with patch.object(api, "MAX_UPLOAD_BYTES", 10):
            oversized = self.upload(content=b"product\n" + b"x" * 20)
        self.assertEqual(oversized.status_code, 413)
        self.assertFalse(
            self.store.get_session(
                self.session_id,
                touch=False
            )["database_path"].exists()
        )

    def test_api_rejects_missing_or_unknown_session(self):
        missing_header = self.client.get("/dataset")
        unknown = self.client.get(
            "/dataset",
            headers=self.headers("00000000-0000-0000-0000-000000000001")
        )
        self.assertEqual(missing_header.status_code, 422)
        self.assertEqual(unknown.status_code, 404)

    def test_delete_requires_matching_owner_and_removes_session_files(self):
        self.upload()
        session = self.store.get_session(self.session_id, touch=False)
        mismatch = self.client.delete(
            f"/sessions/{self.session_id}",
            headers=self.headers("00000000-0000-0000-0000-000000000001")
        )
        deleted = self.client.delete(
            f"/sessions/{self.session_id}",
            headers=self.headers()
        )

        self.assertEqual(mismatch.status_code, 403)
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(session["directory"].exists())

    def test_persistent_session_files_survive_store_reinitialization(self):
        self.upload(content=b"product,quantity\nWidget,7\n")
        first_store_session = self.store.get_session(
            self.session_id,
            touch=False
        )
        restarted_store = SessionStore(self.root / "data", ttl_seconds=3600)
        restored = restarted_store.get_session(
            self.session_id,
            touch=False
        )

        self.assertEqual(
            DatabaseManager(restored["database_path"])
            .get_dataframe().loc[0, "quantity"],
            7
        )
        self.assertTrue(first_store_session["database_path"].exists())


if __name__ == "__main__":
    unittest.main()
