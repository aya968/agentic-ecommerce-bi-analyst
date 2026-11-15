import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from app.database.db_manager import DatabaseManager


class FastAPITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.upload_dir = root / "uploads"
        self.upload_dir.mkdir()
        self.manager = DatabaseManager(root / "test.sqlite")

        for attribute, value in (
            ("db_manager", self.manager),
            ("UPLOAD_DIR", self.upload_dir),
            ("current_schema", None)
        ):
            patcher = patch.object(api, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.client = TestClient(api.app)

    @staticmethod
    def upload_payload(filename="sales.csv", content=None):
        if content is None:
            content = (
                "product,order_date,quantity,stock,lead_time_days\n"
                "Widget,2025-01-06,2,12,5\n"
            ).encode()
        return {
            "file": (filename, content, "text/csv")
        }

    def test_valid_upload_returns_schema_and_keeps_original_separately(self):
        response = self.client.post(
            "/upload-data",
            files=self.upload_payload()
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertIn("product", response.json()["schema_profile"])
        self.assertEqual(
            len(list(self.upload_dir.iterdir())),
            1
        )
        self.assertEqual(self.manager.get_table_names(), ["ecommerce_data"])

    def test_schema_api_response_preserves_original_headers_without_samples(self):
        response = self.client.post(
            "/upload-data",
            files=self.upload_payload(
                content=(
                    b"SKU,OrderDate,Qty\n"
                    b"sku-secret,2025-01-06,2\n"
                )
            )
        )

        schema = response.json()["schema_profile"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            schema["field_mappings"]["product_id"]["original_column"],
            "SKU"
        )
        self.assertNotIn("sku-secret", response.text)
        self.assertTrue(all(
            "sample_values" not in profile
            for profile in schema["column_profiles"]
        ))

    def test_schema_mapping_endpoint_updates_backend_profile(self):
        uploaded = self.client.post(
            "/upload-data",
            files=self.upload_payload()
        )
        self.assertEqual(uploaded.status_code, 200)

        response = self.client.post(
            "/schema-mapping",
            json={
                "overrides": {
                    "product_id": "product",
                    "date": "order_date",
                    "quantity": "quantity"
                }
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["product"], "product")
        self.assertEqual(
            api.current_schema["field_mappings"]["product_id"][
                "detection_method"
            ],
            "manual"
        )

    def test_failed_upload_preserves_active_dataset_and_cleans_failed_file(self):
        first = self.client.post(
            "/upload-data",
            files=self.upload_payload(content=(
                b"product,order_date,quantity\nWidget,2025-01-06,2\n"
            ))
        )
        previous_schema = api.current_schema

        failed = self.client.post(
            "/upload-data",
            files=self.upload_payload(content=b"")
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(failed.status_code, 400)
        self.assertEqual(api.current_schema, previous_schema)
        self.assertEqual(
            self.manager.get_dataframe().loc[0, "quantity"],
            2
        )
        self.assertEqual(len(list(self.upload_dir.iterdir())), 1)

    def test_unsupported_upload_type_is_rejected(self):
        response = self.client.post(
            "/upload-data",
            files=self.upload_payload(filename="sales.json")
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(list(self.upload_dir.iterdir())), 0)

    def test_agent_request_validation_and_expected_result(self):
        api.current_schema = {"product": "product"}
        invalid = self.client.post("/ask-agent", json={"question": "x"})
        self.assertEqual(invalid.status_code, 422)

        expected = {
            "question": "Which product?",
            "sql": "SELECT product FROM ecommerce_data",
            "data": [{"product": "Widget"}],
            "columns": ["product"],
            "insight": "Widget is available.",
            "error": None,
            "success": True
        }
        with patch("api.ask_agent", return_value=expected):
            response = self.client.post(
                "/ask-agent",
                json={"question": "Which product?"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)

    def test_agent_exception_returns_generic_error(self):
        api.current_schema = {"product": "product"}
        with patch(
            "api.ask_agent",
            side_effect=RuntimeError("private internal details")
        ):
            response = self.client.post(
                "/ask-agent",
                json={"question": "Which product?"}
            )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private internal details", response.text)


if __name__ == "__main__":
    unittest.main()
