import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from app.session_store import (
    SessionExpiredError,
    SessionNotFoundError,
    SessionStore
)


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = SessionStore(
            Path(self.temp_dir.name) / "data",
            ttl_seconds=3600
        )

    def test_create_records_timestamps_and_updates_expiration_on_activity(self):
        session = self.store.create_session()
        initial_expiration = session["expires_at"]

        updated = self.store.get_session(session["session_id"])

        self.assertIsNotNone(session["created_at"])
        self.assertEqual(updated["session_id"], session["session_id"])
        self.assertGreaterEqual(
            updated["expires_at"],
            initial_expiration
        )
        self.assertTrue(updated["uploads_directory"].is_dir())
        self.assertEqual(
            updated["database_path"],
            updated["directory"] / "ecommerce.db"
        )

    def test_expired_sessions_are_rejected_and_cleaned(self):
        short_store = SessionStore(
            Path(self.temp_dir.name) / "short-lived",
            ttl_seconds=1
        )
        session = short_store.create_session()
        expired_time = short_store._now() + timedelta(seconds=2)

        with patch.object(
            short_store,
            "_now",
            return_value=expired_time
        ):
            with self.assertRaises(SessionExpiredError):
                short_store.get_session(session["session_id"])
            deleted = short_store.cleanup_expired_sessions()

        self.assertEqual(deleted, [session["session_id"]])
        self.assertFalse(
            (
                short_store.sessions_dir / session["session_id"]
            ).exists()
        )

    def test_cleanup_keeps_nonexpired_sessions(self):
        session = self.store.create_session()

        self.assertEqual(self.store.cleanup_expired_sessions(), [])
        self.assertTrue(
            (self.store.sessions_dir / session["session_id"]).exists()
        )

    def test_invalid_or_traversal_ids_never_resolve_outside_sessions_root(self):
        for session_id in ("../outside", "not-a-session"):
            with self.subTest(session_id=session_id):
                with self.assertRaises(SessionNotFoundError):
                    self.store.get_session(session_id)

    def test_manual_delete_removes_only_the_selected_session(self):
        first = self.store.create_session()
        second = self.store.create_session()

        self.store.delete_session(first["session_id"])

        self.assertFalse(first["directory"].exists())
        self.assertTrue(second["directory"].exists())


if __name__ == "__main__":
    unittest.main()
