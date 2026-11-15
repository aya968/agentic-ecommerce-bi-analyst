import json
import shutil
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import DATA_DIR, SESSION_TTL_SECONDS


class SessionError(ValueError):
    """Base error for missing, expired, or invalid dataset sessions."""


class SessionNotFoundError(SessionError):
    """Raised when a session ID does not identify a stored session."""


class SessionExpiredError(SessionError):
    """Raised when a session has exceeded its inactivity time-to-live."""


class SessionStore:
    """Manage isolated per-session directories and their metadata."""

    def __init__(self, data_dir=DATA_DIR, ttl_seconds=SESSION_TTL_SECONDS):
        if ttl_seconds <= 0:
            raise ValueError("Session TTL must be positive.")
        self.sessions_dir = Path(data_dir).resolve() / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = int(ttl_seconds)
        self._lock = threading.RLock()
        self._session_locks = {}

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    @classmethod
    def _timestamp(cls, value=None):
        return (value or cls._now()).isoformat()

    @staticmethod
    def _parse_timestamp(value):
        return datetime.fromisoformat(value)

    def _validated_session_dir(self, session_id):
        try:
            parsed = uuid.UUID(str(session_id))
        except (ValueError, TypeError, AttributeError) as error:
            raise SessionNotFoundError("Session was not found.") from error

        canonical_id = str(parsed)
        if str(session_id).lower() != canonical_id:
            raise SessionNotFoundError("Session was not found.")

        session_dir = self.sessions_dir / canonical_id
        if (
            session_dir.is_symlink()
            or session_dir.resolve().parent != self.sessions_dir
        ):
            raise SessionNotFoundError("Session was not found.")
        return canonical_id, session_dir

    @contextmanager
    def session_operation(self, session_id):
        canonical_id, _ = self._validated_session_dir(session_id)
        with self._lock:
            session_lock = self._session_locks.setdefault(
                canonical_id,
                threading.RLock()
            )
        with session_lock:
            self.get_session(canonical_id, touch=True)
            yield

    @staticmethod
    def _write_json(path, value):
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        temporary_path.replace(path)

    def _read_metadata(self, session_dir):
        metadata_path = session_dir / "session.json"
        if not metadata_path.is_file():
            raise SessionNotFoundError("Session was not found.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SessionNotFoundError(
                "Session metadata is unavailable."
            ) from error
        return metadata

    def create_session(self):
        with self._lock:
            session_id = str(uuid.uuid4())
            session_dir = self.sessions_dir / session_id
            session_dir.mkdir(mode=0o700)
            (session_dir / "uploads").mkdir(mode=0o700)
            now = self._now()
            metadata = {
                "session_id": session_id,
                "created_at": self._timestamp(now),
                "uploaded_at": None,
                "last_activity": self._timestamp(now),
                "expires_at": self._timestamp(
                    now + timedelta(seconds=self.ttl_seconds)
                ),
                "dataset_filename": None,
                "schema": None
            }
            try:
                self._write_json(session_dir / "session.json", metadata)
            except Exception:
                shutil.rmtree(session_dir, ignore_errors=True)
                raise
            return metadata

    def get_session(self, session_id, touch=True):
        with self._lock:
            canonical_id, session_dir = self._validated_session_dir(
                session_id
            )
            if not session_dir.is_dir():
                raise SessionNotFoundError("Session was not found.")

            metadata = self._read_metadata(session_dir)
            if metadata.get("session_id") != canonical_id:
                raise SessionNotFoundError("Session was not found.")

            now = self._now()
            expires_at = self._parse_timestamp(metadata["expires_at"])
            if expires_at <= now:
                raise SessionExpiredError(
                    "This session expired. Start a new session and upload "
                    "your dataset again."
                )

            if touch:
                metadata["last_activity"] = self._timestamp(now)
                metadata["expires_at"] = self._timestamp(
                    now + timedelta(seconds=self.ttl_seconds)
                )
                self._write_json(session_dir / "session.json", metadata)

            return {
                **metadata,
                "directory": session_dir,
                "database_path": session_dir / "ecommerce.db",
                "uploads_directory": session_dir / "uploads"
            }

    def update_session(self, session_id, **updates):
        with self._lock:
            session = self.get_session(session_id, touch=False)
            session_dir = session["directory"]
            metadata = {
                key: value
                for key, value in session.items()
                if key not in {
                    "directory",
                    "database_path",
                    "uploads_directory"
                }
            }
            metadata.update(updates)
            now = self._now()
            metadata["last_activity"] = self._timestamp(now)
            metadata["expires_at"] = self._timestamp(
                now + timedelta(seconds=self.ttl_seconds)
            )
            self._write_json(session_dir / "session.json", metadata)
            return self.get_session(session_id, touch=False)

    def delete_session(self, session_id):
        with self.session_operation(session_id):
            canonical_id, session_dir = self._validated_session_dir(
                session_id
            )
            if not session_dir.is_dir():
                raise SessionNotFoundError("Session was not found.")
            metadata = self._read_metadata(session_dir)
            if metadata.get("session_id") != canonical_id:
                raise SessionNotFoundError("Session was not found.")
            shutil.rmtree(session_dir)

    def cleanup_expired_sessions(self):
        """Remove only sessions whose persisted expiration time has passed."""
        deleted = []
        with self._lock:
            for session_dir in self.sessions_dir.iterdir():
                if session_dir.is_symlink() or not session_dir.is_dir():
                    continue
                try:
                    metadata = self._read_metadata(session_dir)
                    session_id, validated_dir = self._validated_session_dir(
                        metadata.get("session_id")
                    )
                    if validated_dir != session_dir:
                        continue
                    if (
                        self._parse_timestamp(metadata["expires_at"])
                        <= self._now()
                    ):
                        shutil.rmtree(validated_dir)
                        deleted.append(session_id)
                except (KeyError, TypeError, ValueError, OSError):
                    continue
        return deleted
