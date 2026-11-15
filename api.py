import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile
)
from pydantic import BaseModel, Field

from app.agents.graph import ask_agent
from app.agents.sql_validator import SQLValidator
from app.analytics.forecasting import DemandForecaster
from app.analytics.inventory import InventoryOptimizer
from app.config import DATA_DIR, MAX_UPLOAD_BYTES, SESSION_TTL_SECONDS
from app.database.db_manager import DatabaseManager
from app.session_store import (
    SessionExpiredError,
    SessionNotFoundError,
    SessionStore
)
from app.utils.schema_analyzer import SchemaAnalyzer


logger = logging.getLogger(__name__)
session_store = SessionStore(DATA_DIR, SESSION_TTL_SECONDS)
schema_analyzer = SchemaAnalyzer()
forecaster = DemandForecaster()
inventory_optimizer = InventoryOptimizer()

app = FastAPI(
    title="E-commerce AI Analyst API",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class InventoryRequest(BaseModel):
    default_lead_time_days: float | None = Field(
        default=None,
        gt=0
    )


class SchemaMappingRequest(BaseModel):
    overrides: dict[str, str | None]


def public_schema_profile(profile):
    """Avoid returning sampled uploaded values in API responses."""
    public_profile = {
        key: value
        for key, value in profile.items()
        if key not in {"all_columns", "column_profiles"}
    }
    public_profile["column_profiles"] = [
        {
            key: value
            for key, value in column_profile.items()
            if key != "sample_values"
        }
        for column_profile in profile.get("column_profiles", [])
    ]
    return public_profile


def _session_or_404(session_id):
    try:
        return session_store.get_session(session_id)
    except SessionExpiredError as error:
        raise HTTPException(status_code=410, detail=str(error)) from error
    except SessionNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="Session was not found. Start a new session."
        ) from error


def _schema_for_session(session):
    schema = session.get("schema")
    if not schema:
        raise HTTPException(
            status_code=400,
            detail="Upload a dataset to this session first."
        )
    return schema


def _serialize_records(dataframe):
    return json.loads(
        dataframe.to_json(
            orient="records",
            date_format="iso",
            date_unit="us"
        )
    )


@app.on_event("startup")
def cleanup_expired_sessions():
    deleted = session_store.cleanup_expired_sessions()
    if deleted:
        logger.info("Removed %s expired dataset sessions.", len(deleted))


@app.get("/")
def home():
    return {
        "message": "E-commerce AI Analyst API is running"
    }


@app.post("/sessions")
def create_session():
    try:
        metadata = session_store.create_session()
        return {
            "session_id": metadata["session_id"],
            "created_at": metadata["created_at"],
            "expires_at": metadata["expires_at"]
        }
    except OSError as error:
        logger.exception("Could not create an isolated dataset session.")
        raise HTTPException(
            status_code=500,
            detail="A new dataset session could not be created."
        ) from error


@app.post("/sessions/activity")
def record_session_activity(
    session_id: str = Header(alias="X-Session-ID")
):
    session = _session_or_404(session_id)
    return {
        "session_id": session["session_id"],
        "last_activity": session["last_activity"],
        "expires_at": session["expires_at"]
    }


@app.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    owner_session_id: str = Header(alias="X-Session-ID")
):
    if session_id != owner_session_id:
        raise HTTPException(
            status_code=403,
            detail="The session ID does not match the active session."
        )
    try:
        session_store.delete_session(session_id)
    except SessionExpiredError as error:
        raise HTTPException(status_code=410, detail=str(error)) from error
    except SessionNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="Session was not found."
        ) from error
    return {"success": True, "message": "Session data was deleted."}


@app.post("/upload-data")
async def upload_data(
    file: UploadFile = File(...),
    replace_existing: bool = Form(default=False),
    session_id: str = Header(alias="X-Session-ID")
):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".csv", ".xlsx", ".xls"}:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files are supported."
        )

    try:
        with session_store.session_operation(session_id):
            session = session_store.get_session(session_id, touch=False)
            if session.get("uploaded_at") and not replace_existing:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This session already has a dataset. Confirm "
                        "replacement before uploading another file."
                    )
                )

            upload_path = (
                session["uploads_directory"]
                / f"{uuid.uuid4().hex}{extension}"
            )
            size = 0
            upload_succeeded = False
            try:
                with upload_path.open("wb") as target:
                    while True:
                        chunk = await file.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    "The uploaded file exceeds the "
                                    f"{MAX_UPLOAD_BYTES}-byte limit."
                                )
                            )
                        target.write(chunk)

                dataframe, report = DatabaseManager(
                    session["database_path"]
                ).prepare_file(str(upload_path))
                original_column_names = [
                    mapping["original"]
                    for mapping in report["column_name_mapping"]
                ]
                schema = schema_analyzer.analyze(
                    dataframe,
                    original_column_names=original_column_names
                )

                session_dir = session["directory"]
                staged_database = (
                    session_dir / f".dataset-{uuid.uuid4().hex}.sqlite"
                )
                backup_database = (
                    session_dir / f".backup-{uuid.uuid4().hex}.sqlite"
                )
                database_path = session["database_path"]
                had_previous_database = database_path.exists()
                try:
                    database_manager = DatabaseManager(staged_database)
                    database_manager.save_dataframe_to_sqlite(
                        dataframe,
                        table_name="ecommerce_data",
                        report=report
                    )
                    schema["database_schema"] = (
                        SQLValidator.get_database_schema(staged_database)
                    )
                    if had_previous_database:
                        shutil.copy2(database_path, backup_database)
                    os.replace(
                        staged_database,
                        database_path
                    )
                    try:
                        session_store.update_session(
                            session_id,
                            uploaded_at=datetime.now(
                                timezone.utc
                            ).isoformat(),
                            dataset_filename=Path(
                                file.filename or "dataset"
                            ).name,
                            schema=schema
                        )
                    except Exception:
                        if had_previous_database and backup_database.exists():
                            os.replace(backup_database, database_path)
                        elif not had_previous_database:
                            database_path.unlink(missing_ok=True)
                        raise
                except Exception:
                    staged_database.unlink(missing_ok=True)
                    raise
                finally:
                    backup_database.unlink(missing_ok=True)

                report["table_name"] = "ecommerce_data"
                report["loaded_rows"] = len(dataframe)
                upload_succeeded = True
                return {
                    "success": True,
                    "message": "Dataset uploaded successfully.",
                    "cleaning_report": report,
                    "schema_profile": public_schema_profile(schema),
                    "columns": list(dataframe.columns)
                }
            finally:
                await file.close()
                if not upload_succeeded:
                    upload_path.unlink(missing_ok=True)

    except HTTPException:
        raise
    except (SessionExpiredError, SessionNotFoundError) as error:
        status = 410 if isinstance(error, SessionExpiredError) else 404
        raise HTTPException(
            status_code=status,
            detail=str(error)
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
    except Exception as error:
        logger.exception("Dataset upload failed for a session.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Dataset upload failed. The active session dataset, if any, "
                "was preserved."
            )
        ) from error


@app.post("/schema-mapping")
def update_schema_mapping(
    request: SchemaMappingRequest,
    session_id: str = Header(alias="X-Session-ID")
):
    try:
        with session_store.session_operation(session_id):
            session = session_store.get_session(session_id, touch=False)
            _schema_for_session(session)
            dataframe = DatabaseManager(
                session["database_path"]
            ).get_dataframe()
            schema = schema_analyzer.apply_overrides(
                dataframe,
                request.overrides,
                original_column_names=session["schema"].get(
                    "original_columns"
                )
            )
            schema["database_schema"] = (
                SQLValidator.get_database_schema(
                    session["database_path"]
                )
            )
            session_store.update_session(session_id, schema=schema)
            return public_schema_profile(schema)
    except HTTPException:
        raise
    except (SessionExpiredError, SessionNotFoundError) as error:
        status = 410 if isinstance(error, SessionExpiredError) else 404
        raise HTTPException(status_code=status, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/dataset")
def get_dataset(session_id: str = Header(alias="X-Session-ID")):
    try:
        with session_store.session_operation(session_id):
            session = session_store.get_session(session_id, touch=False)
            _schema_for_session(session)
            dataframe = DatabaseManager(
                session["database_path"]
            ).get_dataframe()
            return {
                "columns": list(dataframe.columns),
                "rows": _serialize_records(dataframe),
                "row_count": len(dataframe)
            }
    except (SessionExpiredError, SessionNotFoundError) as error:
        status = 410 if isinstance(error, SessionExpiredError) else 404
        raise HTTPException(status_code=status, detail=str(error)) from error


@app.post("/ask-agent")
def ask_question(
    request: QueryRequest,
    session_id: str = Header(alias="X-Session-ID")
):
    try:
        with session_store.session_operation(session_id):
            session = session_store.get_session(session_id, touch=False)
            schema = _schema_for_session(session)
            return ask_agent(
                request.question,
                schema,
                db_path=str(session["database_path"])
            )
    except HTTPException:
        raise
    except (SessionExpiredError, SessionNotFoundError) as error:
        status = 410 if isinstance(error, SessionExpiredError) else 404
        raise HTTPException(status_code=status, detail=str(error)) from error
    except Exception as error:
        logger.exception("Agent request failed for a dataset session.")
        raise HTTPException(
            status_code=500,
            detail="The agent could not complete the request."
        ) from error


@app.get("/forecast-readiness")
def forecast_readiness(
    session_id: str = Header(alias="X-Session-ID")
):
    try:
        with session_store.session_operation(session_id):
            session = session_store.get_session(session_id, touch=False)
            schema = _schema_for_session(session)
            dataframe = DatabaseManager(
                session["database_path"]
            ).get_dataframe()
            result = forecaster.check_readiness(dataframe, schema)
            result.pop("data", None)
            return result
    except HTTPException:
        raise
    except (SessionExpiredError, SessionNotFoundError) as error:
        status = 410 if isinstance(error, SessionExpiredError) else 404
        raise HTTPException(status_code=status, detail=str(error)) from error


@app.get("/forecast")
def get_forecast(session_id: str = Header(alias="X-Session-ID")):
    try:
        with session_store.session_operation(session_id):
            session = session_store.get_session(session_id, touch=False)
            schema = _schema_for_session(session)
            dataframe = DatabaseManager(
                session["database_path"]
            ).get_dataframe()
            result = forecaster.forecast(dataframe, schema)
            result.pop("data", None)
            return result
    except HTTPException:
        raise
    except (SessionExpiredError, SessionNotFoundError) as error:
        status = 410 if isinstance(error, SessionExpiredError) else 404
        raise HTTPException(status_code=status, detail=str(error)) from error


@app.post("/inventory")
def get_inventory(
    request: InventoryRequest,
    session_id: str = Header(alias="X-Session-ID")
):
    try:
        with session_store.session_operation(session_id):
            session = session_store.get_session(session_id, touch=False)
            schema = _schema_for_session(session)
            dataframe = DatabaseManager(
                session["database_path"]
            ).get_dataframe()
            forecast_result = forecaster.forecast(dataframe, schema)
            if not forecast_result.get("ready"):
                return forecast_result

            return inventory_optimizer.optimize_from_data(
                df=dataframe,
                forecast_result=forecast_result,
                schema=schema,
                default_lead_time_days=request.default_lead_time_days
            )
    except HTTPException:
        raise
    except (SessionExpiredError, SessionNotFoundError) as error:
        status = 410 if isinstance(error, SessionExpiredError) else 404
        raise HTTPException(status_code=status, detail=str(error)) from error
