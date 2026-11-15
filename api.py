import shutil
import uuid

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from app.database.db_manager import DatabaseManager
from app.utils.schema_analyzer import SchemaAnalyzer
from app.agents.graph import ask_agent
from app.analytics.forecasting import DemandForecaster
from app.analytics.inventory import InventoryOptimizer
from app.config import UPLOAD_DIR


app = FastAPI(
    title="E-commerce AI Analyst API",
    version="1.0.0"
)

db_manager = DatabaseManager()
schema_analyzer = SchemaAnalyzer()
forecaster = DemandForecaster()
inventory_optimizer = InventoryOptimizer()

# The API uses one process-wide active dataset and is intended for one user.
current_schema = None


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


@app.get("/")
def home():
    return {
        "message": "E-commerce AI Analyst API is running"
    }


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    global current_schema

    file_path = None
    upload_succeeded = False
    try:
        extension = Path(file.filename or "").suffix.lower()
        if extension not in [".csv", ".xlsx", ".xls"]:
            raise HTTPException(
                status_code=400,
                detail="Only CSV and Excel files are supported."
            )

        file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        dataframe, report = db_manager.prepare_file(
            str(file_path),
        )
        original_column_names = [
            mapping["original"]
            for mapping in report["column_name_mapping"]
        ]
        new_schema = schema_analyzer.analyze(
            dataframe,
            original_column_names=original_column_names
        )
        report = db_manager.save_dataframe_to_sqlite(
            dataframe,
            table_name="ecommerce_data",
            report=report
        )
        current_schema = new_schema
        upload_succeeded = True

        return {
            "success": True,
            "message": "Dataset uploaded successfully.",
            "cleaning_report": report,
            "schema_profile": public_schema_profile(new_schema),
            "columns": list(dataframe.columns)
        }

    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Dataset upload failed. The previously active dataset "
                "was preserved."
            )
        ) from error

    finally:
        if file_path is not None and not upload_succeeded:
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                pass
        await file.close()


@app.post("/schema-mapping")
def update_schema_mapping(request: SchemaMappingRequest):
    global current_schema

    if current_schema is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a dataset first."
        )

    try:
        dataframe = db_manager.get_dataframe()
        current_schema = schema_analyzer.apply_overrides(
            dataframe,
            request.overrides,
            original_column_names=current_schema.get("original_columns")
        )
        return public_schema_profile(current_schema)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Schema mappings could not be updated."
        ) from error


@app.post("/ask-agent")
def ask_question(request: QueryRequest):
    if current_schema is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a dataset first."
        )

    try:
        result = ask_agent(
            request.question,
            current_schema
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The agent could not complete the request."
        ) from error


@app.get("/forecast-readiness")
def forecast_readiness():
    if current_schema is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a dataset first."
        )

    df = db_manager.get_dataframe()

    result = forecaster.check_readiness(
        df,
        current_schema
    )

    if "data" in result:
        result.pop("data")

    return result


@app.get("/forecast")
def get_forecast():
    if current_schema is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a dataset first."
        )

    df = db_manager.get_dataframe()

    result = forecaster.forecast(
        df,
        current_schema
    )

    if "data" in result:
        result.pop("data")

    return result


@app.post("/inventory")
def get_inventory(request: InventoryRequest):
    if current_schema is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a dataset first."
        )

    df = db_manager.get_dataframe()

    forecast_result = forecaster.forecast(
        df,
        current_schema
    )

    if not forecast_result.get("ready"):
        return forecast_result

    result = inventory_optimizer.optimize_from_data(
        df=df,
        forecast_result=forecast_result,
        schema=current_schema,
        default_lead_time_days=request.default_lead_time_days
    )

    return result
    