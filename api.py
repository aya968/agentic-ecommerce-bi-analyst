import os
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
from app.config import DATA_DIR


app = FastAPI(
    title="E-commerce AI Analyst API",
    version="1.0.0"
)

db_manager = DatabaseManager()
schema_analyzer = SchemaAnalyzer()
forecaster = DemandForecaster()
inventory_optimizer = InventoryOptimizer()

current_schema = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class InventoryRequest(BaseModel):
    default_lead_time_days: float | None = Field(
        default=None,
        gt=0
    )


@app.get("/")
def home():
    return {
        "message": "E-commerce AI Analyst API is running"
    }


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    global current_schema

    extension = Path(file.filename).suffix.lower()

    if extension not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files are supported."
        )

    unique_name = f"{uuid.uuid4().hex}{extension}"
    file_path = DATA_DIR / unique_name

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        report = db_manager.load_file_to_sqlite(
            str(file_path),
            table_name="ecommerce_data"
        )

        df = db_manager.get_dataframe()

        current_schema = schema_analyzer.analyze(df)

        return {
            "success": True,
            "message": "Dataset uploaded successfully.",
            "cleaning_report": report,
            "schema_profile": {
                key: value
                for key, value in current_schema.items()
                if key != "all_columns"
            },
            "columns": list(df.columns)
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    finally:
        await file.close()


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
            detail=str(error)
        )


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
    