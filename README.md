
# E-commerce AI Analyst
### AI-Powered Business Intelligence, Demand Forecasting & Inventory Optimization

An intelligent E-commerce analytics platform that transforms raw sales data into actionable business insights using Google's Gemini API.

The platform allows users to upload E-commerce datasets, explore business performance, ask questions in natural language, generate SQL queries automatically, forecast product demand, and receive inventory replenishment recommendations.

Built with Python, Streamlit, FastAPI, LangGraph, SQLite, and the Google Gen AI Python SDK.

---

## Overview

E-commerce businesses generate large amounts of sales and transaction data. However, extracting meaningful insights from this data often requires SQL knowledge, analytics experience, and separate forecasting tools.

E-commerce AI Analyst simplifies this process by combining traditional data analytics with AI-powered natural language interaction.

Users can upload their datasets and interact with their business data without manually writing SQL queries.

The system also evaluates whether the uploaded data is sufficient for demand forecasting and inventory optimization. If required information is missing, it explains exactly what additional data is needed instead of generating unsupported predictions.

---

## Key Features

### 1. Data Upload & Processing
- Upload CSV and Excel files.
- Automatic column-name normalization.
- Remove duplicate records.
- Remove completely empty rows and columns.
- Trim unnecessary whitespace.
- Detect and convert likely date columns.
- Generate a data-cleaning report.
- Store processed datasets in SQLite.

### 2. Business Intelligence Dashboard
Monitor important E-commerce performance indicators, including:

- Total Revenue.
- Total Orders.
- Total Units Sold.
- Number of Products.
- Monthly Revenue Trends.
- Top-Selling Products.

KPIs are displayed according to the columns available in the uploaded dataset.

### 3. AI-Powered Natural Language to SQL

Ask business questions using natural language.

Example questions:

- What is the total revenue?
- Which are the top 10 products by sales?
- What is the monthly revenue trend?
- Which products generate the highest revenue?
- How many unique orders are available?

The AI Agent:

1. Understands the user's question.
2. Analyzes the available database schema.
3. Generates a SQLite SELECT query.
4. Validates the generated SQL.
5. Executes the query against SQLite.
6. Sends the results to the LLM.
7. Generates a business-oriented explanation.

### 4. AI Business Insights

The system translates query results into understandable business insights.

The generated explanations can include:

- Important performance metrics.
- Sales observations.
- Product performance.
- Business trends supported by the query results.
- Practical recommendations.

The AI is instructed not to invent values or claim unsupported conclusions.

### 5. Demand Forecasting

Forecast future product demand using historical sales data.

Current implementation:
- Product-level demand forecasting.
- Weekly sales aggregation.
- 8-week moving-average, Holt linear, Croston SBA, and 52-week Holt-Winters models.
- Per-product model selection using expanding-window backtesting.
- Four-week forecast horizon.
- Configurable zero-fill or forward-only interpolation for missing weeks.
- Product-level model diagnostics and validation metrics.

The forecasting module does not generate predictions when essential data is unavailable.

Required data:

| Field | Example columns |
|---|---|
| Product identifier | product_id, product_name, sku |
| Transaction date | order_date, transaction_date, date |
| Quantity sold | quantity, qty, units_sold |

At least eight valid weekly periods are required by the current baseline implementation.

More historical data, preferably 6–12 months, is recommended for better analysis.

When data is insufficient, the application explains:
- Which columns are missing.
- Which column names are accepted.
- How many weekly periods are available.
- How many periods are required.

### 6. Inventory Optimization

Generate product-level inventory recommendations based on forecasted demand.

The current calculation includes:

- Current stock.
- Forecasted daily demand.
- Supplier lead time.
- Safety stock.
- Reorder point.
- Suggested order quantity.

The calculation uses a configurable service-level Z-score.

Required data:

| Field | Example columns |
|---|---|
| Product identifier | product_id, product_name, sku |
| Current stock | current_stock, available_stock |
| Transaction date | order_date, transaction_date |
| Quantity sold | quantity, units_sold |
| Supplier lead time | lead_time_days |

A default supplier lead time can also be entered manually.

The application does not invent stock quantities or supplier lead times.

---

## System Architecture

```text
                 USER
                   |
                   v
          STREAMLIT DASHBOARD
                   |
          Upload / Ask Questions
                   |
                   v
          DATA PROCESSING LAYER
                   |
          Pandas + Data Cleaning
                   |
                   v
              SQLITE DB
                   |
          +--------+---------+
          |                  |
          v                  v
    AI ANALYTICS        FORECASTING
          |                  |
          v                  v
     LANGGRAPH         DEMAND FORECAST
          |                  |
          v                  v
     GEMINI API        INVENTORY ENGINE
          |                  |
          v                  v
     SQL VALIDATION    REORDER SUGGESTIONS
          |
          v
     SQL EXECUTION
          |
          v
     BUSINESS INSIGHTS
          |
          v
       DASHBOARD
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing and analysis |
| SQLite | Local analytical database |
| Streamlit | Interactive dashboard |
| FastAPI | Backend REST API |
| LangGraph | AI Agent workflow orchestration |
| Google Gen AI SDK | Gemini API client |
| Gemini 3.5 Flash-Lite | Default language model |
| Statsmodels | Holt and Holt-Winters forecasting |
| OpenPyXL / xlrd | `.xlsx` / legacy `.xls` files |
| Python-dotenv | Environment configuration |

### Gemini API

The application sends prompts to Google's Gemini API using:

- SDK: `google-genai`
- Default model: `gemini-3.5-flash-lite` (configurable with `GEMINI_MODEL`)
- API key: `GEMINI_API_KEY`, read by the backend from `.env`

Gemini inference requires internet access. Dataset schema, the user's question,
and SQL query results used to create insights are sent to Google for inference;
avoid submitting confidential or regulated data unless permitted by your
organization's data-handling policies. Gemini API quotas and pricing depend on
your account, model, and usage. Review Google's current [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
and quotas before use.

---

## Project Structure

```text
ecommerce_ai_analyst/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── sql_validator.py
│   │   └── graph.py
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── forecasting.py
│   │   └── inventory.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── db_manager.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── schema_analyzer.py
│       └── schema_detection.py
│
├── data/
│   └── uploads/              # Original uploaded files (UUID filenames)
│
├── tests/
│
├── api.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Docker Compose Deployment

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine (Linux).
- Docker Compose v2 (`docker compose`).

The Compose setup builds one Python image and runs it as two services:
`api` (FastAPI) and `streamlit` (the dashboard). The Streamlit container calls
FastAPI at `http://api:8000` over the Compose network; `localhost` inside a
container would refer to that same container, not the other service.

### Configure and start

Create a local `.env` file and set your Gemini key. The file is not copied into
the image or committed:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Linux / macOS:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_new_key_here` with your own Gemini API key.
Docker Compose passes it only to the API service. A key is required for
Gemini-backed questions; upload, dashboard, forecasting, and inventory
features can start without one.

Build and launch both services:

```bash
docker compose build
docker compose up -d
docker compose ps
```

Compose waits for the FastAPI health check before starting Streamlit. Open:

- Streamlit dashboard: <http://localhost:8501>
- FastAPI root/health endpoint: <http://localhost:8000/>
- FastAPI interactive docs: <http://localhost:8000/docs>

The host ports can be changed with `STREAMLIT_PORT` and `API_PORT` in `.env`.
The internal service ports stay at 8501 and 8000, and the dashboard's
`API_BASE_URL` is fixed to the Compose service address `http://api:8000`.

### Logs, rebuild, and stop

```bash
docker compose logs -f
docker compose logs -f api
docker compose logs -f streamlit
docker compose up -d --build
docker compose down
```

`docker compose down` stops and removes the containers and network but keeps
the persistent data volume. To remove the database and uploaded files as well:

```bash
docker compose down -v
```

**Warning:** `down -v` permanently deletes the Compose-managed database and
uploaded originals. Do not use it if you want to keep your data.

### Persistent data and configuration

Both services mount the same named volume, `ecommerce-analyst-data`, at
`/data`. The SQLite database is `/data/ecommerce.db` and original uploads are
stored in `/data/uploads`. Uploads from the dashboard are sent to FastAPI,
which stores the file and atomically replaces the processed SQLite table. Since
both containers use the same volume and paths, the dashboard sees the database
created by the API rather than a separate SQLite copy.

Compose overrides the local-development `DATA_DIR`, `DB_PATH`, `UPLOAD_DIR`,
and `API_BASE_URL` values with container paths and service DNS. The Gemini
credential is supplied as a runtime environment variable and is excluded from
the Docker build context by `.dockerignore`.

This project remains a single-active-dataset educational application. The
FastAPI schema state is process-wide and both services share one database;
Compose does not make concurrent users or uploads isolated.

---

## Installation & Setup

### Prerequisites

- Python 3.10 or newer.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
- Git (optional).

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd ecommerce_ai_analyst
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey).
Copy `.env.example` to `.env` in the project root, then set `GEMINI_API_KEY` to
your new key. Keep `.env` private; it is excluded from Git.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Linux / macOS:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder:

```env
GEMINI_API_KEY=your_new_key_here
```

Optional configuration includes the Gemini model, request timeout, retry count,
retry delay, database path, and upload directory; the defaults are listed in
`.env.example`.

### 5. Start the API and Streamlit dashboard

Start FastAPI in the first terminal:

```bash
uvicorn api:app --reload
```

In a second terminal, start Streamlit:

```bash
streamlit run streamlit_app.py
```

The dashboard uses `API_BASE_URL` from `.env` (default
`http://127.0.0.1:8000`) for upload, schema mapping, and AI requests. It will
normally be available at:

http://localhost:8501

Internet access is required for Gemini inference. No local LLM runtime is needed.

---

## REST API Reference

API base URL:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

### Available API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API health message |
| POST | `/upload-data` | Upload and process a dataset |
| POST | `/schema-mapping` | Confirm semantic column mappings |
| POST | `/ask-agent` | Ask a natural-language business question |
| GET | `/forecast-readiness` | Check forecasting data requirements |
| GET | `/forecast` | Generate demand forecasts |
| POST | `/inventory` | Generate inventory recommendations |

### Example: Ask the AI Agent

Request:

```json
{
  "question": "Which are the top 5 products by revenue?"
}
```

Endpoint:

```text
POST /ask-agent
```

---

## Dataset Requirements

The platform works with E-commerce transaction datasets.

A typical dataset may contain:

| Column | Description |
|---|---|
| order_id | Unique order identifier |
| product_id | Product identifier |
| product_name | Product name |
| customer_id | Customer identifier |
| order_date | Transaction date |
| quantity | Units sold |
| unit_price | Price per unit |
| revenue | Transaction revenue |
| current_stock | Current available inventory |
| lead_time_days | Supplier delivery lead time |

Not every feature requires all these columns.

The available KPIs and analytics depend on the actual dataset schema.

### Dynamic Schema Detection

The schema analyzer is not tied to one vendor's header names. It evaluates
normalized aliases alongside bounded data profiles (data type, null rate,
cardinality, numeric/date parse rates, ranges, and representative values).
Profiles are sampled for classification while null counts and cardinality are
computed across the full column. The Streamlit dashboard shows the detected
semantic role, original uploaded header, confidence, evidence, and plausible
alternatives. Use **Review detected fields** in the sidebar to confirm, replace,
or leave a role unmapped. Applied mappings update downstream analysis and clear
previous forecast and inventory results.

Automatic mappings below a 0.65 score are not selected. Scores of at least 0.85
are labeled high confidence; scores from 0.65 through 0.8499 are medium
confidence. Similar candidate scores, duplicate headers, or one column matching
multiple roles are reported as ambiguous instead of being silently assigned.
These scores are evidence indicators, not statistical probabilities.

The roles distinguish product identifiers from product names, sales quantity
from current stock, unit price from revenue, and order IDs from product IDs.
An absent stock field remains absent; historical quantity is never treated as
on-hand inventory. Revenue is derived as `quantity * unit_price` only when no
confident revenue column exists. Profit is derived as `revenue - cost` only
when no confident profit column exists. The expressions and assumptions are
included in the schema profile; discounts, taxes, and shipping are not
automatically applied to derived revenue.

Readiness is evaluated independently for forecasting, inventory, sales, and
customer analysis. Forecast readiness additionally requires at least eight
observed weekly periods for a product; missing weeks are not counted as
observations by this assessment. Inventory readiness requires detected
non-negative stock and a positive lead time (or a positive default explicitly
provided by the caller). Sales readiness requires valid transaction dates and
usable quantity or monetary values. Customer readiness requires customer IDs
and usable transaction data.

Example mapping excerpt:

```json
{
  "field_mappings": {
    "product_id": {
      "column": "stock_code",
      "original_column": "StockCode",
      "confidence": "High confidence",
      "detection_method": "exact semantic alias + data profile",
      "ambiguous": false
    },
    "stock": {
      "column": null,
      "confidence": "Low confidence",
      "reason": "No matching column with sufficient data evidence."
    }
  },
  "readiness": {
    "forecasting": {"ready": true},
    "inventory": {
      "ready": false,
      "missing_fields": ["Inventory stock quantity"]
    }
  }
}
```

For example, `StockCode`, `InvoiceDate`, `Quantity`, and `UnitPrice` can be
recognized as product ID, transaction date, quantity sold, and unit price.
`SKU`, `OrderDate`, `Qty`, and `Price` follow a different naming convention but
map to the same roles. A separate `stock_quantity` or `available_stock` field is
required for inventory recommendations.

### Recommended Data Quality

For better results:
- Use consistent product identifiers.
- Use valid transaction dates.
- Store quantities as numeric values.
- Avoid duplicate transaction records.
- Keep stock values separate from historical sales transactions.
- Identify stockout periods when possible.
- Use consistent currency and measurement units.

---

## Forecasting Methodology

`DemandForecaster` compares four candidate models separately for each product:

- **8-week moving average:** baseline and fallback.
- **Holt linear:** considered when the training window shows a clear trend
  (at least 8 weeks, 1% relative slope, and absolute correlation of at least 0.6).
- **Croston SBA:** considered only for intermittent demand: at least 40% zero
  weeks, at least two positive weeks, and a mean interval of at least 1.5 weeks.
- **Holt-Winters additive ETS:** considered only with at least two observed
  52-week cycles in a training window.

Selection uses up to three chronological expanding-window backtests. Each
validation window spans the forecast horizon (4 weeks by default); each model
is fitted using only the preceding weeks. MAE and WAPE are computed only over
validation weeks with actual recorded observations. WAPE is `null` when actual
demand sums to zero but prediction error is nonzero; if both are zero, WAPE is
0. The selector ranks by WAPE, then MAE. If WAPE is undefined for all candidates,
it ranks by MAE. Exact ties prefer the simpler models in this order: moving
average, Holt, Croston SBA, then seasonal ETS.

Missing calendar weeks default to zero demand for model training, preserving the
original behavior. Set `DemandForecaster(missing_week_policy="interpolate")` to
interpolate interior gaps and carry the last observed value across a trailing
training gap. An imputed period is never treated as validation ground truth.
Without an explicit data-coverage signal, an absent transaction week cannot be
distinguished from a true zero-demand week; choose the policy that fits the
dataset.

Every product's `model_diagnostics` reports model eligibility, validation
scores, selection reason, fallback use, and missing-week counts. When no full
chronological validation horizon exists, the moving average is used and its
validation metrics are `null`; this is not presented as a validated result.
Forecasts are nonnegative estimates, not guarantees. They do not model
promotions, stockouts, price changes, or external market factors.

Example forecast record when no full backtest horizon is available:

```json
{
  "product": "SKU-001",
  "forecast_week": 1,
  "forecast_date": "2025-01-06",
  "forecast_quantity": 12.5,
  "selected_model": "8_week_moving_average",
  "validation_mae": null,
  "validation_wape": null
}
```

Seasonal evaluation needs at least two years of observed weekly demand in a
training window, so it will be unavailable for shorter datasets.

---

## Inventory Optimization Methodology

The inventory module uses forecasted demand and supplier lead time.

### Daily Demand

```text
Daily Demand = Weekly Forecast / 7
```

### Safety Stock

```text
Safety Stock =
Z × Daily Demand Standard Deviation × √Lead Time
```

### Reorder Point

```text
Reorder Point =
Daily Demand × Lead Time + Safety Stock
```

### Target Stock

```text
Target Stock =
Daily Demand × (Lead Time + 7) + Safety Stock
```

### Suggested Order Quantity

```text
Suggested Order Quantity =
MAX(0, Target Stock - Current Stock)
```

The current implementation does not account for:
- Open purchase orders.
- Minimum order quantities.
- Supplier capacity constraints.
- Returns.
- Explicit stockout periods.
- Multiple warehouses.

These should be considered before using recommendations for actual purchasing decisions.

---

## Security & Data Handling

- Gemini API keys are loaded by the backend from `.env`; do not put them in
  frontend code, API responses, logs, or source control.
- Prompts and query results are sent to Google's Gemini API for inference.
- SQL queries are restricted to SELECT statements.
- SQLite is opened in read-only mode for AI-generated queries.
- A SQLite authorizer provides an additional query execution restriction.
- Uploaded datasets are stored in the local SQLite database.
- Original uploads are retained separately in `UPLOAD_DIR` (default
  `data/uploads/`) with generated filenames. The cleaned dataset replaces the
  `ecommerce_data` table in `DB_PATH` (default `data/ecommerce.db`).
- A failed upload is removed and the prior SQLite table is retained. The table
  swap is transactional.
- Internet access and a valid Gemini API key are required for LLM features.
- API quotas, rate limits, and possible usage costs are controlled by Google.

The FastAPI process keeps `current_schema` and the active database table in
process-wide state. Streamlit sessions also share the configured SQLite file.
Each successful upload replaces the one active dataset. This is a
single-user educational design; simultaneous users or concurrent uploads can
observe each other's dataset. A session-scoped/multi-tenant storage redesign is
intentionally out of scope.

The current implementation is intended for local development and demonstration.
Additional authentication, authorization, upload limits, isolation, and
security testing are required before public deployment.

Do not upload confidential business data to an unsecured deployment.

---

## Current Limitations

The current version is a working baseline, with the following limitations:

1. Forecast model selection depends on data history, coverage, and the chosen missing-week policy.
2. Seasonal ETS requires at least two years of observed weekly history for training.
3. Inventory calculations use the latest dated stock/lead-time record and skip products with invalid latest values.
4. Supplier lead time may be entered as a shared default.
5. The application currently processes one active dataset at a time and is not safe for concurrent users.
6. AI-generated SQL and insights require validation before use in critical business decisions.

---

## Future Improvements

Potential next development steps:

- Advanced forecasting models.
- Seasonal demand detection.
- Promotion and price-impact analysis.
- Stockout-aware demand estimation.
- Customer segmentation.
- RFM analysis.
- Product recommendation engine.
- Profit margin and profitability analysis.
- Automated anomaly detection.
- Multi-dataset support.
- User authentication and role management.
- Exportable PDF and Excel reports.
- Interactive Plotly dashboards.
- Automated inventory alerts.
- Model evaluation and forecasting accuracy metrics.

---

## Project Goals

This project demonstrates the integration of:

- Data Engineering fundamentals.
- Exploratory Data Analysis.
- Business Intelligence.
- Natural Language Processing.
- Large Language Models.
- AI Agent workflows.
- Text-to-SQL.
- Time-Series Forecasting fundamentals.
- Inventory Optimization.
- Backend API development.
- Interactive dashboard development.

---

## License

Add your preferred open-source license before public distribution.

---

## Author

Developed as an E-commerce AI Analytics project.

**Technologies:** Python | Streamlit | FastAPI | LangGraph | Gemini API | SQLite
