
# E-commerce AI Analyst
### AI-Powered Business Intelligence, Demand Forecasting & Inventory Optimization

An intelligent E-commerce analytics platform that transforms raw sales data into actionable business insights using a locally hosted Large Language Model (LLM).

The platform allows users to upload E-commerce datasets, explore business performance, ask questions in natural language, generate SQL queries automatically, forecast product demand, and receive inventory replenishment recommendations.

Built with Python, Streamlit, FastAPI, LangGraph, SQLite, and Ollama.

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
- Missing weekly buckets are filled with zero.
- Eight-week moving-average baseline.
- Four-week forecast horizon.
- Product-level data-readiness checks.

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
     OLLAMA LLM        INVENTORY ENGINE
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
| Ollama | Local LLM runtime |
| Qwen 2.5 Coder 3B | Local language model |
| Requests | Communication with Ollama |
| OpenPyXL | Excel file support |
| Python-dotenv | Environment configuration |

### Local AI Architecture

The application uses:

- Runtime: Ollama
- Model: qwen2.5-coder:3b
- Default endpoint: http://127.0.0.1:11434

No external Gemini or OpenAI API key is required.

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
│       └── schema_analyzer.py
│
├── data/
│
├── api.py
├── streamlit_app.py
├── requirements.txt
├── .env
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or newer.
- Ollama installed.
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

### 4. Download the Local LLM

```bash
ollama pull qwen2.5-coder:3b
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5-coder:3b
DB_PATH=data/ecommerce.db
```

### 6. Start Ollama

```bash
ollama serve
```

If Ollama is already running as a background service, this step is not necessary.

### 7. Launch the Streamlit Dashboard

Open another terminal:

```bash
streamlit run streamlit_app.py
```

The dashboard will normally be available at:

http://localhost:8501

---

## Running the REST API

Start FastAPI:

```bash
uvicorn api:app --reload
```

API base URL:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

### Available API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API health message |
| POST | `/upload-data` | Upload and process a dataset |
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

The current forecasting implementation uses a simple eight-week moving average.

For each product:

```text
Forecasted Weekly Demand =
Average Weekly Quantity Sold
Over the Last 8 Weeks
```

The system generates estimates for the next four weeks.

This baseline is designed to provide a transparent starting point.

It does not currently model:
- Annual seasonality.
- Holiday effects.
- Promotions.
- Price changes.
- External market factors.
- Explicit stockout-adjusted demand.

Forecasts should therefore be interpreted as estimates, not guaranteed future sales.

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

- The LLM runs locally through Ollama.
- SQL queries are restricted to SELECT statements.
- SQLite is opened in read-only mode for AI-generated queries.
- A SQLite authorizer provides an additional query execution restriction.
- Uploaded datasets are stored in the local SQLite database.
- No external LLM API key is required.

The current implementation is intended for local development and demonstration. Additional authentication, authorization, upload limits, isolation, and security testing are required before public deployment.

Do not upload confidential business data to an unsecured deployment.

---

## Current Limitations

The current version is a working baseline, with the following limitations:

1. Forecasting uses a moving-average approach rather than an advanced time-series model.
2. Forecast quality depends on historical data quality and completeness.
3. Missing weekly periods are currently treated as zero demand.
4. Inventory calculations assume the selected stock value represents the current product stock.
5. Supplier lead time may be entered as a shared default.
6. The application currently processes one active dataset at a time.
7. AI-generated SQL and insights require validation before use in critical business decisions.

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

**Technologies:** Python | Streamlit | FastAPI | LangGraph | Ollama | SQLite
