import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app.database.db_manager import DatabaseManager
from app.utils.schema_analyzer import SchemaAnalyzer
from app.analytics.forecasting import DemandForecaster
from app.analytics.inventory import InventoryOptimizer
from app.agents.graph import ask_agent


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="E-commerce AI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ELEGANT DARK THEME (NO HARSH BLACKS OR WHITE FLASHES)
# =========================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

/* Main application background */
.stApp {
    background-color: #0F172A;
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Typography */
h1, h2, h3 {
    color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important;
}

h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    padding-bottom: 10px;
    border-bottom: 1px solid #334155;
}

h3 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}

p, label {
    color: #94A3B8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E293B;
    border-right: 1px solid #334155;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}

/* Cards & Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}

/* Metrics */
div[data-testid="stMetric"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
    font-weight: 800 !important;
}

/* Buttons */
.stButton > button {
    background-color: #0D9488;
    color: #FFFFFF !important;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1.1rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #14B8A6;
    color: #FFFFFF !important;
    box-shadow: 0 0 12px rgba(20, 184, 166, 0.3);
}

/* Inputs */
.stTextInput input,
.stNumberInput input {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 9px !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #14B8A6 !important;
    box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.25) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1E293B;
    border: 1px dashed #475569;
    border-radius: 12px;
    padding: 12px;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #334155 !important;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Captions */
.stCaption {
    color: #64748B !important;
}

/* Remove Streamlit footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# INITIALIZE APPLICATION
# =========================================================

db_manager = DatabaseManager()
schema_analyzer = SchemaAnalyzer()
forecaster = DemandForecaster()
inventory_optimizer = InventoryOptimizer()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# =========================================================
# SESSION STATE
# =========================================================

def initialize_session():

    defaults = {
        "uploaded": False,
        "schema": None,
        "cleaning_report": None,
        "dataframe": None,
        "forecast_result": None,
        "dataset_name": None,
        "inventory_result": None
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_numeric_series(df, column):

    if not column or column not in df.columns:
        return None

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def format_number(value):

    if pd.isna(value):
        return "—"

    return f"{value:,.2f}"


def show_missing_data(readiness):

    st.warning(
        readiness.get(
            "message",
            "Forecasting is not available for this dataset."
        )
    )

    missing_fields = readiness.get(
        "missing_fields",
        []
    )

    if missing_fields:

        st.markdown("#### Required Data")

        for field in missing_fields:

            if isinstance(field, dict):

                field_name = field.get(
                    "field",
                    "Unknown field"
                )

                examples = field.get(
                    "accepted_examples",
                    []
                )

                st.markdown(f"**{field_name}**")

                if examples:
                    st.caption(
                        "Accepted column names: "
                        + ", ".join(examples)
                    )

            else:
                st.write(f"- {field}")

    available_weeks = readiness.get("available_weeks")

    required_weeks = readiness.get("required_weeks")

    if available_weeks is not None:

        col1, col2 = st.columns(2)

        col1.metric(
            "Available Weeks",
            available_weeks
        )

        col2.metric(
            "Required Weeks",
            required_weeks or 8
        )

    st.info(
        """
        To enable reliable demand forecasting, provide:
        - A product identifier.
        - A transaction date.
        - A numeric quantity sold column.
        - At least 8 valid weekly periods.

        Preferably provide 6–12 months of historical sales data.
        """
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📊 E-commerce AI Analyst")

    st.caption("Intelligent Business Analytics Platform")

    st.divider()

    st.markdown("### Dataset Management")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload an E-commerce CSV or Excel file."
    )

    if uploaded_file is not None:

        st.caption(
            f"Selected file: {uploaded_file.name}"
        )

        if st.button(
            "Process Dataset",
            type="primary",
            use_container_width=True
        ):

            try:

                extension = Path(
                    uploaded_file.name
                ).suffix.lower()

                temp_path = DATA_DIR / (
                    "uploaded_ecommerce" + extension
                )

                with open(temp_path, "wb") as file:

                    file.write(
                        uploaded_file.getvalue()
                    )

                with st.spinner(
                    "Cleaning and analyzing your dataset..."
                ):

                    report = db_manager.load_file_to_sqlite(
                        str(temp_path),
                        table_name="ecommerce_data"
                    )

                    df = db_manager.get_dataframe()

                    schema = schema_analyzer.analyze(df)

                    st.session_state.uploaded = True

                    st.session_state.dataframe = df

                    st.session_state.schema = schema

                    st.session_state.cleaning_report = report

                    st.session_state.dataset_name = uploaded_file.name

                    st.session_state.forecast_result = None

                    st.session_state.inventory_result = None

                st.success(
                    "Dataset processed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Dataset processing failed: {error}"
                )

    st.divider()

    if st.session_state.uploaded:

        st.markdown("### Current Dataset")

        st.success(
            st.session_state.dataset_name
        )

        if st.button(
            "Clear Session",
            use_container_width=True
        ):

            for key in [
                "uploaded",
                "schema",
                "cleaning_report",
                "dataframe",
                "forecast_result",
                "dataset_name",
                "inventory_result"
            ]:

                st.session_state.pop(key, None)

            st.rerun()

    st.divider()

    st.caption(
        "Powered by Python · Streamlit · LangGraph · Ollama"
    )


# =========================================================
# MAIN HEADER
# =========================================================

with st.container(border=True):

    st.caption(
        "INTELLIGENT E-COMMERCE ANALYTICS"
    )

    st.title(
        "E-commerce AI Analyst"
    )

    st.markdown(
        """
        Turn your E-commerce data into business insights.

        Analyze sales, understand product performance,
        forecast demand, and make smarter inventory decisions.
        """
    )

    st.caption(
        "Local AI powered by Ollama and Qwen 2.5 Coder"
    )


# =========================================================
# EMPTY STATE
# =========================================================

if not st.session_state.uploaded:

    st.divider()

    st.markdown("## Welcome to your analytics workspace")

    st.info(
        "Upload an E-commerce CSV or Excel dataset "
        "using the sidebar to get started."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.markdown("### 📈 Business Analytics")

            st.write(
                "Understand revenue, orders, products "
                "and sales performance."
            )

    with col2:

        with st.container(border=True):

            st.markdown("### 🤖 AI Analyst")

            st.write(
                "Ask questions about your business "
                "using natural language."
            )

    with col3:

        with st.container(border=True):

            st.markdown("### 🔮 Demand Forecasting")

            st.write(
                "Predict future demand and optimize "
                "your inventory decisions."
            )

    st.stop()


# =========================================================
# LOAD ACTIVE DATASET
# =========================================================

df = st.session_state.dataframe

schema = st.session_state.schema

report = st.session_state.cleaning_report


# =========================================================
# 1. DATASET OVERVIEW
# =========================================================

st.divider()

st.header("1. Dataset Overview")

st.caption(
    "A summary of your uploaded E-commerce dataset."
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    with st.container(border=True):

        st.metric(
            "Total Rows",
            f"{len(df):,}"
        )

with col2:

    with st.container(border=True):

        st.metric(
            "Total Columns",
            f"{len(df.columns):,}"
        )

with col3:

    with st.container(border=True):

        st.metric(
            "Duplicates Removed",
            f"{report.get('duplicates_removed', 0):,}"
        )

with col4:

    with st.container(border=True):

        st.metric(
            "Removed Rows",
            f"{report.get('removed_rows', 0):,}"
        )


with st.expander("View Dataset Preview"):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


with st.expander("View Detected Schema"):

    st.json({
        key: value
        for key, value in schema.items()
        if key != "all_columns"
    })


# =========================================================
# 2. BUSINESS KPIS
# =========================================================

st.divider()

st.header("2. Business KPIs")

st.caption(
    "Key performance indicators calculated from your dataset."
)

revenue_col = schema.get("revenue")

quantity_col = schema.get("quantity")

price_col = schema.get("price")

order_col = schema.get("order_id")

product_col = schema.get("product")

customer_col = schema.get("customer_id")


# Revenue

revenue = None

if revenue_col and revenue_col in df.columns:

    revenue_series = get_numeric_series(
        df,
        revenue_col
    )

    revenue = revenue_series.sum()

elif (
    quantity_col
    and price_col
    and quantity_col in df.columns
    and price_col in df.columns
):

    quantity_series = get_numeric_series(
        df,
        quantity_col
    )

    price_series = get_numeric_series(
        df,
        price_col
    )

    revenue = (
        quantity_series * price_series
    ).sum()


# Orders

total_orders = None

if order_col and order_col in df.columns:

    total_orders = df[order_col].nunique()


# Quantity

total_quantity = None

if quantity_col and quantity_col in df.columns:

    total_quantity = get_numeric_series(
        df,
        quantity_col
    ).sum()


# Products

total_products = None

if product_col and product_col in df.columns:

    total_products = df[product_col].nunique()


# Display KPIs

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    with st.container(border=True):

        st.metric(
            "Total Revenue",
            format_number(revenue)
            if revenue is not None
            else "—"
        )

with kpi2:

    with st.container(border=True):

        st.metric(
            "Total Orders",
            f"{total_orders:,}"
            if total_orders is not None
            else "—"
        )

with kpi3:

    with st.container(border=True):

        st.metric(
            "Units Sold",
            f"{total_quantity:,.0f}"
            if total_quantity is not None
            else "—"
        )

with kpi4:

    with st.container(border=True):

        st.metric(
            "Products",
            f"{total_products:,}"
            if total_products is not None
            else "—"
        )


st.caption(
    "Metrics are calculated according to the available dataset columns."
)


# =========================================================
# 3. SALES ANALYSIS
# =========================================================

st.divider()

st.header("3. Sales Analysis")

st.caption(
    "Explore sales trends and product performance."
)

date_col = schema.get("date")


# Monthly Revenue / Quantity

if date_col and date_col in df.columns:

    chart_df = df.copy()

    chart_df[date_col] = pd.to_datetime(
        chart_df[date_col],
        errors="coerce"
    )

    chart_df = chart_df.dropna(
        subset=[date_col]
    )

    if revenue_col and revenue_col in chart_df.columns:

        chart_df["_metric"] = get_numeric_series(
            chart_df,
            revenue_col
        )

        metric_name = "Revenue"

    elif (
        quantity_col
        and quantity_col in chart_df.columns
    ):

        chart_df["_metric"] = get_numeric_series(
            chart_df,
            quantity_col
        )

        metric_name = "Units Sold"

    elif (
        quantity_col
        and price_col
        and quantity_col in chart_df.columns
        and price_col in chart_df.columns
    ):

        chart_df["_metric"] = (
            get_numeric_series(chart_df, quantity_col)
            * get_numeric_series(chart_df, price_col)
        )

        metric_name = "Revenue"

    else:

        chart_df = pd.DataFrame()

        metric_name = None

    if not chart_df.empty and metric_name:

        chart_df["Month"] = (
            chart_df[date_col]
            .dt.to_period("M")
            .astype(str)
        )

        monthly_data = (
            chart_df.groupby("Month")["_metric"]
            .sum()
            .sort_index()
        )

        st.subheader(
            f"Monthly {metric_name}"
        )

        st.line_chart(
            monthly_data,
            use_container_width=True
        )

    else:

        st.warning(
            "A revenue or quantity column is required "
            "to display monthly performance."
        )

else:

    st.warning(
        "A valid transaction date column is required "
        "to display monthly trends."
    )


# Top products

if (
    product_col
    and quantity_col
    and product_col in df.columns
    and quantity_col in df.columns
):

    st.subheader(
        "Top Products by Quantity Sold"
    )

    product_df = df.copy()

    product_df["_quantity"] = get_numeric_series(
        product_df,
        quantity_col
    )

    top_products = (
        product_df.groupby(product_col)["_quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(
        top_products,
        use_container_width=True
    )

else:

    st.info(
        "Product and quantity columns are required "
        "to display top-selling products."
    )


# =========================================================
# 4. AI BUSINESS ANALYST
# =========================================================

st.divider()

st.header("4. Ask Your AI Analyst")

st.caption(
    "Ask questions about your E-commerce business "
    "using natural language."
)

with st.container(border=True):

    st.markdown("#### 💬 Business Question")

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What are the top 5 products by revenue?"
        ),
        key="ai_question"
    )

    analyze_button = st.button(
        "Analyze Question",
        type="primary"
    )

    if analyze_button:

        if len(question.strip()) < 3:

            st.warning(
                "Please enter a valid business question."
            )

        else:

            with st.spinner(
                "Your AI analyst is analyzing the dataset..."
            ):

                try:

                    result = ask_agent(
                        question,
                        schema
                    )

                    if result.get("success"):

                        st.success(
                            "Analysis completed successfully."
                        )

                        st.markdown(
                            "### AI Business Insights"
                        )

                        st.write(
                            result.get("insight")
                        )

                        with st.expander(
                            "View Generated SQL"
                        ):

                            st.code(
                                result.get("sql") or "",
                                language="sql"
                            )

                        st.markdown(
                            "### Query Results"
                        )

                        result_data = result.get(
                            "data",
                            []
                        )

                        if result_data:

                            st.dataframe(
                                pd.DataFrame(result_data),
                                use_container_width=True
                            )

                        else:

                            st.info(
                                "The query returned no rows."
                            )

                    else:

                        st.error(
                            result.get("error")
                            or "The AI Agent could not complete the analysis."
                        )

                except Exception as error:

                    st.error(
                        f"AI Agent Error: {error}"
                    )


# =========================================================
# 5. DEMAND FORECASTING
# =========================================================

st.divider()

st.header("5. Demand Forecasting")

st.caption(
    "Forecast future product demand based on historical sales."
)

readiness = forecaster.check_readiness(
    df,
    schema
)

if not readiness.get("ready", False):

    show_missing_data(readiness)

else:

    st.success(
        readiness.get(
            "message",
            "Your dataset is ready for forecasting."
        )
    )

    if st.button(
        "Generate Demand Forecast",
        type="primary"
    ):

        with st.spinner(
            "Calculating product demand forecasts..."
        ):

            try:

                forecast_result = forecaster.forecast(
                    df,
                    schema
                )

                st.session_state.forecast_result = forecast_result

            except Exception as error:

                st.error(
                    f"Forecasting Error: {error}"
                )

    forecast_result = st.session_state.forecast_result

    if forecast_result:

        if forecast_result.get("ready"):

            st.markdown(
                "### Product Demand Forecast"
            )

            forecast_df = pd.DataFrame(
                forecast_result.get(
                    "forecasts",
                    []
                )
            )

            if not forecast_df.empty:

                st.dataframe(
                    forecast_df,
                    use_container_width=True
                )

            else:

                st.info(
                    "No forecast records were generated."
                )

            if forecast_result.get("warning"):

                st.info(
                    forecast_result["warning"]
                )

        else:

            show_missing_data(
                forecast_result
            )


# =========================================================
# 6. INVENTORY OPTIMIZATION
# =========================================================

st.divider()

st.header("6. Inventory Optimization")

st.caption(
    "Get inventory replenishment recommendations "
    "based on forecasted demand."
)

stock_col = schema.get("stock")

lead_time_col = schema.get("lead_time")


with st.container(border=True):

    st.markdown(
        "#### Supplier Lead Time"
    )

    st.write(
        "If supplier lead time is not available in your dataset, "
        "enter a default value."
    )

    default_lead_time = st.number_input(
        "Default lead time in days",
        min_value=0,
        value=0,
        step=1,
        key="default_lead_time"
    )


if not stock_col:

    st.warning(
        "Current stock information is missing. "
        "Add a column such as current_stock, available_stock "
        "or inventory."
    )

elif not lead_time_col and default_lead_time <= 0:

    st.warning(
        "Supplier lead time is missing. "
        "Add a lead_time_days column or enter a default value."
    )

elif not readiness.get("ready", False):

    st.warning(
        "Inventory optimization requires sufficient "
        "historical demand data."
    )

else:

    if st.button(
        "Optimize Inventory",
        type="primary"
    ):

        with st.spinner(
            "Calculating inventory recommendations..."
        ):

            try:

                forecast_result = forecaster.forecast(
                    df,
                    schema
                )

                if not forecast_result.get("ready"):

                    st.warning(
                        forecast_result.get("message")
                    )

                else:

                    inventory_result = (
                        inventory_optimizer.optimize_from_data(
                            df=df,
                            forecast_result=forecast_result,
                            schema=schema,
                            default_lead_time_days=(
                                default_lead_time
                                if default_lead_time > 0
                                else None
                            )
                        )
                    )

                    st.session_state.inventory_result = (
                        inventory_result
                    )

            except Exception as error:

                st.error(
                    f"Inventory Optimization Error: {error}"
                )


inventory_result = st.session_state.inventory_result

if inventory_result:

    if inventory_result.get("ready"):

        st.markdown(
            "### Inventory Recommendations"
        )

        recommendations = pd.DataFrame(
            inventory_result.get(
                "recommendations",
                []
            )
        )

        if not recommendations.empty:

            st.dataframe(
                recommendations,
                use_container_width=True
            )

        else:

            st.info(
                "No inventory recommendations were generated."
            )

        with st.expander(
            "View Inventory Calculation Assumptions"
        ):

            st.json(
                inventory_result.get(
                    "assumptions",
                    {}
                )
            )

    else:

        st.warning(
            inventory_result.get(
                "message",
                "Inventory optimization could not be completed."
            )
        )

        if inventory_result.get("missing_fields"):

            st.write(
                inventory_result["missing_fields"]
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 20px;
        color: #64748B;
        font-size: 12px;
    ">
        E-commerce AI Analyst<br>
        Intelligent Business Analytics Platform<br>
        Powered by Python, Streamlit, LangGraph and Ollama
    </div>
    """,
    unsafe_allow_html=True
)
