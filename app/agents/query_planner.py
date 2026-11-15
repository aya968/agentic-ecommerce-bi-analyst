import json
import re

from app.utils.schema_detection import ROLE_LABELS


ROLE_ALIASES = {
    "customer": "customer_id",
    "customer_identifier": "customer_id",
    "client": "customer_id",
    "client_id": "customer_id",
    "transaction_date": "date",
    "order_date": "date",
    "invoice_date": "date",
    "quantity_sold": "quantity",
    "units_sold": "quantity",
    "unit_price": "price",
    "sales_amount": "revenue",
    "product": "product_id",
    "item": "product_id",
    "country": "country_region"
}

PLAN_SYSTEM_PROMPT = """
You are an analytical query planner. Convert the business question into a
concise JSON plan using the supplied live SQLite schema and semantic mappings.

Return one JSON object with exactly these keys:
{
  "intent": "short description",
  "entities": ["semantic entity names"],
  "metrics": ["semantic metric names"],
  "dimensions": ["semantic roles or dimensions"],
  "filters": ["requested conditions"],
  "time_granularity": "day|week|month|quarter|year|null",
  "required_concepts": ["role keys required from the semantic mapping"],
  "supported": true,
  "unsupported_reason": null
}

Use only role keys listed in the supplied semantic mapping for
required_concepts. Interpret equivalent headers through the mappings, not
literal header spelling. A metric such as revenue is available when the schema
documents a safe derived expression and its source concepts are available.
Do not declare a request unsupported merely because its physical column names
differ from familiar names. Set supported=false only for non-analytical or
unsupported requests; if required data is absent, list the missing role key(s)
in required_concepts and explain that in unsupported_reason. Do not write SQL.
Return JSON only.
"""

_CONCEPT_ALIASES = {
    **{role: role for role in ROLE_LABELS},
    **ROLE_ALIASES
}


def _column_names(database_schema):
    names = set()
    for table in database_schema.get("tables", []):
        names.update(
            column.get("name")
            for column in table.get("columns", [])
            if column.get("name")
        )
    return names


def _mapped_fields(schema_profile):
    mappings = schema_profile.get("field_mappings", {})
    legacy = {
        "product_id": (
            schema_profile.get("product_id")
            or (
                schema_profile.get("product")
                if schema_profile.get("product_grouping_source")
                != "product_name"
                else None
            )
        ),
        "product_name": schema_profile.get("product_name"),
        "date": schema_profile.get("date"),
        "quantity": schema_profile.get("quantity"),
        "price": schema_profile.get("price") or schema_profile.get("unit_price"),
        "revenue": schema_profile.get("revenue"),
        "order_id": schema_profile.get("order_id"),
        "customer_id": schema_profile.get("customer_id"),
        "stock": schema_profile.get("stock"),
        "lead_time": schema_profile.get("lead_time"),
        "supplier_id": schema_profile.get("supplier_id"),
        "category": schema_profile.get("category"),
        "discount": schema_profile.get("discount"),
        "cost": schema_profile.get("cost"),
        "profit": schema_profile.get("profit"),
        "country_region": schema_profile.get("country_region")
    }

    fields = {}
    for role in ROLE_LABELS:
        mapping = mappings.get(role, {})
        column = mapping.get("column")
        if column is None and not mapping:
            column = legacy[role]
        fields[role] = {
            "column": column,
            "ambiguous": bool(mapping.get("ambiguous", False)),
            "confidence": mapping.get("confidence"),
            "reason": mapping.get("reason"),
            "candidates": [
                candidate.get("column")
                for candidate in mapping.get("candidates", [])
                if candidate.get("column")
            ]
        }

    if fields["product_id"]["column"] or fields["product_name"]["column"]:
        product_column = (
            fields["product_id"]["column"]
            or fields["product_name"]["column"]
        )
        fields["product"] = {
            "column": product_column,
            "ambiguous": (
                fields["product_id"]["ambiguous"]
                and fields["product_name"]["ambiguous"]
            ),
            "confidence": None,
            "reason": "Legacy product grouping role."
        }

    revenue_definition = schema_profile.get(
        "derived_metrics",
        {}
    ).get("revenue", {})
    if revenue_definition.get("available"):
        fields["revenue"]["derived"] = {
            "source": revenue_definition.get("source"),
            "expression": revenue_definition.get("expression"),
            "calculation": revenue_definition.get("calculation")
        }

    database_schema = schema_profile.get("database_schema")
    if database_schema:
        actual_columns = _column_names(database_schema)
        for field in fields.values():
            column = field["column"]
            if column and column not in actual_columns:
                field["column"] = None
                field["reason"] = (
                    "Detected mapping does not exist in the active database."
                )
            field["candidates"] = [
                candidate
                for candidate in field.get("candidates", [])
                if candidate in actual_columns
            ]

    return fields


def build_schema_context(schema_profile):
    """Build a compact, live-schema-grounded context without sample values."""
    database_schema = schema_profile.get("database_schema", {})
    if not database_schema:
        columns = []
        for profile in schema_profile.get("column_profiles", []):
            columns.append({
                "name": profile.get("column"),
                "type": profile.get("dtype", profile.get("data_type"))
            })
        if not columns:
            columns = [
                {"name": column, "type": "unknown"}
                for column in schema_profile.get("all_columns", [])
            ]
        database_schema = {
            "dialect": "SQLite",
            "tables": [{
                "name": schema_profile.get("table_name", "ecommerce_data"),
                "columns": columns
            }]
        }

    return {
        "database": database_schema,
        "semantic_mappings": _mapped_fields(schema_profile),
        "derived_metrics": {
            metric: definition
            for metric, definition in schema_profile.get(
                "derived_metrics",
                {}
            ).items()
            if definition.get("available")
        }
    }


def query_plan_prompt(question, schema_context):
    return (
        "Live database schema and semantic mappings:\n"
        + json.dumps(schema_context, ensure_ascii=False, indent=2)
        + "\n\nBusiness question:\n"
        + question
        + "\n\nCreate the requested JSON plan."
    )


def parse_query_plan(response):
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("The query planner returned invalid JSON.")
        text = text[start:end + 1]

    plan = json.loads(text)
    if not isinstance(plan, dict):
        raise ValueError("The query planner response must be a JSON object.")

    required_keys = {
        "intent",
        "entities",
        "metrics",
        "dimensions",
        "filters",
        "time_granularity",
        "required_concepts",
        "supported",
        "unsupported_reason"
    }
    if not required_keys.issubset(plan):
        raise ValueError("The query planner response is missing required fields.")
    if not isinstance(plan["supported"], bool):
        raise ValueError("The query planner's supported flag must be boolean.")
    if not isinstance(plan["required_concepts"], list):
        raise ValueError("The planner must list required semantic concepts.")

    normalized_requirements = []
    for concept in plan["required_concepts"]:
        if not isinstance(concept, str):
            raise ValueError("Required semantic concepts must be strings.")
        normalized = _CONCEPT_ALIASES.get(concept.strip().lower())
        if normalized is None:
            raise ValueError(
                f"The planner returned an unknown semantic role: {concept}."
            )
        if normalized not in normalized_requirements:
            normalized_requirements.append(normalized)
    plan["required_concepts"] = normalized_requirements

    for key in ("entities", "metrics", "dimensions", "filters"):
        if not isinstance(plan[key], list):
            raise ValueError(f"The planner field '{key}' must be a list.")
    return plan


def check_plan_support(plan, schema_context):
    """Resolve planned roles against verified physical columns and mappings."""
    mappings = schema_context["semantic_mappings"]
    missing = []
    ambiguous = []
    for role in plan["required_concepts"]:
        mapping = mappings.get(role, {})
        if mapping.get("ambiguous") and not mapping.get("derived"):
            ambiguous.append(role)
        elif not mapping.get("column") and not mapping.get("derived"):
            missing.append(role)

    return {
        **plan,
        "missing_concepts": missing,
        "ambiguous_concepts": ambiguous,
        "supported": (
            plan["supported"]
            and not missing
            and not ambiguous
        )
    }


def semantic_role_label(role):
    return ROLE_LABELS.get(role, role.replace("_", " ").title())
