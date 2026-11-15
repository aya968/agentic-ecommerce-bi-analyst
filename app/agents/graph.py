import json
import sqlite3
import time

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langgraph.graph import StateGraph, END

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_DELAY_SECONDS
)

from app.agents.state import AgentState
from app.agents.prompts import (
    SQL_SYSTEM_PROMPT,
    INSIGHT_SYSTEM_PROMPT
)

from app.agents.sql_validator import SQLValidator


MAX_SQL_RETRIES = 2
MAX_SQL_ERROR_CONTEXT_LENGTH = 500


def query_llm(prompt, system_prompt="", temperature=0.1):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY in your .env file."
        )

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=GEMINI_TIMEOUT_SECONDS * 1000
        )
    )

    try:
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt or None,
                        temperature=temperature
                    )
                )
                break

            except genai_errors.APIError as error:
                status_code = error.code
                retryable = (
                    status_code == 429
                    or status_code in {500, 502, 503, 504}
                )
                if retryable and attempt < GEMINI_MAX_RETRIES:
                    time.sleep(GEMINI_RETRY_DELAY_SECONDS * (2 ** attempt))
                    continue

                if status_code in {401, 403}:
                    message = (
                        "Gemini access was denied. Check the API key, "
                        "API access, and configured model."
                    )
                elif status_code == 429:
                    message = (
                        "Gemini rate limit exceeded. Wait a moment and retry."
                    )
                elif status_code in {408, 504}:
                    message = "Gemini request timed out. Please try again."
                elif status_code >= 500:
                    message = (
                        "Gemini is temporarily unavailable. Please try again."
                    )
                else:
                    message = (
                        "Gemini request failed. Check your API configuration "
                        "and try again."
                    )
                raise RuntimeError(message) from None

            except httpx.TimeoutException:
                if attempt < GEMINI_MAX_RETRIES:
                    time.sleep(GEMINI_RETRY_DELAY_SECONDS * (2 ** attempt))
                    continue
                raise RuntimeError(
                    "Gemini request timed out. Please try again."
                ) from None

            except httpx.TransportError:
                if attempt < GEMINI_MAX_RETRIES:
                    time.sleep(GEMINI_RETRY_DELAY_SECONDS * (2 ** attempt))
                    continue
                raise RuntimeError(
                    "Could not connect to Gemini. Check your internet "
                    "connection and try again."
                ) from None

        text = response.text
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                "Gemini returned an empty or non-text response. Please retry."
            )
        return text.strip()
    finally:
        client.close()


def generate_sql_node(state: AgentState):
    schema = state.get("schema_description", "")
    correction_context = ""
    previous_sql = state.get("previous_sql")
    sql_error = state.get("sql_error")

    if previous_sql or sql_error:
        correction_context = f"""
Previous SQL that failed:
{previous_sql or "(no previous SQL was generated)"}

SQL execution error:
{sql_error or "(no execution error was recorded)"}

Correct this specific problem. Preserve the user's original intent and
return a corrected SQLite SELECT query, not an unrelated replacement.
This is correction attempt {state.get("retry_count", 0)} of
{MAX_SQL_RETRIES}.
"""

    prompt = f"""
Database schema:

{schema}

User question:
{state.get("question", "")}

{correction_context}

Generate one SQLite SELECT query.
Return SQL only.
"""

    try:
        generated_sql = query_llm(
            prompt,
            SQL_SYSTEM_PROMPT
        )

        if generated_sql.strip() == "CANNOT_ANSWER":
            return {
                "error": (
                    "The available dataset does not contain "
                    "the columns needed to answer this question."
                ),
                "success": False,
                "sql_ready": False,
                "generated_sql": None
            }

        return {
            "generated_sql": generated_sql,
            "error": None,
            "sql_error": None,
            "previous_sql": None,
            "sql_ready": True
        }

    except Exception:
        return {
            "error": (
                "SQL generation failed. Check Gemini configuration "
                "and try again."
            ),
            "generated_sql": None,
            "sql_ready": False,
            "success": False
        }


def route_after_generation(state: AgentState):
    return "execute_sql" if state.get("sql_ready") else END


def _sql_error_context(error):
    if isinstance(error, (ValueError, sqlite3.Error)):
        return str(error)[:MAX_SQL_ERROR_CONTEXT_LENGTH]
    return f"SQL execution failed ({type(error).__name__})."


def execute_sql_node(state: AgentState):
    try:
        database_path = state.get("db_path")
        if database_path is None:
            result = SQLValidator.execute_read_only(
                state["generated_sql"]
            )
        else:
            result = SQLValidator.execute_read_only(
                state["generated_sql"],
                db_path=database_path
            )

        return {
            "sql_result": result["rows"],
            "columns": result["columns"],
            "result_metadata": {
                "row_count": result["row_count"],
                "truncated": result["truncated"]
            },
            "error": None,
            "sql_error": None,
            "previous_sql": None,
            "success": True
        }

    except Exception as error:
        return {
            "error": (
                "Generated SQL could not be executed. "
                "Attempting a correction."
            ),
            "previous_sql": state.get("generated_sql"),
            "sql_error": _sql_error_context(error),
            "success": False,
            "retry_count": state.get("retry_count", 0) + 1
        }


def generate_insight_node(state: AgentState):
    prompt = f"""
User question:
{state.get("question")}

SQL query:
{state.get("generated_sql")}

Query result:
{json.dumps(state.get("sql_result", []), ensure_ascii=False, default=str)}

Result columns:
{state.get("columns", [])}

Explain the result as an E-commerce business analyst.
"""

    try:
        insight = query_llm(
            prompt,
            INSIGHT_SYSTEM_PROMPT,
            temperature=0.3
        )

        return {
            "insight": insight,
            "success": True
        }

    except Exception:
        return {
            "insight": "SQL query executed successfully, but insight generation failed.",
            "success": True
        }


def route_after_execution(state: AgentState):
    if state.get("success"):
        return "insight"

    if state.get("retry_count", 0) <= MAX_SQL_RETRIES:
        return "generate_sql"

    return END


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("generate_insight", generate_insight_node)

    workflow.set_entry_point("generate_sql")

    workflow.add_conditional_edges(
        "generate_sql",
        route_after_generation,
        {
            "execute_sql": "execute_sql",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "insight": "generate_insight",
            "generate_sql": "generate_sql",
            END: END
        }
    )

    workflow.add_edge("generate_insight", END)

    return workflow.compile()


# Build the agent once at module load time to avoid repeated compilation
agent = build_agent()


def ask_agent(question, schema_profile, db_path=None):
    schema_description = json.dumps(
        schema_profile,
        ensure_ascii=False,
        indent=2
    )

    initial_state = {
        "question": question,
        "schema_profile": schema_profile,
        "schema_description": schema_description,
        "retry_count": 0,
        "success": False,
        "error": None,
        "previous_sql": None,
        "sql_error": None,
        "sql_ready": False
    }
    if db_path is not None:
        initial_state["db_path"] = str(db_path)

    result = agent.invoke(initial_state)
    error = result.get("error")
    if (
        not result.get("success")
        and result.get("retry_count", 0) > MAX_SQL_RETRIES
    ):
        error = (
            "Unable to execute the generated SQL after "
            f"{MAX_SQL_RETRIES} correction attempts. "
            "Revise the question and try again."
        )

    return {
        "question": question,
        "sql": result.get("generated_sql"),
        "data": result.get("sql_result", []),
        "columns": result.get("columns", []),
        "insight": result.get("insight"),
        "error": error,
        "success": result.get("success", False)
    }
    