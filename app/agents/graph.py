import json
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

    prompt = f"""
Database schema:

{schema}

User question:
{state.get("question", "")}

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
                "success": False
            }

        return {
            "generated_sql": generated_sql,
            "error": None
        }

    except Exception as error:
        return {
            "error": str(error),
            "success": False
        }


def execute_sql_node(state: AgentState):
    try:
        result = SQLValidator.execute_read_only(
            state["generated_sql"]
        )

        return {
            "sql_result": result["rows"],
            "columns": result["columns"],
            "result_metadata": {
                "row_count": result["row_count"],
                "truncated": result["truncated"]
            },
            "error": None,
            "success": True
        }

    except Exception as error:
        return {
            "error": str(error),
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

    except Exception as error:
        return {
            "insight": (
                "SQL query executed successfully, but insight "
                f"generation failed: {error}"
            ),
            "success": True
        }


def route_after_execution(state: AgentState):
    if state.get("success"):
        return "insight"

    if state.get("retry_count", 0) <= 2:
        return "generate_sql"

    return END


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("generate_insight", generate_insight_node)

    workflow.set_entry_point("generate_sql")

    workflow.add_edge("generate_sql", "execute_sql")

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


def ask_agent(question, schema_profile):
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
        "error": None
    }

    result = agent.invoke(initial_state)

    return {
        "question": question,
        "sql": result.get("generated_sql"),
        "data": result.get("sql_result", []),
        "columns": result.get("columns", []),
        "insight": result.get("insight"),
        "error": result.get("error"),
        "success": result.get("success", False)
    }
    