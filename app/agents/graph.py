import requests
import json

from langgraph.graph import StateGraph, END

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT
)

from app.agents.state import AgentState
from app.agents.prompts import (
    SQL_SYSTEM_PROMPT,
    INSIGHT_SYSTEM_PROMPT
)

from app.agents.sql_validator import SQLValidator


def query_local_llm(prompt, system_prompt="", temperature=0.1):
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )

        response.raise_for_status()

        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Run: ollama run qwen2.5-coder:3b"
        )

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama request timed out."
        )

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Ollama request failed: {error}"
        )


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
        generated_sql = query_local_llm(
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
        insight = query_local_llm(
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
    