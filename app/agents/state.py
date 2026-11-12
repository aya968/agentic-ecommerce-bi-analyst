from typing import TypedDict, Optional, Any


class AgentState(TypedDict, total=False):
    question: str
    schema_profile: dict
    schema_description: str
    generated_sql: str
    sql_result: list
    columns: list
    insight: str
    error: Optional[str]
    retry_count: int
    success: bool
    