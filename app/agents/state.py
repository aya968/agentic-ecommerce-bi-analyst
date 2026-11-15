from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    schema_profile: dict
    schema_description: str
    generated_sql: str
    sql_result: list
    columns: list
    insight: str
    error: Optional[str]
    previous_sql: Optional[str]
    sql_error: Optional[str]
    retry_count: int
    success: bool
    sql_ready: bool
    