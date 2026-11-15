import re
import sqlite3
from pathlib import Path

from app.config import DB_PATH, MAX_SQL_ROWS


FORBIDDEN_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "analyze"
}


class SQLValidator:

    @staticmethod
    def _connect_read_only(db_path):
        database_uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
        return sqlite3.connect(
            database_uri,
            uri=True,
            timeout=10
        )

    @staticmethod
    def _authorizer(action, arg1, arg2, database, trigger):
        allowed_actions = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION
        }
        return (
            sqlite3.SQLITE_OK
            if action in allowed_actions
            else sqlite3.SQLITE_DENY
        )

    @staticmethod
    def validate(sql):
        if not sql or not isinstance(sql, str):
            raise ValueError("SQL query is empty.")

        cleaned = sql.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:sql)?\s*|\s*```$",
                "",
                cleaned,
                flags=re.IGNORECASE
            ).strip()

        # Remove SQL comments
        cleaned = re.sub(
            r"/\*.*?\*/",
            "",
            cleaned,
            flags=re.DOTALL
        )

        cleaned = re.sub(
            r"--[^\n]*",
            "",
            cleaned
        ).strip()

        if ";" in cleaned.rstrip(";"):
            raise ValueError("Only one SQL statement is allowed.")

        cleaned = cleaned.rstrip(";").strip()

        if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
            raise ValueError("Only read-only SELECT queries are allowed.")

        tokens = re.findall(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
            cleaned.lower()
        )

        forbidden = FORBIDDEN_KEYWORDS.intersection(tokens)

        if forbidden:
            raise ValueError(
                f"Forbidden SQL keyword: {', '.join(forbidden)}"
            )

        if re.search(
            r"\b(load_extension|readfile|writefile)\s*\(",
            cleaned,
            re.IGNORECASE
        ):
            raise ValueError("Unsafe SQLite function detected.")

        return cleaned

    @staticmethod
    def get_database_schema(db_path):
        """Return live SQLite tables, columns, declared types, and row counts."""
        connection = SQLValidator._connect_read_only(db_path)
        try:
            table_names = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table'
                      AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall()
            ]
            tables = []
            for table_name in table_names:
                escaped_name = table_name.replace('"', '""')
                columns = [
                    {
                        "name": row[1],
                        "type": row[2] or "unknown",
                        "nullable": not bool(row[3]),
                        "primary_key": bool(row[5])
                    }
                    for row in connection.execute(
                        f'PRAGMA table_info("{escaped_name}")'
                    ).fetchall()
                ]
                row_count = connection.execute(
                    f'SELECT COUNT(*) FROM "{escaped_name}"'
                ).fetchone()[0]
                tables.append({
                    "name": table_name,
                    "columns": columns,
                    "row_count": row_count
                })
            return {"dialect": "SQLite", "tables": tables}
        finally:
            connection.close()

    @staticmethod
    def validate_schema(sql, db_path):
        """Compile a read-only query against the active database schema."""
        safe_sql = SQLValidator.validate(sql)
        connection = SQLValidator._connect_read_only(db_path)
        connection.set_authorizer(SQLValidator._authorizer)
        try:
            connection.execute(
                f"EXPLAIN QUERY PLAN {safe_sql}"
            ).fetchall()
        finally:
            connection.close()
        return safe_sql

    @staticmethod
    def execute_read_only(sql, db_path=DB_PATH):
        safe_sql = SQLValidator.validate_schema(sql, db_path)
        connection = SQLValidator._connect_read_only(db_path)

        connection.row_factory = sqlite3.Row
        connection.set_authorizer(SQLValidator._authorizer)

        try:
            cursor = connection.execute(safe_sql)

            columns = [
                description[0]
                for description in cursor.description or []
            ]

            rows = cursor.fetchmany(MAX_SQL_ROWS)

            result = [
                dict(row)
                for row in rows
            ]

            return {
                "columns": columns,
                "rows": result,
                "row_count": len(result),
                "truncated": len(result) == MAX_SQL_ROWS
            }

        finally:
            connection.close()
            