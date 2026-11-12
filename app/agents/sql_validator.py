import re
import sqlite3

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

        if not re.match(r"^SELECT\b", cleaned, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed.")

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
    def execute_read_only(sql, db_path=DB_PATH):
        safe_sql = SQLValidator.validate(sql)

        uri = f"file:{db_path}?mode=ro"

        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=10
        )

        connection.row_factory = sqlite3.Row

        def authorizer(action, arg1, arg2, database, trigger):
            allowed_actions = {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION
            }

            if action in allowed_actions:
                return sqlite3.SQLITE_OK

            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorizer)

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
            