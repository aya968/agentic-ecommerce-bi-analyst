import sqlite3
import unittest
from unittest.mock import patch

from app.agents.graph import (
    MAX_SQL_RETRIES,
    ask_agent,
    route_after_execution
)


SCHEMA_PROFILE = {
    "product": "product",
    "quantity": "quantity",
    "revenue": "revenue"
}

SQL_RESULT = {
    "rows": [{"total_revenue": 15}],
    "columns": ["total_revenue"],
    "row_count": 1,
    "truncated": False
}


class LangGraphRetryTests(unittest.TestCase):
    def test_successful_generation_execution_and_insight(self):
        with patch(
            "app.agents.graph.query_llm",
            side_effect=[
                "SELECT SUM(revenue) AS total_revenue FROM ecommerce_data",
                "Revenue totaled 15."
            ]
        ) as query:
            with patch(
                "app.agents.graph.SQLValidator.execute_read_only",
                return_value=SQL_RESULT
            ) as execute:
                result = ask_agent(
                    "What is total revenue?",
                    SCHEMA_PROFILE
                )

        self.assertTrue(result["success"])
        self.assertEqual(result["data"], SQL_RESULT["rows"])
        self.assertEqual(result["insight"], "Revenue totaled 15.")
        self.assertEqual(query.call_count, 2)
        execute.assert_called_once()

    def test_execution_error_is_passed_to_corrective_prompt(self):
        failed_sql = "SELECT SUM(total) FROM ecommerce_data"
        corrected_sql = "SELECT SUM(revenue) FROM ecommerce_data"

        with patch(
            "app.agents.graph.query_llm",
            side_effect=[failed_sql, corrected_sql, "Corrected insight."]
        ) as query:
            with patch(
                "app.agents.graph.SQLValidator.execute_read_only",
                side_effect=[
                    sqlite3.OperationalError("no such column: total"),
                    SQL_RESULT
                ]
            ) as execute:
                result = ask_agent(
                    "What is total revenue?",
                    SCHEMA_PROFILE
                )

        self.assertTrue(result["success"])
        self.assertEqual(result["sql"], corrected_sql)
        self.assertEqual(query.call_count, 3)
        self.assertEqual(execute.call_count, 2)
        correction_prompt = query.call_args_list[1].args[0]
        self.assertIn(failed_sql, correction_prompt)
        self.assertIn("no such column: total", correction_prompt)
        self.assertIn("correction attempt 1 of", correction_prompt)

    def test_repeated_execution_failures_stop_after_two_retries(self):
        with patch(
            "app.agents.graph.query_llm",
            side_effect=[
                "SELECT missing FROM ecommerce_data",
                "SELECT still_missing FROM ecommerce_data",
                "SELECT also_missing FROM ecommerce_data"
            ]
        ) as query:
            with patch(
                "app.agents.graph.SQLValidator.execute_read_only",
                side_effect=sqlite3.OperationalError(
                    "no such column: missing"
                )
            ) as execute:
                result = ask_agent(
                    "What is total revenue?",
                    SCHEMA_PROFILE
                )

        self.assertFalse(result["success"])
        self.assertEqual(query.call_count, MAX_SQL_RETRIES + 1)
        self.assertEqual(execute.call_count, MAX_SQL_RETRIES + 1)
        self.assertIn("2 correction attempts", result["error"])
        self.assertNotIn("no such column", result["error"])

    def test_route_stops_when_retry_limit_is_reached(self):
        self.assertEqual(
            route_after_execution({
                "success": False,
                "retry_count": MAX_SQL_RETRIES
            }),
            "generate_sql"
        )
        self.assertEqual(
            route_after_execution({
                "success": False,
                "retry_count": MAX_SQL_RETRIES + 1
            }),
            "__end__"
        )


if __name__ == "__main__":
    unittest.main()
