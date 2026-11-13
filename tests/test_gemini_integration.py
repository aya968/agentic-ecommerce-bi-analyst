import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from google.genai import errors as genai_errors

from app.agents.graph import ask_agent, query_llm
from app.agents.sql_validator import SQLValidator


class GeminiClientTests(unittest.TestCase):
    def test_missing_api_key_fails_before_client_creation(self):
        with patch("app.agents.graph.GEMINI_API_KEY", None):
            with patch("app.agents.graph.genai.Client") as client_factory:
                with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                    query_llm("prompt")

        client_factory.assert_not_called()

    def test_sends_prompts_temperature_and_timeout_to_gemini(self):
        client = Mock()
        client.models.generate_content.return_value.text = "  response  "

        with patch("app.agents.graph.GEMINI_API_KEY", "local-test-key"):
            with patch(
                "app.agents.graph.genai.Client",
                return_value=client
            ) as factory:
                result = query_llm("prompt", "system", temperature=0.4)

        self.assertEqual(result, "response")
        factory.assert_called_once()
        self.assertEqual(
            factory.call_args.kwargs["http_options"].timeout,
            60_000
        )
        generation_config = client.models.generate_content.call_args.kwargs[
            "config"
        ]
        self.assertEqual(generation_config.system_instruction, "system")
        self.assertEqual(generation_config.temperature, 0.4)

    def test_retries_network_errors(self):
        client = Mock()
        client.models.generate_content.side_effect = [
            httpx.ConnectError("private connection details"),
            type("Response", (), {"text": "ok"})()
        ]

        with patch("app.agents.graph.GEMINI_API_KEY", "local-test-key"):
            with patch("app.agents.graph.genai.Client", return_value=client):
                with patch("app.agents.graph.time.sleep"):
                    self.assertEqual(query_llm("prompt"), "ok")

        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_api_errors_do_not_expose_provider_details(self):
        client = Mock()
        client.models.generate_content.side_effect = genai_errors.ClientError(
            403,
            {"message": "sensitive provider details"}
        )

        with patch("app.agents.graph.GEMINI_API_KEY", "local-test-key"):
            with patch("app.agents.graph.genai.Client", return_value=client):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "access was denied"
                ) as raised:
                    query_llm("prompt")

        self.assertNotIn("sensitive provider details", str(raised.exception))

    def test_empty_model_response_is_reported(self):
        client = Mock()
        client.models.generate_content.return_value.text = "  "

        with patch("app.agents.graph.GEMINI_API_KEY", "local-test-key"):
            with patch("app.agents.graph.genai.Client", return_value=client):
                with self.assertRaisesRegex(RuntimeError, "empty or non-text"):
                    query_llm("prompt")

    def test_rate_limit_has_a_friendly_error_after_retries(self):
        client = Mock()
        client.models.generate_content.side_effect = genai_errors.ClientError(
            429,
            {"message": "provider details"}
        )

        with patch("app.agents.graph.GEMINI_API_KEY", "local-test-key"):
            with patch("app.agents.graph.genai.Client", return_value=client):
                with patch("app.agents.graph.time.sleep"):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "rate limit exceeded"
                    ) as raised:
                        query_llm("prompt")

        self.assertNotIn("provider details", str(raised.exception))
        self.assertEqual(client.models.generate_content.call_count, 3)

    def test_timeout_has_a_friendly_error_after_retries(self):
        client = Mock()
        client.models.generate_content.side_effect = httpx.ReadTimeout(
            "private timeout details"
        )

        with patch("app.agents.graph.GEMINI_API_KEY", "local-test-key"):
            with patch("app.agents.graph.genai.Client", return_value=client):
                with patch("app.agents.graph.time.sleep"):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "request timed out"
                    ) as raised:
                        query_llm("prompt")

        self.assertNotIn("private timeout details", str(raised.exception))


class LangGraphIntegrationTests(unittest.TestCase):
    def test_full_workflow_uses_both_llm_nodes_and_read_only_sql(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "ecommerce.db"
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    "CREATE TABLE ecommerce_data (product TEXT, revenue REAL)"
                )
                connection.execute(
                    "INSERT INTO ecommerce_data VALUES (?, ?)",
                    ("Widget", 42.5)
                )

            original_execute = SQLValidator.execute_read_only
            with patch(
                "app.agents.graph.query_llm",
                side_effect=[
                    "SELECT product, revenue FROM ecommerce_data",
                    "Widget generated 42.5 in revenue."
                ]
            ) as query:
                with patch(
                    "app.agents.graph.SQLValidator.execute_read_only",
                    side_effect=lambda sql: original_execute(sql, str(db_path))
                ):
                    result = ask_agent(
                        "What is the revenue?",
                        {"table": "ecommerce_data"}
                    )

            self.assertTrue(result["success"])
            self.assertEqual(
                result["data"],
                [{"product": "Widget", "revenue": 42.5}]
            )
            self.assertEqual(result["insight"], "Widget generated 42.5 in revenue.")
            self.assertEqual(query.call_count, 2)

            with self.assertRaisesRegex(ValueError, "Only SELECT"):
                SQLValidator.execute_read_only(
                    "DELETE FROM ecommerce_data",
                    str(db_path)
                )
            with sqlite3.connect(db_path) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM ecommerce_data"
                    ).fetchone()[0],
                    1
                )


if __name__ == "__main__":
    unittest.main()
