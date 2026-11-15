import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.agents.sql_validator import SQLValidator


class SQLValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.database_path = Path(self.temp_dir.name) / "test.sqlite"
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE ecommerce_data "
                    "(product TEXT, quantity INTEGER)"
                )
                connection.execute(
                    "INSERT INTO ecommerce_data VALUES ('Widget', 3)"
                )

    def test_accepts_single_select_and_optional_trailing_semicolon(self):
        self.assertEqual(
            SQLValidator.validate(
                "SELECT product FROM ecommerce_data;"
            ),
            "SELECT product FROM ecommerce_data"
        )

    def test_rejects_data_modification_and_schema_statements(self):
        for statement in (
            "INSERT INTO ecommerce_data VALUES ('x', 1)",
            "UPDATE ecommerce_data SET quantity = 0",
            "DELETE FROM ecommerce_data",
            "DROP TABLE ecommerce_data"
        ):
            with self.subTest(statement=statement):
                with self.assertRaises(ValueError):
                    SQLValidator.validate(statement)

    def test_rejects_multiple_statements(self):
        with self.assertRaisesRegex(ValueError, "one SQL statement"):
            SQLValidator.validate(
                "SELECT 1; SELECT 2"
            )

    def test_read_only_select_returns_rows(self):
        result = SQLValidator.execute_read_only(
            "SELECT product, quantity FROM ecommerce_data",
            str(self.database_path)
        )

        self.assertEqual(result["columns"], ["product", "quantity"])
        self.assertEqual(
            result["rows"],
            [{"product": "Widget", "quantity": 3}]
        )

    def test_read_only_execution_does_not_modify_database(self):
        with self.assertRaises(ValueError):
            SQLValidator.execute_read_only(
                "DELETE FROM ecommerce_data",
                str(self.database_path)
            )

        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM ecommerce_data"
                ).fetchone()[0],
                1
            )

    def test_invalid_sql_is_reported(self):
        with self.assertRaises(sqlite3.Error):
            SQLValidator.execute_read_only(
                "SELECT FROM ecommerce_data",
                str(self.database_path)
            )


if __name__ == "__main__":
    unittest.main()
