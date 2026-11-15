import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.agents.graph import ask_agent
from app.database.db_manager import DatabaseManager
from app.utils.schema_analyzer import SchemaAnalyzer


def plan_response(required, **overrides):
    plan = {
        "intent": "answer the analytical question",
        "entities": [],
        "metrics": [],
        "dimensions": [],
        "filters": [],
        "time_granularity": None,
        "required_concepts": required,
        "supported": True,
        "unsupported_reason": None
    }
    plan.update(overrides)
    return json.dumps(plan)


class TextToSQLTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def write_dataset(self, frame, name):
        manager = DatabaseManager(self.root / name)
        cleaned, report = manager.clean_dataframe(frame)
        schema = SchemaAnalyzer().analyze(
            cleaned,
            original_column_names=[
                mapping["original"]
                for mapping in report["column_name_mapping"]
            ]
        )
        manager.save_dataframe_to_sqlite(
            cleaned,
            report=report
        )
        return manager, schema

    def run_agent(self, manager, schema, question, required, sql, insight):
        with patch(
            "app.agents.graph.query_llm",
            side_effect=[
                plan_response(required),
                sql,
                insight
            ]
        ) as query:
            result = ask_agent(
                question,
                schema,
                db_path=manager.db_path
            )
        return result, query

    def test_customerid_invoicedate_and_derived_revenue_run_real_monthly_sql(self):
        frame = pd.DataFrame({
            "CustomerID": ["C1", "C1", "C2"],
            "OrderID": ["O1", "O2", "O3"],
            "InvoiceDate": [
                "2025-01-03",
                "2025-01-21",
                "2025-02-02"
            ],
            "Quantity": [2, 1, 4],
            "UnitPrice": [10, 5, 10]
        })
        manager, schema = self.write_dataset(frame, "customer-sales.sqlite")
        sql = """
            SELECT strftime('%Y-%m', invoice_date) AS month,
                   SUM(quantity * unit_price) AS revenue
            FROM ecommerce_data
            GROUP BY month
            ORDER BY month
        """

        result, query = self.run_agent(
            manager,
            schema,
            "Show monthly revenue",
            ["date", "revenue"],
            sql,
            "Revenue was 25 in January and 40 in February."
        )

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(
            result["data"],
            [
                {"month": "2025-01", "revenue": 25},
                {"month": "2025-02", "revenue": 40}
            ]
        )
        planning_prompt = query.call_args_list[0].args[0]
        self.assertIn('"invoice_date"', planning_prompt)
        self.assertIn('"unit_price"', planning_prompt)
        self.assertIn('"dialect": "SQLite"', planning_prompt)
        self.assertNotIn("C1", planning_prompt)

    def test_client_id_orderdate_and_repeat_purchases_run_real_sql(self):
        frame = pd.DataFrame({
            "client_id": ["A", "A", "B", "A"],
            "order_date": [
                "2024-01-03",
                "2024-02-03",
                "2024-01-10",
                "2024-02-20"
            ],
            "invoice_no": ["1", "2", "3", "4"],
            "sales_qty": [1, 2, 1, 3]
        })
        manager, schema = self.write_dataset(frame, "retention.sqlite")
        sql = """
            SELECT customer_id,
                   COUNT(DISTINCT strftime('%Y-%m', order_date))
                       AS active_months,
                   COUNT(DISTINCT invoice_no) AS order_count
            FROM ecommerce_data
            GROUP BY customer_id
            HAVING COUNT(DISTINCT invoice_no) > 1
            ORDER BY customer_id
        """

        result, _ = self.run_agent(
            manager,
            schema,
            "Which customers purchased in multiple months?",
            ["customer_id", "date", "order_id"],
            sql,
            "Customer A ordered in multiple months."
        )

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(
            result["data"],
            [{
                "customer_id": "A",
                "active_months": 2,
                "order_count": 3
            }]
        )

    def test_yearly_aggregation_uses_sqlite_live_schema(self):
        frame = pd.DataFrame({
            "SKU": ["S1", "S1", "S2"],
            "TransactionDate": [
                "2023-02-01",
                "2024-02-01",
                "2024-06-01"
            ],
            "UnitsSold": [2, 3, 5],
            "Revenue": [20, 30, 70]
        })
        manager, schema = self.write_dataset(frame, "yearly.sqlite")
        sql = """
            SELECT strftime('%Y', transaction_date) AS year,
                   SUM(revenue) AS revenue
            FROM ecommerce_data
            GROUP BY year
            ORDER BY year
        """

        result, _ = self.run_agent(
            manager,
            schema,
            "Compare revenue by year",
            ["date", "revenue"],
            sql,
            "Revenue increased."
        )

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(
            result["data"],
            [
                {"year": "2023", "revenue": 20},
                {"year": "2024", "revenue": 100}
            ]
        )

    def test_missing_customer_identifier_is_not_confused_with_alias_mismatch(self):
        frame = pd.DataFrame({
            "SKU": ["S1", "S2"],
            "InvoiceDate": ["2025-01-01", "2025-01-02"],
            "Quantity": [1, 2]
        })
        manager, schema = self.write_dataset(frame, "no-customer.sqlite")

        with patch(
            "app.agents.graph.query_llm",
            return_value=plan_response(
                ["customer_id", "date"],
                supported=True
            )
        ) as query:
            result = ask_agent(
                "Find repeat customers",
                schema,
                db_path=manager.db_path
            )

        self.assertFalse(result["success"])
        self.assertIn("Customer ID", result["error"])
        self.assertNotIn("CANNOT_ANSWER", result["error"])
        self.assertEqual(query.call_count, 1)

    def test_invalid_sql_is_schema_checked_and_recovered(self):
        frame = pd.DataFrame({
            "SKU": ["S1", "S2"],
            "InvoiceDate": ["2025-01-01", "2025-01-02"],
            "UnitsSold": [2, 3],
            "Revenue": [20, 30]
        })
        manager, schema = self.write_dataset(frame, "retry.sqlite")
        with patch(
            "app.agents.graph.query_llm",
            side_effect=[
                plan_response(["revenue"]),
                "SELECT SUM(missing_revenue) FROM ecommerce_data",
                "SELECT SUM(revenue) AS total FROM ecommerce_data",
                "Total revenue is 50."
            ]
        ) as query:
            result = ask_agent(
                "What is total revenue?",
                schema,
                db_path=manager.db_path
            )

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["data"], [{"total": 50}])
        correction_prompt = query.call_args_list[2].args[0]
        self.assertIn("missing_revenue", correction_prompt)
        self.assertIn("no such column", correction_prompt.lower())

    def test_dataset_schema_is_refreshed_per_database_and_not_reused(self):
        first_frame = pd.DataFrame({
            "product": ["A"],
            "quantity": [2],
            "unit_price": [10]
        })
        second_frame = pd.DataFrame({
            "sku": ["B"],
            "units_sold": [3],
            "revenue": [99]
        })
        first_db, first_schema = self.write_dataset(
            first_frame,
            "first-session.sqlite"
        )
        second_db, second_schema = self.write_dataset(
            second_frame,
            "second-session.sqlite"
        )

        with patch(
            "app.agents.graph.query_llm",
            side_effect=[
                plan_response(["quantity", "price"]),
                "SELECT SUM(quantity * unit_price) AS sales "
                "FROM ecommerce_data",
                "The first dataset produced 20.",
                plan_response(["revenue"]),
                "SELECT SUM(revenue) AS sales FROM ecommerce_data",
                "The second dataset produced 99."
            ]
        ) as query:
            first = ask_agent(
                "Total sales?",
                first_schema,
                db_path=first_db.db_path
            )
            second = ask_agent(
                "Total sales?",
                second_schema,
                db_path=second_db.db_path
            )

        self.assertTrue(first["success"], first["error"])
        self.assertTrue(second["success"], second["error"])
        self.assertEqual(first["data"], [{"sales": 20}])
        self.assertEqual(second["data"], [{"sales": 99}])
        first_plan = query.call_args_list[0].args[0]
        second_plan = query.call_args_list[3].args[0]
        self.assertIn('"unit_price"', first_plan)
        self.assertNotIn('"unit_price"', second_plan)
        self.assertIn('"units_sold"', second_plan)


if __name__ == "__main__":
    unittest.main()
