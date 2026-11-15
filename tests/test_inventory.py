import unittest

import pandas as pd

from app.analytics.inventory import InventoryOptimizer


FORECAST_RESULT = {
    "ready": True,
    "forecasts": [
        {"product": "A", "forecast_quantity": 14},
        {"product": "B", "forecast_quantity": 7}
    ]
}

SCHEMA = {
    "product": "product",
    "date": "date",
    "quantity": "quantity",
    "stock": "stock",
    "lead_time": "lead_time"
}


class InventoryDataTests(unittest.TestCase):
    def test_latest_stock_and_lead_time_use_chronological_order(self):
        frame = pd.DataFrame([
            ["A", "2025-01-15", 2, 10, 5],
            ["A", "2025-01-01", 1, 30, 2],
            ["A", "2025-01-08", 2, 20, 3]
        ], columns=SCHEMA.values())

        result = InventoryOptimizer().optimize_from_data(
            frame,
            FORECAST_RESULT,
            SCHEMA
        )

        self.assertTrue(result["ready"])
        recommendation = result["recommendations"][0]
        self.assertEqual(recommendation["current_stock"], 10)
        self.assertEqual(recommendation["lead_time_days"], 5)

    def test_missing_latest_stock_is_reported_and_not_replaced_by_old_value(self):
        frame = pd.DataFrame([
            ["A", "2025-01-01", 1, 30, 2],
            ["A", "2025-01-08", 2, None, 3]
        ], columns=SCHEMA.values())

        result = InventoryOptimizer().optimize_from_data(
            frame,
            FORECAST_RESULT,
            SCHEMA
        )

        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["skipped_products"][0]["product"], "A")
        self.assertIn("latest dated record", result["skipped_products"][0]["reason"])

    def test_negative_stock_and_invalid_lead_time_are_skipped(self):
        frame = pd.DataFrame([
            ["A", "2025-01-01", 1, -3, 5],
            ["B", "2025-01-01", 1, 10, -2]
        ], columns=SCHEMA.values())

        result = InventoryOptimizer().optimize_from_data(
            frame,
            FORECAST_RESULT,
            SCHEMA
        )

        self.assertEqual(result["recommendations"], [])
        self.assertEqual(
            {item["product"] for item in result["skipped_products"]},
            {"A", "B"}
        )

    def test_invalid_date_prevents_using_an_older_inventory_record(self):
        frame = pd.DataFrame([
            ["A", "2025-01-01", 1, 30, 5],
            ["A", "invalid-date", 2, 1, 1]
        ], columns=SCHEMA.values())

        result = InventoryOptimizer().optimize_from_data(
            frame,
            FORECAST_RESULT,
            SCHEMA
        )

        self.assertEqual(result["recommendations"], [])
        self.assertIn("invalid date", result["skipped_products"][0]["reason"])

    def test_multiple_products_are_recommended_independently(self):
        frame = pd.DataFrame([
            ["B", "2025-01-01", 1, 5, 2],
            ["A", "2025-01-01", 1, 30, 5]
        ], columns=SCHEMA.values())

        result = InventoryOptimizer().optimize_from_data(
            frame,
            FORECAST_RESULT,
            SCHEMA
        )

        self.assertEqual(
            {item["product"] for item in result["recommendations"]},
            {"A", "B"}
        )

    def test_invalid_default_lead_time_is_reported(self):
        frame = pd.DataFrame(
            [["A", "2025-01-01", 1, 12]],
            columns=["product", "date", "quantity", "stock"]
        )
        schema = {
            "product": "product",
            "date": "date",
            "quantity": "quantity",
            "stock": "stock"
        }

        result = InventoryOptimizer().optimize_from_data(
            frame,
            FORECAST_RESULT,
            schema,
            default_lead_time_days=-1
        )

        self.assertEqual(result["recommendations"], [])
        self.assertIn("positive", result["skipped_products"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
