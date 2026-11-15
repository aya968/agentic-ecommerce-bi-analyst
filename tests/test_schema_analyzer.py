import unittest

import pandas as pd

from app.utils.schema_analyzer import SchemaAnalyzer
from app.agents.metric_planner import MetricPlanner


def weekly_rows(weeks=12, products=("P-1", "P-2")):
    rows = []
    for product in products:
        for date in pd.date_range("2024-01-01", periods=weeks, freq="W-MON"):
            rows.append({
                "product_id": product,
                "transaction_date": date,
                "quantity": 3,
                "unit_price": 12.5,
                "revenue": 37.5,
                "order_id": f"{product}-{date:%Y%m%d}",
                "customer_id": f"C-{date:%Y%m%d}"
            })
    return pd.DataFrame(rows)


class SchemaAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = SchemaAnalyzer()

    def test_detects_standard_ecommerce_fields_and_readiness_independently(self):
        frame = weekly_rows()

        schema = self.analyzer.analyze(frame)

        self.assertEqual(schema["product_id"], "product_id")
        self.assertEqual(schema["product"], "product_id")
        self.assertEqual(schema["date"], "transaction_date")
        self.assertEqual(schema["quantity"], "quantity")
        self.assertEqual(schema["price"], "unit_price")
        self.assertEqual(schema["revenue"], "revenue")
        self.assertEqual(
            schema["field_mappings"]["product_id"]["confidence"],
            "High confidence"
        )
        self.assertGreaterEqual(
            schema["field_mappings"]["product_id"]["confidence_score"],
            0.85
        )
        self.assertTrue(schema["readiness"]["forecasting"]["ready"])
        self.assertTrue(schema["readiness"]["sales_analysis"]["ready"])
        self.assertTrue(schema["readiness"]["customer_analysis"]["ready"])
        self.assertFalse(schema["readiness"]["inventory"]["ready"])
        self.assertIn(
            "Inventory stock quantity",
            schema["readiness"]["inventory"]["missing_fields"]
        )

    def test_retail_column_names_do_not_confuse_product_code_with_stock(self):
        frame = weekly_rows().rename(columns={
            "product_id": "StockCode",
            "transaction_date": "InvoiceDate",
            "quantity": "Quantity",
            "unit_price": "UnitPrice",
            "revenue": "Sales"
        })

        schema = self.analyzer.analyze(frame)

        self.assertEqual(schema["product_id"], "StockCode")
        self.assertIsNone(schema["stock"])
        self.assertEqual(schema["date"], "InvoiceDate")
        self.assertEqual(schema["quantity"], "Quantity")
        self.assertEqual(schema["price"], "UnitPrice")
        self.assertEqual(schema["revenue"], "Sales")

    def test_sku_orderdate_and_qty_variant_is_detected(self):
        frame = weekly_rows().rename(columns={
            "product_id": "SKU",
            "transaction_date": "OrderDate",
            "quantity": "Qty",
            "unit_price": "Price"
        }).drop(columns="revenue")

        schema = self.analyzer.analyze(frame)

        self.assertEqual(schema["product_id"], "SKU")
        self.assertEqual(schema["date"], "OrderDate")
        self.assertEqual(schema["quantity"], "Qty")
        self.assertEqual(schema["price"], "Price")
        self.assertEqual(schema["derived_metrics"]["revenue"]["source"], "derived")
        self.assertNotIn("revenue", schema["business_concepts"])
        self.assertTrue(schema["metric_capabilities"]["total_revenue"])
        self.assertTrue(
            MetricPlanner.plan("total_revenue", schema)["supported"]
        )

    def test_productid_transactiondate_units_sold_and_revenue_variant(self):
        frame = weekly_rows().rename(columns={
            "product_id": "ProductID",
            "transaction_date": "TransactionDate",
            "quantity": "UnitsSold"
        }).drop(columns="unit_price")

        schema = self.analyzer.analyze(frame)

        self.assertEqual(schema["product_id"], "ProductID")
        self.assertEqual(schema["date"], "TransactionDate")
        self.assertEqual(schema["quantity"], "UnitsSold")
        self.assertEqual(schema["revenue"], "revenue")
        self.assertEqual(schema["derived_metrics"]["revenue"]["source"], "column")

    def test_inventory_readiness_requires_stock_and_lead_time(self):
        frame = weekly_rows().assign(
            stock_quantity=20,
            lead_time_days=7
        )

        schema = self.analyzer.analyze(frame)

        self.assertEqual(schema["stock"], "stock_quantity")
        self.assertEqual(schema["lead_time"], "lead_time_days")
        self.assertTrue(schema["readiness"]["inventory"]["ready"])

    def test_inventory_stock_is_not_inferred_from_product_or_sales_quantity(self):
        frame = weekly_rows().rename(columns={
            "product_id": "item_code",
            "quantity": "units_sold"
        }).drop(columns=["unit_price", "revenue"])

        schema = self.analyzer.analyze(frame)

        self.assertEqual(schema["product_id"], "item_code")
        self.assertEqual(schema["quantity"], "units_sold")
        self.assertIsNone(schema["stock"])
        self.assertFalse(schema["inventory_ready"])

    def test_ambiguous_revenue_is_not_silently_selected(self):
        frame = weekly_rows().rename(columns={"revenue": "sales_amount"})
        frame["total_revenue"] = frame["sales_amount"]

        schema = self.analyzer.analyze(frame)

        self.assertIsNone(schema["revenue"])
        mapping = schema["field_mappings"]["revenue"]
        self.assertTrue(mapping["ambiguous"])
        self.assertGreaterEqual(len(mapping["candidates"]), 2)

    def test_missing_values_and_invalid_dates_reduce_readiness(self):
        frame = weekly_rows()
        frame["transaction_date"] = frame["transaction_date"].astype(object)
        frame["quantity"] = frame["quantity"].astype(object)
        frame.loc[:8, "transaction_date"] = "not-a-date"
        frame.loc[:8, "quantity"] = "unknown"

        schema = self.analyzer.analyze(frame)

        self.assertFalse(schema["readiness"]["forecasting"]["ready"])
        self.assertIsNone(schema["date"])
        self.assertIsNone(schema["quantity"])
        self.assertIn(
            "Transaction date",
            schema["readiness"]["forecasting"]["missing_fields"]
        )

    def test_duplicate_column_names_are_ambiguous(self):
        frame = weekly_rows().rename(columns={"product_id": "SKU"})
        frame["sku"] = frame["SKU"]

        schema = self.analyzer.analyze(frame)

        self.assertIsNone(schema["product_id"])
        self.assertTrue(schema["field_mappings"]["product_id"]["ambiguous"])

    def test_unrelated_columns_do_not_create_semantic_false_positives(self):
        frame = pd.DataFrame({
            "campaign_label": ["spring", "summer", "winter"],
            "employee_count": [12, 14, 11],
            "random_measure": [0.2, 0.8, 0.4],
            "warehouse_zone": ["A", "B", "A"]
        })

        schema = self.analyzer.analyze(frame)

        self.assertIsNone(schema["product"])
        self.assertIsNone(schema["quantity"])
        self.assertIsNone(schema["stock"])
        self.assertIsNone(schema["revenue"])
        self.assertFalse(schema["readiness"]["sales_analysis"]["ready"])

    def test_manual_overrides_validate_and_rebuild_mappings(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="W-MON")
        frame = pd.DataFrame({
            "key": ["P-1"] * 12,
            "when": dates,
            "amount_of_units": [2] * 12,
            "display_name": ["Widget"] * 12
        })

        schema = self.analyzer.apply_overrides(
            frame,
            {
                "product_id": "key",
                "date": "when",
                "quantity": "amount_of_units"
            }
        )

        self.assertEqual(schema["product"], "key")
        self.assertTrue(schema["field_mappings"]["product_id"]["confidence_score"] == 1.0)
        self.assertEqual(
            schema["field_mappings"]["product_id"]["detection_method"],
            "manual"
        )
        self.assertTrue(schema["readiness"]["forecasting"]["ready"])

        with self.assertRaisesRegex(ValueError, "mostly numeric"):
            self.analyzer.apply_overrides(
                frame,
                {"quantity": "display_name"}
            )

    def test_existing_revenue_is_preserved_and_profit_can_be_derived(self):
        frame = weekly_rows().assign(
            cost=10,
            profit=99,
            actual_sales=100
        )

        schema = self.analyzer.analyze(frame)

        self.assertEqual(
            schema["derived_metrics"]["revenue"]["source"],
            "column"
        )
        self.assertEqual(schema["derived_metrics"]["revenue"]["column"], "revenue")
        self.assertEqual(
            schema["derived_metrics"]["profit"]["source"],
            "column"
        )
        self.assertEqual(schema["profit"], "profit")

        frame = frame.drop(columns="profit")
        schema = self.analyzer.analyze(frame)
        self.assertEqual(
            schema["derived_metrics"]["profit"]["source"],
            "derived"
        )
        self.assertEqual(
            schema["derived_metrics"]["profit"]["expression"],
            '("revenue") - "cost"'
        )

    def test_insufficient_history_is_reported_without_forcing_readiness(self):
        frame = weekly_rows(weeks=4)

        schema = self.analyzer.analyze(frame)

        self.assertFalse(schema["forecast_ready"])
        self.assertEqual(
            schema["readiness"]["forecasting"]["usable_weeks"],
            4
        )
        self.assertTrue(
            any(
                "8 usable weekly observations" in item
                for item in schema["readiness"]["forecasting"]["missing_fields"]
            )
        )

    def test_original_headers_are_retained_in_field_mappings(self):
        frame = weekly_rows().rename(columns={
            "product_id": "product_id",
            "transaction_date": "transaction_date"
        })

        schema = self.analyzer.analyze(
            frame,
            original_column_names=[
                "ProductID",
                "TransactionDate",
                "Quantity Sold",
                "UnitPrice",
                "Revenue",
                "OrderID",
                "CustomerID"
            ]
        )

        self.assertEqual(
            schema["field_mappings"]["date"]["original_column"],
            "TransactionDate"
        )


if __name__ == "__main__":
    unittest.main()
