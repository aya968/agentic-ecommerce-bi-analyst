import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.database.db_manager import DatabaseManager


class DatabaseManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.manager = DatabaseManager(root / "analytics.sqlite")
        self.input_dir = root / "input"
        self.input_dir.mkdir()

    def write_csv(self, name, content):
        path = self.input_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_csv_and_creates_processed_table(self):
        path = self.write_csv(
            "sales.csv",
            "Product,Order Date,Quantity\nWidget,2025-01-06,2\n"
        )

        report = self.manager.load_file_to_sqlite(path)

        self.assertEqual(report["loaded_rows"], 1)
        self.assertEqual(report["table_name"], "ecommerce_data")
        self.assertEqual(
            self.manager.get_dataframe().to_dict("records"),
            [{
                "product": "Widget",
                "order_date": "2025-01-06 00:00:00",
                "quantity": 2
            }]
        )
        self.assertEqual(self.manager.get_table_names(), ["ecommerce_data"])

    def test_loads_xlsx(self):
        path = self.input_dir / "sales.xlsx"
        pd.DataFrame({
            "product": ["Widget"],
            "quantity": [3]
        }).to_excel(path, index=False)

        report = self.manager.load_file_to_sqlite(path)

        self.assertEqual(report["loaded_rows"], 1)
        self.assertEqual(
            self.manager.get_dataframe().loc[0, "quantity"],
            3
        )

    def test_rejects_unsupported_extension(self):
        path = self.write_csv("sales.txt", "product,quantity\nWidget,1\n")

        with self.assertRaisesRegex(ValueError, "Unsupported file"):
            self.manager.load_file_to_sqlite(path)

    def test_duplicate_columns_and_rows_are_cleaned(self):
        frame = pd.DataFrame(
            [[" Widget ", 2, " Widget ", 2], [None, None, None, None]],
            columns=["Product", "Quantity", "Product", "Quantity"]
        )

        cleaned, report = self.manager.clean_dataframe(frame)

        self.assertEqual(
            list(cleaned.columns),
            ["product", "quantity", "product_1", "quantity_1"]
        )
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.loc[0, "product"], "Widget")
        self.assertEqual(report["duplicates_removed"], 0)
        self.assertEqual(report["cleaned_rows"], 1)

    def test_camel_case_headers_keep_source_names_and_normalize_semantically(self):
        frame = pd.DataFrame({
            "StockCode": ["S-1"],
            "InvoiceDate": ["2025-01-06"],
            "UnitPrice": [12.5]
        })

        cleaned, report = self.manager.clean_dataframe(frame)

        self.assertEqual(
            list(cleaned.columns),
            ["stock_code", "invoice_date", "unit_price"]
        )
        self.assertEqual(
            report["column_name_mapping"],
            [
                {"original": "StockCode", "cleaned": "stock_code"},
                {"original": "InvoiceDate", "cleaned": "invoice_date"},
                {"original": "UnitPrice", "cleaned": "unit_price"}
            ]
        )

    def test_empty_file_does_not_create_processed_table(self):
        path = self.write_csv("empty.csv", "product,quantity\n")

        with self.assertRaisesRegex(ValueError, "no usable rows"):
            self.manager.load_file_to_sqlite(path)

        self.assertEqual(self.manager.get_table_names(), [])

    def test_failed_table_write_preserves_previous_processed_data(self):
        first = self.write_csv(
            "first.csv",
            "product,quantity\nWidget,2\n"
        )
        second = self.write_csv(
            "second.csv",
            "product,quantity\nWidget,9\n"
        )
        self.manager.load_file_to_sqlite(first)

        with patch.object(
            pd.DataFrame,
            "to_sql",
            side_effect=RuntimeError("simulated write failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.manager.load_file_to_sqlite(second)

        self.assertEqual(
            self.manager.get_dataframe().loc[0, "quantity"],
            2
        )
        self.assertEqual(self.manager.get_table_names(), ["ecommerce_data"])

    def test_database_table_names_are_validated(self):
        with self.assertRaisesRegex(ValueError, "Invalid SQLite table name"):
            self.manager.get_dataframe('ecommerce_data"; DROP TABLE x;--')


if __name__ == "__main__":
    unittest.main()
