import pandas as pd


COLUMN_HINTS = {
    "product": [
        "product", "product_name", "item", "item_name",
        "sku", "product_id"
    ],
    "date": [
        "date", "order_date", "created_at",
        "transaction_date", "timestamp"
    ],
    "quantity": [
        "quantity", "qty", "units_sold",
        "order_quantity", "sales_quantity"
    ],
    "price": [
        "price", "unit_price", "selling_price"
    ],
    "revenue": [
        "revenue", "sales", "total_sales",
        "amount", "total_amount"
    ],
    "order_id": [
        "order_id", "invoice_id", "transaction_id"
    ],
    "stock": [
        "stock", "inventory", "current_stock",
        "available_stock", "stock_quantity"
    ],
    "lead_time": [
        "lead_time", "lead_time_days",
        "supplier_lead_time"
    ],
    "customer_id": [
        "customer_id",
        "client_id",
        "buyer_id"
    ]
}


class SchemaAnalyzer:

    @staticmethod
    def detect_column(columns, category):
        normalized = {
            str(column).lower().strip(): column
            for column in columns
        }

        hints = COLUMN_HINTS.get(category, [])

        # Exact match first
        for hint in hints:
            if hint in normalized:
                return normalized[hint]

        # Partial match second
        for column in columns:
            name = str(column).lower()

            for hint in hints:
                if hint in name:
                    return column

        return None

    def analyze(self, df):
        columns = list(df.columns)

        profile = {
            "product": self.detect_column(columns, "product"),
            "date": self.detect_column(columns, "date"),
            "quantity": self.detect_column(columns, "quantity"),
            "price": self.detect_column(columns, "price"),
            "revenue": self.detect_column(columns, "revenue"),
            "order_id": self.detect_column(columns, "order_id"),
            "stock": self.detect_column(columns, "stock"),
            "lead_time": self.detect_column(columns, "lead_time"),
            "all_columns": columns,
            "row_count": len(df),
            "customer_id": self.detect_column(columns, "customer_id"),
            "column_count": len(columns)
        }

        profile["forecast_ready"] = all(
            profile[key] is not None
            for key in ["product", "date", "quantity"]
        )

        profile["inventory_ready"] = (
            profile["forecast_ready"]
            and profile["stock"] is not None
        )

        return profile
