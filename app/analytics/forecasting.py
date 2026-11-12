import pandas as pd


class DemandForecaster:

    MIN_WEEKS = 8
    FORECAST_WEEKS = 4

    def __init__(self, min_weeks=None):
        self.min_weeks = min_weeks or self.MIN_WEEKS

    def check_readiness(self, df, schema):
        missing = []

        product_col = schema.get("product")
        date_col = schema.get("date")
        quantity_col = schema.get("quantity")

        if not product_col:
            missing.append({
                "field": "Product identifier",
                "accepted_examples": [
                    "product_id",
                    "product_name",
                    "sku"
                ]
            })

        if not date_col:
            missing.append({
                "field": "Sales date",
                "accepted_examples": [
                    "order_date",
                    "transaction_date",
                    "date"
                ]
            })

        if not quantity_col:
            missing.append({
                "field": "Quantity sold",
                "accepted_examples": [
                    "quantity",
                    "qty",
                    "units_sold"
                ]
            })

        if missing:
            return {
                "ready": False,
                "missing_fields": missing,
                "message": (
                    "Forecasting cannot run because required "
                    "columns are missing."
                )
            }

        working = df[
            [product_col, date_col, quantity_col]
        ].copy()

        working[date_col] = pd.to_datetime(
            working[date_col],
            errors="coerce"
        )

        working[quantity_col] = pd.to_numeric(
            working[quantity_col],
            errors="coerce"
        )

        working = working.dropna(
            subset=[product_col, date_col, quantity_col]
        )

        if working.empty:
            return {
                "ready": False,
                "missing_fields": [],
                "message": (
                    "No valid product, date and quantity records "
                    "were found."
                )
            }

        working["week"] = (
            working[date_col]
            .dt.to_period("W")
            .apply(lambda period: period.start_time)
        )

        total_weeks = working["week"].nunique()

        if total_weeks < self.min_weeks:
            return {
                "ready": False,
                "missing_fields": [],
                "available_weeks": int(total_weeks),
                "required_weeks": self.min_weeks,
                "message": (
                    f"Only {total_weeks} weeks of valid data found. "
                    f"At least {self.min_weeks} weeks are required "
                    "for this baseline forecast."
                )
            }

        return {
            "ready": True,
            "data": working,
            "available_weeks": int(total_weeks),
            "required_weeks": self.min_weeks,
            "message": "Data is sufficient for baseline forecasting."
        }

    def forecast(self, df, schema):
        readiness = self.check_readiness(df, schema)

        if not readiness["ready"]:
            return readiness

        working = readiness["data"]

        product_col = schema["product"]
        quantity_col = schema["quantity"]

        results = []

        for product, group in working.groupby(product_col):
            weekly = (
                group.groupby("week")[quantity_col]
                .sum()
                .sort_index()
            )

            # Fill missing weeks with zero.
            # Interpret carefully if stockout periods exist.
            full_index = pd.date_range(
                weekly.index.min(),
                weekly.index.max(),
                freq="W-MON"
            )

            weekly = weekly.reindex(
                full_index,
                fill_value=0
            )

            if len(weekly) < self.min_weeks:
                results.append({
                    "product": str(product),
                    "status": "insufficient_history",
                    "available_weeks": len(weekly),
                    "required_weeks": self.min_weeks
                })
                continue

            recent = weekly.tail(8)

            average_demand = float(recent.mean())

            # Simple transparent baseline:
            # average of the last 8 observed weekly buckets.
            for week_number in range(1, self.FORECAST_WEEKS + 1):
                results.append({
                    "product": str(product),
                    "forecast_week": week_number,
                    "forecast_quantity": round(
                        max(0, average_demand), 2
                    ),
                    "method": "8-week moving average"
                })

        return {
            "ready": True,
            "message": "Baseline forecast generated.",
            "available_weeks": readiness["available_weeks"],
            "forecasts": results,
            "warning": (
                "This is a baseline estimate, not a guarantee. "
                "Promotions, seasonality, stockouts and price changes "
                "are not modeled."
            )
        }
        