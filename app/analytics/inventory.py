import math

import pandas as pd


class InventoryOptimizer:

    def optimize(
        self,
        forecast_result,
        schema,
        service_level_z=1.65,
        default_lead_time_days=None
    ):
        if not forecast_result.get("ready"):
            return {
                "ready": False,
                "message": (
                    "Inventory optimization requires valid "
                    "demand forecasting data."
                ),
                "forecast_status": forecast_result
            }

        stock_col = schema.get("stock")
        lead_time_col = schema.get("lead_time")

        if not stock_col:
            return {
                "ready": False,
                "missing_fields": [
                    {
                        "field": "Current stock",
                        "accepted_examples": [
                            "current_stock",
                            "available_stock",
                            "inventory"
                        ]
                    }
                ],
                "message": (
                    "Current stock is required to calculate "
                    "reorder recommendations."
                )
            }

        if not lead_time_col and default_lead_time_days is None:
            return {
                "ready": False,
                "missing_fields": [
                    {
                        "field": "Supplier lead time in days",
                        "accepted_examples": [
                            "lead_time_days",
                            "supplier_lead_time"
                        ]
                    }
                ],
                "message": (
                    "Provide a lead time column or a default "
                    "lead time in days."
                )
            }

        return {
            "ready": False,
            "message": (
                "Inventory calculation needs a product-level "
                "stock table. Use optimize_from_data() with the "
                "original dataframe."
            )
        }

    def optimize_from_data(
        self,
        df,
        forecast_result,
        schema,
        default_lead_time_days=None,
        service_level_z=1.65
    ):
        if not forecast_result.get("ready"):
            return self.optimize(
                forecast_result,
                schema,
                service_level_z,
                default_lead_time_days
            )

        readiness = self.optimize(
            forecast_result,
            schema,
            service_level_z,
            default_lead_time_days
        )

        if not readiness.get("ready") and (
            "Current stock is required" in readiness.get("message", "")
            or "Provide a lead time" in readiness.get("message", "")
            or "Inventory calculation needs" in readiness.get("message", "")
        ):
            if "Inventory calculation needs" not in readiness.get("message", ""):
                return readiness

        product_col = schema.get("product")
        date_col = schema.get("date")
        quantity_col = schema.get("quantity")
        stock_col = schema.get("stock")
        lead_time_col = schema.get("lead_time")

        if not all([product_col, date_col, quantity_col, stock_col]):
            return {
                "ready": False,
                "message": "Required product, date, quantity or stock column is missing."
            }
        required_columns = [
            product_col,
            date_col,
            quantity_col,
            stock_col
        ]
        if lead_time_col:
            required_columns.append(lead_time_col)
        if any(column not in df.columns for column in required_columns):
            return {
                "ready": False,
                "message": "One or more inventory columns are unavailable."
            }

        data = df.copy()

        data[date_col] = pd.to_datetime(
            data[date_col],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        data[quantity_col] = pd.to_numeric(
            data[quantity_col],
            errors="coerce"
        )

        data[stock_col] = pd.to_numeric(
            data[stock_col],
            errors="coerce"
        )

        if lead_time_col:
            data[lead_time_col] = pd.to_numeric(
                data[lead_time_col],
                errors="coerce"
            )

        product_is_valid = data[product_col].map(
            lambda value: pd.notna(value) and bool(str(value).strip())
        )

        undated_products = set(
            data.loc[
                data[date_col].isna() & product_is_valid,
                product_col
            ].astype(str)
        )

        finite_quantity = data[quantity_col].map(
            lambda value: pd.notna(value) and math.isfinite(float(value))
        )
        data = data.loc[
            product_is_valid
            & data[date_col].notna()
            & finite_quantity
        ].copy()

        if data.empty:
            return {
                "ready": False,
                "message": "No valid inventory records found."
            }

        output = []

        forecasts = forecast_result.get("forecasts", [])

        forecast_df = pd.DataFrame(forecasts)

        if forecast_df.empty or "forecast_quantity" not in forecast_df:
            return {
                "ready": False,
                "message": "No usable product forecasts are available."
            }

        forecast_df["forecast_quantity"] = pd.to_numeric(
            forecast_df["forecast_quantity"],
            errors="coerce"
        )
        forecast_df = forecast_df.loc[
            forecast_df["forecast_quantity"].map(
                lambda value: (
                    pd.notna(value)
                    and math.isfinite(float(value))
                    and float(value) >= 0
                )
            )
        ]
        forecast_by_product = (
            forecast_df[
                forecast_df["product"].notna()
            ]
            .groupby("product")["forecast_quantity"]
            .mean()
            .to_dict()
        )

        skipped_products = []
        for product, group in data.groupby(product_col):
            product_key = str(product)

            if product_key not in forecast_by_product:
                continue

            if product_key in undated_products:
                skipped_products.append({
                    "product": product_key,
                    "reason": (
                        "A record has an invalid date, so the latest "
                        "inventory values cannot be determined safely."
                    )
                })
                continue

            group = group.sort_values(date_col, kind="stable")
            latest = group.iloc[-1]
            stock_value = latest[stock_col]

            if (
                pd.isna(stock_value)
                or not math.isfinite(float(stock_value))
                or float(stock_value) < 0
            ):
                skipped_products.append({
                    "product": product_key,
                    "reason": (
                        "The latest dated record has missing or invalid "
                        "stock; no recommendation was generated."
                    )
                })
                continue

            current_stock = float(stock_value)

            if lead_time_col:
                lead_value = latest[lead_time_col]
                if (
                    pd.isna(lead_value)
                    or not math.isfinite(float(lead_value))
                    or float(lead_value) <= 0
                ):
                    skipped_products.append({
                        "product": product_key,
                        "reason": (
                            "The latest dated record has missing or "
                            "invalid lead time; no recommendation was generated."
                        )
                    })
                    continue
                lead_time = float(lead_value)
            else:
                lead_time = float(default_lead_time_days)

            if not math.isfinite(lead_time) or lead_time <= 0:
                skipped_products.append({
                    "product": product_key,
                    "reason": (
                        "Lead time must be a finite positive number; "
                        "no recommendation was generated."
                    )
                })
                continue

            daily_forecast = (
                float(forecast_by_product[product_key]) / 7
            )

            product_history = group.sort_values(date_col)

            weekly_demand = (
                product_history
                .set_index(date_col)[quantity_col]
                .resample("W")
                .sum()
            )

            daily_std = (
                float(weekly_demand.std()) / (7 ** 0.5)
                if len(weekly_demand) > 1
                and pd.notna(weekly_demand.std())
                else 0.0
            )

            safety_stock = (
                service_level_z
                * daily_std
                * (lead_time ** 0.5)
            )

            reorder_point = (
                daily_forecast * lead_time
                + safety_stock
            )

            target_stock = (
                daily_forecast * (lead_time + 7)
                + safety_stock
            )

            suggested_order = max(
                0,
                target_stock - current_stock
            )

            output.append({
                "product": product_key,
                "current_stock": round(current_stock, 2),
                "daily_forecast": round(daily_forecast, 2),
                "lead_time_days": round(lead_time, 2),
                "safety_stock": round(safety_stock, 2),
                "reorder_point": round(reorder_point, 2),
                "suggested_order_quantity": round(
                    suggested_order, 2
                ),
                "action": (
                    "REORDER"
                    if current_stock <= reorder_point
                    else "STOCK SUFFICIENT"
                )
            })

        return {
            "ready": True,
            "recommendations": output,
            "skipped_products": skipped_products,
            "assumptions": {
                "service_level_z": service_level_z,
                "target_coverage": "Lead time + 7 days",
                "note": (
                    "Calculations do not account for open purchase "
                    "orders, MOQ, returns or supplier constraints."
                )
            }
        }
        