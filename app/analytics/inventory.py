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

        data = df.copy()

        data[date_col] = pd.to_datetime(
            data[date_col],
            errors="coerce"
        )

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

        data = data.dropna(
            subset=[
                product_col,
                date_col,
                quantity_col,
                stock_col
            ]
        )

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

        forecast_by_product = (
            forecast_df[
                forecast_df["forecast_quantity"].notna()
            ]
            .groupby("product")["forecast_quantity"]
            .mean()
            .to_dict()
        )

        for product, group in data.groupby(product_col):
            product_key = str(product)

            if product_key not in forecast_by_product:
                continue

            current_stock = float(group[stock_col].iloc[-1])

            if lead_time_col:
                lead_values = group[lead_time_col].dropna()

                if lead_values.empty:
                    continue

                lead_time = float(lead_values.iloc[-1])
            else:
                lead_time = float(default_lead_time_days)

            if lead_time < 0:
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
            "assumptions": {
                "service_level_z": service_level_z,
                "target_coverage": "Lead time + 7 days",
                "note": (
                    "Calculations do not account for open purchase "
                    "orders, MOQ, returns or supplier constraints."
                )
            }
        }
        