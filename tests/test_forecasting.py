import math
import unittest
from unittest.mock import patch

import pandas as pd

from app.analytics.forecasting import (
    DemandForecaster,
    _fit_model_forecast
)
from app.analytics.inventory import InventoryOptimizer


SCHEMA = {
    "product": "product",
    "date": "date",
    "quantity": "quantity"
}


def sales_frame(quantities, product="SKU-1", start="2022-01-03"):
    return pd.DataFrame({
        "product": [product] * len(quantities),
        "date": pd.date_range(start, periods=len(quantities), freq="W-MON"),
        "quantity": quantities
    })


class ForecastPreparationTests(unittest.TestCase):
    def test_invalid_dates_and_quantities_are_dropped(self):
        frame = sales_frame([2] * 12)
        frame.loc[len(frame)] = ["SKU-1", "not-a-date", 4]
        frame.loc[len(frame)] = ["SKU-1", "2022-04-04", "not-a-number"]

        result = DemandForecaster().forecast(frame, SCHEMA)

        self.assertTrue(result["ready"])
        self.assertEqual(result["dropped_records"], 2)
        self.assertEqual(len(result["forecasts"]), 4)

    def test_duplicate_product_date_records_are_aggregated(self):
        frame = sales_frame([2] * 12)
        duplicate = frame.iloc[[-1]].copy()
        duplicate.loc[:, "quantity"] = 3
        frame = pd.concat([frame, duplicate], ignore_index=True)

        result = DemandForecaster().forecast(frame, SCHEMA)

        self.assertEqual(
            result["forecasts"][0]["forecast_quantity"],
            2.38
        )

    def test_zero_and_interpolate_missing_week_policies(self):
        frame = sales_frame([2] * 12)
        frame = frame.drop(index=5).reset_index(drop=True)
        zero_forecaster = DemandForecaster(missing_week_policy="zero")
        interpolate_forecaster = DemandForecaster(
            missing_week_policy="interpolate"
        )

        zero_readiness = zero_forecaster.check_readiness(frame, SCHEMA)
        zero_weekly, _, missing_count = zero_forecaster._complete_weekly_series(
            zero_readiness["data"],
            "quantity"
        )
        interpolate_readiness = interpolate_forecaster.check_readiness(
            frame,
            SCHEMA
        )
        interpolated, _, _ = (
            interpolate_forecaster._complete_weekly_series(
                interpolate_readiness["data"],
                "quantity"
            )
        )

        self.assertEqual(missing_count, 1)
        self.assertEqual(zero_weekly.iloc[5], 0)
        self.assertEqual(interpolated.iloc[5], 2)
        self.assertEqual(
            zero_forecaster.forecast(frame, SCHEMA)["missing_week_policy"],
            "zero"
        )

    def test_invalid_schema_and_insufficient_history_are_reported(self):
        frame = sales_frame([1] * 7)
        missing_column_result = DemandForecaster().forecast(
            frame,
            {"product": "product", "date": "missing", "quantity": "quantity"}
        )
        insufficient_result = DemandForecaster().forecast(frame, SCHEMA)

        self.assertFalse(missing_column_result["ready"])
        self.assertEqual(len(missing_column_result["missing_fields"]), 1)
        self.assertFalse(insufficient_result["ready"])
        self.assertEqual(insufficient_result["available_weeks"], 7)


class ModelSelectionTests(unittest.TestCase):
    def diagnostics(self, result):
        return result["model_diagnostics"][0]

    def test_regular_demand_uses_baseline_without_overfitting_trend(self):
        result = DemandForecaster().forecast(sales_frame([12] * 24), SCHEMA)
        diagnostic = self.diagnostics(result)

        self.assertEqual(
            diagnostic["selected_model"],
            "8_week_moving_average"
        )
        self.assertEqual(
            next(
                item for item in diagnostic["candidate_models"]
                if item["model"] == "holt_linear"
            )["status"],
            "skipped"
        )
        self.assertEqual(result["forecasts"][0]["validation_mae"], 0)

    def test_upward_and_downward_trends_can_select_holt(self):
        for quantities in (
            [5 + 2 * week for week in range(32)],
            [100 - 2 * week for week in range(32)]
        ):
            with self.subTest(direction=quantities[-1] > quantities[0]):
                result = DemandForecaster().forecast(
                    sales_frame(quantities),
                    SCHEMA
                )
                diagnostic = self.diagnostics(result)
                self.assertEqual(
                    diagnostic["selected_model"],
                    "holt_linear"
                )
                self.assertTrue(
                    all(item["forecast_quantity"] >= 0
                        for item in result["forecasts"])
                )

    def test_seasonal_ets_is_evaluated_after_multiple_years(self):
        quantities = [
            20 + 8 * math.sin(2 * math.pi * (week % 52) / 52)
            for week in range(168)
        ]
        result = DemandForecaster().forecast(sales_frame(quantities), SCHEMA)
        diagnostic = self.diagnostics(result)
        seasonal = next(
            item for item in diagnostic["candidate_models"]
            if item["model"] == "holt_winters_additive_52_week"
        )

        self.assertEqual(seasonal["status"], "evaluated")
        self.assertEqual(
            diagnostic["selected_model"],
            "holt_winters_additive_52_week"
        )

    def test_intermittent_candidate_uses_explicit_sparsity_rule(self):
        quantities = [18 if week % 9 == 0 else 0 for week in range(180)]
        result = DemandForecaster().forecast(sales_frame(quantities), SCHEMA)
        diagnostic = self.diagnostics(result)
        croston = next(
            item for item in diagnostic["candidate_models"]
            if item["model"] == "croston_sba"
        )

        self.assertEqual(croston["status"], "evaluated")
        self.assertIn(
            diagnostic["selected_model"],
            {"croston_sba", "8_week_moving_average"}
        )

    def test_all_zero_demand_has_safe_metrics_and_nonnegative_forecast(self):
        result = DemandForecaster().forecast(sales_frame([0] * 20), SCHEMA)

        self.assertEqual(
            {item["forecast_quantity"] for item in result["forecasts"]},
            {0}
        )
        self.assertEqual(result["forecasts"][0]["validation_wape"], 0)

    def test_no_full_backtest_uses_moving_average_with_null_metrics(self):
        result = DemandForecaster().forecast(sales_frame([7] * 10), SCHEMA)
        diagnostic = self.diagnostics(result)

        self.assertTrue(diagnostic["fallback_used"])
        self.assertEqual(
            diagnostic["selected_model"],
            "8_week_moving_average"
        )
        self.assertIsNone(result["forecasts"][0]["validation_mae"])
        self.assertIsNone(result["forecasts"][0]["validation_wape"])

    def test_candidate_fit_failure_is_reported_without_aborting(self):
        frame = sales_frame([5 + 2 * week for week in range(24)])
        original_fit = _fit_model_forecast

        def fail_holt(model, *args, **kwargs):
            if model == "holt_linear":
                raise ValueError("synthetic fitting failure")
            return original_fit(model, *args, **kwargs)

        with patch(
            "app.analytics.forecasting._fit_model_forecast",
            side_effect=fail_holt
        ):
            result = DemandForecaster().forecast(frame, SCHEMA)

        diagnostic = self.diagnostics(result)
        failed = next(
            item for item in diagnostic["candidate_models"]
            if item["model"] == "holt_linear"
        )
        self.assertTrue(result["ready"])
        self.assertEqual(failed["status"], "failed")
        self.assertIn("synthetic fitting failure", failed["reason"])
        self.assertTrue(result["forecasts"])

    def test_forecast_contract_and_inventory_compatibility(self):
        frame = sales_frame([8] * 20)
        frame["stock"] = 3
        frame["lead_time"] = 5
        schema = {
            **SCHEMA,
            "stock": "stock",
            "lead_time": "lead_time"
        }
        forecasts = DemandForecaster().forecast(frame, schema)
        required_fields = {
            "product",
            "forecast_week",
            "forecast_date",
            "forecast_quantity",
            "selected_model",
            "validation_mae",
            "validation_wape"
        }

        self.assertTrue(all(
            required_fields.issubset(record)
            for record in forecasts["forecasts"]
        ))
        self.assertEqual(
            forecasts["forecasts"][0]["forecast_date"],
            "2022-05-23"
        )

        inventory = InventoryOptimizer().optimize_from_data(
            frame,
            forecasts,
            schema
        )
        self.assertTrue(inventory["ready"])
        self.assertEqual(
            inventory["recommendations"][0]["product"],
            "SKU-1"
        )


if __name__ == "__main__":
    unittest.main()
