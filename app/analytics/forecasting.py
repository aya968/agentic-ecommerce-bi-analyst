import math

import pandas as pd


INTERMITTENT_ZERO_FRACTION = 0.4

MODEL_ORDER = (
    "moving_average",
    "holt_linear",
    "croston_sba",
    "holt_winters_52",
)

MODEL_LABELS = {
    "moving_average": "8_week_moving_average",
    "holt_linear": "holt_linear",
    "croston_sba": "croston_sba",
    "holt_winters_52": "holt_winters_additive_52_week",
}


def _intermittent_demand_reason(series):
    positive = series[series > 0]
    zero_fraction = float((series == 0).mean())

    if len(positive) < 2:
        return "Intermittent demand requires at least two positive weeks."

    event_weeks = [
        index for index, value in enumerate(series)
        if value > 0
    ]
    intervals = [
        later - earlier
        for earlier, later in zip(event_weeks, event_weeks[1:])
    ]
    mean_interval = sum(intervals) / len(intervals)

    if zero_fraction < INTERMITTENT_ZERO_FRACTION:
        return (
            "Intermittent demand requires at least "
            f"{INTERMITTENT_ZERO_FRACTION:.0%} zero-demand weeks."
        )

    if mean_interval < 1.5:
        return "Mean interval between positive-demand weeks is below 1.5."

    return None


def _trend_reason(series):
    if len(series) < 8:
        return "Holt's method requires at least 8 training weeks."

    x = pd.Series(range(len(series)), dtype=float)
    y = pd.Series(series.to_numpy(dtype=float))
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = float((centered_x ** 2).sum())

    if denominator == 0:
        return "No measurable trend is present."

    slope = float((centered_x * centered_y).sum() / denominator)
    correlation = float(x.corr(y))
    scale = max(float(y.abs().mean()), 1.0)

    if (
        not math.isfinite(correlation)
        or abs(slope) < scale * 0.05
        or abs(correlation) < 0.6
    ):
        return "A clear trend was not detected in the training weeks."

    return None


def _eligibility_reason(
    model,
    series,
    observed_weeks,
    min_weeks,
    seasonal_period,
    min_seasonal_cycles
):
    if model == "moving_average":
        if len(series) < min_weeks:
            return f"At least {min_weeks} training weeks are required."
        return None

    if model == "holt_linear":
        return _trend_reason(series)

    if model == "croston_sba":
        return _intermittent_demand_reason(series)

    if model == "holt_winters_52":
        required = seasonal_period * min_seasonal_cycles
        if len(series) < required or observed_weeks < required:
            return (
                f"Seasonal ETS requires at least {min_seasonal_cycles} "
                f"complete {seasonal_period}-week cycles "
                f"({required} observed weeks)."
            )
        return None

    raise ValueError(f"Unknown forecasting model: {model}")


def _croston_sba_forecast(series, horizon, alpha=0.1):
    values = [float(value) for value in series]
    events = [
        (index, value)
        for index, value in enumerate(values)
        if value > 0
    ]
    if not events:
        return [0.0] * horizon

    first_index, first_demand = events[0]
    demand_estimate = first_demand
    interval_estimate = float(first_index + 1)
    previous_index = first_index

    for index, demand in events[1:]:
        interval = float(index - previous_index)
        demand_estimate += alpha * (demand - demand_estimate)
        interval_estimate += alpha * (interval - interval_estimate)
        previous_index = index

    forecast = (
        (1 - alpha / 2)
        * demand_estimate
        / interval_estimate
    )
    return [forecast] * horizon


def _fit_model_forecast(
    model,
    series,
    horizon,
    min_weeks,
    seasonal_period,
    min_seasonal_cycles
):
    if model == "moving_average":
        return [float(series.tail(min_weeks).mean())] * horizon

    if model == "croston_sba":
        return _croston_sba_forecast(series, horizon)

    if model == "holt_linear":
        from statsmodels.tsa.holtwinters import Holt

        fitted = Holt(
            series.astype(float),
            damped_trend=True,
            initialization_method="estimated"
        ).fit(optimized=True)
        return fitted.forecast(horizon).tolist()

    if model == "holt_winters_52":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        fitted = ExponentialSmoothing(
            series.astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_period,
            initialization_method="estimated"
        ).fit(optimized=True)
        return fitted.forecast(horizon).tolist()

    raise ValueError(f"Unknown forecasting model: {model}")


def _validated_predictions(predictions, horizon):
    values = [float(value) for value in predictions]
    if len(values) != horizon or not all(
        math.isfinite(value) for value in values
    ):
        raise ValueError("Model returned invalid forecast values.")
    return [max(0.0, value) for value in values]


def _forecast_metrics(actual, predicted):
    errors = [
        abs(float(observed) - float(forecast))
        for observed, forecast in zip(actual, predicted)
    ]
    absolute_actual = sum(abs(float(value)) for value in actual)
    absolute_error = sum(errors)

    if absolute_actual == 0:
        wape = 0.0 if absolute_error == 0 else None
    else:
        wape = absolute_error / absolute_actual

    return {
        "validation_mae": absolute_error / len(errors),
        "validation_wape": wape
    }


class DemandForecaster:
    """Select and explain weekly demand forecasts using chronological tests."""

    MIN_WEEKS = 8
    FORECAST_WEEKS = 4
    SEASONAL_PERIOD = 52
    MIN_SEASONAL_CYCLES = 2
    MAX_BACKTEST_FOLDS = 3
    def __init__(
        self,
        min_weeks=None,
        forecast_weeks=None,
        missing_week_policy="zero",
        backtest_folds=None
    ):
        self.min_weeks = (
            self.MIN_WEEKS if min_weeks is None else int(min_weeks)
        )
        self.forecast_weeks = (
            self.FORECAST_WEEKS
            if forecast_weeks is None
            else int(forecast_weeks)
        )
        self.backtest_folds = (
            self.MAX_BACKTEST_FOLDS
            if backtest_folds is None
            else int(backtest_folds)
        )
        if missing_week_policy not in {"zero", "interpolate"}:
            raise ValueError(
                "missing_week_policy must be 'zero' or 'interpolate'."
            )
        if self.min_weeks < 1 or self.forecast_weeks < 1:
            raise ValueError("Week counts must be positive integers.")
        if self.backtest_folds < 1:
            raise ValueError("backtest_folds must be a positive integer.")
        self.missing_week_policy = missing_week_policy

    @staticmethod
    def _required_columns(schema):
        return {
            "Product identifier": schema.get("product"),
            "Sales date": schema.get("date"),
            "Quantity sold": schema.get("quantity")
        }

    def check_readiness(self, df, schema):
        required = self._required_columns(schema)
        missing = [
            {
                "field": field,
                "accepted_examples": examples
            }
            for field, column, examples in (
                (
                    "Product identifier",
                    required["Product identifier"],
                    ["product_id", "product_name", "sku"]
                ),
                (
                    "Sales date",
                    required["Sales date"],
                    ["order_date", "transaction_date", "date"]
                ),
                (
                    "Quantity sold",
                    required["Quantity sold"],
                    ["quantity", "qty", "units_sold"]
                )
            )
            if not column or column not in df.columns
        ]

        if missing:
            return {
                "ready": False,
                "missing_fields": missing,
                "message": (
                    "Forecasting cannot run because required "
                    "columns are missing."
                )
            }

        product_col = required["Product identifier"]
        date_col = required["Sales date"]
        quantity_col = required["Quantity sold"]
        working = df[[product_col, date_col, quantity_col]].copy()
        original_count = len(working)

        working[date_col] = pd.to_datetime(
            working[date_col],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)
        working[quantity_col] = pd.to_numeric(
            working[quantity_col],
            errors="coerce"
        )
        finite_quantity = working[quantity_col].map(
            lambda value: pd.notna(value) and math.isfinite(float(value))
        )
        valid_product = working[product_col].map(
            lambda value: pd.notna(value) and bool(str(value).strip())
        )
        working = working.loc[
            working[date_col].notna()
            & finite_quantity
            & valid_product
        ].copy()

        if working.empty:
            return {
                "ready": False,
                "missing_fields": [],
                "dropped_records": original_count,
                "message": (
                    "No valid product, date and quantity records "
                    "were found."
                )
            }

        working["week"] = (
            working[date_col]
            .dt.to_period("W-SUN")
            .dt.start_time
        )
        total_weeks = int(working["week"].nunique())
        dropped_records = original_count - len(working)

        if total_weeks < self.min_weeks:
            return {
                "ready": False,
                "missing_fields": [],
                "available_weeks": total_weeks,
                "required_weeks": self.min_weeks,
                "dropped_records": dropped_records,
                "message": (
                    f"Only {total_weeks} weeks of valid data found. "
                    f"At least {self.min_weeks} weeks are required "
                    "for forecasting."
                )
            }

        return {
            "ready": True,
            "data": working,
            "available_weeks": total_weeks,
            "required_weeks": self.min_weeks,
            "dropped_records": dropped_records,
            "message": "Data is sufficient for weekly forecasting."
        }

    def _complete_weekly_series(self, group, quantity_col):
        observed = (
            group.groupby("week")[quantity_col]
            .sum()
            .sort_index()
        )
        full_index = pd.date_range(
            observed.index.min(),
            observed.index.max(),
            freq="W-MON"
        )
        observed = observed.reindex(full_index)
        missing_weeks = int(observed.isna().sum())

        if self.missing_week_policy == "zero":
            weekly = observed.fillna(0.0)
        else:
            weekly = observed.interpolate(
                method="linear",
                limit_direction="both"
            )

        return weekly.astype(float), observed.notna(), missing_weeks

    def _backtest_splits(self, series_length):
        available_origins = (
            series_length - self.min_weeks
        ) // self.forecast_weeks
        fold_count = min(self.backtest_folds, available_origins)
        if fold_count < 1:
            return []

        first_origin = series_length - fold_count * self.forecast_weeks
        return [
            (
                first_origin + fold * self.forecast_weeks,
                first_origin + (fold + 1) * self.forecast_weeks
            )
            for fold in range(fold_count)
        ]

    def _evaluate_candidate(self, model, series, observed_mask, splits):
        if not splits:
            return {
                "model": MODEL_LABELS[model],
                "status": "not_evaluated",
                "folds": 0,
                "validation_mae": None,
                "validation_wape": None,
                "reason": (
                    "Insufficient history for one complete chronological "
                    "backtest horizon."
                )
            }

        actual = []
        predicted = []

        for train_end, validation_end in splits:
            training = series.iloc[:train_end]
            observed_weeks = int(observed_mask.iloc[:train_end].sum())
            ineligible_reason = _eligibility_reason(
                model,
                training,
                observed_weeks,
                self.min_weeks,
                self.SEASONAL_PERIOD,
                self.MIN_SEASONAL_CYCLES
            )
            if ineligible_reason:
                return {
                    "model": MODEL_LABELS[model],
                    "status": "skipped",
                    "folds": 0,
                    "validation_mae": None,
                    "validation_wape": None,
                    "reason": ineligible_reason
                }

            try:
                forecast = _fit_model_forecast(
                    model,
                    training,
                    self.forecast_weeks,
                    self.min_weeks,
                    self.SEASONAL_PERIOD,
                    self.MIN_SEASONAL_CYCLES
                )
                forecast = _validated_predictions(
                    forecast,
                    self.forecast_weeks
                )
            except ImportError as error:
                return {
                    "model": MODEL_LABELS[model],
                    "status": "skipped",
                    "folds": 0,
                    "validation_mae": None,
                    "validation_wape": None,
                    "reason": (
                        f"Optional model dependency is unavailable: {error}"
                    )
                }
            except Exception as error:
                return {
                    "model": MODEL_LABELS[model],
                    "status": "failed",
                    "folds": 0,
                    "validation_mae": None,
                    "validation_wape": None,
                    "reason": (
                        f"Model fitting failed: {type(error).__name__}: "
                        f"{error}"
                    )
                }

            actual.extend(
                float(value)
                for value in series.iloc[train_end:validation_end]
            )
            predicted.extend(forecast)

        metrics = _forecast_metrics(actual, predicted)
        return {
            "model": MODEL_LABELS[model],
            "status": "evaluated",
            "folds": len(splits),
            **metrics,
            "reason": None
        }

    @staticmethod
    def _candidate_sort_key(candidate, has_defined_wape):
        wape = candidate["validation_wape"]
        return (
            round(wape, 10) if has_defined_wape and wape is not None else (
                math.inf if has_defined_wape else 0
            ),
            round(candidate["validation_mae"], 10),
            MODEL_ORDER.index(candidate["model_id"])
        )

    def _select_model(self, series, observed_mask):
        splits = self._backtest_splits(len(series))
        candidates = []

        for model in MODEL_ORDER:
            result = self._evaluate_candidate(
                model,
                series,
                observed_mask,
                splits
            )
            result["model_id"] = model
            candidates.append(result)

        successful = [
            candidate
            for candidate in candidates
            if candidate["status"] == "evaluated"
        ]
        if not successful:
            return "moving_average", candidates, (
                "No eligible model could be validated on a complete "
                "chronological horizon; using the moving-average fallback."
            ), True

        has_defined_wape = any(
            candidate["validation_wape"] is not None
            for candidate in successful
        )
        selected = min(
            successful,
            key=lambda candidate: self._candidate_sort_key(
                candidate,
                has_defined_wape
            )
        )

        if selected["model_id"] == "moving_average":
            reason = (
                "The moving average had the best validation score, or tied "
                "for best score and won the simpler-model tie-break."
            )
        elif has_defined_wape:
            reason = "Selected by lowest validation WAPE, with MAE as tie-break."
        else:
            reason = (
                "Validation WAPE was undefined for zero-demand holdouts; "
                "selected by lowest validation MAE."
            )
        return selected["model_id"], candidates, reason, False

    def _forecast_product(self, product, group, quantity_col):
        weekly, observed_mask, missing_weeks = self._complete_weekly_series(
            group,
            quantity_col
        )
        warnings = []

        if missing_weeks:
            warnings.append(
                f"{missing_weeks} missing calendar weeks were handled using "
                f"the '{self.missing_week_policy}' policy."
            )

        diagnostics = {
            "product": str(product),
            "missing_weeks": missing_weeks,
            "missing_week_policy": self.missing_week_policy
        }

        if int(observed_mask.sum()) < self.min_weeks:
            diagnostics.update({
                "status": "insufficient_history",
                "selected_model": None,
                "selection_reason": (
                    f"At least {self.min_weeks} observed weekly periods "
                    "are required for the baseline."
                ),
                "fallback_used": False,
                "candidate_models": [
                    {
                        "model": MODEL_LABELS[model],
                        "status": "not_evaluated",
                        "folds": 0,
                        "validation_mae": None,
                        "validation_wape": None,
                        "reason": (
                            f"At least {self.min_weeks} observed weekly "
                            "periods are required for the baseline."
                        )
                    }
                    for model in MODEL_ORDER
                ],
                "warnings": warnings
            })
            return [], diagnostics

        selected_model, candidates, selection_reason, fallback_used = (
            self._select_model(weekly, observed_mask)
        )
        full_series_reason = _eligibility_reason(
            selected_model,
            weekly,
            int(observed_mask.sum()),
            self.min_weeks,
            self.SEASONAL_PERIOD,
            self.MIN_SEASONAL_CYCLES
        )

        if full_series_reason:
            selected_model = "moving_average"
            selection_reason = (
                f"The selected model was unsuitable for the final fit "
                f"({full_series_reason}); using the moving-average fallback."
            )
            fallback_used = True

        try:
            predictions = _fit_model_forecast(
                selected_model,
                weekly,
                self.forecast_weeks,
                self.min_weeks,
                self.SEASONAL_PERIOD,
                self.MIN_SEASONAL_CYCLES
            )
            predictions = _validated_predictions(
                predictions,
                self.forecast_weeks
            )
        except Exception as error:
            if selected_model == "moving_average":
                raise
            failed_model = selected_model
            selected_model = "moving_average"
            predictions = _validated_predictions(
                _fit_model_forecast(
                    selected_model,
                    weekly,
                    self.forecast_weeks,
                    self.min_weeks,
                    self.SEASONAL_PERIOD,
                    self.MIN_SEASONAL_CYCLES
                ),
                self.forecast_weeks
            )
            selection_reason = (
                f"Final {MODEL_LABELS[failed_model]} fit failed "
                f"({type(error).__name__}); using the moving-average fallback."
            )
            fallback_used = True
            warnings.append(selection_reason)

        selected_candidate = next(
            (
                candidate for candidate in candidates
                if candidate["model_id"] == selected_model
            ),
            None
        )
        validation_mae = (
            selected_candidate["validation_mae"]
            if selected_candidate
            and selected_candidate["status"] == "evaluated"
            else None
        )
        validation_wape = (
            selected_candidate["validation_wape"]
            if selected_candidate
            and selected_candidate["status"] == "evaluated"
            else None
        )

        for candidate in candidates:
            candidate.pop("model_id", None)

        if any(candidate["status"] == "failed" for candidate in candidates):
            warnings.append(
                "One or more candidate models failed to fit; see candidate "
                "diagnostics. Other eligible models were retained."
            )

        diagnostics.update({
            "status": "forecasted",
            "selected_model": MODEL_LABELS[selected_model],
            "selection_reason": selection_reason,
            "fallback_used": fallback_used,
            "validation_mae": validation_mae,
            "validation_wape": validation_wape,
            "candidate_models": candidates,
            "warnings": warnings
        })

        next_week = weekly.index[-1] + pd.Timedelta(days=7)
        forecasts = [
            {
                "product": str(product),
                "forecast_week": week_number,
                "forecast_date": (
                    next_week
                    + pd.Timedelta(weeks=week_number - 1)
                ).date().isoformat(),
                "forecast_quantity": round(quantity, 2),
                "selected_model": MODEL_LABELS[selected_model],
                "validation_mae": (
                    round(validation_mae, 4)
                    if validation_mae is not None
                    else None
                ),
                "validation_wape": (
                    round(validation_wape, 4)
                    if validation_wape is not None
                    else None
                ),
                "method": MODEL_LABELS[selected_model]
            }
            for week_number, quantity in enumerate(predictions, start=1)
        ]
        return forecasts, diagnostics

    def forecast(self, df, schema):
        readiness = self.check_readiness(df, schema)
        if not readiness["ready"]:
            return readiness

        working = readiness["data"]
        product_col = schema["product"]
        quantity_col = schema["quantity"]
        forecasts = []
        diagnostics = []

        for product, group in working.groupby(product_col, sort=False):
            product_forecasts, product_diagnostics = self._forecast_product(
                product,
                group,
                quantity_col
            )
            forecasts.extend(product_forecasts)
            diagnostics.append(product_diagnostics)

        warnings = [
            warning
            for item in diagnostics
            for warning in item["warnings"]
        ]
        overall_warning = (
            "Forecasts are estimates, not guarantees. "
            f"Missing calendar weeks use the '{self.missing_week_policy}' "
            "policy unless changed in DemandForecaster."
        )
        if warnings:
            overall_warning += " " + " ".join(dict.fromkeys(warnings))

        return {
            "ready": True,
            "message": "Weekly demand forecasts generated.",
            "available_weeks": readiness["available_weeks"],
            "required_weeks": readiness["required_weeks"],
            "forecasts": forecasts,
            "model_diagnostics": diagnostics,
            "missing_week_policy": self.missing_week_policy,
            "dropped_records": readiness["dropped_records"],
            "warning": overall_warning
        }
