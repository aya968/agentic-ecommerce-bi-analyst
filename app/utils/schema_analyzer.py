import math

import pandas as pd

from app.utils.schema_detection import (
    NUMERIC_ROLES,
    ROLE_LABELS,
    detect_role_candidates,
    normalize_column_name,
    profile_column
)


class SchemaAnalyzer:
    """Detect, explain, and validate semantic mappings for uploaded tables."""

    MIN_FORECAST_WEEKS = 8
    def __init__(
        self,
        min_forecast_weeks: int = MIN_FORECAST_WEEKS
    ) -> None:
        if int(min_forecast_weeks) < 1:
            raise ValueError("min_forecast_weeks must be positive.")
        self.min_forecast_weeks = int(min_forecast_weeks)

    @staticmethod
    def normalize_column_name(name):
        return normalize_column_name(name)

    @staticmethod
    def detect_column(columns, category):
        """Compatibility helper for callers that only have column names."""
        legacy_roles = {
            "product": ("product_id", "product_name"),
            "date": ("date",),
            "quantity": ("quantity",),
            "price": ("price",),
            "revenue": ("revenue",),
            "order_id": ("order_id",),
            "customer_id": ("customer_id",),
            "stock": ("stock",),
            "lead_time": ("lead_time",)
        }
        roles = legacy_roles.get(category, ())
        matches = []
        from app.utils.schema_detection import ROLE_ALIASES

        for column in columns:
            normalized = normalize_column_name(column)
            if any(normalized in ROLE_ALIASES[role] for role in roles):
                matches.append(column)
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _profile_dataframe(df, original_column_names=None):
        source_names = (
            list(original_column_names)
            if original_column_names is not None
            else list(df.columns)
        )
        if len(source_names) != len(df.columns):
            raise ValueError(
                "Original and cleaned column lists must have equal lengths."
            )
        return [
            profile_column(
                df.iloc[:, index],
                column,
                index,
                original_name=source_names[index]
            )
            for index, column in enumerate(df.columns)
        ]

    @staticmethod
    def _numeric_values(series):
        values = pd.to_numeric(series, errors="coerce")
        return values.where(values.map(
            lambda value: pd.isna(value) or math.isfinite(float(value))
        ))

    @staticmethod
    def _date_values(series):
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
            utc=True
        ).dt.tz_convert(None)

    @staticmethod
    def _column(df, column):
        if column is None:
            return None
        positions = [
            index for index, name in enumerate(df.columns)
            if name == column
        ]
        if len(positions) != 1:
            return None
        return df.iloc[:, positions[0]]

    @staticmethod
    def _field_column(mappings, role):
        field = mappings[role]
        if field["ambiguous"]:
            return None
        return field["column"]

    @staticmethod
    def _column_quality_warnings(mappings):
        warnings = []
        for role, field in mappings.items():
            if field["ambiguous"]:
                warnings.append(
                    f"{ROLE_LABELS[role]} has ambiguous column candidates."
                )
        return warnings

    def _derived_metrics(self, mappings):
        quantity = self._field_column(mappings, "quantity")
        price = self._field_column(mappings, "price")
        revenue = self._field_column(mappings, "revenue")
        cost = self._field_column(mappings, "cost")
        profit = self._field_column(mappings, "profit")

        metrics = {}
        if revenue is not None:
            metrics["revenue"] = {
                "available": True,
                "source": "column",
                "column": revenue,
                "expression": None,
                "calculation": "Existing revenue column; retained as supplied."
            }
        elif quantity is not None and price is not None:
            derivation_note = (
                "No unambiguous existing revenue column was selected; "
                if mappings["revenue"]["ambiguous"]
                else ""
            )
            metrics["revenue"] = {
                "available": True,
                "source": "derived",
                "column": None,
                "expression": (
                    f"{self._quote_column(quantity)} * "
                    f"{self._quote_column(price)}"
                ),
                "calculation": (
                    derivation_note
                    + "quantity sold multiplied by unit price; discount, tax, "
                    "shipping, and refunds beyond negative quantities are "
                    "not adjusted."
                )
            }
        else:
            metrics["revenue"] = {
                "available": False,
                "source": None,
                "column": None,
                "expression": None,
                "calculation": None
            }

        if profit is not None:
            metrics["profit"] = {
                "available": True,
                "source": "column",
                "column": profit,
                "expression": None,
                "calculation": "Existing profit column; retained as supplied."
            }
        elif metrics["revenue"]["available"] and cost is not None:
            derivation_note = (
                "No unambiguous existing profit column was selected; "
                if mappings["profit"]["ambiguous"]
                else ""
            )
            revenue_expression = (
                self._quote_column(revenue)
                if revenue is not None
                else metrics["revenue"]["expression"]
            )
            metrics["profit"] = {
                "available": True,
                "source": "derived",
                "column": None,
                "expression": (
                    f"({revenue_expression}) - {self._quote_column(cost)}"
                ),
                "calculation": (
                    derivation_note
                    + "revenue minus cost; the cost column is assumed to "
                    "represent the same row-level granularity."
                )
            }
        else:
            metrics["profit"] = {
                "available": False,
                "source": None,
                "column": None,
                "expression": None,
                "calculation": None
            }
        return metrics

    @staticmethod
    def _quote_column(column):
        return '"' + str(column).replace('"', '""') + '"'

    def _readiness(
        self,
        df,
        mappings,
        derived_metrics,
        default_lead_time_days=None
    ):
        warnings = self._column_quality_warnings(mappings)
        date_col = self._field_column(mappings, "date")
        quantity_col = self._field_column(mappings, "quantity")
        revenue_col = self._field_column(mappings, "revenue")
        product_id = self._field_column(mappings, "product_id")
        product_name = self._field_column(mappings, "product_name")
        product_col = product_id or product_name

        date_values = (
            self._date_values(self._column(df, date_col))
            if date_col is not None
            else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        )
        quantity_values = (
            self._numeric_values(self._column(df, quantity_col))
            if quantity_col is not None
            else None
        )
        revenue_values = (
            self._numeric_values(self._column(df, revenue_col))
            if revenue_col is not None
            else None
        )
        if revenue_values is None and derived_metrics["revenue"]["available"]:
            revenue_sources = [
                self._field_column(mappings, "quantity"),
                self._field_column(mappings, "price")
            ]
            if all(source is not None for source in revenue_sources):
                left = self._numeric_values(
                    self._column(df, revenue_sources[0])
                )
                right = self._numeric_values(
                    self._column(df, revenue_sources[1])
                )
                revenue_values = left * right

        transaction_value_valid = pd.Series(
            False,
            index=df.index,
            dtype=bool
        )
        if quantity_values is not None:
            transaction_value_valid |= quantity_values.notna()
        if revenue_values is not None:
            transaction_value_valid |= revenue_values.notna()
        valid_transactions = date_values.notna() & transaction_value_valid

        base_warnings = list(warnings)
        if date_col is not None and date_values.isna().any():
            base_warnings.append(
                "Some transaction dates are missing or could not be parsed."
            )
        if quantity_col is not None:
            negative_count = int(
                (quantity_values < 0).fillna(False).sum()
            )
            if negative_count:
                base_warnings.append(
                    f"{negative_count} negative quantities may represent "
                    "returns and are retained with their sign."
                )
            if quantity_values.isna().any():
                base_warnings.append(
                    "Some quantity values are missing, non-numeric, or "
                    "non-finite."
                )

        forecast_missing = []
        if product_col is None:
            forecast_missing.append("Product identifier or product name")
        if date_col is None:
            forecast_missing.append("Transaction date")
        if quantity_col is None:
            forecast_missing.append("Quantity sold")

        usable_weeks = 0
        if (
            not forecast_missing
            and product_col is not None
            and quantity_values is not None
        ):
            valid_product = self._column(df, product_col).notna()
            usable = pd.DataFrame({
                "_product": self._column(df, product_col),
                "_date": date_values,
                "_quantity": quantity_values
            }).loc[valid_product & date_values.notna() & quantity_values.notna()]
            if not usable.empty:
                usable["_week"] = (
                    usable["_date"].dt.to_period("W-SUN").dt.start_time
                )
                usable_weeks = int(
                    usable.groupby("_product")["_week"].nunique().max()
                )
                if usable_weeks < self.min_forecast_weeks:
                    forecast_missing.append(
                        f"{self.min_forecast_weeks} usable weekly observations "
                        "for at least one product"
                    )

        forecast_warnings = list(base_warnings)
        if product_id is None and product_name is not None:
            forecast_warnings.append(
                "No product identifier was detected; forecasts are grouped "
                "by product name."
            )
        forecasting_ready = not forecast_missing and usable_weeks >= (
            self.min_forecast_weeks
        )
        forecasting = {
            "ready": forecasting_ready,
            "missing_fields": forecast_missing,
            "warnings": forecast_warnings,
            "explanation": (
                "Product, date, and quantity data contain sufficient weekly "
                "observations."
                if forecasting_ready
                else "Forecasting needs valid product, date, and quantity data "
                "with sufficient weekly history."
            ),
            "usable_weeks": usable_weeks,
            "required_weeks": self.min_forecast_weeks
        }

        inventory_missing = list(forecast_missing)
        stock_col = self._field_column(mappings, "stock")
        lead_time_col = self._field_column(mappings, "lead_time")
        if stock_col is None:
            inventory_missing.append("Inventory stock quantity")
        elif len(df):
            stock_values = self._numeric_values(self._column(df, stock_col))
            valid_stock = stock_values.notna() & (stock_values >= 0)
            if not valid_stock.any():
                inventory_missing.append("Valid non-negative stock values")
            elif not valid_stock.all():
                base_warnings.append(
                    "Some stock values are missing, non-numeric, non-finite, "
                    "or negative."
                )

        try:
            numeric_default_lead_time = float(default_lead_time_days)
        except (TypeError, ValueError):
            numeric_default_lead_time = None
        valid_default_lead_time = (
            numeric_default_lead_time is not None
            and math.isfinite(numeric_default_lead_time)
            and numeric_default_lead_time > 0
        )
        if lead_time_col is None and not valid_default_lead_time:
            inventory_missing.append(
                "Valid lead time column or positive configured default"
            )
        elif lead_time_col is not None and len(df):
            lead_values = self._numeric_values(
                self._column(df, lead_time_col)
            )
            valid_lead_time = lead_values.notna() & (lead_values > 0)
            if not valid_lead_time.any():
                inventory_missing.append("Valid positive lead-time values")
            elif not valid_lead_time.all():
                base_warnings.append(
                    "Some lead-time values are missing, non-numeric, or "
                    "non-positive."
                )

        inventory_ready = (
            forecasting_ready
            and stock_col is not None
            and (
                lead_time_col is not None
                or valid_default_lead_time
            )
            and len(inventory_missing) == 0
        )
        inventory = {
            "ready": inventory_ready,
            "missing_fields": inventory_missing,
            "warnings": list(base_warnings),
            "explanation": (
                "Demand history, stock, and lead-time inputs are available."
                if inventory_ready
                else "Inventory recommendations need valid demand history, "
                "stock, and lead-time data."
            )
        }

        sales_missing = []
        if date_col is None:
            sales_missing.append("Transaction date")
        if (
            quantity_col is None
            and not derived_metrics["revenue"]["available"]
        ):
            sales_missing.append("Quantity or monetary sales data")
        valid_sales_count = int(valid_transactions.sum())
        if not valid_sales_count:
            sales_missing.append("At least one valid transaction")
        sales_analysis = {
            "ready": not sales_missing,
            "missing_fields": sales_missing,
            "warnings": list(base_warnings),
            "explanation": (
                "Valid transaction and sales measures are available."
                if not sales_missing
                else "Sales analysis requires valid transactions and "
                "quantity or monetary values."
            )
        }

        customer_missing = []
        customer_col = self._field_column(mappings, "customer_id")
        if customer_col is None:
            customer_missing.append("Customer identifier")
        if not valid_sales_count:
            customer_missing.append("Usable transaction data")
        if (
            customer_col is not None
            and not (
                valid_transactions
                & self._column(df, customer_col).notna()
            ).any()
        ):
            customer_missing.append("Non-null customer identifiers")
        customer_analysis = {
            "ready": not customer_missing,
            "missing_fields": customer_missing,
            "warnings": list(base_warnings),
            "explanation": (
                "Customer identifiers and usable transaction data are present."
                if not customer_missing
                else "Customer analysis needs a customer identifier and "
                "usable transactions."
            )
        }

        return {
            "forecasting": forecasting,
            "inventory": inventory,
            "sales_analysis": sales_analysis,
            "customer_analysis": customer_analysis
        }

    def _assemble_profile(
        self,
        df,
        column_profiles,
        field_mappings,
        default_lead_time_days=None
    ):
        mapped = {
            role: (
                field_mappings[role]["column"]
                if not field_mappings[role]["ambiguous"]
                else None
            )
            for role in ROLE_LABELS
        }
        derived_metrics = self._derived_metrics(field_mappings)

        profile = {
            "product_id": mapped["product_id"],
            "product_name": mapped["product_name"],
            "product": mapped["product_id"] or mapped["product_name"],
            "product_grouping_source": (
                "product_id"
                if mapped["product_id"]
                else "product_name"
                if mapped["product_name"]
                else None
            ),
            "date": mapped["date"],
            "quantity": mapped["quantity"],
            "price": mapped["price"],
            "unit_price": mapped["price"],
            "revenue": mapped["revenue"],
            "order_id": mapped["order_id"],
            "customer_id": mapped["customer_id"],
            "stock": mapped["stock"],
            "lead_time": mapped["lead_time"],
            "supplier_id": mapped["supplier_id"],
            "category": mapped["category"],
            "discount": mapped["discount"],
            "cost": mapped["cost"],
            "profit": mapped["profit"],
            "country_region": mapped["country_region"],
            "country": mapped["country_region"],
            "all_columns": list(df.columns),
            "original_columns": [
                profile["original_column"]
                for profile in column_profiles
            ],
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "column_profiles": column_profiles,
            "field_mappings": field_mappings,
            "ambiguous_mappings": [
                field_mappings[role]
                for role in ROLE_LABELS
                if field_mappings[role]["ambiguous"]
            ],
            "derived_metrics": derived_metrics
        }

        business_concepts = {}
        for role, column in mapped.items():
            if column is not None:
                business_concepts[role] = column
        if mapped["price"] is not None:
            business_concepts["unit_price"] = mapped["price"]
        profile["business_concepts"] = business_concepts
        profile["metric_capabilities"] = {
            "total_revenue": derived_metrics["revenue"]["available"],
            "total_profit": derived_metrics["profit"]["available"],
            "total_orders": mapped["order_id"] is not None,
            "total_quantity": mapped["quantity"] is not None,
            "unique_customers": mapped["customer_id"] is not None,
            "top_products": (
                profile["product"] is not None
                and mapped["quantity"] is not None
            )
        }

        readiness = self._readiness(
            df,
            field_mappings,
            derived_metrics,
            default_lead_time_days
        )
        profile["readiness"] = readiness
        profile["forecast_ready"] = readiness["forecasting"]["ready"]
        profile["inventory_ready"] = readiness["inventory"]["ready"]
        profile["sales_analysis_ready"] = readiness["sales_analysis"]["ready"]
        profile["customer_analysis_ready"] = (
            readiness["customer_analysis"]["ready"]
        )
        return profile

    def analyze(
        self,
        df,
        default_lead_time_days=None,
        original_column_names=None
    ):
        column_profiles = self._profile_dataframe(
            df,
            original_column_names
        )
        field_mappings = detect_role_candidates(column_profiles)
        return self._assemble_profile(
            df,
            column_profiles,
            field_mappings,
            default_lead_time_days
        )

    @staticmethod
    def _validate_manual_column(df, role, column, column_profiles):
        matching = [
            profile for profile in column_profiles
            if profile["column"] == column
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Select one unique existing column for {ROLE_LABELS[role]}."
            )
        profile = matching[0]
        if profile["non_null_count"] == 0:
            raise ValueError(
                f"{ROLE_LABELS[role]} cannot use an all-null column."
            )
        if role in NUMERIC_ROLES and profile["numeric_success_rate"] < 0.7:
            raise ValueError(
                f"{ROLE_LABELS[role]} must contain mostly numeric values."
            )
        if role == "date" and profile["date_parse_success_rate"] < 0.7:
            raise ValueError(
                "Transaction Date must contain mostly parseable date values."
            )
        return profile

    def apply_overrides(
        self,
        df,
        overrides,
        default_lead_time_days=None,
        original_column_names=None
    ):
        """Validate user-selected mappings and rebuild the readiness profile."""
        unknown_roles = set(overrides) - set(ROLE_LABELS)
        if unknown_roles:
            raise ValueError(
                "Unknown semantic mapping(s): "
                + ", ".join(sorted(unknown_roles))
            )

        column_profiles = self._profile_dataframe(
            df,
            original_column_names
        )
        field_mappings = detect_role_candidates(column_profiles)
        for role, column in overrides.items():
            if column is None:
                field_mappings[role] = {
                    "column": None,
                    "original_column": None,
                    "semantic_role": ROLE_LABELS[role],
                    "confidence": "Low confidence",
                    "confidence_score": 0.0,
                    "detection_method": "manual",
                    "reason": "Manually left unmapped.",
                    "ambiguous": False,
                    "candidates": []
                }
                continue

            candidate_profile = self._validate_manual_column(
                df,
                role,
                column,
                column_profiles
            )
            field_mappings[role] = {
                "column": column,
                "original_column": candidate_profile["original_column"],
                "semantic_role": ROLE_LABELS[role],
                "confidence": "High confidence",
                "confidence_score": 1.0,
                "detection_method": "manual",
                "reason": "Selected and validated manually by the user.",
                "ambiguous": False,
                "candidates": [{
                    "column": column,
                    "original_column": candidate_profile["original_column"],
                    "column_index": candidate_profile["column_index"],
                    "confidence_score": 1.0,
                    "detection_method": "manual",
                    "reason": "Selected by the user."
                }]
            }

        assigned_columns = {}
        for role, field in field_mappings.items():
            if field["column"] is not None and not field["ambiguous"]:
                assigned_columns.setdefault(field["column"], []).append(role)
        conflicts = [
            roles for roles in assigned_columns.values()
            if len(roles) > 1
        ]
        if conflicts:
            labels = ", ".join(
                ROLE_LABELS[role]
                for roles in conflicts
                for role in roles
            )
            raise ValueError(
                "A physical column cannot be mapped to multiple semantic "
                f"roles: {labels}."
            )

        return self._assemble_profile(
            df,
            column_profiles,
            field_mappings,
            default_lead_time_days
        )
