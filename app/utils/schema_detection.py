import math
import re
import unicodedata

import pandas as pd


ROLE_LABELS = {
    "product_id": "Product Identifier",
    "product_name": "Product Name / Description",
    "date": "Transaction Date",
    "quantity": "Quantity Sold",
    "price": "Unit Price",
    "revenue": "Revenue / Sales Amount",
    "order_id": "Order ID",
    "customer_id": "Customer ID",
    "stock": "Inventory Stock Quantity",
    "lead_time": "Lead Time",
    "supplier_id": "Supplier ID",
    "category": "Category",
    "discount": "Discount",
    "cost": "Cost",
    "profit": "Profit",
    "country_region": "Country / Region"
}

ROLE_ALIASES = {
    "product_id": {
        "product id": 0.91,
        "product identifier": 0.91,
        "product code": 0.9,
        "productcode": 0.9,
        "item code": 0.9,
        "itemcode": 0.9,
        "item id": 0.9,
        "itemid": 0.9,
        "stock code": 0.9,
        "stockcode": 0.9,
        "sku": 0.94,
        "stock keeping unit": 0.92,
        "part number": 0.88,
        "article id": 0.86,
        "model number": 0.86,
        "product": 0.5,
        "item": 0.5
    },
    "product_name": {
        "product name": 0.92,
        "productname": 0.92,
        "item name": 0.91,
        "itemname": 0.91,
        "product description": 0.91,
        "item description": 0.9,
        "product title": 0.9,
        "description": 0.78,
        "title": 0.75,
        "product": 0.5,
        "item": 0.5
    },
    "date": {
        "transaction date": 0.92,
        "transactiondate": 0.92,
        "order date": 0.92,
        "orderdate": 0.92,
        "invoice date": 0.92,
        "invoicedate": 0.92,
        "purchase date": 0.9,
        "created at": 0.9,
        "createdat": 0.9,
        "created date": 0.9,
        "timestamp": 0.88,
        "date": 0.82,
        "day": 0.7
    },
    "quantity": {
        "quantity sold": 0.94,
        "quantitysold": 0.94,
        "sales quantity": 0.94,
        "salesquantity": 0.94,
        "units sold": 0.94,
        "unitssold": 0.94,
        "order quantity": 0.9,
        "items sold": 0.9,
        "qty": 0.88,
        "quantity": 0.84,
        "units": 0.68
    },
    "price": {
        "unit price": 0.94,
        "unitprice": 0.94,
        "selling price": 0.91,
        "sellingprice": 0.91,
        "item price": 0.88,
        "price per unit": 0.93,
        "price": 0.8
    },
    "revenue": {
        "sales amount": 0.93,
        "total sales": 0.92,
        "total revenue": 0.94,
        "line total": 0.91,
        "order total": 0.9,
        "total amount": 0.88,
        "revenue": 0.9,
        "sales": 0.66,
        "amount": 0.62
    },
    "order_id": {
        "order id": 0.93,
        "orderid": 0.93,
        "order number": 0.91,
        "invoice id": 0.92,
        "invoiceid": 0.92,
        "invoice number": 0.92,
        "invoiceno": 0.91,
        "invoice no": 0.91,
        "transaction id": 0.92,
        "transactionid": 0.92,
        "transaction number": 0.9,
        "order": 0.62,
        "invoice": 0.66
    },
    "customer_id": {
        "customer id": 0.93,
        "customerid": 0.93,
        "customer number": 0.91,
        "client id": 0.9,
        "buyer id": 0.9,
        "consumer id": 0.9
    },
    "stock": {
        "inventory quantity": 0.94,
        "stock quantity": 0.94,
        "available stock": 0.94,
        "inventory level": 0.93,
        "on hand quantity": 0.94,
        "on hand": 0.87,
        "current stock": 0.93,
        "stock level": 0.92,
        "stock": 0.84,
        "inventory": 0.8
    },
    "lead_time": {
        "supplier lead time": 0.94,
        "supplierleadtime": 0.94,
        "delivery lead time": 0.94,
        "lead time days": 0.92,
        "lead time": 0.9
    },
    "supplier_id": {
        "supplier id": 0.93,
        "supplierid": 0.93,
        "vendor id": 0.92,
        "supplier number": 0.9,
        "vendor number": 0.9
    },
    "category": {
        "product category": 0.93,
        "productcategory": 0.93,
        "item category": 0.91,
        "category": 0.82,
        "department": 0.8,
        "product type": 0.84,
        "segment": 0.72
    },
    "discount": {
        "discount amount": 0.94,
        "discount percent": 0.94,
        "discount percentage": 0.94,
        "discount rate": 0.92,
        "discount": 0.84,
        "coupon discount": 0.88
    },
    "cost": {
        "unit cost": 0.94,
        "cost price": 0.91,
        "purchase cost": 0.91,
        "cost of goods sold": 0.94,
        "cogs": 0.92,
        "cost": 0.82
    },
    "profit": {
        "gross profit": 0.94,
        "net profit": 0.94,
        "profit amount": 0.92,
        "profit": 0.88,
        "profit margin": 0.86
    },
    "country_region": {
        "country region": 0.92,
        "country": 0.88,
        "region": 0.84,
        "territory": 0.82,
        "market": 0.7
    }
}

NUMERIC_ROLES = {
    "quantity",
    "price",
    "revenue",
    "stock",
    "lead_time",
    "discount",
    "cost",
    "profit"
}
IDENTIFIER_ROLES = {
    "product_id",
    "order_id",
    "customer_id",
    "supplier_id"
}
CATEGORICAL_ROLES = {
    "product_name",
    "category",
    "country_region"
}


def normalize_column_name(name):
    """Normalize spaces, underscores, punctuation, and CamelCase consistently."""
    value = unicodedata.normalize("NFKC", str(name))
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _sample(series, limit=1000):
    if len(series) <= limit:
        return series
    step = max(1, len(series) // limit)
    return series.iloc[::step].head(limit)


def _sample_text(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)[:100]


def profile_column(series, name, index, original_name=None):
    """Collect bounded type evidence and whole-column cardinality statistics."""
    non_null = series.dropna()
    sample = _sample(non_null)
    numeric_sample = pd.to_numeric(sample, errors="coerce")
    numeric_sample_valid = numeric_sample.map(
        lambda value: pd.notna(value) and math.isfinite(float(value))
    )
    numeric_rate = (
        float(numeric_sample_valid.mean())
        if len(sample)
        else 0.0
    )
    normalized_name = normalize_column_name(name)
    datetime_dtype = pd.api.types.is_datetime64_any_dtype(series.dtype)
    date_name = any(
        word in normalized_name.split()
        for word in ("date", "time", "timestamp", "day")
    )
    sample_has_date_shape = sample.astype(str).str.contains(
        r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[T ]\d{1,2}:\d{2}",
        regex=True
    ).mean() if len(sample) else 0.0

    if datetime_dtype:
        date_rate = 1.0
    elif date_name or sample_has_date_shape >= 0.5:
        parsed_dates = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed",
            utc=True
        )
        date_rate = (
            float(parsed_dates.notna().mean())
            if len(sample)
            else 0.0
        )
    else:
        date_rate = 0.0

    unique_count = int(non_null.nunique(dropna=True))
    non_null_count = int(len(non_null))
    numeric_values = None
    if pd.api.types.is_numeric_dtype(series.dtype) or numeric_rate >= 0.8:
        numeric_values = pd.to_numeric(non_null, errors="coerce")
        numeric_values = numeric_values.where(numeric_values.map(
            lambda value: pd.isna(value) or math.isfinite(float(value))
        ))
    numeric_valid_count = (
        int(numeric_values.notna().sum())
        if numeric_values is not None
        else 0
    )
    numeric_range = (
        numeric_values.dropna()
        if numeric_values is not None
        else pd.Series(dtype=float)
    )

    if datetime_dtype or date_rate >= 0.8:
        data_type = "datetime"
    elif numeric_rate >= 0.8:
        data_type = "numeric"
    else:
        data_type = "categorical"

    text_lengths = [
        len(str(value))
        for value in sample
        if isinstance(value, str)
    ]
    mean_text_length = (
        sum(text_lengths) / len(text_lengths)
        if text_lengths
        else 0.0
    )
    text_with_spaces = (
        sum(" " in str(value).strip() for value in sample)
        / len(sample)
        if len(sample)
        else 0.0
    )

    return {
        "column": str(name),
        "original_column": str(
            name if original_name is None else original_name
        ),
        "column_index": int(index),
        "normalized_name": normalized_name,
        "dtype": str(series.dtype),
        "data_type": data_type,
        "row_count": int(len(series)),
        "non_null_count": non_null_count,
        "null_percentage": (
            round(1 - non_null_count / len(series), 4)
            if len(series)
            else 1.0
        ),
        "unique_count": unique_count,
        "unique_ratio": (
            round(unique_count / non_null_count, 4)
            if non_null_count
            else 0.0
        ),
        "mostly_unique": bool(
            non_null_count and unique_count / non_null_count >= 0.9
        ),
        "numeric_success_rate": round(numeric_rate, 4),
        "numeric_valid_count": numeric_valid_count,
        "numeric_min": (
            float(numeric_range.min())
            if len(numeric_range)
            else None
        ),
        "numeric_max": (
            float(numeric_range.max())
            if len(numeric_range)
            else None
        ),
        "negative_count": (
            int((numeric_values < 0).sum())
            if numeric_values is not None
            else 0
        ),
        "zero_count": (
            int((numeric_values == 0).sum())
            if numeric_values is not None
            else 0
        ),
        "date_parse_success_rate": round(date_rate, 4),
        "mean_text_length": round(mean_text_length, 2),
        "text_with_spaces_ratio": round(text_with_spaces, 4),
        "sample_values": [
            _sample_text(value)
            for value in sample.head(5)
        ]
    }


def _name_score(role, normalized_name):
    aliases = ROLE_ALIASES[role]
    if normalized_name in aliases:
        return aliases[normalized_name], "exact semantic alias"

    tokens = normalized_name.split()
    for alias, score in aliases.items():
        alias_tokens = alias.split()
        if len(alias_tokens) > 1:
            for start in range(len(tokens) - len(alias_tokens) + 1):
                if tokens[start:start + len(alias_tokens)] == alias_tokens:
                    return score - 0.08, "semantic alias phrase"
    return None, None


def _score_candidate(role, profile):
    name_score, match_method = _name_score(
        role,
        profile["normalized_name"]
    )
    if name_score is None or profile["non_null_count"] == 0:
        return None

    score = name_score
    reasons = [match_method]

    if role in NUMERIC_ROLES:
        numeric_rate = profile["numeric_success_rate"]
        if numeric_rate < 0.7:
            return None
        score += 0.1 * numeric_rate
        reasons.append(
            f"{numeric_rate:.0%} of sampled values are numeric"
        )

    if role == "date":
        date_rate = profile["date_parse_success_rate"]
        if date_rate < 0.7:
            return None
        score += 0.12 * date_rate
        reasons.append(
            f"{date_rate:.0%} of sampled values parse as dates"
        )

    if role in IDENTIFIER_ROLES:
        if profile["unique_count"] <= 1:
            score -= 0.2
            reasons.append("only one distinct non-null value")
        elif profile["unique_ratio"] >= 0.9:
            score += 0.08
            reasons.append("values are mostly unique")
        elif profile["unique_ratio"] >= 0.4:
            score += 0.04
            reasons.append("moderate identifier cardinality")

    if role == "product_id":
        if profile["text_with_spaces_ratio"] >= 0.3:
            score -= 0.12
            reasons.append("values resemble descriptions rather than codes")
        elif (
            profile["mostly_unique"]
            and profile["mean_text_length"] > 0
            and profile["mean_text_length"] <= 24
        ):
            score += 0.06
            reasons.append("short, mostly unique values resemble product codes")

    if role == "product_name":
        if profile["data_type"] == "numeric":
            return None
        elif profile["text_with_spaces_ratio"] >= 0.3:
            score += 0.1
            reasons.append("text values resemble product descriptions")

    if role in CATEGORICAL_ROLES and profile["data_type"] == "numeric":
        score -= 0.12
        reasons.append("numeric-only values are atypical for this role")

    null_percentage = profile["null_percentage"]
    if null_percentage >= 0.8:
        score -= 0.2
        reasons.append("at least 80% of values are null")
    elif null_percentage >= 0.5:
        score -= 0.1
        reasons.append("at least 50% of values are null")

    score = max(0.0, min(0.99, score))
    return {
        "column": profile["column"],
        "original_column": profile["original_column"],
        "column_index": profile["column_index"],
        "confidence_score": round(score, 4),
        "detection_method": f"{match_method} + data profile",
        "reason": "; ".join(reasons)
    }


def detect_role_candidates(profiles):
    """Score columns for every role and mark weak or ambiguous assignments."""
    name_counts = {}
    for profile in profiles:
        source_name = normalize_column_name(profile["original_column"])
        name_counts[source_name] = (
            name_counts.get(source_name, 0) + 1
        )

    candidates_by_role = {
        role: sorted(
            (
                candidate
                for profile in profiles
                if (candidate := _score_candidate(role, profile)) is not None
            ),
            key=lambda candidate: (
                candidate["confidence_score"],
                candidate["column_index"]
            ),
            reverse=True
        )
        for role in ROLE_LABELS
    }

    selected = {}
    ambiguous_roles = set()
    for role, candidates in candidates_by_role.items():
        top = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        if not top or top["confidence_score"] < 0.65:
            continue
        if name_counts[
            normalize_column_name(top["original_column"])
        ] > 1:
            ambiguous_roles.add(role)
            continue
        if (
            second
            and top["confidence_score"] - second["confidence_score"] < 0.08
        ):
            ambiguous_roles.add(role)
            continue
        selected[role] = top

    assigned_columns = {}
    for role, candidate in selected.items():
        assigned_columns.setdefault(candidate["column_index"], []).append(role)

    for column_roles in assigned_columns.values():
        if len(column_roles) < 2:
            continue
        ranked_roles = sorted(
            column_roles,
            key=lambda role: selected[role]["confidence_score"],
            reverse=True
        )
        best_score = selected[ranked_roles[0]]["confidence_score"]
        next_score = selected[ranked_roles[1]]["confidence_score"]
        if best_score - next_score < 0.15:
            ambiguous_roles.update(column_roles)
            for role in column_roles:
                selected.pop(role, None)
        else:
            for role in ranked_roles[1:]:
                ambiguous_roles.add(role)
                selected.pop(role, None)

    mappings = {}
    for role, label in ROLE_LABELS.items():
        candidates = candidates_by_role[role]
        chosen = selected.get(role)
        if chosen:
            score = chosen["confidence_score"]
            confidence = "High confidence" if score >= 0.85 else "Medium confidence"
            reason = chosen["reason"]
            method = chosen["detection_method"]
            column = chosen["column"]
            original_column = chosen["original_column"]
            is_ambiguous = False
        else:
            score = (
                candidates[0]["confidence_score"]
                if candidates
                else 0.0
            )
            confidence = "Low confidence"
            column = None
            original_column = None
            method = "name and data profile"
            is_ambiguous = role in ambiguous_roles
            if is_ambiguous:
                reason = (
                    "Multiple columns or semantic roles are similarly "
                    "plausible; confirm the mapping manually."
                )
            elif candidates:
                reason = (
                    "Evidence is below the automatic-mapping confidence "
                    "threshold; confirm or leave unmapped."
                )
            else:
                reason = "No matching column with sufficient data evidence."

        mappings[role] = {
            "column": column,
            "original_column": original_column,
            "semantic_role": label,
            "confidence": confidence,
            "confidence_score": round(score, 4),
            "detection_method": method,
            "reason": reason,
            "ambiguous": is_ambiguous,
            "candidates": candidates[:5]
        }

    return mappings


def confidence_for_score(score):
    if score >= 0.85:
        return "High confidence"
    if score >= 0.65:
        return "Medium confidence"
    return "Low confidence"
