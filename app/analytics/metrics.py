class MetricPlanner:

    METRICS = {
        "total_revenue": {
            "description": "Total revenue generated",
            "formula": "SUM(revenue)"
        },
        "total_orders": {
            "description": "Number of orders",
            "formula": "COUNT(DISTINCT order_id)"
        },
        "total_quantity": {
            "description": "Total units sold",
            "formula": "SUM(quantity)"
        },
        "average_order_value": {
            "description": "Average revenue per order",
            "formula": "SUM(revenue) / COUNT(DISTINCT order_id)"
        },
        "unique_customers": {
            "description": "Number of unique customers",
            "formula": "COUNT(DISTINCT customer_id)"
        }
    }

    @staticmethod
    def get_available_metrics(schema_profile):
        available = []

        if schema_profile.get("revenue"):
            available.append("total_revenue")

        if schema_profile.get("order_id"):
            available.append("total_orders")

        if schema_profile.get("quantity"):
            available.append("total_quantity")

        if (
            schema_profile.get("revenue")
            and schema_profile.get("order_id")
        ):
            available.append("average_order_value")

        if schema_profile.get("customer_id"):
            available.append("unique_customers")

        return available
        