
class MetricPlanner:

    @staticmethod
    def plan(metric, profile):

        concepts = profile.get(
            "business_concepts", {}
        )

        capabilities = profile.get(
            "metric_capabilities", {}
        )

        available = set(concepts.keys())

        plan = {
            "metric": metric,
            "supported": False,
            "required_concepts": [],
            "available_concepts": [],
            "missing_concepts": [],
            "calculation_logic": "",
            "warnings": []
        }

        def finish(required, logic):

            plan["required_concepts"] = required

            plan["available_concepts"] = [
                concept for concept in required
                if concept in available
            ]

            plan["missing_concepts"] = [
                concept for concept in required
                if concept not in available
            ]

            plan["supported"] = (
                len(plan["missing_concepts"]) == 0
            )

            plan["calculation_logic"] = logic

            return plan

        if metric == "average_order_value":

            revenue_sources = [
                concept for concept in [
                    "line_total",
                    "revenue",
                    "quantity",
                    "unit_price",
                    "order_total",
                    "total_amount"
                ]
                if concept in available
            ]

            if "order_id" not in available:

                return finish(
                    ["order_id"],
                    "Cannot calculate AOV without an order identifier."
                )

            if not revenue_sources:

                return finish(
                    ["order_id", "revenue"],
                    "Order revenue is not available."
                )

            plan["supported"] = True

            plan["required_concepts"] = [
                "order_id",
                "revenue"
            ]

            plan["available_concepts"] = [
                "order_id"
            ] + revenue_sources

            plan["calculation_logic"] = """
            Determine the revenue representation first.

            If each row represents an order line:
                Aggregate line revenue by order ID,
                then calculate AVG(order_total).

            If each row represents one complete order:
                Calculate AVG(order_total) directly.

            If the revenue field is a pre-aggregated
            sales amount, verify its granularity before
            using it.

            Never assume that every row is one order.
            """

            return plan

        if metric == "total_revenue":

            if "revenue" in available:
                return finish(
                    ["revenue"],
                    "Aggregate revenue using the verified revenue column."
                )

            if "line_total" in available:
                return finish(
                    ["line_total"],
                    "SUM(line_total)."
                )

            if (
                "quantity" in available
                and "unit_price" in available
            ):
                return finish(
                    ["quantity", "unit_price"],
                    "SUM(quantity * unit_price)."
                )

            return finish(
                ["revenue"],
                "Revenue cannot be determined reliably."
            )

        if metric == "total_orders":

            return finish(
                ["order_id"],
                "COUNT(DISTINCT order_id)."
            )

        if metric == "total_customers":

            return finish(
                ["customer_id"],
                "COUNT(DISTINCT customer_id)."
            )

        if metric == "top_products":

            if "product_id" in available or "product_name" in available:

                if "quantity" in available:

                    return finish(
                        [
                            "quantity",
                            (
                                "product_id"
                                if "product_id" in available
                                else "product_name"
                            )
                        ],
                        "Aggregate quantity by product and rank descending."
                    )

            return finish(
                ["product_id", "quantity"],
                "Product sales ranking is not supported by the available fields."
            )

        return {
            **plan,
            "supported": False,
            "warnings": [
                "No specialized metric rule found. "
                "Use schema-grounded analytical reasoning."
            ]
        }
