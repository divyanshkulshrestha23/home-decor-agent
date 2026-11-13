"""BigQuery schemas for the synthetic sales tables."""

CUSTOMER_FIELDS = [
    ("customer_id", "STRING", "REQUIRED", "Customer id"),
    ("city", "STRING", "REQUIRED", "Delivery city"),
    ("state", "STRING", "REQUIRED", "Indian state or union territory"),
    ("city_tier", "INTEGER", "REQUIRED", "1 = metro, 2 = large city, 3 = smaller city"),
    ("signup_date", "DATE", "REQUIRED", "Date the customer created an account"),
]

ORDER_FIELDS = [
    ("order_id", "STRING", "REQUIRED", "Order id"),
    ("customer_id", "STRING", "REQUIRED", "customers.customer_id"),
    ("order_date", "DATE", "REQUIRED", "Date the order was placed"),
    ("channel", "STRING", "REQUIRED", "web or app"),
    ("status", "STRING", "REQUIRED", "delivered or cancelled"),
    ("city", "STRING", "REQUIRED", "Delivery city"),
    ("state", "STRING", "REQUIRED", "Delivery state"),
    ("city_tier", "INTEGER", "REQUIRED", "1 = metro, 2 = large city, 3 = smaller city"),
    ("delivered_date", "DATE", "NULLABLE", "Delivery date; empty for cancelled orders"),
]

ORDER_ITEM_FIELDS = [
    ("order_item_id", "STRING", "REQUIRED", "Order line id"),
    ("order_id", "STRING", "REQUIRED", "orders.order_id"),
    ("order_date", "DATE", "REQUIRED", "Copied from orders for convenience"),
    ("product_id", "STRING", "REQUIRED", "products.product_id"),
    ("category", "STRING", "REQUIRED", "products.category at the time of sale"),
    ("brand", "STRING", "NULLABLE", "products.brand"),
    ("quantity", "INTEGER", "REQUIRED", "Units on this line"),
    ("list_price_inr", "INTEGER", "REQUIRED", "Catalog price per unit, rupees"),
    ("discount_pct", "INTEGER", "REQUIRED", "Discount applied, percent"),
    ("sale_price_inr", "INTEGER", "REQUIRED", "Price paid per unit after discount, rupees"),
    ("line_total_inr", "INTEGER", "REQUIRED", "sale_price_inr x quantity; revenue for this line"),
    ("returned", "BOOLEAN", "REQUIRED", "Whether the line was returned"),
    ("return_reason", "STRING", "NULLABLE", "Reason given for the return"),
    ("return_date", "DATE", "NULLABLE", "Date the return was received"),
]

TABLES = {
    "customers": CUSTOMER_FIELDS,
    "orders": ORDER_FIELDS,
    "order_items": ORDER_ITEM_FIELDS,
}
