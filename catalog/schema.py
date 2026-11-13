"""BigQuery schema for the products table, plus a local row check.

The schema is plain data so it can be checked without the BigQuery client
installed; load_bigquery.py turns it into SchemaField objects.
"""

import math

# (name, type, mode, description)
PRODUCT_FIELDS = [
    ("product_id", "STRING", "REQUIRED", "ABO item_id"),
    ("name", "STRING", "REQUIRED", "English product name"),
    ("brand", "STRING", "NULLABLE", "Brand name"),
    ("product_type", "STRING", "REQUIRED", "ABO product_type"),
    ("category", "STRING", "REQUIRED", "Store category derived from product_type"),
    ("colors", "STRING", "REPEATED", "Colour text as listed"),
    ("color_families", "STRING", "REPEATED", "Standardized colour names from ABO"),
    ("materials", "STRING", "REPEATED", "Material text as listed"),
    ("styles", "STRING", "REPEATED", "Style text as listed"),
    ("height_cm", "FLOAT", "NULLABLE", "Height in centimetres"),
    ("width_cm", "FLOAT", "NULLABLE", "ABO width in cm (not always the long side)"),
    ("length_cm", "FLOAT", "NULLABLE", "ABO length in cm (not always the depth)"),
    ("longest_side_cm", "FLOAT", "NULLABLE", "Longer of width and length: use for 'how wide'"),
    ("shortest_side_cm", "FLOAT", "NULLABLE", "Shorter of width and length: use for 'how deep'"),
    ("weight_kg", "FLOAT", "NULLABLE", "Weight in kilograms"),
    ("bullet_points", "STRING", "REPEATED", "English bullet points (max 10)"),
    ("keywords", "STRING", "REPEATED", "English search keywords (max 30)"),
    ("browse_path", "STRING", "NULLABLE", "ABO browse node path"),
    ("main_image_id", "STRING", "NULLABLE", "ABO main image id"),
    ("other_image_ids", "STRING", "REPEATED", "ABO other image ids"),
    ("has_3d_model", "BOOLEAN", "REQUIRED", "Whether ABO has a 3D model"),
    ("main_image_path", "STRING", "NULLABLE", "Path of the main image in ABO's image folders"),
    ("price_inr", "INTEGER", "REQUIRED", "Synthetic price in rupees (see catalog/pricing.py)"),
]

FIELD_NAMES = [name for name, *_ in PRODUCT_FIELDS]


def _type_ok(value, field_type):
    if field_type == "DATE":
        return isinstance(value, str) and len(value) == 10 and value[4] == "-" and value[7] == "-"
    if field_type == "STRING":
        return isinstance(value, str)
    if field_type == "INTEGER":
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type == "FLOAT":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if field_type == "BOOLEAN":
        return isinstance(value, bool)
    raise ValueError(f"Unhandled field type {field_type}")


def row_errors(row, fields=PRODUCT_FIELDS):
    """Return a list of problems that would make BigQuery reject this row."""
    errors = []
    extra = set(row) - {name for name, *_ in fields}
    if extra:
        errors.append(f"unknown fields: {sorted(extra)}")
    for name, field_type, mode, _ in fields:
        value = row.get(name)
        if mode == "REPEATED":
            if value is None:
                continue
            if not isinstance(value, list):
                errors.append(f"{name}: expected a list")
            elif not all(_type_ok(v, field_type) for v in value):
                errors.append(f"{name}: list items must be {field_type}")
        elif value is None:
            if mode == "REQUIRED":
                errors.append(f"{name}: missing")
        elif not _type_ok(value, field_type):
            errors.append(f"{name}: expected {field_type}")
    return errors
