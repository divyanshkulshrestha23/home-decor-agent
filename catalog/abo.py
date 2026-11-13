"""Read Amazon Berkeley Objects (ABO) listings and turn them into flat rows.

ABO stores most text fields as lists of {"language_tag", "value"} entries,
and measurements as {"value", "unit"} entries, sometimes with a
"normalized_value". The helpers here are defensive about those shapes
because not every listing has every field.
"""

import gzip
import json
import tarfile

# Product types kept for the store, mapped to the category shown to users.
# Type names come from the dataset's full product_type list
# (python -m catalog.prepare --list-types).
CATEGORY_BY_TYPE = {
    "SOFA": "sofa",
    "CHAIR": "chair",
    "BEAN_BAG_CHAIR": "chair",
    "TABLE": "table",
    "RUG": "rug",
    "HOME_FURNITURE_AND_DECOR": "furniture_and_decor",
    "FURNITURE": "furniture_and_decor",
    "HOME_BED_AND_BATH": "bed_and_bath",
    "FLAT_SHEET": "bed_and_bath",
    "HOME": "home",
    "BED": "bed",
    "BED_FRAME": "bed",
    "HEADBOARD": "bed",
    "OTTOMAN": "ottoman",
    "BENCH": "bench",
    "STOOL_SEATING": "stool",
    "DESK": "desk",
    "CABINET": "storage",
    "SHELF": "storage",
    "DRESSER": "storage",
    "AV_FURNITURE": "storage",
    "STORAGE_BOX": "small_storage",
    "BASKET": "small_storage",
    "LAMP": "lighting",
    "LIGHT_FIXTURE": "lighting",
    "HOME_LIGHTING_AND_LAMPS": "lighting",
    "STRING_LIGHT": "lighting",
    "PILLOW": "soft_furnishing",
    "CURTAIN": "soft_furnishing",
    "WINDOW_SHADE": "soft_furnishing",
    "HOME_MIRROR": "decor",
    "VASE": "decor",
    "PLANTER": "decor",
    "WALL_ART": "decor",
    "CLOCK": "decor",
    "CANDLE": "decor",
    "CANDLE_HOLDER": "decor",
    "PICTURE_FRAME": "decor",
    "SCULPTURE": "decor",
    "FIGURINE": "decor",
    "ARTIFICIAL_PLANT": "decor",
    "ROOM_DIVIDER": "decor",
}

LENGTH_TO_CM = {
    "centimeters": 1.0,
    "centimetres": 1.0,
    "cm": 1.0,
    "millimeters": 0.1,
    "millimetres": 0.1,
    "mm": 0.1,
    "meters": 100.0,
    "metres": 100.0,
    "m": 100.0,
    "inches": 2.54,
    "inch": 2.54,
    "in": 2.54,
    "feet": 30.48,
    "foot": 30.48,
    "ft": 30.48,
}

WEIGHT_TO_KG = {
    "kilograms": 1.0,
    "kg": 1.0,
    "grams": 0.001,
    "g": 0.001,
    "pounds": 0.45359237,
    "pound": 0.45359237,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "ounces": 0.028349523125,
    "ounce": 0.028349523125,
    "oz": 0.028349523125,
}

MAX_BULLET_POINTS = 10
MAX_KEYWORDS = 30


def iter_listings(tar_path):
    """Yield every listing dict from the .json.gz files inside abo-listings.tar."""
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".json.gz")):
                continue
            with gzip.open(tar.extractfile(member), "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def _is_english(entry):
    return isinstance(entry, dict) and str(entry.get("language_tag", "")).startswith("en")


def english_values(entries):
    """Return the English values from a list of language-tagged entries.

    Entries without a language tag are kept too, since some fields
    (such as product_type) never carry one.
    """
    if not isinstance(entries, list):
        return []
    values = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tag = entry.get("language_tag")
        value = entry.get("value")
        if (tag is None or _is_english(entry)) and isinstance(value, str):
            value = value.strip()
            if value and value not in values:
                values.append(value)
    return values


def first_english(entries):
    values = english_values(entries)
    return values[0] if values else None


def has_english_name(listing):
    return any(_is_english(e) for e in listing.get("item_name") or [])


def standard_colors(entries):
    """Collect the standardized colour names ABO provides alongside free text."""
    if not isinstance(entries, list):
        return []
    colors = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for value in entry.get("standardized_values") or []:
            if isinstance(value, str) and value.strip():
                value = value.strip().lower()
                if value not in colors:
                    colors.append(value)
    return colors


def _value_unit(entry):
    if not isinstance(entry, dict):
        return None, None
    value = entry.get("value")
    unit = entry.get("unit")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None, None
    return float(value), str(unit).strip().lower() if unit else None


def convert(entry, table):
    """Convert a measurement entry with the given unit table.

    Uses the value as listed when its unit is known, and falls back to
    ABO's normalized_value (usually inches or pounds) otherwise. The listed
    value is preferred because it is often already metric, which avoids a
    round trip through inches.

    Returns (converted_value, unknown_unit). unknown_unit is the unit string
    when a value exists but no unit is convertible, so the caller can count
    it instead of silently dropping it.
    """
    if isinstance(entry, list):
        entry = entry[0] if entry else None
    if not isinstance(entry, dict):
        return None, None
    candidates = [_value_unit(entry), _value_unit(entry.get("normalized_value"))]
    unknown = None
    for value, unit in candidates:
        if value is None:
            continue
        if unit not in table:
            unknown = unknown or unit
            continue
        if value <= 0:
            return None, None
        return round(value * table[unit], 2), None
    return None, unknown


def dimensions_cm(listing):
    """Return ({"height_cm", "width_cm", "length_cm"}, [unknown units])."""
    dims = listing.get("item_dimensions")
    if isinstance(dims, list):
        dims = dims[0] if dims else None
    result = {"height_cm": None, "width_cm": None, "length_cm": None}
    unknown = []
    if not isinstance(dims, dict):
        return result, unknown
    for axis in ("height", "width", "length"):
        value, bad_unit = convert(dims.get(axis), LENGTH_TO_CM)
        result[f"{axis}_cm"] = value
        if bad_unit:
            unknown.append(bad_unit)
    return result, unknown


def horizontal_span(dims):
    """Return (longest_side_cm, shortest_side_cm) of the floor footprint.

    ABO's "width" and "length" are not used consistently: for a sofa,
    "length" can be the long side and "width" the depth. Filters such as
    "narrower" should use these derived values instead.
    """
    sides = [v for v in (dims["width_cm"], dims["length_cm"]) if v is not None]
    if not sides:
        return None, None
    return max(sides), min(sides)


def weight_kg(listing):
    return convert(listing.get("item_weight"), WEIGHT_TO_KG)


def product_type(listing):
    types = listing.get("product_type")
    if isinstance(types, list) and types and isinstance(types[0], dict):
        value = types[0].get("value")
        if isinstance(value, str):
            return value
    return None


def browse_path(listing):
    """Return the first browse-node path (for example a category trail), if any."""
    nodes = listing.get("node")
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and isinstance(node.get("node_name"), str):
                return node["node_name"]
    return None


def to_row(listing):
    """Flatten one listing into a catalog row (without price).

    Returns (row, unknown_units).
    """
    ptype = product_type(listing)
    dims, unknown = dimensions_cm(listing)
    weight, bad_weight_unit = weight_kg(listing)
    if bad_weight_unit:
        unknown.append(bad_weight_unit)

    longest, shortest = horizontal_span(dims)
    other_images = listing.get("other_image_id") or []
    row = {
        "product_id": listing["item_id"],
        "name": first_english(listing.get("item_name")),
        "brand": first_english(listing.get("brand")),
        "product_type": ptype,
        "category": CATEGORY_BY_TYPE.get(ptype),
        "colors": english_values(listing.get("color")),
        "color_families": standard_colors(listing.get("color")),
        "materials": english_values(listing.get("material")),
        "styles": english_values(listing.get("style")),
        **dims,
        "longest_side_cm": longest,
        "shortest_side_cm": shortest,
        "weight_kg": weight,
        "bullet_points": english_values(listing.get("bullet_point"))[:MAX_BULLET_POINTS],
        "keywords": english_values(listing.get("item_keywords"))[:MAX_KEYWORDS],
        "browse_path": browse_path(listing),
        "main_image_id": listing.get("main_image_id"),
        "other_image_ids": [i for i in other_images if isinstance(i, str)],
        "has_3d_model": bool(listing.get("3dmodel_id")),
        "main_image_path": None,
    }
    return row, unknown
