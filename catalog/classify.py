"""Assign store categories to products whose ABO product_type is too broad.

Types such as HOME or HOME_FURNITURE_AND_DECOR mix picture frames,
bedsheets, storage boxes, tools and phone accessories. For those types the
category comes from the product's browse path (its category trail), and,
when the path is generic (just "/Categories" or missing), from words in
the product name. Anything that matches no rule is left out of the store.

The rules were written from the most common browse paths per broad type
(python -m catalog.prepare --browse-paths ...). Rules are checked in order,
so more specific rules come before more general ones.
"""

import re

BROAD_TYPES = {
    "HOME",
    "HOME_BED_AND_BATH",
    "HOME_FURNITURE_AND_DECOR",
    "FURNITURE",
    "OUTDOOR_LIVING",
}

# Paths that mean "not something this store sells", checked first.
EXCLUDE_PATHS = [
    "bean bag covers", "clothes covers", "grill", "fire pit covers", "abdeckungen",
    "garden hoses", "watering", "riveters", "tools & equipment", "event & party",
    "smartwatch", "携帯電話", "outdoor heating", "outdoor carts", "outdoor cooking",
]
# Excluded only when the path ends with them: "Photo Albums" alone is not
# decor, but ".../Photo Albums, Frames & Accessories/Picture Frames" is.
EXCLUDE_PATH_ENDINGS = ["/photo albums"]

# (category, path substrings), lowercase, checked in order.
PATH_RULES = [
    # Umbrellas sit under "Patio Furniture & Accessories", so decor goes first.
    ("outdoor_decor", ["umbrellas & shade", "outdoor décor", "birdbath"]),
    ("outdoor_furniture", ["patio furniture", "patio seating", "patio sofas",
                           "garden & outdoor furniture", "hammock"]),
    ("lighting", ["outdoor lighting", "lighting & ceiling fans", "lamps"]),
    ("chair", ["bean bags", "desk chairs", "recliners", "/chairs"]),
    ("rug", ["area rugs", "rugs, pads", "runners"]),
    ("bath_linen", ["bath mats", "towels", "bath linen"]),
    ("window_hardware", ["window rods", "finials", "window treatment hardware"]),
    ("curtains", ["curtains", "vorhänge"]),
    ("cushions", ["cushions", "クッション"]),
    ("bedding", ["bedding", "bedsheets", "sheets & pillowcases", "duvet", "comforter",
                 "mattresses protectors", "blankets, throws", "bettwaren", "bettwäsche",
                 "bettlaken", "bettdecken", "spannbetttücher", "pillow shams"]),
    ("bed", ["mattresses", "beds, frames", "bedroom furniture"]),
    ("small_storage", ["boxes, baskets & bins", "laundry bins", "laundry hampers"]),
    ("storage", ["tv & media furniture", "media storage", "wardrobe", "wall shelves",
                 "kitchen islands", "bar & serving carts", "racks, shelves & drawers",
                 "closet systems", "closet storage"]),
    ("desk", ["desks"]),
    ("table", ["dining room sets", "table & chair sets", "/tables"]),
    ("ottoman", ["ottomans", "footstools"]),
    ("sofa", ["sofas & couches"]),
    ("decor", ["clocks", "frames", "bilderrahmen", "wall stickers", "posters & prints",
               "artwork", "wall decor", "decorative accessories", "home décor",
               "home & décor"]),
    ("accent_furniture", ["accent furniture", "living room furniture",
                          "office furniture", "/furniture"]),
]

# Paths excluded only if no rule above matched them first. "Kitchen & Dining"
# appears in a rug archive path and in "Kitchen & Dining Room Furniture",
# which the rules above already claim.
EXCLUDE_PATHS_LATE = ["kitchen & dining", "gardening", "painting supplies, tools"]

# Names of covers are left out (bean bag covers, sofa covers), except
# bedding and cushion covers, which are products in their own right.
EXCLUDE_NAME = re.compile(r"\bcovers?\b", re.IGNORECASE)
COVER_OK = re.compile(r"\b(duvet|cushion|pillow|quilt)\s+covers?\b", re.IGNORECASE)

# (category, regex) on the product name, checked in order. Word boundaries
# stop "table" matching "tablet" and "light" matching "lightweight".
NAME_RULES = [
    ("curtains", r"\bcurtains?\b"),
    ("bedding", r"\b(bed ?sheets?|fitted sheets?|flat sheets?|duvets?|comforters?|quilts?"
                r"|pillowcases?|pillow cases?|bedspreads?|blankets?|sheet sets?)\b"),
    ("bath_linen", r"\b(towels?|bath mats?)\b"),
    ("cushions", r"\b(cushions?|pillows?)\b"),
    ("lighting", r"\b(lamps?|lights?|chandeliers?|lanterns?|sconces?)\b"),
    ("rug", r"\b(rugs?|runners?|doormats?)\b"),
    ("decor", r"\b(frames?|clocks?|posters?|paintings?|wall art|stickers?|decals?"
              r"|mirrors?|vases?|candles?|figurines?|sculptures?)\b"),
    ("small_storage", r"\b(storage box(es)?|baskets?|bins?)\b"),
    ("storage", r"\b(shelf|shelves|cabinets?|wardrobes?|bookcases?|dressers?)\b"),
    ("chair", r"\b(bean ?bags?|chairs?|recliners?)\b"),
    ("sofa", r"\b(sofas?|couch(es)?)\b"),
    ("desk", r"\bdesks?\b"),
    ("table", r"\btables?\b"),
    ("ottoman", r"\b(ottomans?|poufs?)\b"),
    ("stool", r"\bstools?\b"),
    ("bench", r"\bbench(es)?\b"),
    ("bed", r"\b(beds?|mattress(es)?|headboards?)\b"),
]
NAME_RULES = [(category, re.compile(pattern, re.IGNORECASE)) for category, pattern in NAME_RULES]


def classify_by_path(path):
    """Return (category, excluded) from a browse path."""
    lowered = (path or "").lower()
    if any(word in lowered for word in EXCLUDE_PATHS):
        return None, True
    if any(lowered.endswith(ending) for ending in EXCLUDE_PATH_ENDINGS):
        return None, True
    for category, words in PATH_RULES:
        if any(word in lowered for word in words):
            return category, False
    if any(word in lowered for word in EXCLUDE_PATHS_LATE):
        return None, True
    return None, False


def classify_by_name(name):
    if not name:
        return None
    if EXCLUDE_NAME.search(name) and not COVER_OK.search(name):
        return None
    for category, pattern in NAME_RULES:
        if pattern.search(name):
            return category
    return None


def classify(path, name):
    """Return (category, reason) for a broad-type product.

    reason is "path", "name", "excluded" or "unclassified". The name is
    used whenever the path matched no rule, which covers generic paths
    ("/Categories", missing) and paths in other languages.
    """
    category, excluded = classify_by_path(path)
    if excluded:
        return None, "excluded"
    if category:
        return category, "path"
    category = classify_by_name(name)
    if category:
        return category, "name"
    return None, "unclassified"
