"""Generate prices for the catalog.

ABO has no prices, so every price in this store is synthetic. Prices are
deterministic: the same product always gets the same price, because the
random draw comes from a hash of the product id rather than a random seed.

Each category has an assumed price range in rupees. A product's base price
is drawn log-uniformly from that range (so cheap items are as common as
expensive ones on a log scale), then adjusted for material. The ranges are
assumptions chosen to look plausible, not market data.
"""

import hashlib
import math

# (low, high) in INR. Assumed, not sourced.
PRICE_RANGE_INR = {
    "sofa": (15000, 90000),
    "bed": (12000, 70000),
    "table": (3000, 40000),
    "desk": (5000, 35000),
    "chair": (2500, 25000),
    "storage": (4000, 45000),
    "small_storage": (300, 4000),
    "bench": (3000, 20000),
    "ottoman": (2000, 15000),
    "stool": (1200, 8000),
    "rug": (1500, 25000),
    "accent_furniture": (2000, 25000),
    "outdoor_furniture": (5000, 60000),
    "outdoor_decor": (800, 15000),
    "lighting": (800, 12000),
    "bedding": (600, 8000),
    "bath_linen": (300, 3000),
    "curtains": (500, 6000),
    "window_hardware": (300, 3000),
    "cushions": (300, 3000),
    "decor": (400, 8000),
}
DEFAULT_RANGE_INR = (500, 10000)

# Multipliers for materials that usually cost more or less. Matched as
# lowercase substrings of the material text; the largest match wins.
MATERIAL_FACTOR = {
    "leather": 1.3,
    "marble": 1.3,
    "teak": 1.25,
    "walnut": 1.25,
    "oak": 1.2,
    "solid wood": 1.2,
    "velvet": 1.1,
    "metal": 1.05,
    "plastic": 0.8,
    "polypropylene": 0.8,
}


def _unit_draw(product_id):
    """A number in [0, 1) derived from the product id."""
    digest = hashlib.sha256(product_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def material_factor(materials):
    text = " ".join(materials).lower()
    factors = [f for name, f in MATERIAL_FACTOR.items() if name in text]
    return max(factors) if factors else 1.0


def round_price(value):
    """Round to a retail-looking price ending in 99 (minimum 99)."""
    hundreds = max(1, round(value / 100))
    return hundreds * 100 - 1


def price_inr(product_id, category, materials):
    low, high = PRICE_RANGE_INR.get(category, DEFAULT_RANGE_INR)
    u = _unit_draw(product_id)
    base = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
    return round_price(base * material_factor(materials))
