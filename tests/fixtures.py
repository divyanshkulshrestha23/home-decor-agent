"""Constructed ABO-style listings for tests (no real dataset rows)."""

import gzip
import json
import tarfile
from pathlib import Path


def text(value, tag="en_US"):
    return [{"language_tag": tag, "value": value}]


def listing(item_id="SOFA1", ptype="SOFA", name="Grey three-seat sofa", **overrides):
    base = {
        "item_id": item_id,
        "item_name": text(name),
        "brand": text("Rivet"),
        "product_type": [{"value": ptype}],
        "color": [{"language_tag": "en_US", "value": "Slate Grey", "standardized_values": ["Grey"]}],
        "material": text("Velvet"),
        "style": text("Mid-Century Modern"),
        "item_dimensions": {
            "height": {"normalized_value": {"unit": "inches", "value": 30}, "unit": "inches", "value": 30},
            "width": {"unit": "centimeters", "value": 200},
            "length": {"unit": "inches", "value": 35},
        },
        "item_weight": [{"normalized_value": {"unit": "pounds", "value": 100}, "unit": "pounds", "value": 100}],
        "bullet_point": text("Seats three") + text("Sitzt drei", "de_DE"),
        "item_keywords": text("sofa") + text("couch"),
        "node": [{"node_id": 1, "node_name": "/Categories/Furniture/Living Room/Sofas"}],
        "main_image_id": "img-" + item_id,
        "other_image_id": ["other-" + item_id],
        "country": "US",
        "marketplace": "Amazon",
        "domain_name": "amazon.com",
    }
    base.update(overrides)
    return base


def write_listings_tar(path, listings):
    """Write listings into a tar shaped like abo-listings.tar."""
    path = Path(path)
    gz = path.parent / "listings_0.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        for item in listings:
            f.write(json.dumps(item) + "\n")
    with tarfile.open(path, "w") as tar:
        tar.add(gz, arcname="listings/metadata/listings_0.json.gz")
    gz.unlink()
    return path
