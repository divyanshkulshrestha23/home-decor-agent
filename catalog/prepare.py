"""Filter ABO listings into the home decor catalog.

Usage:
    python -m catalog.prepare --listings abo-listings.tar
    python -m catalog.prepare --listings abo-listings.tar --images-metadata images.csv.gz
    python -m catalog.prepare --listings abo-listings.tar --list-types
    python -m catalog.prepare --listings abo-listings.tar --inspect 3 --type SOFA

Writes data/products.jsonl (one product per line, ready for BigQuery) and
data/catalog_report.json, and prints a summary.
"""

import argparse
import csv
import gzip
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from catalog.abo import (
    CATEGORY_BY_TYPE,
    browse_path,
    has_english_name,
    iter_listings,
    product_type,
    to_row,
)
from catalog.classify import BROAD_TYPES, classify
from catalog.pricing import price_inr
from catalog.schema import row_errors

COVERAGE_FIELDS = [
    "brand", "colors", "color_families", "materials", "styles",
    "height_cm", "width_cm", "length_cm", "longest_side_cm", "weight_kg",
    "bullet_points", "keywords", "browse_path", "main_image_path",
]


def load_image_paths(path):
    """Map image_id -> path from ABO's images.csv.gz."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        if "image_id" not in columns or "path" not in columns:
            sys.exit(
                f"{path} has columns {columns}; expected 'image_id' and 'path'."
            )
        return {r["image_id"]: r["path"] for r in reader}


def build_catalog(listings, image_paths=None):
    """Filter and flatten listings. Returns (rows, report)."""
    image_paths = image_paths or {}
    dropped = Counter()
    unknown_units = Counter()
    classified_by = Counter()
    unclassified = Counter()
    seen = set()
    rows = []
    scanned = 0

    for listing in listings:
        scanned += 1
        ptype = product_type(listing)
        if ptype not in CATEGORY_BY_TYPE and ptype not in BROAD_TYPES:
            dropped["type_not_in_store"] += 1
            continue
        if not has_english_name(listing):
            dropped["no_english_name"] += 1
            continue
        if not listing.get("main_image_id"):
            dropped["no_image"] += 1
            continue
        item_id = listing.get("item_id")
        if not isinstance(item_id, str) or item_id in seen:
            dropped["duplicate_or_missing_id"] += 1
            continue

        row, unknown = to_row(listing)
        if ptype in BROAD_TYPES:
            category, reason = classify(row["browse_path"], row["name"])
            classified_by[reason] += 1
            if category is None:
                dropped[f"broad_type_{reason}"] += 1
                if reason == "unclassified":
                    unclassified[f"{row['browse_path'] or '(none)'} | {row['name'][:60]}"] += 1
                continue
            row["category"] = category
        unknown_units.update(unknown)
        row["main_image_path"] = image_paths.get(row["main_image_id"])
        row["price_inr"] = price_inr(row["product_id"], row["category"], row["materials"])

        errors = row_errors(row)
        if errors:
            dropped["invalid_row"] += 1
            continue
        seen.add(item_id)
        rows.append(row)

    report = summarize(rows, scanned, dropped, unknown_units)
    report["broad_types_classified_by"] = dict(classified_by)
    report["unclassified_examples"] = [text for text, _ in unclassified.most_common(30)]
    return rows, report


def _present(value):
    return value not in (None, [], "")


def summarize(rows, scanned, dropped, unknown_units):
    total = len(rows)
    by_category = Counter(r["category"] for r in rows)
    by_type = Counter(r["product_type"] for r in rows)
    coverage = {
        field: round(100 * sum(_present(r[field]) for r in rows) / total, 1) if total else 0.0
        for field in COVERAGE_FIELDS
    }
    all_dims = sum(
        all(r[f] is not None for f in ("height_cm", "width_cm", "length_cm")) for r in rows
    )
    search_ready = sum(
        bool(r["colors"]) and bool(r["materials"]) and r["longest_side_cm"] is not None
        for r in rows
    )

    prices = {}
    for category in by_category:
        values = sorted(r["price_inr"] for r in rows if r["category"] == category)
        prices[category] = {
            "min": values[0],
            "median": int(statistics.median(values)),
            "max": values[-1],
        }

    return {
        "listings_scanned": scanned,
        "products_kept": total,
        "dropped": dict(dropped),
        "store_types_with_no_products": sorted(
            t for t in set(CATEGORY_BY_TYPE) | BROAD_TYPES if t not in by_type
        ),
        "products_by_category": dict(by_category.most_common()),
        "products_by_type": dict(by_type.most_common()),
        "field_coverage_percent": coverage,
        "all_three_dimensions_percent": round(100 * all_dims / total, 1) if total else 0.0,
        "colour_material_width_percent": round(100 * search_ready / total, 1) if total else 0.0,
        "unknown_units": dict(unknown_units),
        "price_inr_by_category": prices,
    }


def print_report(report):
    print(f"Listings scanned: {report['listings_scanned']}")
    print(f"Products kept:    {report['products_kept']}")
    print(f"Dropped:          {report['dropped']}")
    if report["store_types_with_no_products"]:
        print(f"Store types with no products: {report['store_types_with_no_products']}")
    print("\nProducts by category:")
    for category, count in report["products_by_category"].items():
        p = report["price_inr_by_category"][category]
        print(
            f"  {category:22s} {count:6d}   price Rs {p['min']:,} / {p['median']:,} / {p['max']:,}"
            "  (min / median / max)"
        )
    print("\nField coverage among kept products:")
    for field, value in report["field_coverage_percent"].items():
        print(f"  {field:18s} {value:5.1f}%")
    print(f"  {'all 3 dimensions':18s} {report['all_three_dimensions_percent']:5.1f}%")
    print(f"  {'colour+material+size':18s} {report['colour_material_width_percent']:5.1f}%")
    if report["unknown_units"]:
        print(f"\nUnits not converted: {report['unknown_units']}")
    if report.get("broad_types_classified_by"):
        print(f"\nBroad types (HOME etc.) classified by: {report['broad_types_classified_by']}")
    if report.get("unclassified_examples"):
        print("Most common unclassified (path | name), left out of the store:")
        for text in report["unclassified_examples"][:15]:
            print(f"  {text}")


def list_types(tar_path):
    """Print every product type with its count and English-name count."""
    total, english = Counter(), Counter()
    for listing in iter_listings(tar_path):
        ptype = product_type(listing) or "UNKNOWN"
        total[ptype] += 1
        if has_english_name(listing):
            english[ptype] += 1
    print(f"{'product_type':40s} {'all':>7s} {'english':>8s}  in store")
    for ptype, count in total.most_common():
        mark = "yes" if ptype in CATEGORY_BY_TYPE else ""
        print(f"{ptype:40s} {count:7d} {english[ptype]:8d}  {mark}")


def browse_paths(tar_path, types, top):
    """Print the most common browse paths for each given product type."""
    wanted = set(types)
    paths = {t: Counter() for t in types}
    for listing in iter_listings(tar_path):
        ptype = product_type(listing)
        if ptype in wanted and has_english_name(listing):
            paths[ptype][browse_path(listing) or "(none)"] += 1
    for ptype in types:
        total = sum(paths[ptype].values())
        print(f"\n{ptype} ({total} English listings)")
        for path, count in paths[ptype].most_common(top):
            print(f"  {count:6d}  {path}")


def inspect(tar_path, ptype, count):
    """Print raw listings of one type, to check field shapes."""
    shown = 0
    for listing in iter_listings(tar_path):
        if product_type(listing) == ptype:
            print(json.dumps(listing, indent=2, ensure_ascii=False))
            shown += 1
            if shown >= count:
                return
    print(f"Found {shown} listings of type {ptype}.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--listings", required=True, help="Path to abo-listings.tar")
    parser.add_argument("--images-metadata", help="Optional path to ABO images.csv.gz")
    parser.add_argument("--out", default="data/products.jsonl")
    parser.add_argument("--report", default="data/catalog_report.json")
    parser.add_argument("--list-types", action="store_true", help="Print all product types and exit")
    parser.add_argument("--inspect", type=int, metavar="N", help="Print N raw listings of --type and exit")
    parser.add_argument("--type", help="Product type for --inspect")
    parser.add_argument(
        "--browse-paths", nargs="+", metavar="TYPE",
        help="Print the most common browse paths for these product types and exit",
    )
    parser.add_argument("--top", type=int, default=25, help="Paths per type for --browse-paths")
    args = parser.parse_args(argv)

    if args.list_types:
        list_types(args.listings)
        return
    if args.browse_paths:
        browse_paths(args.listings, args.browse_paths, args.top)
        return
    if args.inspect:
        if not args.type:
            parser.error("--inspect needs --type")
        inspect(args.listings, args.type, args.inspect)
        return

    image_paths = load_image_paths(args.images_metadata) if args.images_metadata else {}
    rows, report = build_catalog(iter_listings(args.listings), image_paths)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(report)
    print(f"\nWrote {len(rows)} products to {out} and the report to {report_path}")


if __name__ == "__main__":
    main()
