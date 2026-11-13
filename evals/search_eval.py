"""Evaluation A: how well does search find a product from a different photo of it?

For a sample of products, one of the product's *other* photos (a different
angle, or the item in a room) is used as the query, with no text and no
filters. A good search should rank that same product near the top.

Steps (Cloud Shell, from the repo root):
    python3 -m evals.search_eval prepare --images-metadata images.csv.gz
    curl -s https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar \
      | python3 -m catalog.fetch_images --paths-file evals/data/query_paths.txt --out data/eval_images
    python3 -m evals.search_eval run --project YOUR_PROJECT_ID

Metrics:
- exact@k: the queried product itself is in the top k.
- variant@k: the product or one of its colour/size variants (same main image or
  same name stem) is in the top k. Variants look nearly identical, so a shopper
  would accept one.
- category@1: the top result is in the same store category.
"""

import argparse
import csv
import gzip
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent.search import Filters, build_query, variant_key

CASES_FILE = Path("evals/data/search_cases.jsonl")
PATHS_FILE = Path("evals/data/query_paths.txt")
RESULTS_FILE = Path("evals/results/search_eval.json")
PER_CATEGORY = 10
SEED = 7
POOL = 20  # rows fetched per query; enough to score top 5 after the fact
KS = (1, 5)


def load_image_paths(images_metadata):
    with gzip.open(images_metadata, "rt", encoding="utf-8", newline="") as f:
        return {r["image_id"]: r["path"] for r in csv.DictReader(f)}


def pick_cases(products, image_paths, per_category=PER_CATEGORY, seed=SEED):
    """Choose up to per_category products per category that have a usable other photo."""
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for row in products:
        main = row.get("main_image_path")
        for image_id in row.get("other_image_ids") or []:
            path = image_paths.get(image_id)
            if path and path != main:
                by_category[row["category"]].append((row, image_id, path))
                break
    cases = []
    for category in sorted(by_category):
        candidates = by_category[category]
        rng.shuffle(candidates)
        for row, image_id, path in candidates[:per_category]:
            cases.append({
                "product_id": row["product_id"],
                "category": row["category"],
                "name": row["name"],
                "main_image_path": row.get("main_image_path"),
                "query_image_id": image_id,
                "query_image_path": path,
            })
    return cases


def score_case(case, rows):
    """Score one query's ranked rows (dicts with product_id, name, category, main_image_path)."""
    target_image, target_stem = variant_key({"name": case["name"], "main_image_path": case["main_image_path"]})

    def is_variant(row):
        image, stem = variant_key(row)
        return row["product_id"] == case["product_id"] or image == target_image or stem == target_stem

    scores = {}
    for k in KS:
        top = rows[:k]
        scores[f"exact@{k}"] = any(r["product_id"] == case["product_id"] for r in top)
        scores[f"variant@{k}"] = any(is_variant(r) for r in top)
    scores["category@1"] = bool(rows) and rows[0]["category"] == case["category"]
    rank = next((i + 1 for i, r in enumerate(rows) if r["product_id"] == case["product_id"]), None)
    scores["exact_rank"] = rank
    return scores


def summarize(scored):
    """scored: list of (case, scores). Returns overall and per-category percentages."""
    metrics = [f"{kind}@{k}" for kind in ("exact", "variant") for k in KS] + ["category@1"]

    def pct(items, metric):
        return round(100 * sum(s[metric] for _, s in items) / len(items), 1) if items else None

    by_category = defaultdict(list)
    for case, scores in scored:
        by_category[case["category"]].append((case, scores))
    return {
        "queries": len(scored),
        "overall": {m: pct(scored, m) for m in metrics},
        "by_category": {c: {"queries": len(items), **{m: pct(items, m) for m in metrics}}
                        for c, items in sorted(by_category.items())},
    }


def run_search(search, case, images_dir):
    image = Path(images_dir) / case["query_image_path"]
    vector = search.query_vector(image_bytes=image.read_bytes(), image_mime="image/jpeg")
    sql, params = build_query(Filters(), search.project, search.dataset, POOL)
    return search.run_sql(sql, params, vector)


def evaluate(search, cases, images_dir, workers=4):
    usable = [c for c in cases if (Path(images_dir) / c["query_image_path"]).is_file()]
    missing = len(cases) - len(usable)
    with ThreadPoolExecutor(workers) as pool:
        all_rows = list(pool.map(lambda c: run_search(search, c, images_dir), usable))
    scored = [(case, score_case(case, rows)) for case, rows in zip(usable, all_rows)]
    summary = summarize(scored)
    summary["missing_query_images"] = missing
    summary["misses"] = [
        {"product": c["name"][:80], "category": c["category"],
         "top1": rows[0]["name"][:80] if rows else None, "exact_rank": s["exact_rank"]}
        for (c, s), rows in zip(scored, all_rows) if not s["variant@5"]
    ][:25]
    return summary


def print_summary(summary):
    print(f"Queries: {summary['queries']} (missing query images: {summary['missing_query_images']})")
    print("Overall: " + ", ".join(f"{k} {v}%" for k, v in summary["overall"].items()))
    print(f"\n{'category':20s} {'n':>3s} {'exact@1':>8s} {'exact@5':>8s} {'variant@5':>10s} {'cat@1':>6s}")
    for category, m in summary["by_category"].items():
        print(f"{category:20s} {m['queries']:3d} {m['exact@1']:8.1f} {m['exact@5']:8.1f} "
              f"{m['variant@5']:10.1f} {m['category@1']:6.1f}")
    if summary["misses"]:
        print("\nSome misses (product -> what came first):")
        for miss in summary["misses"][:10]:
            print(f"  {miss['product']}\n    -> {miss['top1']}  (exact rank: {miss['exact_rank']})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Pick query photos and write the cases")
    prep.add_argument("--images-metadata", required=True)
    prep.add_argument("--products", default="data/products.jsonl")
    prep.add_argument("--per-category", type=int, default=PER_CATEGORY)
    run = sub.add_parser("run", help="Run the searches and score them")
    run.add_argument("--project", required=True)
    run.add_argument("--images", default="data/eval_images")
    run.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        with open(args.products, encoding="utf-8") as f:
            products = [json.loads(line) for line in f if line.strip()]
        cases = pick_cases(products, load_image_paths(args.images_metadata), args.per_category)
        CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
        CASES_FILE.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
        PATHS_FILE.write_text("".join(c["query_image_path"] + "\n" for c in cases), encoding="utf-8")
        print(f"Wrote {len(cases)} cases to {CASES_FILE} and their photo paths to {PATHS_FILE}.")
        return

    from agent import config
    from agent.search import ProductSearch

    cases = [json.loads(line) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    search = ProductSearch.from_environment(args.project, config.DATASET, config.BQ_LOCATION, config.MODEL_LOCATION)
    summary = evaluate(search, cases, args.images, args.workers)
    if summary["queries"] == 0:
        sys.exit("No query images found. Run the curl | fetch_images step first.")
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
