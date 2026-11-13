"""Embed catalog products with Gemini Embedding 2 on Vertex AI.

Each product becomes one vector built from its main image plus a short text
line (name, category, colour, material, style), so a shopper's photo plus
spoken details ("like this, in navy") can be matched in the same space.

Usage (Cloud Shell; uses your gcloud login, no API key):
    python3 -m catalog.embed_images --project YOUR_PROJECT --limit 100   # test run
    python3 -m catalog.embed_images --check                               # sanity check
    python3 -m catalog.embed_images --project YOUR_PROJECT                # the rest
    python3 -m catalog.embed_images --project YOUR_PROJECT --load         # to BigQuery

Results go to data/embeddings.jsonl. Runs resume: products already in that
file are skipped, so an interrupted run can be started again.
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "gemini-embedding-2"
DIMENSIONS = 768
MAX_TEXT_CHARS = 300
RETRY_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
STOP_AFTER_FAILURES = 5  # stop early if the first calls all fail (e.g. no access)
# httpx network errors seen in practice ("Temporary failure in name resolution").
NETWORK_ERRORS = {"ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError",
                  "RemoteProtocolError", "WriteError", "PoolTimeout"}

EMBEDDING_FIELDS = [
    ("product_id", "STRING", "REQUIRED", "products.product_id"),
    ("embedding", "FLOAT", "REPEATED", "L2-normalised vector"),
    ("model", "STRING", "REQUIRED", "Embedding model"),
    ("dims", "INTEGER", "REQUIRED", "Vector length"),
    ("content_mode", "STRING", "REQUIRED", "image or image_text"),
]


def product_text(row):
    """Short description embedded alongside the image."""
    parts = [row["name"], f"Category: {row['category'].replace('_', ' ')}"]
    for label, field in (("Colour", "colors"), ("Material", "materials"), ("Style", "styles")):
        if row.get(field):
            parts.append(f"{label}: {', '.join(row[field][:3])}")
    return ". ".join(parts)[:MAX_TEXT_CHARS]


def mime_type(path):
    suffix = Path(path).suffix.lower()
    return {".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")


def build_content(image_bytes, image_mime, text=None):
    """One Content = one embedding. Image first, then optional text."""
    from google.genai import types

    parts = [types.Part.from_bytes(data=image_bytes, mime_type=image_mime)]
    if text:
        parts.append(types.Part.from_text(text=text))
    return types.Content(parts=parts)


def normalize(values):
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        raise ValueError("zero vector")
    return [v / norm for v in values]


def make_embedder(client, model=MODEL, dims=DIMENSIONS, sleep=time.sleep):
    """Return embed(content) -> normalised vector, retrying rate limits and server errors."""
    from google.genai import errors, types

    config = types.EmbedContentConfig(output_dimensionality=dims)

    def embed(content):
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = client.models.embed_content(model=model, contents=[content], config=config)
                return normalize(response.embeddings[0].values)
            except errors.APIError as exc:
                if exc.code not in RETRY_CODES or attempt == MAX_ATTEMPTS - 1:
                    raise
            except (ConnectionError, OSError) as exc:
                # Network blips (DNS failures, resets). httpx's ConnectError is
                # not a ConnectionError, so it is matched by name below.
                if attempt == MAX_ATTEMPTS - 1:
                    raise
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ not in NETWORK_ERRORS or attempt == MAX_ATTEMPTS - 1:
                    raise
            sleep(min(60, 2 ** attempt))
        raise RuntimeError("unreachable")

    return embed


def load_rows(products_file):
    with Path(products_file).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_ids(out_path):
    path = Path(out_path)
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["product_id"] for line in f if line.strip()}


def embed_catalog(rows, images_dir, embed, out_path, *, mode="image_text", limit=None,
                  workers=4, model=MODEL, dims=DIMENSIONS):
    """Embed products not yet in out_path. Returns a summary dict."""
    images_dir = Path(images_dir)
    already = done_ids(out_path)
    todo, no_image = [], 0
    for row in rows:
        if row["product_id"] in already:
            continue
        image = images_dir / (row.get("main_image_path") or "")
        if not row.get("main_image_path") or not image.is_file():
            no_image += 1
            continue
        todo.append((row, image))
    if limit is not None:
        todo = todo[:limit]

    def job(row, image):
        text = product_text(row) if mode == "image_text" else None
        content = build_content(image.read_bytes(), mime_type(image), text)
        return embed(content)

    written, failures, first_errors = 0, 0, []
    start = time.monotonic()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(job, row, image): row for row, image in todo}
        for future in as_completed(futures):
            row = futures[future]
            try:
                vector = future.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures += 1
                if len(first_errors) < 5:
                    first_errors.append(f"{row['product_id']}: {type(exc).__name__}: {exc}"[:300])
                if written == 0 and failures >= STOP_AFTER_FAILURES:
                    for pending in futures:
                        pending.cancel()
                    break
                continue
            out.write(json.dumps({
                "product_id": row["product_id"], "embedding": vector,
                "model": model, "dims": dims, "content_mode": mode,
            }) + "\n")
            out.flush()
            written += 1
            if written % 500 == 0:
                print(f"  {written}/{len(todo)} embedded", file=sys.stderr)

    return {
        "already_done": len(already),
        "attempted": len(todo),
        "written": written,
        "failed": failures,
        "skipped_no_image": no_image,
        "seconds": round(time.monotonic() - start, 1),
        "first_errors": first_errors,
    }


def nearest_neighbour_check(embeddings_file, products_file, examples=5):
    """Share of products whose closest other product is in the same category.

    A rough sanity check that the vectors carry meaning, not an accuracy
    score: similar products should usually share a category.
    """
    import numpy as np

    by_id = {r["product_id"]: r for r in load_rows(products_file)}
    records = [r for r in load_rows(embeddings_file) if r["product_id"] in by_id]
    if len(records) < 2:
        return {"products": len(records), "same_category_top1": None, "examples": []}
    ids = [r["product_id"] for r in records]
    matrix = np.array([r["embedding"] for r in records], dtype=np.float32)
    scores = matrix @ matrix.T
    np.fill_diagonal(scores, -np.inf)
    nearest = scores.argmax(axis=1)
    same = sum(by_id[ids[i]]["category"] == by_id[ids[j]]["category"] for i, j in enumerate(nearest))
    sample = [
        (by_id[ids[i]]["name"][:60], by_id[ids[nearest[i]]]["name"][:60], round(float(scores[i, nearest[i]]), 3))
        for i in range(min(examples, len(ids)))
    ]
    return {"products": len(ids), "same_category_top1": round(100 * same / len(ids), 1), "examples": sample}


def load_to_bigquery(embeddings_file, project, dataset="home_decor", table="product_embeddings",
                     location="asia-south1"):
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField(n, t, mode=m, description=d) for n, t, m, d in EMBEDDING_FIELDS],
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    with Path(embeddings_file).open("rb") as f:
        client.load_table_from_file(f, table_id, job_config=job_config, location=location).result()
    return table_id, client.get_table(table_id).num_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project")
    parser.add_argument("--location", default="global",
                        help="Vertex AI location. gemini-embedding-2 is offered in global, us and eu only")
    parser.add_argument("--products", default="data/products.jsonl")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--out", default="data/embeddings.jsonl")
    parser.add_argument("--limit", type=int, help="Embed at most this many new products")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=["image_text", "image"], default="image_text")
    parser.add_argument("--dims", type=int, default=DIMENSIONS)
    parser.add_argument("--check", action="store_true", help="Run the nearest-neighbour check and exit")
    parser.add_argument("--load", action="store_true", help="Load the embeddings into BigQuery and exit")
    args = parser.parse_args(argv)

    if args.check:
        result = nearest_neighbour_check(args.out, args.products)
        print(f"Products checked: {result['products']}")
        print(f"Closest product is in the same category: {result['same_category_top1']}%")
        for name, neighbour, score in result["examples"]:
            print(f"  {name}\n    -> {neighbour}  (similarity {score})")
        return
    if not args.project:
        parser.error("--project is required unless --check is used")
    if args.load:
        table_id, rows = load_to_bigquery(args.out, args.project)
        print(f"Loaded {rows} embeddings into {table_id}.")
        return

    from google import genai

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    embed = make_embedder(client, dims=args.dims)
    summary = embed_catalog(
        load_rows(args.products), args.images, embed, args.out,
        mode=args.mode, limit=args.limit, workers=args.workers, dims=args.dims,
    )
    for key, value in summary.items():
        if key != "first_errors":
            print(f"{key:18s} {value}")
    for error in summary["first_errors"]:
        print(f"  error: {error}")
    if summary["written"] == 0 and summary["failed"]:
        sys.exit("No embeddings were written. See the errors above.")


if __name__ == "__main__":
    main()
