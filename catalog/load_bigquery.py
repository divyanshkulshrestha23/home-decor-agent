"""Load data/products.jsonl into BigQuery.

Usage:
    python -m catalog.load_bigquery --project YOUR_PROJECT --dry-run
    python -m catalog.load_bigquery --project YOUR_PROJECT

--dry-run checks every row against the schema locally and does not need
Google Cloud credentials. A real run creates the dataset if needed and
replaces the products table.
"""

import argparse
import json
import sys
from pathlib import Path

from catalog.schema import PRODUCT_FIELDS, row_errors

TABLE_DESCRIPTION = (
    "Home decor catalog built from Amazon Berkeley Objects listings "
    "(CC BY-NC 4.0). Prices are synthetic."
)


def check_file(path):
    """Return (row_count, [(line_number, errors)]) for a JSONL file."""
    count = 0
    problems = []
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append((number, [f"invalid JSON: {exc}"]))
                continue
            errors = row_errors(row)
            if errors:
                problems.append((number, errors))
    return count, problems


def bigquery_schema():
    from google.cloud import bigquery

    return [
        bigquery.SchemaField(name, field_type, mode=mode, description=description)
        for name, field_type, mode, description in PRODUCT_FIELDS
    ]


def load(path, project, dataset, table, location):
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    dataset_ref = bigquery.Dataset(f"{project}.{dataset}")
    dataset_ref.location = location
    client.create_dataset(dataset_ref, exists_ok=True)

    table_id = f"{project}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        schema=bigquery_schema(),
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    with Path(path).open("rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config, location=location)
    job.result()

    loaded = client.get_table(table_id)
    loaded.description = TABLE_DESCRIPTION
    client.update_table(loaded, ["description"])
    return table_id, loaded.num_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default="data/products.jsonl")
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", default="home_decor")
    parser.add_argument("--table", default="products")
    parser.add_argument("--location", default="asia-south1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    count, problems = check_file(args.file)
    if problems:
        for number, errors in problems[:20]:
            print(f"line {number}: {'; '.join(errors)}")
        sys.exit(f"{len(problems)} of {count} rows would be rejected. Nothing loaded.")
    print(f"{count} rows match the schema.")

    if args.dry_run:
        return
    table_id, rows = load(args.file, args.project, args.dataset, args.table, args.location)
    print(f"Loaded {rows} rows into {table_id}.")


if __name__ == "__main__":
    main()
