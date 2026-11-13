import gzip
import json

import pytest

from catalog import load_bigquery, prepare
from catalog.abo import to_row
from catalog.pricing import price_inr
from catalog.schema import FIELD_NAMES, PRODUCT_FIELDS, row_errors
from tests.fixtures import listing, text, write_listings_tar


def valid_row():
    row, _ = to_row(listing())
    row["price_inr"] = price_inr(row["product_id"], row["category"], row["materials"])
    return row


def test_build_catalog_filters_and_counts():
    listings = [
        listing("A"),
        listing("A"),  # duplicate id
        listing("B", ptype="CELLULAR_PHONE_CASE"),
        listing("C", item_name=text("Sofa", "de_DE")),
        listing("D", main_image_id=None),
        listing("E", ptype="RUG", name="Wool rug"),
    ]
    rows, report = prepare.build_catalog(listings)
    assert [r["product_id"] for r in rows] == ["A", "E"]
    assert report["listings_scanned"] == 6
    assert report["dropped"] == {
        "duplicate_or_missing_id": 1,
        "type_not_in_store": 1,
        "no_english_name": 1,
        "no_image": 1,
    }
    assert report["products_by_category"] == {"sofa": 1, "rug": 1}
    assert "TABLE" in report["store_types_with_no_products"]
    assert report["field_coverage_percent"]["materials"] == 100.0
    assert report["field_coverage_percent"]["main_image_path"] == 0.0


def test_build_catalog_attaches_image_paths():
    rows, report = prepare.build_catalog([listing("A")], {"img-A": "8c/8ccb5859.jpg"})
    assert rows[0]["main_image_path"] == "8c/8ccb5859.jpg"
    assert report["field_coverage_percent"]["main_image_path"] == 100.0


def test_rows_match_schema():
    assert row_errors(valid_row()) == []
    assert set(valid_row()) == set(FIELD_NAMES)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"name": None}, "name: missing"),
        ({"price_inr": 10.5}, "price_inr: expected INTEGER"),
        ({"height_cm": "tall"}, "height_cm: expected FLOAT"),
        ({"height_cm": float("nan")}, "height_cm: expected FLOAT"),
        ({"colors": "grey"}, "colors: expected a list"),
        ({"colors": ["grey", 3]}, "colors: list items must be STRING"),
        ({"has_3d_model": 1}, "has_3d_model: expected BOOLEAN"),
        ({"surprise": 1}, "unknown fields"),
    ],
)
def test_row_errors_catch_bad_values(change, message):
    row = valid_row()
    row.update(change)
    assert any(message in e for e in row_errors(row))


def test_end_to_end_prepare_then_dry_run(tmp_path, capsys):
    tar = write_listings_tar(
        tmp_path / "abo-listings.tar",
        [listing("A"), listing("B", ptype="TABLE", name="Oak table"), listing("C", ptype="SHOES")],
    )
    images = tmp_path / "images.csv.gz"
    with gzip.open(images, "wt", encoding="utf-8") as f:
        f.write("image_id,height,width,path\nimg-A,100,100,aa/img-A.jpg\n")

    out = tmp_path / "products.jsonl"
    report = tmp_path / "report.json"
    prepare.main([
        "--listings", str(tar), "--images-metadata", str(images),
        "--out", str(out), "--report", str(report),
    ])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["main_image_path"] == "aa/img-A.jpg"
    assert json.loads(report.read_text())["products_kept"] == 2

    load_bigquery.main(["--file", str(out), "--project", "test", "--dry-run"])
    assert "2 rows match the schema." in capsys.readouterr().out


def test_dry_run_rejects_bad_rows(tmp_path):
    bad = valid_row()
    bad["price_inr"] = "free"
    path = tmp_path / "products.jsonl"
    path.write_text(json.dumps(valid_row()) + "\n" + json.dumps(bad) + "\n{oops\n")
    count, problems = load_bigquery.check_file(path)
    assert count == 3
    assert [line for line, _ in problems] == [2, 3]
    with pytest.raises(SystemExit):
        load_bigquery.main(["--file", str(path), "--project", "test", "--dry-run"])


def test_images_metadata_needs_expected_columns(tmp_path):
    path = tmp_path / "images.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write("id,file\nx,y\n")
    with pytest.raises(SystemExit):
        prepare.load_image_paths(path)


def test_list_types_and_inspect(tmp_path, capsys):
    tar = write_listings_tar(tmp_path / "abo-listings.tar", [listing("A"), listing("B", ptype="SHOES")])
    prepare.main(["--listings", str(tar), "--list-types"])
    out = capsys.readouterr().out
    assert "SOFA" in out and "SHOES" in out
    prepare.main(["--listings", str(tar), "--inspect", "1", "--type", "SHOES"])
    assert '"item_id": "B"' in capsys.readouterr().out


def test_bigquery_schema_builds():
    pytest.importorskip("google.cloud.bigquery")
    schema = load_bigquery.bigquery_schema()
    assert [f.name for f in schema] == FIELD_NAMES
    assert {f.mode for f in schema} == {mode for _, _, mode, _ in PRODUCT_FIELDS}
