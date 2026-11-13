import pytest

from catalog import abo
from tests.fixtures import listing, text


def test_english_values_keeps_english_and_untagged_only():
    entries = text("Oak") + text("Eiche", "de_DE") + [{"value": "Teak"}] + text("Oak", "en_GB")
    assert abo.english_values(entries) == ["Oak", "Teak"]


def test_english_values_ignores_bad_shapes():
    assert abo.english_values(None) == []
    assert abo.english_values([{"language_tag": "en_US", "value": "  "}, "x", 3]) == []


def test_dimensions_prefer_normalized_and_convert_to_cm():
    dims, unknown = abo.dimensions_cm(listing())
    assert dims == {"height_cm": 76.2, "width_cm": 200.0, "length_cm": 88.9}
    assert unknown == []


def test_dimensions_report_unknown_units():
    item = listing(item_dimensions={"height": {"unit": "cubits", "value": 2}})
    dims, unknown = abo.dimensions_cm(item)
    assert dims["height_cm"] is None
    assert unknown == ["cubits"]


def test_dimensions_ignore_non_positive_and_missing():
    item = listing(item_dimensions={"height": {"unit": "inches", "value": 0}})
    dims, unknown = abo.dimensions_cm(item)
    assert dims == {"height_cm": None, "width_cm": None, "length_cm": None}
    assert unknown == []
    assert abo.dimensions_cm(listing(item_dimensions=None))[0]["width_cm"] is None


def test_weight_converts_pounds_to_kg():
    assert abo.weight_kg(listing()) == (45.36, None)


def test_standard_colors_lowercased_and_deduplicated():
    entries = [
        {"language_tag": "en_US", "value": "Navy", "standardized_values": ["Blue"]},
        {"language_tag": "en_US", "value": "Sky", "standardized_values": ["blue", "White"]},
    ]
    assert abo.standard_colors(entries) == ["blue", "white"]


def test_to_row_flattens_listing():
    row, unknown = abo.to_row(listing())
    assert unknown == []
    assert row["product_id"] == "SOFA1"
    assert row["name"] == "Grey three-seat sofa"
    assert row["category"] == "sofa"
    assert row["colors"] == ["Slate Grey"]
    assert row["color_families"] == ["grey"]
    assert row["materials"] == ["Velvet"]
    assert row["bullet_points"] == ["Seats three"]
    assert row["keywords"] == ["sofa", "couch"]
    assert row["browse_path"] == "/Categories/Furniture/Living Room/Sofas"
    assert row["has_3d_model"] is False
    assert row["other_image_ids"] == ["other-SOFA1"]


def test_to_row_caps_keywords():
    keywords = [{"language_tag": "en_US", "value": f"k{i}"} for i in range(50)]
    row, _ = abo.to_row(listing(item_keywords=keywords))
    assert len(row["keywords"]) == abo.MAX_KEYWORDS


@pytest.mark.parametrize(
    "item,expected",
    [
        (listing(), True),
        (listing(item_name=text("Sofa", "de_DE")), False),
        (listing(item_name=[]), False),
    ],
)
def test_has_english_name(item, expected):
    assert abo.has_english_name(item) is expected
