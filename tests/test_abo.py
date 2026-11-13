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


def test_real_sofa_shape_uses_listed_metric_values():
    # Same structure as ABO listing B086B5P473: millimetres listed,
    # inches in normalized_value, and "length" is the long side.
    def mm(value, inches):
        return {"normalized_value": {"unit": "inches", "value": inches}, "unit": "millimeters", "value": value}

    item = listing(
        item_dimensions={"height": mm(810, 31.8897), "length": mm(1800, 70.866), "width": mm(860, 33.8582)},
        item_weight=[{"normalized_value": {"unit": "pounds", "value": 63.934}, "unit": "grams", "value": 29000}],
    )
    row, unknown = abo.to_row(item)
    assert unknown == []
    assert (row["height_cm"], row["length_cm"], row["width_cm"]) == (81.0, 180.0, 86.0)
    assert (row["longest_side_cm"], row["shortest_side_cm"]) == (180.0, 86.0)
    assert row["weight_kg"] == 29.0


def test_falls_back_to_normalized_value_for_unknown_listed_unit():
    entry = {"normalized_value": {"unit": "inches", "value": 10}, "unit": "hands", "value": 2.5}
    assert abo.convert(entry, abo.LENGTH_TO_CM) == (25.4, None)


def test_horizontal_span_with_one_side_or_none():
    assert abo.horizontal_span({"width_cm": 50.0, "length_cm": None}) == (50.0, 50.0)
    assert abo.horizontal_span({"width_cm": None, "length_cm": None}) == (None, None)


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
    assert (row["longest_side_cm"], row["shortest_side_cm"]) == (200.0, 88.9)
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
