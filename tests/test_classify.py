"""Classification of broad product types, using browse paths seen in ABO."""

import pytest

from catalog import prepare
from catalog.classify import classify, classify_by_name
from tests.fixtures import listing


@pytest.mark.parametrize(
    "path,category",
    [
        ("/Categories/Home Storage & Organization/Boxes, Baskets & Bins/Storage Boxes & Chests/Storage Boxes", "small_storage"),
        ("/Categories/Home Décor/Photo Albums, Frames & Accessories/Picture Frames/Wall & Tabletop Frames", "decor"),
        ("/Home & Garden/Home & Kitchen/Categories/Bedding & Linens/Bedding/Sheets & Pillowcases/Fitted Sheets", "bedding"),
        ("/Home & Garden/Home & Kitchen/Categories/Home Accessories/Window Treatments/Curtains", "curtains"),
        ("/Categories/Home Furnishing/Bathroom Linen/Bath Mats", "bath_linen"),
        ("/Kategorien/Wohnaccessoires & Deko/Fensterdekoration/Vorhänge", "curtains"),
        ("/Kategorien/Bettwaren & Bettwäsche/Bettlaken & Kissenbezüge/Spannbetttücher", "bedding"),
        ("/Categories/Home Décor/Window Treatment Hardware/Finials", "window_hardware"),
        ("/Categories/Home Décor/Rugs, Pads & Protectors/Area Rugs", "rug"),
        ("/Kitchen & Dining Features/Kitchen & Dining Archives/2020-10/222425 - Area Rugs, Runners & Pads", "rug"),
        ("/Categories/Home & Décor/Clocks/Wall Clocks", "decor"),
        ("/Categories/Painting Supplies, Tools & Wall Treatments/Wall Stickers & Murals", "decor"),
        ("/Categories/Furniture/Living Room Furniture/TV & Media Furniture/Media Storage", "storage"),
        ("/Categories/Furniture/Kitchen & Dining Room Furniture/Kitchen Islands & Carts", "storage"),
        ("/Categories/Furniture/Living Room Furniture/Bean Bags, Covers & Refills/Filled Bean Bags", "chair"),
        ("/Categories/Furniture/Study & Home Office Furniture/Chairs & Sofas/Desk Chairs", "chair"),
        ("/Home & Garden/Home & Kitchen/Categories/Furniture/Home Office Furniture/Desks & Workstations/Desks", "desk"),
        ("/Categories/Furniture/Accent Furniture/Ottomans", "ottoman"),
        ("/Categories/Furniture/Accent Furniture", "accent_furniture"),
        ("/Categories/Furniture/Bedroom Furniture/Mattresses & Box Springs/Mattresses", "bed"),
        ("/Categories/Patio Furniture & Accessories/Patio Furniture Sets", "outdoor_furniture"),
        ("/Categories/Furniture/Garden & Outdoor Furniture/Patio Sofas", "outdoor_furniture"),
        ("/Categories/Patio Furniture & Accessories/Umbrellas & Shade/Umbrellas", "outdoor_decor"),
        ("/Categories/Lighting & Ceiling Fans/Outdoor Lighting/Torches", "lighting"),
        ("/Categories/Home Storage & Organization/Racks, Shelves & Drawers/Storage Drawer Units", "storage"),
        ("/Categories/Home Storage & Organization/Laundry Organization/Laundry Bins", "small_storage"),
        ("/Categories/Storage & Organization/Clothing & Closet Storage/Closet Systems", "storage"),
    ],
)
def test_paths_map_to_categories(path, category):
    assert classify(path, "anything") == (category, "path")


@pytest.mark.parametrize(
    "path",
    [
        "/Categories/Furniture/Living Room Furniture/Bean Bags, Covers & Refills/Bean Bag Covers",
        "/Categories/Home Storage & Organization/Clothing & Wardrobe Storage/Clothes Covers",
        "/Categories/Grills & Outdoor Cooking/Outdoor Cooking Tools & Accessories/Grill Covers",
        "/Categories/Tools & Equipment/Riveters",
        "/カテゴリー別/携帯電話・スマートフォン/携帯電話・スマートフォンアクセサリ/スマートウォッチアクセサリ",
        "/Categories/Home & Décor/Photo Albums",
        "/Categories/Kitchen & Dining",
        "/Home & Garden/Garden & Outdoors/Categories/Gardening/Watering Equipment/Hoses & Accessories/Garden Hoses",
        "/Categories/Outdoor Heating & Cooling/Patio Heaters",
    ],
)
def test_paths_excluded(path):
    assert classify(path, "Oak Table Lamp") == (None, "excluded")


@pytest.mark.parametrize(
    "name,category",
    [
        ("Cotton Double Bedsheet with 2 Pillow Covers", "bedding"),
        ("Velvet Cushion Cover Set of 5", "cushions"),
        ("Blackout Curtains for Bedroom", "curtains"),
        ("Brass Table Lamp", "lighting"),
        ("Solid Wood Coffee Table", "table"),
        ("Wall Clock for Living Room", "decor"),
        ("Bean Bag XXL", "chair"),
        ("Bath Towel Set", "bath_linen"),
    ],
)
def test_generic_paths_fall_back_to_name(name, category):
    assert classify("/Categories", name) == (category, "name")
    assert classify(None, name) == (category, "name")


def test_name_rules_use_word_boundaries():
    assert classify_by_name("Tablet Stand Lightweight") is None


def test_covers_left_out_except_bedding_and_cushion_covers():
    assert classify_by_name("Stretch Sofa Cover") is None
    assert classify_by_name("Duvet Cover Set, Queen") == "bedding"


def test_unclassified_when_nothing_matches():
    assert classify("/Categories", "Silicone Spatula") == (None, "unclassified")


def test_build_catalog_classifies_broad_types():
    node = lambda path: [{"node_id": 1, "node_name": path}]  # noqa: E731
    listings = [
        listing("A", ptype="HOME", name="Photo frame", node=node("/Categories/Home Décor/Photo Albums, Frames & Accessories/Picture Frames")),
        listing("B", ptype="HOME", name="Watch strap", node=node("/カテゴリー別/携帯電話・スマートフォン/スマートウォッチアクセサリ")),
        listing("C", ptype="HOME", name="Linen Curtains", node=node("/Categories")),
        listing("D", ptype="HOME", name="Silicone Spatula", node=node("/Categories")),
    ]
    rows, report = prepare.build_catalog(listings)
    assert {r["product_id"]: r["category"] for r in rows} == {"A": "decor", "C": "curtains"}
    assert report["broad_types_classified_by"] == {"path": 1, "excluded": 1, "name": 1, "unclassified": 1}
    assert report["dropped"]["broad_type_excluded"] == 1
    assert report["unclassified_examples"] == ["/Categories | Silicone Spatula"]
