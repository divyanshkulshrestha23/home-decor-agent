import datetime as dt
import json
import random
from collections import Counter

import pytest

from catalog.pricing import PRICE_RANGE_INR
from catalog.schema import row_errors
from sales import generate as gen
from sales import load
from sales.schema import TABLES


def fake_catalog():
    rng = random.Random(1)
    brands = ["Rivet", "Stone & Beam", "nuLOOM", "Pinzon"]
    return [
        {"product_id": f"{cat}-{i}", "category": cat, "brand": rng.choice(brands),
         "price_inr": rng.randint(500, 40000)}
        for cat in PRICE_RANGE_INR for i in range(20)
    ]


@pytest.fixture(scope="module")
def sales():
    # A shorter window that still covers festive season, New Year and both stories.
    return gen.generate(fake_catalog(), start=dt.date(2025, 9, 1), end=dt.date(2026, 8, 31))


def test_generation_is_deterministic(sales):
    again = gen.generate(fake_catalog(), start=dt.date(2025, 9, 1), end=dt.date(2026, 8, 31))
    assert again[3] == sales[3]
    assert again[2][:50] == sales[2][:50]


def test_rows_match_schemas(sales):
    customers, orders, items, _ = sales
    for name, rows in (("customers", customers), ("orders", orders), ("order_items", items)):
        assert all(row_errors(r, TABLES[name]) == [] for r in rows), name


def test_orders_and_lines_are_consistent(sales):
    customers, orders, items, facts = sales
    by_id = {o["order_id"]: o for o in orders}
    assert facts["orders"] == len(orders) and facts["order_items"] == len(items)
    assert set(i["order_id"] for i in items) <= set(by_id)
    for item in items:
        assert item["line_total_inr"] == item["sale_price_inr"] * item["quantity"]
        assert item["sale_price_inr"] <= item["list_price_inr"]
        if by_id[item["order_id"]]["status"] == "cancelled":
            assert item["returned"] is False
        if item["returned"]:
            assert item["return_reason"] and item["return_date"] > item["order_date"]
    for order in orders:
        assert (order["delivered_date"] is None) == (order["status"] == "cancelled")
    first = {}
    for order in orders:
        first.setdefault(order["customer_id"], order["order_date"])
    assert all(c["signup_date"] <= first[c["customer_id"]] for c in customers)


def test_festive_season_is_busier(sales):
    monthly = Counter(o["order_date"][:7] for o in sales[1])
    assert monthly["2025-10"] > 1.6 * monthly["2025-09"]


def test_rug_return_story(sales):
    items, brand = sales[2], sales[3]["rug_story_brand"]

    def rate(story_brand, after):
        rows = [i for i in items if i["category"] == "rug"
                and (i["brand"] == brand) == story_brand and (i["order_date"] >= "2026-06-01") == after]
        return sum(i["returned"] for i in rows) / len(rows)

    assert rate(True, True) > 0.18
    assert rate(True, False) < 0.12 and rate(False, True) < 0.12


def test_desk_story_reduces_share():
    weights_2025 = gen.category_weights(dt.date(2025, 5, 5))
    weights_2026 = gen.category_weights(dt.date(2026, 5, 5))
    assert weights_2026["desk"] == pytest.approx(weights_2025["desk"] * gen.DESK_STORY_FACTOR)


def test_load_check_reports_invalid_rows(tmp_path):
    good = {"customer_id": "C1", "city": "Pune", "state": "Maharashtra", "city_tier": 1, "signup_date": "2025-01-02"}
    bad = dict(good, signup_date="2 Jan 2025")
    path = tmp_path / "customers.jsonl"
    path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    count, problems = load.check(path, TABLES["customers"])
    assert count == 2 and problems[0][0] == 2
