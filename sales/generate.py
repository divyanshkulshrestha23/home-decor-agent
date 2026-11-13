"""Generate synthetic sales for the catalog: customers, orders and order lines.

ABO has products but no sales, so the store's sales history is simulated.
Everything here is an assumption chosen to look plausible, not market data.
The generator is seeded, so the same catalog always gives the same sales.

Patterns built in (the analytics agent has to find them from the data):
- steady growth of about 2% a month, busier weekends;
- a festive-season peak in October and early November, a New Year sale in
  the first week of January, and a monsoon dip in July and August;
- outdoor furniture and decor sell more from March to May and less in the
  monsoon;
- return rates and reasons differ by category (furniture arrives damaged,
  textiles differ in colour from the photo).

Two hidden events, for "why" questions (see sales/STORIES.md):
- from 1 June 2026, rugs from one brand are returned far more often, mostly
  because the colour differs from the photo;
- from 1 January 2026, desk sales fall by about 40%.

Usage:
    python3 -m sales.generate                 # writes data/sales/*.jsonl
"""

import argparse
import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

SEED = 2026
START = dt.date(2025, 1, 1)
END = dt.date(2026, 10, 31)
BASE_DAILY_ORDERS = 55
MONTHLY_GROWTH = 0.02
CUSTOMERS = 9000

# (city, state, tier, weight)
CITIES = [
    ("Mumbai", "Maharashtra", 1, 14), ("Delhi", "Delhi", 1, 14), ("Bengaluru", "Karnataka", 1, 13),
    ("Hyderabad", "Telangana", 1, 8), ("Pune", "Maharashtra", 1, 8), ("Chennai", "Tamil Nadu", 1, 7),
    ("Kolkata", "West Bengal", 1, 6), ("Ahmedabad", "Gujarat", 2, 5), ("Jaipur", "Rajasthan", 2, 4),
    ("Lucknow", "Uttar Pradesh", 2, 4), ("Chandigarh", "Chandigarh", 2, 3), ("Kochi", "Kerala", 2, 3),
    ("Indore", "Madhya Pradesh", 2, 3), ("Surat", "Gujarat", 2, 2), ("Nagpur", "Maharashtra", 2, 2),
    ("Coimbatore", "Tamil Nadu", 2, 2), ("Bhubaneswar", "Odisha", 3, 1), ("Dehradun", "Uttarakhand", 3, 1),
    ("Guwahati", "Assam", 3, 1), ("Mysuru", "Karnataka", 3, 1),
]
DELIVERY_DAYS = {1: (2, 5), 2: (4, 8), 3: (5, 10)}

# Relative share of order lines by category.
CATEGORY_WEIGHT = {
    "cushions": 12, "bedding": 12, "decor": 12, "lighting": 8, "rug": 7, "chair": 7,
    "curtains": 6, "bath_linen": 6, "small_storage": 5, "table": 5, "storage": 4,
    "ottoman": 3, "stool": 3, "sofa": 3, "bed": 2.5, "accent_furniture": 2,
    "window_hardware": 2, "desk": 1.5, "bench": 1, "outdoor_furniture": 1, "outdoor_decor": 1,
}
MULTI_UNIT = {"cushions", "bath_linen", "small_storage", "bedding", "curtains"}

RETURN_RATE = {
    "sofa": 0.05, "bed": 0.04, "chair": 0.06, "table": 0.05, "rug": 0.08, "bedding": 0.07,
    "curtains": 0.08, "cushions": 0.05, "lighting": 0.06, "bath_linen": 0.05,
}
DEFAULT_RETURN_RATE = 0.04
FURNITURE = {"sofa", "bed", "chair", "table", "desk", "storage", "bench", "ottoman", "stool",
             "accent_furniture", "outdoor_furniture"}
REASONS = ["damaged in transit", "colour differs from photo", "size not as expected",
           "changed mind", "quality not as expected"]
FURNITURE_REASON_WEIGHTS = [40, 10, 20, 15, 15]
OTHER_REASON_WEIGHTS = [10, 35, 15, 20, 20]

RUG_STORY_START = dt.date(2026, 6, 1)
RUG_STORY_RETURN_RATE = 0.27
DESK_STORY_START = dt.date(2026, 1, 1)
DESK_STORY_FACTOR = 0.6


def festive(day):
    return dt.date(day.year, 10, 1) <= day <= dt.date(day.year, 11, 10)


def new_year_sale(day):
    return day.month == 1 and day.day <= 7


def monsoon(day):
    return day.month in (7, 8)


def daily_order_mean(day):
    months = (day.year - START.year) * 12 + day.month - START.month
    mean = BASE_DAILY_ORDERS * (1 + MONTHLY_GROWTH) ** months
    mean *= {4: 1.1, 5: 1.3, 6: 1.3}.get(day.weekday(), 1.0)  # Fri, Sat, Sun
    if festive(day):
        mean *= 2.0
    elif new_year_sale(day):
        mean *= 1.4
    elif monsoon(day):
        mean *= 0.85
    return mean


def category_weights(day):
    weights = dict(CATEGORY_WEIGHT)
    if day.month in (3, 4, 5):
        weights["outdoor_furniture"] *= 2
        weights["outdoor_decor"] *= 2
    if monsoon(day):
        weights["outdoor_furniture"] *= 0.5
        weights["outdoor_decor"] *= 0.5
    if festive(day):
        for category in ("bedding", "curtains", "decor", "lighting"):
            weights[category] *= 1.3
    if day >= DESK_STORY_START:
        weights["desk"] *= DESK_STORY_FACTOR
    return weights


def discount_for(day, rng):
    if festive(day):
        return rng.choice([10, 15, 20])
    if new_year_sale(day):
        return 10
    return 5 if rng.random() < 0.2 else 0


def poisson(rng, mean):
    """Poisson draw via the normal approximation; fine for means above ~20."""
    return max(0, round(rng.gauss(mean, mean ** 0.5)))


def story_rug_brand(products):
    """The brand with the most rugs carries the return-rate story."""
    brands = Counter(p["brand"] for p in products if p["category"] == "rug" and p.get("brand"))
    return brands.most_common(1)[0][0] if brands else None


def generate(products, seed=SEED, start=START, end=END):
    """Return (customers, orders, order_items, facts) as lists of dicts."""
    rng = random.Random(seed)
    by_category = {}
    for product in sorted(products, key=lambda p: p["product_id"]):
        by_category.setdefault(product["category"], []).append(product)
    # Zipf-like popularity within each category: a few products sell most.
    popularity = {}
    for category, items in by_category.items():
        rng.shuffle(items)
        popularity[category] = [1 / (rank + 1) ** 1.1 for rank in range(len(items))]

    customer_ids = [f"C{n:05d}" for n in range(1, CUSTOMERS + 1)]
    customer_weight = [1 / (rank + 1) ** 0.6 for rank in range(CUSTOMERS)]  # some come back often
    city_of = {cid: rng.choices(CITIES, weights=[c[3] for c in CITIES])[0] for cid in customer_ids}
    rug_brand = story_rug_brand(products)

    orders, items, first_order = [], [], {}
    day, order_no, line_no = start, 0, 0
    while day <= end:
        weights = category_weights(day)
        categories = [c for c in weights if c in by_category]
        category_w = [weights[c] for c in categories]
        for _ in range(poisson(rng, daily_order_mean(day))):
            order_no += 1
            order_id = f"O{order_no:06d}"
            customer = rng.choices(customer_ids, weights=customer_weight)[0]
            first_order.setdefault(customer, day)
            city, state, tier, _ = city_of[customer]
            months = (day.year - start.year) * 12 + day.month - start.month
            channel = "app" if rng.random() < min(0.65, 0.35 + 0.01 * months) else "web"
            cancelled = rng.random() < 0.03
            delivered = None if cancelled else day + dt.timedelta(days=rng.randint(*DELIVERY_DAYS[tier]))
            orders.append({
                "order_id": order_id, "customer_id": customer, "order_date": day.isoformat(),
                "channel": channel, "status": "cancelled" if cancelled else "delivered",
                "city": city, "state": state, "city_tier": tier,
                "delivered_date": delivered.isoformat() if delivered else None,
            })
            discount = discount_for(day, rng)
            for _ in range(rng.choices([1, 2, 3], weights=[65, 25, 10])[0]):
                category = rng.choices(categories, weights=category_w)[0]
                product = rng.choices(by_category[category], weights=popularity[category])[0]
                quantity = rng.choices([1, 2, 4], weights=[70, 20, 10])[0] if category in MULTI_UNIT else 1
                sale_price = round(product["price_inr"] * (100 - discount) / 100)
                returned, reason, return_date = False, None, None
                if not cancelled:
                    rate = RETURN_RATE.get(category, DEFAULT_RETURN_RATE)
                    story = category == "rug" and product.get("brand") == rug_brand and day >= RUG_STORY_START
                    if story:
                        rate = RUG_STORY_RETURN_RATE
                    if rng.random() < rate:
                        returned = True
                        if story and rng.random() < 0.8:
                            reason = "colour differs from photo"
                        else:
                            reason = rng.choices(
                                REASONS,
                                weights=FURNITURE_REASON_WEIGHTS if category in FURNITURE else OTHER_REASON_WEIGHTS,
                            )[0]
                        return_date = (delivered + dt.timedelta(days=rng.randint(1, 14))).isoformat()
                line_no += 1
                items.append({
                    "order_item_id": f"L{line_no:07d}", "order_id": order_id, "order_date": day.isoformat(),
                    "product_id": product["product_id"], "category": category, "brand": product.get("brand"),
                    "quantity": quantity, "list_price_inr": product["price_inr"], "discount_pct": discount,
                    "sale_price_inr": sale_price, "line_total_inr": sale_price * quantity,
                    "returned": returned, "return_reason": reason, "return_date": return_date,
                })
        day += dt.timedelta(days=1)

    customers = []
    for customer in customer_ids:
        if customer not in first_order:
            continue
        city, state, tier, _ = city_of[customer]
        signup = first_order[customer] - dt.timedelta(days=rng.randint(0, 60))
        customers.append({"customer_id": customer, "city": city, "state": state,
                          "city_tier": tier, "signup_date": signup.isoformat()})
    facts = {"rug_story_brand": rug_brand, "orders": len(orders), "order_items": len(items),
             "customers": len(customers)}
    return customers, orders, items, facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--products", default="data/products.jsonl")
    parser.add_argument("--out", default="data/sales")
    args = parser.parse_args(argv)

    with open(args.products, encoding="utf-8") as f:
        products = [json.loads(line) for line in f if line.strip()]
    customers, orders, items, facts = generate(products)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("customers", customers), ("orders", orders), ("order_items", items)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"Wrote {facts['customers']} customers, {facts['orders']} orders and "
          f"{facts['order_items']} order lines to {out}.")


if __name__ == "__main__":
    main()
