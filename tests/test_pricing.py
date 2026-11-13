import pytest

from catalog import pricing


def test_price_is_deterministic():
    assert pricing.price_inr("ABC", "sofa", []) == pricing.price_inr("ABC", "sofa", [])


def test_prices_differ_between_products():
    prices = {pricing.price_inr(f"id{i}", "chair", []) for i in range(50)}
    assert len(prices) > 25


@pytest.mark.parametrize("category", list(pricing.PRICE_RANGE_INR) + ["unknown_category"])
def test_price_stays_in_range_and_ends_in_99(category):
    low, high = pricing.PRICE_RANGE_INR.get(category, pricing.DEFAULT_RANGE_INR)
    for i in range(200):
        price = pricing.price_inr(f"p{i}", category, [])
        assert price % 100 == 99
        assert low - 100 <= price <= high + 100


def test_material_factor_takes_largest_match():
    assert pricing.material_factor(["Leather", "Metal frame"]) == 1.3
    assert pricing.material_factor(["Plastic"]) == 0.8
    assert pricing.material_factor([]) == 1.0


def test_premium_material_raises_price():
    plain = pricing.price_inr("same-id", "chair", ["Fabric"])
    leather = pricing.price_inr("same-id", "chair", ["Leather"])
    assert leather > plain


def test_round_price_minimum():
    assert pricing.round_price(10) == 99
    assert pricing.round_price(12_460) == 12_499
