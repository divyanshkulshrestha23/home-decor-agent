"""Tools the agent can call."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

from agent.config import CATEGORIES, TOP_K
from agent.search import Filters


def shopper_image(content):
    """Return (bytes, mime_type) of the first image in a message, or (None, None)."""
    for part in (getattr(content, "parts", None) or []):
        data = getattr(part, "inline_data", None)
        if data is not None and str(data.mime_type or "").startswith("image/") and data.data:
            return data.data, data.mime_type
    return None, None


def _clean_list(values):
    return [v.strip() for v in (values or []) if isinstance(v, str) and v.strip()]


def make_find_similar_tool(search):
    """Build the find_similar_products tool around a ProductSearch."""

    def find_similar_products(
        description: str,
        categories: Optional[list[str]] = None,
        min_price_inr: Optional[int] = None,
        max_price_inr: Optional[int] = None,
        colors: Optional[list[str]] = None,
        materials: Optional[list[str]] = None,
        max_width_cm: Optional[float] = None,
        max_depth_cm: Optional[float] = None,
        max_height_cm: Optional[float] = None,
        tool_context=None,
    ) -> dict:
        """Search the store for products similar to the shopper's photo and description.

        The shopper's photo, if they sent one, is read automatically from their
        message; do not describe the photo's pixels here, describe what they want.

        Args:
          description: Short English description of what the shopper wants, including
            any changes from the photo, e.g. "mid-century armchair in navy velvet".
          categories: Store categories to restrict to. Allowed values: accent_furniture,
            bath_linen, bed, bedding, bench, chair, curtains, cushions, decor, desk,
            lighting, ottoman, outdoor_decor, outdoor_furniture, rug, small_storage,
            sofa, stool, storage, table, window_hardware.
          min_price_inr: Lowest price in rupees, only if the shopper gave one.
          max_price_inr: Highest price in rupees, only if the shopper gave a budget.
          colors: Colours the product must have, only if the shopper insisted on them.
          materials: Materials the product must have, only if the shopper insisted.
          max_width_cm: Maximum width (longest side) in cm, only if the shopper gave space limits.
          max_depth_cm: Maximum depth (shorter side) in cm.
          max_height_cm: Maximum height in cm.
        """
        unknown = [c for c in _clean_list(categories) if c not in CATEGORIES]
        if unknown:
            return {"error": f"Unknown categories {unknown}. Allowed: {CATEGORIES}"}

        image_bytes, image_mime = (None, None)
        if tool_context is not None:
            image_bytes, image_mime = shopper_image(tool_context.user_content)
        if not image_bytes and not (description or "").strip():
            return {"error": "Need a photo or a description to search."}

        filters = Filters(
            categories=_clean_list(categories),
            min_price_inr=min_price_inr,
            max_price_inr=max_price_inr,
            colors=_clean_list(colors),
            materials=_clean_list(materials),
            max_width_cm=max_width_cm,
            max_depth_cm=max_depth_cm,
            max_height_cm=max_height_cm,
        )
        try:
            results = search.search(
                filters, image_bytes=image_bytes, image_mime=image_mime,
                text=(description or "").strip() or None, top_k=TOP_K,
            )
        except Exception as exc:  # noqa: BLE001 - report to the agent instead of crashing the run
            logger.exception("find_similar_products failed")
            return {"error": f"The product search failed ({type(exc).__name__}). Please try again."}
        response = {
            "used_photo": bool(image_bytes),
            "filters_applied": filters.describe(),
            "results": results,
        }
        if not results:
            response["note"] = (
                "No products matched these filters. Products with no listed size are "
                "excluded by size filters. Consider relaxing a filter and searching again."
            )
        return response

    return find_similar_products
