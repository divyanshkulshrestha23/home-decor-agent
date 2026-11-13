"""Settings, read from environment variables with defaults."""

import os

from catalog.pricing import PRICE_RANGE_INR

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
DATASET = os.environ.get("BQ_DATASET", "home_decor")
BQ_LOCATION = os.environ.get("BQ_LOCATION", "asia-south1")

# Gemini models run in the global location; the data stays in BQ_LOCATION.
MODEL_LOCATION = os.environ.get("MODEL_LOCATION", "global")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash-lite")
AGENT_TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.2"))

CATEGORIES = sorted(PRICE_RANGE_INR)
TOP_K = 5
