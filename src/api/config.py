MODELS = {"cheap": "allam-2-7b", "mid": "openai/gpt-oss-20b", "strong": "openai/gpt-oss-120b"}
# USD per 1M tokens (input, output). Keep these in sync with PRICES in src/labeling/grade.py.
PRICES = {"cheap": (0.05, 0.08), "mid": (0.075, 0.30), "strong": (0.15, 0.60)}
DEFAULT_POLICY = "mid"   # always-mid is the best measured trade-off
MAX_TOKENS = 1024