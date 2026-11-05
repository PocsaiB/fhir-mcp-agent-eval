"""Helpers shared by the runner and the scorer."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

# USD per 1M tokens (input, output); used for cost estimates and the --max-cost limit.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-4-5": (1.00, 5.00)}
LABELS = {"rule-router-v1": "Szabályalapú router", "claude-haiku-5-5": "Haiku 5.5 (kicsi)",
          "claude-sonnet-5-5": "Sonnet 5.5 (nagy)", "llama3.2:3b": "Llama 3.2 3B (helyi, butább)",
          "claude-haiku-4-5": "Haiku 4.5 (régebbi)"}


def norm(s: str) -> str:
    """Lower case, accents removed (for comparisons)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def load_questions(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("A kérdés-azonosítóknak egyedinek kell lenniük.")
    return rows


def run_cost(rec: dict) -> float:
    p = PRICES.get(rec.get("model"))
    return 0.0 if not p else (rec.get("input_tokens", 0) * p[0] + rec.get("output_tokens", 0) * p[1]) / 1e6
