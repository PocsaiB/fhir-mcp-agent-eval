"""Shared log-row schema, written by both the agent and the baseline so the scorer reads them the same way.

One run (question x repeat) = one JSONL row; the steps (model turns / rule calls) are in the "steps" list.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_record(*, qid: str, repeat: int, system: str, provider: str, model: str, effort: str | None, question: str,
                 steps: list[dict], final_answer: str, end_reason: str, duration_s: float,
                 error: str | None = None) -> dict:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question_id": qid, "repeat": repeat, "system": system, "provider": provider, "model": model,
        "effort": effort, "question": question, "steps": steps, "final_answer": final_answer,
        "end_reason": end_reason,  # final_answer | truncated | max_steps | error
        "n_steps": len(steps), "n_tool_calls": sum(len(s["tool_calls"]) for s in steps),
        "duration_s": round(duration_s, 2),
        "input_tokens": sum(s.get("input_tokens", 0) for s in steps),
        "output_tokens": sum(s.get("output_tokens", 0) for s in steps),
    }
    if error:
        rec["error"] = error
    return rec


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
