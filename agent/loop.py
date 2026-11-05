"""Agent loop with JSONL logging. One question = one log row (the steps are in the "steps" list)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from agent.mcp_client import open_tools
from agent.models import make_adapter
from agent.records import append_record, build_record

SYSTEM_PROMPT = """Egy kórházi adatbázis (FHIR) kérdezőasszisztense vagy. A felhasználó kérdéseire kizárólag a \
rendelkezésre álló toolok kimenete alapján válaszolsz.
- A betegeket azonosító (id) alapján lehet lekérdezni; névből előbb a search_patients toollal keress azonosítót.
- Ha több beteg is megfelel a keresésnek, és a kérdés nem dönti el, melyik a kérdezett, kérdezz vissza, ne találgass.
- Ha az adat nincs meg a toolok kimenetében, mondd ki, hogy nincs ilyen adat. Soha ne találj ki értéket.
- Röviden, magyarul válaszolj, a konkrét értékeket és dátumokat a tool-kimenetből idézve."""

MODELS = {  # system name -> (provider, model id)
    "small": ("anthropic", os.environ.get("SMALL_MODEL", "claude-haiku-5-5")),
    "large": ("anthropic", os.environ.get("LARGE_MODEL", "claude-sonnet-5-5")),
    "old": ("anthropic", os.environ.get("OLD_MODEL", "claude-haiku-4-5")),  # older generation (the 3.x Haiku models are retired)
    "local": ("ollama", os.environ.get("LOCAL_MODEL", "llama3.2:3b")),
}
MAX_STEPS = 8


async def run_question(tools, system_name: str, qid: str, question: str, log_path: Path, repeat: int = 0) -> dict:
    provider, model = MODELS[system_name]
    adapter = make_adapter(provider, model, SYSTEM_PROMPT, tools.tools)
    adapter.start(question)
    steps, t0 = [], time.time()
    final, end_reason, error = "", "max_steps", None
    try:
        for i in range(1, MAX_STEPS + 1):
            ts = time.time()
            turn = await adapter.step()
            step = {"step": i, "assistant_text": turn.text, "tool_calls": [], "latency_s": round(time.time() - ts, 2),
                    "input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens,
                    "stop_reason": turn.stop_reason}
            if not turn.tool_calls:
                steps.append(step)
                final = turn.text
                end_reason = "truncated" if turn.stop_reason in ("max_tokens", "length") else "final_answer"
                break
            results = []
            for c in turn.tool_calls:
                out = await tools.call(c["name"], c["args"])
                step["tool_calls"].append({"tool": c["name"], "args": c["args"], "result": out})
                results.append((c["id"], json.dumps(out, ensure_ascii=False)))
            adapter.add_tool_results(results)
            steps.append(step)
    except Exception as e:  # API/network error: log it and carry on with the next question
        end_reason, error = "error", f"{type(e).__name__}: {str(e)[:300]}"
    record = build_record(qid=qid, repeat=repeat, system=system_name, provider=provider, model=model,
                          effort=getattr(adapter, "effort", None), question=question, steps=steps,
                          final_answer=final, end_reason=end_reason, duration_s=time.time() - t0, error=error)
    append_record(log_path, record)
    return record


async def main() -> None:
    ap = argparse.ArgumentParser(description="Egy kérdés futtatása az ágenssel.")
    ap.add_argument("question")
    ap.add_argument("--system", choices=list(MODELS), default="small")
    ap.add_argument("--id", default="adhoc")
    ap.add_argument("--log", default="logs/adhoc.jsonl")
    a = ap.parse_args()
    async with open_tools() as tools:
        rec = await run_question(tools, a.system, a.id, a.question, Path(a.log))
    for s in rec["steps"]:
        for c in s["tool_calls"]:
            print(f"[{s['step']}] {c['tool']}({json.dumps(c['args'], ensure_ascii=False)})")
    print(f"\nVÁLASZ ({rec['end_reason']}, {rec['n_steps']} lépés, {rec['duration_s']} s):\n{rec['final_answer']}")


if __name__ == "__main__":
    asyncio.run(main())
