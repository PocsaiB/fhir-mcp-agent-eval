"""Runner: runs the test questions through one system (baseline | small | large | old | local) and writes a log.

  python -m eval.run --system baseline
  python -m eval.run --system small --repeat 3 --dry-run     # plan and estimated cost only
  python -m eval.run --system large --repeat 2 --max-cost 3  # real API calls

The log defaults to logs/<system>.jsonl. After an interruption the same command resumes (finished question/repeat
pairs are skipped, failed ones are rerun). The API key comes from the ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from agent.mcp_client import open_tools
from agent.records import load_records
from eval.common import PRICES, load_questions, run_cost

# Estimated tokens per question (mean of a 3-question Haiku trial run); only used for the --dry-run plan.
EST_TOKENS = (12000, 600)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=["baseline", "small", "large", "old", "local"])
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--repeat", type=int, default=1, help="hányszor futtassa végig a készletet")
    ap.add_argument("--ids", help="csak ezek a kérdések (vesszővel elválasztva), pl. q01,q05")
    ap.add_argument("--out", help="naplófájl (alapból logs/<rendszer>.jsonl)")
    ap.add_argument("--max-cost", type=float, default=3.0, help="USD-korlát erre a futtatásra (LLM-rendszereknél)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    questions = load_questions(a.questions)
    if a.ids:
        want = set(a.ids.split(","))
        questions = [q for q in questions if q["id"] in want]
    out = Path(a.out or f"logs/{a.system}.jsonl")
    done = {(r["question_id"], r.get("repeat", 0)) for r in load_records(out) if r["end_reason"] != "error"}
    todo = [(q, r) for r in range(a.repeat) for q in questions if (q["id"], r) not in done]
    llm = a.system != "baseline"

    if llm:
        from agent.loop import MODELS
        model = MODELS[a.system][1]
        pin, pout = PRICES.get(model, (0.0, 0.0))  # local model: free
        est = len(todo) * (EST_TOKENS[0] * pin + EST_TOKENS[1] * pout) / 1e6
        print(f"{a.system} ({model}): {len(todo)} futtatás vár ({len(done)} kész), becsült költség ~${est:.2f}, korlát ${a.max_cost}")
    else:
        print(f"baseline: {len(todo)} futtatás vár ({len(done)} kész)")
    if a.dry_run or not todo:
        return 0
    if a.system in ("small", "large", "old") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Hiányzik az ANTHROPIC_API_KEY környezeti változó (új terminálban/újraindított alkalmazásban látszik).")
        return 1

    total = 0.0
    async with open_tools() as tools:
        for i, (q, r) in enumerate(todo, 1):
            if llm:
                from agent.loop import run_question
                rec = await run_question(tools, a.system, q["id"], q["question"], out, repeat=r)
            else:
                from baseline.router import run_question
                rec = await run_question(tools, q["id"], q["question"], out, repeat=r)
            total += run_cost(rec)
            print(f"[{i}/{len(todo)}] {q['id']} r{r} | {rec['end_reason']} | {rec['n_tool_calls']} hívás | {rec['duration_s']} s"
                  + (f" | in {rec['input_tokens']} out {rec['output_tokens']} | összesen ${total:.3f}" if llm else ""), flush=True)
            if rec["end_reason"] == "error":
                print("   hiba:", rec.get("error"))
            if llm and total > a.max_cost:
                print(f"A költségkorlát (${a.max_cost}) átlépve, leállok; ugyanezzel a paranccsal folytatható.")
                return 2
    print(f"Kész. Napló: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
