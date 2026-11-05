"""Scorer: metrics per system and category plus error categories, computed from the logs (no reruns).

  python -m eval.score                              # whichever of the default logs exist
  python -m eval.score --logs logs/baseline.jsonl logs/small.jsonl
  python -m eval.score --review small --only-failed # manual review sheet: question, expected/actual answer, calls
  python -m eval.score --per-question               # per-question hit matrix

The automatic verdicts (tool, arguments, final answer, hallucination) are first-pass estimates; manual corrections go into
eval/overrides.jsonl (per row: system, question_id, repeat, plus any of answer_ok / halluc / error_category / comment)
and take precedence. See eval/README.md for the rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from agent.records import load_records
from eval.common import LABELS, load_questions, norm, run_cost

CATS = ["simple", "multi_step", "unanswerable", "ambiguous", "hard"]
CAT_HU = {"simple": "egyszerű", "multi_step": "többlépéses", "unanswerable": "megválaszolhatatlan",
          "ambiguous": "félreérthető", "hard": "nehéz", "ÖSSZES": "ÖSSZES"}
NO_DATA = [norm(x) for x in ("nincs", "nem talál", "nem találtam", "nem áll rendelkezésre", "nem szerepel", "nem elérhető",
                             "nem érhető el", "nem tudok", "nem tudom", "hiányzik", "nem létezik", "nem rendelkez",
                             "nem adnak", "nem tartalmaz", "nem kérhető le", "nem lehet", "nem található")]
ASK_WORDS = [norm(x) for x in ("melyik", "pontosít", "pontosabb", "kérem", "megadná", "szeretné", "melyikükre",
                               "melyikre", "több beteg", "több találat", "két beteg", "három beteg", "kérdezz")]
MONTHS = {"januar": 1, "februar": 2, "marcius": 3, "aprilis": 4, "majus": 5, "junius": 6, "julius": 7,
          "augusztus": 8, "szeptember": 9, "oktober": 10, "november": 11, "december": 12}
ERR_CATS = ["futtatási hiba", "idő előtti befejezés", "rossz találat kiválasztása", "kitalált válasz (hallucináció)",
            "kihagyott lépés", "rossz paraméter", "félreértett kérdés"]

DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
DATE_HU = re.compile(r"(?<!\d)(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})(?!\d)|(?<!\d)(\d{4})\.\s*([^\W\d_]+)\s+(\d{1,2})(?!\d)")
NUM = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)")


def _tokens(s) -> set[str]:
    return set(re.findall(r"\w+", norm(str(s))))


def arg_ok(key: str, exp, act) -> bool:
    if act is None:
        return False
    if key == "patient_id":
        return str(act).split("/")[-1] == str(exp)
    if key == "name":  # the call's words are a subset or superset of the expected name's words ("Treutel" ~ "Agnes Treutel")
        e, a = _tokens(exp), _tokens(act)
        return bool(a) and (a <= e or e <= a)
    if key == "date_from":
        return str(act)[:10] <= str(exp)[:10]
    if key == "date_to":
        return str(act)[:10] >= str(exp)[:10]
    return norm(str(act)) == norm(str(exp))


def call_matches(exp: dict, act: dict, pids: list[str]) -> bool:
    if exp["tool"] != act["tool"]:
        return False
    args = act.get("args") or {}
    if not all(arg_ok(k, v, args.get(k)) for k, v in exp["args"].items()):
        return False
    if "patient_id" not in exp["args"] and exp["tool"] != "search_patients" and pids:
        return str(args.get("patient_id", "")).split("/")[-1] in pids
    return True


def match_calls(need: list[dict], calls: list[dict], pids: list[str]) -> bool:
    used: set[int] = set()
    for e in need:
        for i, c in enumerate(calls):
            if i not in used and call_matches(e, c, pids):
                used.add(i)
                break
        else:
            return False
    return True


def _found(alt: str, text_n: str) -> bool:
    a = norm(alt)
    if re.fullmatch(r"[\d.,\-: ]+", a):  # number/date: must not match inside a larger number
        return re.search(r"(?<![\d.,])" + re.escape(a) + r"(?!\d|[.,]\d)", text_n) is not None
    return a in text_n


def _has(item, text_n: str) -> bool:
    return any(_found(a, text_n) for a in ([item] if isinstance(item, str) else item))


def candidate_families(q: dict, calls: list[dict]) -> list[str]:
    fams = set()
    for c in calls:
        for p in (c.get("result") or {}).get("patients", []) or []:
            if str(p.get("id")) in q["patient_ids"]:
                fams.add(norm(p["name"].split()[-1]))
    return sorted(fams)


def answer_verdict(q: dict, rec: dict, calls: list[dict]) -> tuple[bool, str]:
    """(is the final answer correct, confidence of the automatic verdict: high | low)."""
    text = rec.get("final_answer") or ""
    if rec.get("end_reason") == "error" or not text.strip():
        return False, "high"
    tn, beh = norm(text), q["expected_behavior"]
    if beh in ("answer", "disambiguate_and_answer"):
        return all(_has(it, tn) for it in q["must_include"]), ("high" if q["must_include"] else "low")
    if beh == "say_no_data":
        return any(p in tn for p in NO_DATA), "low"
    # clarify: asks back, or names at least two candidates
    asks = "?" in text or any(w in tn for w in ASK_WORDS)
    lists = sum(1 for f in candidate_families(q, calls) if f in tn) >= 2
    return asks or lists, "low"


def _dates(text: str) -> set[str]:
    out = {m.group(0) for m in DATE_ISO.finditer(text)}
    for m in DATE_HU.finditer(text):
        if m.group(1):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            mo = MONTHS.get(norm(m.group(5)))
            if not mo:
                continue
            y, d = int(m.group(4)), int(m.group(6))
        out.add(f"{y:04d}-{mo:02d}-{d:02d}")
    return out


def _strip_dates(text: str) -> str:
    return DATE_HU.sub(" ", DATE_ISO.sub(" ", text))


def unsupported_claims(text: str, question: str, calls: list[dict]) -> list[str]:
    """Dates/numbers in the answer that appear in neither the tool outputs nor the question; flagged for manual review."""
    # Arguments of the model's own calls (e.g. LOINC codes) are not claims in the answer, so they count as supported.
    blob = question + " " + " ".join(json.dumps(c.get("result") or {}, ensure_ascii=False) + " "
                                     + json.dumps(c.get("args") or {}, ensure_ascii=False) for c in calls)
    known_dates = _dates(blob) | _dates(question)
    nums: set[float] = {float(y) for y in re.findall(r"(?<!\d)(\d{4})-\d{2}-\d{2}", blob)}
    stack = [c.get("result") or {} for c in calls]
    while stack:  # list lengths (the model may count items)
        x = stack.pop()
        if isinstance(x, dict):
            stack += list(x.values())
        elif isinstance(x, list):
            nums.add(float(len(x)))
            stack += x
    nums |= {float(m) for m in re.findall(r"\d+(?:\.\d+)?", _strip_dates(re.sub(r"\d{4}-\d{2}-\d{2}T[\d:+.\-Z]*", " ", blob)))}
    bad = [d for d in sorted(_dates(text)) if d not in known_dates]
    body = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", _strip_dates(text))
    body = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?", " ", body)  # times of day (23:59) are not measurements
    for m in NUM.finditer(body):
        tok = m.group(1)
        x = float(tok.replace(",", "."))
        dec = len(re.split(r"[.,]", tok)[1]) if re.search(r"[.,]", tok) else 0
        if not any(abs(x - y) <= 0.5 * 10 ** (-dec) + 1e-9 for y in nums):
            bad.append(tok)
    return bad


def auto_error(q: dict, rec: dict, ev: dict, calls: list[dict]) -> str | None:
    if ev["answer_ok"] and not ev["halluc"]:
        return None
    if rec.get("end_reason") == "error":
        return ERR_CATS[0]
    if rec.get("end_reason") in ("truncated", "max_steps") or not (rec.get("final_answer") or "").strip():
        return ERR_CATS[1]
    pids = q["patient_ids"]
    if any(c["args"].get("patient_id") and not re.fullmatch(r"(?:Patient/)?\d+", str(c["args"]["patient_id"])) for c in calls):
        return ERR_CATS[5]  # name or expression in place of the id
    if pids and any(str(c["args"].get("patient_id", "")).split("/")[-1] not in pids
                    for c in calls if c["tool"] != "search_patients" and c["args"].get("patient_id")):
        return ERR_CATS[2]
    if q["expected_behavior"] == "clarify" and not ev["answer_ok"] and any(c["tool"] != "search_patients" for c in calls):
        return ERR_CATS[2]
    if ev["halluc"]:
        return ERR_CATS[3]
    if not ev["tool_ok"]:
        return ERR_CATS[4]
    if not ev["args_ok"]:
        return ERR_CATS[5]
    return ERR_CATS[6]


def evaluate(q: dict, rec: dict, override: dict | None) -> dict:
    calls = [c for s in rec["steps"] for c in s["tool_calls"]]
    need = q["expected_calls"][:1] if q["expected_behavior"] in ("say_no_data", "clarify") else q["expected_calls"]
    want, have = Counter(e["tool"] for e in need), Counter(c["tool"] for c in calls)
    tool_ok = all(have[t] >= n for t, n in want.items())
    args_ok = match_calls(need, calls, q["patient_ids"])
    answer_ok, conf = answer_verdict(q, rec, calls)
    unsupported = unsupported_claims(rec.get("final_answer") or "", q["question"], calls)
    ev = {"tool_ok": tool_ok, "args_ok": args_ok, "answer_ok": answer_ok, "confidence": "auto-" + conf,
          "halluc": bool(unsupported), "unsupported": unsupported, "comment": ""}
    ov = override or {}
    for k in ("answer_ok", "halluc"):
        if k in ov:
            ev[k] = bool(ov[k])
            ev["confidence"] = "manual"
    ev["err_cat"] = ov.get("error_category") or auto_error(q, rec, ev, calls)
    ev["comment"] = ov.get("comment", "")
    return ev


def run_row(q: dict, rec: dict, ev: dict) -> dict:
    return {"system": rec["system"], "label": LABELS.get(rec["model"], rec["model"]), "qid": q["id"], "category": q["category"],
            "repeat": rec.get("repeat", 0), "tool_ok": ev["tool_ok"], "args_ok": ev["args_ok"], "answer_ok": ev["answer_ok"],
            "confidence": ev["confidence"], "halluc": ev["halluc"], "unsupported": " | ".join(ev["unsupported"]),
            "err_cat": ev["err_cat"] or "", "n_tool_calls": rec["n_tool_calls"], "n_steps": rec["n_steps"],
            "input_tokens": rec["input_tokens"], "output_tokens": rec["output_tokens"], "duration_s": rec["duration_s"],
            "cost_usd": round(run_cost(rec), 5), "end_reason": rec["end_reason"], "comment": ev["comment"]}


def pct(xs: list[bool]) -> float:
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stats(rows: list[dict]) -> dict:
    return {"n": len(rows), "tool": pct([r["tool_ok"] for r in rows]), "args": pct([r["args_ok"] for r in rows]),
            "success": pct([r["answer_ok"] for r in rows]), "calls": avg([r["n_tool_calls"] for r in rows]),
            "turns": avg([r["n_steps"] for r in rows]), "halluc": pct([r["halluc"] for r in rows]),
            "tokens_in": avg([r["input_tokens"] for r in rows]), "tokens_out": avg([r["output_tokens"] for r in rows]),
            "time": avg([r["duration_s"] for r in rows]), "cost": avg([r["cost_usd"] for r in rows])}


def fmt(x: float, nd: int = 1) -> str:
    return "–" if x != x else f"{x:.{nd}f}"


def md_table(rows_by_system: dict[str, list[dict]]) -> tuple[str, list[list]]:
    head = ["Rendszer", "Kategória", "Futás", "Eszköz %", "Argum. %", "Siker %", "Tool-hívás", "Fordulók", "Halluc. %"]
    lines, csv_rows = [], []
    for label, rows in rows_by_system.items():
        for cat in ["ÖSSZES"] + CATS:
            sub = rows if cat == "ÖSSZES" else [r for r in rows if r["category"] == cat]
            if not sub:
                continue
            s = stats(sub)
            vals = [label, CAT_HU[cat], s["n"], fmt(s["tool"]), fmt(s["args"]), fmt(s["success"]), fmt(s["calls"], 2),
                    fmt(s["turns"], 2), fmt(s["halluc"])]
            lines.append("| " + " | ".join(f"**{v}**" if cat == "ÖSSZES" and i > 0 else str(v) for i, v in enumerate(vals)) + " |")
            csv_rows.append(vals)
    table = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + lines
    return "\n".join(table), csv_rows


def cost_table(rows_by_system: dict[str, list[dict]]) -> str:
    head = "| Rendszer | Be-token/kérdés | Ki-token/kérdés | Idő/kérdés (s) | Költség/kérdés ($) | Költség összesen ($) |"
    out = [head, "|---|---|---|---|---|---|"]
    for label, rows in rows_by_system.items():
        s = stats(rows)
        out.append(f"| {label} | {fmt(s['tokens_in'], 0)} | {fmt(s['tokens_out'], 0)} | {fmt(s['time'], 1)} | "
                   f"{fmt(s['cost'], 4)} | {fmt(sum(r['cost_usd'] for r in rows), 3)} |")
    return "\n".join(out)


def error_table(rows_by_system: dict[str, list[dict]]) -> str:
    labels = list(rows_by_system)
    out = ["| Hibakategória | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for cat in ERR_CATS:
        counts = [sum(1 for r in rows if r["err_cat"] == cat) for rows in rows_by_system.values()]
        if any(counts):
            out.append(f"| {cat} | " + " | ".join(map(str, counts)) + " |")
    out.append("| (hibátlan futás) | " + " | ".join(str(sum(1 for r in rows if not r["err_cat"])) for rows in rows_by_system.values()) + " |")
    return "\n".join(out)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--logs", nargs="*", help="alapból logs/baseline.jsonl, logs/small.jsonl, logs/large.jsonl")
    ap.add_argument("--overrides", default="eval/overrides.jsonl")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--review", metavar="SYSTEM", help="kézi átnézési lap erre a rendszerre (baseline|small|large)")
    ap.add_argument("--only-failed", action="store_true")
    ap.add_argument("--per-question", action="store_true")
    a = ap.parse_args()

    questions = {q["id"]: q for q in load_questions(a.questions)}
    paths = [Path(p) for p in (a.logs or ["logs/baseline.jsonl", "logs/local.jsonl", "logs/old.jsonl", "logs/small.jsonl", "logs/large.jsonl"])]
    latest: dict[tuple, dict] = {}
    for p in paths:
        for r in load_records(p):
            if r["question_id"] in questions:
                latest[(r["system"], r["question_id"], r.get("repeat", 0))] = r  # a later row overrides an earlier one
    if not latest:
        print("Nincs értékelhető napló (futtasd előbb: python -m eval.run --system baseline).")
        return 1
    overrides = {(o["system"], o["question_id"], o.get("repeat", 0)): o
                 for o in (json.loads(l) for l in Path(a.overrides).read_text(encoding="utf-8").splitlines() if l.strip())
                 } if Path(a.overrides).exists() else {}

    rows, evs = [], {}
    for key, rec in sorted(latest.items()):
        q = questions[key[1]]
        ev = evaluate(q, rec, overrides.get(key))
        evs[key] = (q, rec, ev)
        rows.append(run_row(q, rec, ev))

    if a.review:
        for key, (q, rec, ev) in evs.items():
            if key[0] != a.review or (a.only_failed and ev["answer_ok"] and not ev["halluc"]):
                continue
            calls = [f"{c['tool']}({json.dumps(c['args'], ensure_ascii=False)})" for s in rec["steps"] for c in s["tool_calls"]]
            print(f"=== {q['id']} r{key[2]} [{q['category']}/{q['expected_behavior']}] "
                  f"auto: siker={ev['answer_ok']} ({ev['confidence']}), halluc={ev['halluc']} {ev['unsupported']}, hiba={ev['err_cat']}")
            print(f"K: {q['question']}\nVÁRT: {q['expected_answer']}\nHÍVÁSOK: {' -> '.join(calls) or '(nincs)'}")
            print(f"VÁLASZ: {(rec.get('final_answer') or '').strip()[:600]}\n")
        return 0

    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["label"]].append(r)
    order = ["Szabályalapú router", "Llama 3.2 3B (helyi, butább)", "Haiku 4.5 (régebbi)", "Haiku 5.5 (kicsi)", "Sonnet 5.5 (nagy)"]
    by_sys = {k: by_sys[k] for k in sorted(by_sys, key=lambda k: order.index(k) if k in order else 99)}

    table, csv_rows = md_table(by_sys)
    review_n = {lbl: sum(1 for r in rs if r["confidence"] == "auto-low") for lbl, rs in by_sys.items()}
    report = ("## Eredmények\n\n" + table + "\n\n## Költség és idő\n\n" + cost_table(by_sys) + "\n\n## Hibakategóriák (futásszám)\n\n"
              + error_table(by_sys) + "\n\nKézi ellenőrzést igénylő (alacsony megbízhatóságú) automata ítélet futásonként: "
              + ", ".join(f"{k}: {v}" for k, v in review_n.items()) + "\n")
    print(report)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(report, encoding="utf-8")
    write_csv(out / "results.csv", ["rendszer", "kategória", "futás", "eszköz_%", "argumentum_%", "siker_%", "tool_hívás", "fordulók", "halluc_%"], csv_rows)
    cols = list(rows[0])
    write_csv(out / "runs.csv", cols, [[r[c] for c in cols] for r in rows])

    pq: dict[str, dict] = defaultdict(dict)
    for r in rows:
        pq[r["qid"]].setdefault(r["label"], []).append(r)
    labels = list(by_sys)
    header = ["id", "kategória"] + [f"{l}: siker/eszköz/arg" for l in labels]
    pq_rows = []
    for qid in sorted(pq):
        cells = [qid, questions[qid]["category"]]
        for l in labels:
            rs = pq[qid].get(l, [])
            cells.append(f"{sum(r['answer_ok'] for r in rs)}/{sum(r['tool_ok'] for r in rs)}/{sum(r['args_ok'] for r in rs)} ({len(rs)})" if rs else "")
        pq_rows.append(cells)
    write_csv(out / "per_question.csv", header, pq_rows)
    if a.per_question:
        print("## Kérdésenként (siker/eszköz/argumentum a futásszámból)\n")
        print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
        for c in pq_rows:
            print("| " + " | ".join(c) + " |")
    print(f"\nFájlok: {out}/results.md, results.csv, runs.csv, per_question.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
