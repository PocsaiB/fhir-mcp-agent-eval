"""Rule-based router (no LLM): keyword intent detection, regex entity extraction, fixed flow.

It calls the same MCP tools as the agent and gets the same output format; only the decision making differs:
  1. extract entities (id, name, birth date, year) with regexes;
  2. if there is no id: search_patients, then continue with the FIRST hit (no disambiguation);
  3. call ONE targeted tool chosen by keywords and fill the output into a template.
Deterministic, no learning. The rules encode general Hungarian language patterns and none is written for a specific test
question, but the test set already existed when they were written, so the score may be slightly optimistic.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from agent.mcp_client import open_tools
from agent.records import append_record, build_record

MODEL_NAME = "rule-router-v1"


def plain(s: str) -> str:
    """Lower case, accents removed."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


# Order = priority; the first matching intent wins. Keywords are accent-free and matched as substrings.
INTENTS = [
    ("medications", ["gyogyszer", "recept", "rendeles", "rendelt", "szed"]),
    ("conditions", ["diagnozis", "diagnosztiz", "betegseg", "korkep"]),
    ("observations", ["labor", "vernyomas", "koleszterin", "ldl", "hdl", "trigliceri", "hba1c", "vercukor", "cukorszint",
                      "testtomeg", "suly", "magassag", "bmi", "pulzus", "szivfrekvencia", "meres", "ertek", "szint"]),
    ("patient", ["szulet", "neve", "lakik", "varos", "lakhely", "neme"]),
]
# Measurement type -> LOINC code (order matters: more specific first).
CODES = [("testtomegindex", "39156-5", "testtömegindex (BMI)"), ("bmi", "39156-5", "testtömegindex (BMI)"),
         ("testtomeg", "29463-7", "testtömeg"), ("suly", "29463-7", "testtömeg"),
         ("magassag", "8302-2", "testmagasság"), ("vernyomas", "85354-9", "vérnyomás"),
         ("ldl", "18262-6", "LDL-koleszterin"), ("hdl", "2085-9", "HDL-koleszterin"),
         ("trigliceri", "2571-8", "triglicerid"), ("koleszterin", "2093-3", "összkoleszterin"),
         ("hba1c", "4548-4", "HbA1c"), ("vercukor", "2339-0", "vércukor"), ("cukorszint", "2339-0", "vércukor"),
         ("pulzus", "8867-4", "pulzus"), ("szivfrekvencia", "8867-4", "pulzus")]
ACTIVE_WORDS = ["jelenleg", "aktiv", "jelenlegi", "most", "szed"]
# Question and imperative words that are not names when capitalised at the start of a sentence.
STOP = {"mi", "mit", "ki", "kit", "kik", "kinek", "hany", "mennyi", "mekkora", "mikor", "miota", "melyik", "melyek",
        "milyen", "hol", "hogyan", "miert", "van", "vannak", "volt", "foglald", "mondd", "add", "adj", "sorold",
        "mutasd", "keress", "kerem", "kerlek", "nezd"}
# Common Hungarian suffixes (longer first); stripped from the last word of a name when the full form finds nothing.
ACCENT_SUFFIXES =sorted(["ban", "ben", "ból", "ből", "ba", "be", "hoz", "hez", "höz", "nak", "nek", "nál", "nél",
                          "ról", "ről", "tól", "től", "ra", "re", "ért", "ig", "on", "en", "ön", "t"], key=len, reverse=True)
MONTHS = {"januar": 1, "februar": 2, "marcius": 3, "aprilis": 4, "majus": 5, "junius": 6, "julius": 7,
          "augusztus": 8, "szeptember": 9, "oktober": 10, "november": 11, "december": 12}

NO_RULE = "Ehhez a kérdéshez nincs szabályom, ezért nem tudok válaszolni."
NO_ENTITY = "A kérdésben nem találok beteget (azonosítót, nevet vagy születési dátumot), ezért nem tudok válaszolni."

WORD = re.compile(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+(?:-[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+)*")
ID_RE = re.compile(r"(\d{1,7})\s*-?\s*(?:es|as|os|ös)?\s*azonosít|azonosít\w*\s*[:=]?\s*(\d{1,7})|\bid\s*[:=]?\s*(\d{1,7})", re.I)
ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
HU_DATE_RE = re.compile(r"\b(\d{4})\.\s*([A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+)\s+(\d{1,2})\b")
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\s*-?\s*(?:ben|ban)\b")


@dataclass
class Entities:
    patient_id: str | None = None
    name_forms: list[str] = field(default_factory=list)  # search forms: full form, then suffix-stripped
    birthdate: str | None = None
    year: int | None = None
    active: bool = False


def detect_intent(p: str) -> str | None:
    for intent, words in INTENTS:
        if any(w in p for w in words):
            return intent
    return None


def detect_code(p: str) -> tuple[str, str] | None:
    for key, code, label in CODES:
        if key in p:
            return code, label
    return None


def _base(tok: str) -> str:
    """Strip a hyphenated suffix ('Zieme-nek' -> 'Zieme'); otherwise unchanged."""
    head, _, tail = tok.partition("-")
    return head if tail and tail.lower() in ACCENT_SUFFIXES + ["val", "vel", "nak", "nek"] else tok


def _is_name_token(tok: str) -> bool:
    """Capitalised word; in hyphenated forms every part is capitalised or the tail is a suffix (so 'Szed-e', 'D-vitamin' are rejected)."""
    tok = _base(tok)
    return (all(part[:1].isupper() for part in tok.split("-")) and len(tok) >= 2 and plain(tok) not in STOP
            and not tok.isupper())  # all caps = acronym (BMI, LDL), not a name


def find_name(text: str) -> list[str]:
    ms = list(WORD.finditer(text))
    flags = [_is_name_token(m.group()) for m in ms]
    # A capitalised sentence-initial word is a name only if the next word is one too (e.g. "Agnes Treutel gyógyszerei?").
    if flags and flags[0] and not (len(flags) > 1 and flags[1]):
        flags[0] = False
    group: list[str] = []
    prev_end = 0
    for m, f in zip(ms, flags):
        if not f:
            if group:
                break
            continue
        # Name words may only be separated by a space: punctuation (bracket, colon, comma) ends the group.
        if group and text[prev_end:m.start()].strip():
            break
        group.append(_base(m.group()))
        prev_end = m.end()
    return group


def name_forms(group: list[str]) -> list[str]:
    if not group:
        return []
    forms = [" ".join(group)]
    last = group[-1]
    for suf in ACCENT_SUFFIXES:
        if last.lower().endswith(suf) and len(last) - len(suf) >= 3:
            forms.append(" ".join(group[:-1] + [last[: -len(suf)]]))
    return list(dict.fromkeys(forms))


def extract_entities(text: str, p: str) -> Entities:
    ent = Entities()
    m = ID_RE.search(text)
    if m:
        ent.patient_id = next(g for g in m.groups() if g)
    iso = ISO_RE.search(text)
    hu = HU_DATE_RE.search(text)
    if iso:
        ent.birthdate = iso.group(0)
    elif hu and plain(hu.group(2)) in MONTHS:
        ent.birthdate = f"{hu.group(1)}-{MONTHS[plain(hu.group(2))]:02d}-{int(hu.group(3)):02d}"
    ym = YEAR_RE.search(text)
    if ym and "szulet" not in p:  # "2009-ben született" is a birth year, not a measurement period
        ent.year = int(ym.group(1))
    ent.name_forms = name_forms(find_name(text))
    ent.active = any(w in p for w in ACTIVE_WORDS)
    return ent


def _d(s: str | None) -> str:
    return (s or "")[:10]


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def fmt_obs(o: dict) -> str:
    if o.get("components"):
        body = "; ".join(f"{c['display']}: {_num(c['value'])} {c.get('unit') or ''}".strip() for c in o["components"])
    else:
        body = f"{o.get('display')}: {_num(o.get('value'))} {o.get('unit') or ''}".strip()
    return f"{body} ({_d(o.get('date'))})"


def err_text(res: dict) -> str:
    return f"Hiba történt a lekérdezés során: {res.get('error', {}).get('message', 'ismeretlen hiba')}"


def tpl_patient(res: dict, pid: str) -> str:
    p = res.get("patient")
    if not p:
        return f"Nincs {pid} azonosítójú beteg az adatbázisban."
    return (f"{p['name']} (azonosító: {p['id']}): nem: {p.get('gender')}, születési dátum: {p.get('birth_date')}, "
            f"lakhely: {p.get('city')}, {p.get('state')}, családi állapot: {p.get('marital_status')}.")


def tpl_search(res: dict) -> str:
    ps = res.get("patients", [])
    return f"Találatok ({res.get('total_matches') or len(ps)}): " + "; ".join(
        f"{p['name']} (született: {p.get('birth_date')}, azonosító: {p['id']})" for p in ps) + "."


def tpl_list(who: str, res: dict, key: str, label: str, line) -> str:
    items = res.get(key, [])
    if not items:
        return f"{who}: nincs a feltételeknek megfelelő {label}."
    shown = items[:10]
    return (f"{who} – {label} (összesen {res.get('total_matches') or len(items)}, az első {len(shown)} látszik): "
            + "; ".join(line(x) for x in shown) + ".")


def tpl_obs(who: str, res: dict, code_label: str | None) -> str:
    obs = res.get("observations", [])
    if not obs:
        return f"{who}: nincs a feltételeknek megfelelő mérés."
    if code_label:
        return (f"{who} legutóbbi {code_label} mérése: {fmt_obs(obs[0])}. "
                f"A megadott feltételekkel összesen {res.get('total_matches') or len(obs)} mérés van.")
    return f"{who} legutóbbi mérései: " + "; ".join(fmt_obs(o) for o in obs[:5]) + "."


async def answer(tools, question: str) -> tuple[list[dict], str]:
    steps: list[dict] = []

    async def call(tool: str, args: dict) -> dict:
        t0 = time.time()
        res = await tools.call(tool, args)
        steps.append({"step": len(steps) + 1, "assistant_text": "", "tool_calls": [{"tool": tool, "args": args, "result": res}],
                      "latency_s": round(time.time() - t0, 2), "input_tokens": 0, "output_tokens": 0, "stop_reason": "rule"})
        return res

    p = plain(question)
    intent = detect_intent(p)
    if intent is None:
        return steps, NO_RULE
    ent = extract_entities(question, p)

    pid, pname = ent.patient_id, None
    if pid is None:
        if ent.name_forms:
            hits, res = [], {}
            for form in ent.name_forms:
                args = {"name": form}
                if ent.birthdate:
                    args["birthdate"] = ent.birthdate
                res = await call("search_patients", args)
                if not res.get("ok"):
                    return steps, err_text(res)
                hits = res.get("patients", [])
                if hits:
                    break
            if not hits:
                return steps, f"Nincs ilyen nevű beteg az adatbázisban ({ent.name_forms[0]})."
        elif ent.birthdate:
            res = await call("search_patients", {"birthdate": ent.birthdate})
            if not res.get("ok"):
                return steps, err_text(res)
            hits = res.get("patients", [])
            if not hits:
                return steps, f"Nincs {ent.birthdate} születési dátumú beteg az adatbázisban."
            if intent == "patient":  # asks which patients match a condition, not about one patient's data
                return steps, tpl_search(res)
        else:
            return steps, NO_ENTITY
        pid, pname = hits[0]["id"], hits[0]["name"]  # FIRST hit; no disambiguation

    who = pname or f"A(z) {pid} azonosítójú beteg"
    if intent == "patient":
        res = await call("get_patient", {"patient_id": pid})
        return steps, err_text(res) if not res.get("ok") else tpl_patient(res, pid)
    if intent == "medications":
        args = {"patient_id": pid, **({"status": "active"} if ent.active else {})}
        res = await call("get_medications", args)
        if not res.get("ok"):
            return steps, err_text(res)
        return steps, tpl_list(who, res, "medications", "gyógyszerrendelés",
                               lambda m: f"{m.get('display')} ({_d(m.get('authored_on'))}, {m.get('status')})")
    if intent == "conditions":
        args = {"patient_id": pid, **({"clinical_status": "active"} if ent.active else {})}
        res = await call("get_conditions", args)
        if not res.get("ok"):
            return steps, err_text(res)
        return steps, tpl_list(who, res, "conditions", "diagnózis",
                               lambda c: f"{c.get('display')} ({_d(c.get('onset'))}, {c.get('clinical_status')})")
    code = detect_code(p)
    args = {"patient_id": pid}
    if code:
        args["code"] = code[0]
    if ent.year:
        args["date_from"], args["date_to"] = f"{ent.year}-01-01", f"{ent.year}-12-31"
    res = await call("get_observations", args)
    if not res.get("ok"):
        return steps, err_text(res)
    return steps, tpl_obs(who, res, code[1] if code else None)


async def run_question(tools, qid: str, question: str, log_path: Path, repeat: int = 0) -> dict:
    t0 = time.time()
    steps, final = await answer(tools, question)
    rec = build_record(qid=qid, repeat=repeat, system="baseline", provider="rules", model=MODEL_NAME, effort=None,
                       question=question, steps=steps, final_answer=final, end_reason="final_answer",
                       duration_s=time.time() - t0)
    append_record(log_path, rec)
    return rec


async def main() -> None:
    ap = argparse.ArgumentParser(description="Egy kérdés futtatása a szabályalapú routerrel (próbához).")
    ap.add_argument("question")
    a = ap.parse_args()
    async with open_tools() as tools:
        steps, final = await answer(tools, a.question)
    for s in steps:
        for c in s["tool_calls"]:
            print(f"[{s['step']}] {c['tool']}({c['args']})")
    print(f"\nVÁLASZ:\n{final}")


if __name__ == "__main__":
    asyncio.run(main())
