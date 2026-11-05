"""FHIR query tool logic: shortened, clean JSON instead of the raw Bundle.

Every function returns a dict and never raises:
  success:  {"ok": True, "count": N, "<kulcs>": [...], "note": "..."?}
  error:   {"ok": False, "error": {"type": "...", "message": "..."}}
"""
from __future__ import annotations

import os
from typing import Any

import httpx

FHIR_BASE = os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir").rstrip("/")
TIMEOUT = float(os.environ.get("FHIR_TIMEOUT", "30"))
MAX_LIMIT = 50


def _error(kind: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"type": kind, "message": message}}


def _fetch(resource: str, params: dict[str, Any]) -> tuple[dict | None, dict | None]:
    """FHIR search returning the raw Bundle. Returns (bundle, error dict | None)."""
    clean = {k: v for k, v in params.items() if v not in (None, "")}
    try:
        r = httpx.get(f"{FHIR_BASE}/{resource}", params=clean, timeout=TIMEOUT,
                      headers={"Accept": "application/fhir+json"})
    except httpx.HTTPError as e:
        return None, _error("connection_error", f"A FHIR szerver nem érhető el: {e.__class__.__name__}")
    if r.status_code >= 400:
        detail = ""
        try:
            detail = r.json()["issue"][0].get("diagnostics", "")
        except Exception:
            pass
        return None, _error("fhir_error", f"HTTP {r.status_code}: {detail or r.reason_phrase}")
    return r.json(), None


def _search(resource: str, params: dict[str, Any]) -> tuple[list[dict], int | None, dict | None]:
    """FHIR search. Returns (resources, total, error dict | None)."""
    bundle, err = _fetch(resource, params)
    if err:
        return [], None, err
    entries = [e["resource"] for e in bundle.get("entry", []) if e.get("resource", {}).get("resourceType") == resource]
    return entries, bundle.get("total"), None


def _limit(limit: int | None, default: int = 20) -> int:
    try:
        n = int(limit) if limit is not None else default
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, MAX_LIMIT))


def _coding(cc: dict | None) -> tuple[str | None, str | None]:
    """CodeableConcept -> (code, display text)."""
    if not cc:
        return None, None
    text = cc.get("text")
    for c in cc.get("coding", []):
        return c.get("code"), c.get("display") or text
    return None, text


def _patient_name(p: dict) -> str:
    names = p.get("name", [])
    official = next((n for n in names if n.get("use") == "official"), names[0] if names else {})
    given = " ".join(official.get("given", []))
    return f"{given} {official.get('family', '')}".strip()


def _patient_ref(patient_id: str) -> str:
    return patient_id if patient_id.startswith("Patient/") else f"Patient/{patient_id}"


def _bare_id(patient_id: str) -> str:
    return patient_id.split("/")[-1]


def _short_patient(p: dict) -> dict:
    return {"id": p.get("id"), "name": _patient_name(p), "gender": p.get("gender"),
            "birth_date": p.get("birthDate"), "deceased": bool(p.get("deceasedDateTime") or p.get("deceasedBoolean"))}



def search_patients(name: str | None = None, birthdate: str | None = None,
                    gender: str | None = None, limit: int | None = 10) -> dict:
    # FHIR "name" matches a single string ("Agnes Treutel" would find nothing), so split into words:
    # repeated parameters are ANDed, which makes both "Agnes Treutel" and "Treutel" work.
    tokens = name.split() if name and name.strip() else None
    if not any([tokens, birthdate, gender]):
        return _error("invalid_arguments", "Legalább egy keresési feltétel (name, birthdate, gender) kell.")
    res, total, err = _search("Patient", {"name": tokens, "birthdate": birthdate, "gender": gender,
                                          "_count": _limit(limit, 10), "_total": "accurate"})
    if err:
        return err
    out: dict[str, Any] = {"ok": True, "count": len(res), "total_matches": total,
                           "patients": [_short_patient(p) for p in res]}
    if not res:
        out["note"] = "Nincs a feltételeknek megfelelő beteg."
    elif total and total > len(res):
        out["note"] = f"Összesen {total} találat van, csak az első {len(res)} látszik."
    return out


def get_patient(patient_id: str) -> dict:
    if not patient_id:
        return _error("invalid_arguments", "patient_id kötelező.")
    res, _, err = _search("Patient", {"_id": _bare_id(patient_id)})
    if err:
        return err
    if not res:
        return {"ok": True, "count": 0, "patient": None, "note": f"Nincs {patient_id} azonosítójú beteg."}
    p = res[0]
    out = {"ok": True, "count": 1, "patient": _short_patient(p)}
    addr = (p.get("address") or [{}])[0]
    out["patient"]["city"] = addr.get("city")
    out["patient"]["state"] = addr.get("state")
    out["patient"]["marital_status"] = _coding(p.get("maritalStatus"))[1]
    return out


def get_conditions(patient_id: str, clinical_status: str | None = None, limit: int | None = 30) -> dict:
    if not patient_id:
        return _error("invalid_arguments", "patient_id kötelező.")
    res, total, err = _search("Condition", {"patient": _bare_id(patient_id), "clinical-status": clinical_status,
                                            "_sort": "-onset-date", "_count": _limit(limit, 30),
                                            "_total": "accurate"})
    if err:
        return err
    items = []
    for c in res:
        code, display = _coding(c.get("code"))
        items.append({"code": code, "display": display,
                      "clinical_status": _coding(c.get("clinicalStatus"))[0],
                      "onset": c.get("onsetDateTime"), "abatement": c.get("abatementDateTime"),
                      "recorded": c.get("recordedDate")})
    out: dict[str, Any] = {"ok": True, "count": len(items), "total_matches": total, "conditions": items}
    if not items:
        out["note"] = "Ehhez a beteghez nincs diagnózis a megadott feltételekkel."
    elif total and total > len(items):
        out["note"] = f"Összesen {total} diagnózis van, csak a legfrissebb {len(items)} látszik (a limit legfeljebb 50-ig növelhető, vagy szűrhető)."
    return out


def _obs_value(o: dict) -> dict:
    if "valueQuantity" in o:
        q = o["valueQuantity"]
        return {"value": q.get("value"), "unit": q.get("unit")}
    if "valueCodeableConcept" in o:
        return {"value": _coding(o["valueCodeableConcept"])[1]}
    for k in ("valueString", "valueBoolean", "valueInteger"):
        if k in o:
            return {"value": o[k]}
    if o.get("component"):  # e.g. blood pressure: systolic + diastolic component
        comps = []
        for c in o["component"]:
            q = c.get("valueQuantity", {})
            comps.append({"display": _coding(c.get("code"))[1], "value": q.get("value"), "unit": q.get("unit")})
        return {"components": comps}
    return {"value": None}


def get_observations(patient_id: str, code: str | None = None, category: str | None = None,
                     date_from: str | None = None, date_to: str | None = None,
                     limit: int | None = 20) -> dict:
    if not patient_id:
        return _error("invalid_arguments", "patient_id kötelező.")
    # Several date parameters with the same name: httpx takes them as a list.
    params: dict[str, Any] = {"patient": _bare_id(patient_id), "code": code, "category": category,
                              "_sort": "-date", "_count": _limit(limit, 20), "_total": "accurate"}
    dates = []
    if date_from:
        dates.append(f"ge{date_from}")
    if date_to:
        dates.append(f"le{date_to}")
    if dates:
        params["date"] = dates
    res, total, err = _search("Observation", params)
    if err:
        return err
    items = []
    for o in res:
        c, display = _coding(o.get("code"))
        items.append({"code": c, "display": display, "category": _coding((o.get("category") or [None])[0])[0],
                      "date": o.get("effectiveDateTime"), **_obs_value(o)})
    out: dict[str, Any] = {"ok": True, "count": len(items), "total_matches": total, "observations": items}
    if not items:
        out["note"] = "Nincs a feltételeknek megfelelő mérés."
    elif total and total > len(items):
        out["note"] = f"Összesen {total} mérés van, a legfrissebb {len(items)} látszik."
    return out


def get_medications(patient_id: str, status: str | None = None, limit: int | None = 30) -> dict:
    if not patient_id:
        return _error("invalid_arguments", "patient_id kötelező.")
    # Some Synthea orders use medicationReference instead of medicationCodeableConcept:
    # fetch the referenced Medication resource in the same search (_include).
    bundle, err = _fetch("MedicationRequest", {"patient": _bare_id(patient_id), "status": status,
                                               "_include": "MedicationRequest:medication", "_sort": "-authoredon",
                                               "_count": _limit(limit, 30), "_total": "accurate"})
    if err:
        return err
    entries = [e["resource"] for e in bundle.get("entry", [])]
    res = [x for x in entries if x.get("resourceType") == "MedicationRequest"]
    total = bundle.get("total")
    medications = {f"Medication/{x.get('id')}": x for x in entries if x.get("resourceType") == "Medication"}
    items = []
    for m in res:
        ref = m.get("medicationReference") or {}
        concept = m.get("medicationCodeableConcept") or medications.get(ref.get("reference", ""), {}).get("code")
        code, display = _coding(concept)
        display = display or ref.get("display")
        items.append({"code": code, "display": display, "status": m.get("status"),
                      "authored_on": m.get("authoredOn"),
                      "reason": _coding((m.get("reasonCode") or [None])[0])[1]})
    out: dict[str, Any] = {"ok": True, "count": len(items), "total_matches": total, "medications": items}
    if not items:
        out["note"] = "Ehhez a beteghez nincs gyógyszerrendelés a megadott feltételekkel."
    elif total and total > len(items):
        out["note"] = f"Összesen {total} rendelés van, csak a legfrissebb {len(items)} látszik (a limit legfeljebb 50-ig növelhető, vagy szűrhető)."
    return out


TOOL_FUNCS = {f.__name__: f for f in (search_patients, get_patient, get_conditions, get_observations, get_medications)}
