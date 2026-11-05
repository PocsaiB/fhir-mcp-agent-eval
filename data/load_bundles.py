"""Load Synthea FHIR R4 bundles into the local HAPI server.

Usage: python data/load_bundles.py [--dir data/synthea_output] [--force]
Skips loading if the server already holds Patient resources (to avoid duplicates).
The hospital*/practitioner* bundles go first because the patient bundles reference them.
"""
import argparse
import os
import sys
from pathlib import Path

import httpx

BASE = os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir").rstrip("/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/synthea_output")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    files = sorted(Path(a.dir).rglob("*.json"), key=lambda p: (not p.name.startswith(("hospital", "practitioner")), p.name))
    if not files:
        print(f"Nincs JSON a(z) {a.dir} mappában.", file=sys.stderr)
        return 1
    with httpx.Client(timeout=300) as c:
        total = c.get(f"{BASE}/Patient", params={"_summary": "count"}).json().get("total", 0)
        if total and not a.force:
            print(f"A szerveren már {total} beteg van, kihagyom (--force-szal újratölthető).")
            return 0
        for i, f in enumerate(files, 1):
            r = c.post(BASE, content=f.read_bytes(), headers={"Content-Type": "application/fhir+json"})
            status = "ok" if r.status_code < 300 else f"HIBA {r.status_code}: {r.text[:200]}"
            print(f"[{i}/{len(files)}] {f.name}: {status}")
    total = httpx.get(f"{BASE}/Patient", params={"_summary": "count"}).json().get("total")
    print(f"Betöltés kész, betegek száma: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
