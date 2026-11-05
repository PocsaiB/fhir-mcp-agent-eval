# fhir-mcp-agent-eval

LLM-alapú eszközhasználat mérése egy helyi HAPI FHIR szerveren. Az ágens magyar nyelvű kérdésekre válaszol egy saját MCP-szerver (5 tool) segítségével.
Öt rendszert hasonlít össze ugyanazokkal a toolokkal és ugyanazzal a 42 kérdéses tesztkészlettel:

| Rendszer | Leírás |
|---|---|
| `baseline` | szabályalapú router, LLM nélkül (alapvonal) |
| `local` | Llama 3.2 3B (Ollama, helyi GPU) |
| `old` | Claude Haiku 4.5 |
| `small` | Claude Haiku 5.5 |
| `large` | Claude Sonnet 5.5 |

Kurzusfeladat: NLP és Transformer-alapú modellek (MSc).

## Eredmény röviden (42 kérdés, helyes végső válasz, kézi javítás után)

| Rendszer | Futás | Siker % | Költség / kérdés (USD) |
|---|---|---|---|
| Szabályalapú router | 42 | 47,6 | 0 |
| Llama 3.2 3B | 42 | 11,9 | 0 |
| Haiku 4.5 | 84 | 95,2 | 0,0097 |
| Haiku 5.5 | 126 | 100,0 | 0,0011 |
| Sonnet 5.5 | 84 | 100,0 | 0,0190 |

A teljes táblázat (eszközválasztás, argumentumok, lépésszám, hallucináció, kategóriánként) az `eval/results/` mappában van.

## Felépítés

| Mappa | Tartalom |
|---|---|
| `server/` | MCP szerver (FastMCP, stdio), 5 tool: `search_patients`, `get_patient`, `get_conditions`, `get_observations`, `get_medications` |
| `agent/` | ágens-ciklus, modell-adapterek (Anthropic, Ollama, OpenAI), közös naplósor-séma |
| `baseline/` | szabályalapú router |
| `eval/` | 42 tesztkérdés, futtató (`run.py`), értékelő (`score.py`), eredmények, kézi javítások (`overrides.jsonl`) |
| `data/` | docker-compose a HAPI-hoz, betöltő szkript, a Synthea-adat generálásának leírása |
| `logs/` | futtatási naplók (JSONL), ezekből készül az értékelés és a hibaelemzés |

## Futtatás

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# adat: HAPI + Synthea (lásd data/README.md), majd
docker compose -f data/docker-compose.yml up -d
python data/load_bundles.py --dir data/synthea_output

# futtatás és értékelés
python -m eval.run --system baseline                      # ingyenes
python -m eval.run --system small --repeat 3 --dry-run    # terv és költségbecslés API-hívás nélkül
python -m eval.run --system small --repeat 3
python -m eval.score                                      # eredmények: eval/results/
```

Az Anthropic-modellekhez az `ANTHROPIC_API_KEY` környezeti változó kell (a repóba nem kerül). A helyi modellhez: `ollama serve`, majd `python -m eval.run --system local`.
A pontozás részletei: `eval/README.md`, az adatról: `data/README.md`.

## Megjegyzések

- A Synthea által generált nyers adat (`data/synthea_output/`, ~144 MB) nem része a repónak. A generálás paraméterei és a használt jar SHA-256 értéke a `data/README.md`-ben van.
- A `logs/` mappa a mérés nyers naplóit tartalmazza, az értékelés ezekből újrafuttatás nélkül reprodukálható (`python -m eval.score`).
- A kézi javításokat (`eval/overrides.jsonl`, 45 futás) az MI-asszisztens (Claude) javasolta a naplók átnézése után, a szerző átnézte és jóváhagyta.
