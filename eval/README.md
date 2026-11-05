# Tesztkészlet (`questions.jsonl`)

30 magyar nyelvű kérdés a helyi HAPI-adatkészletre (Synthea, seed 42, referenciadátum 2026-10-01, lásd `data/README.md`).
Egy sor = egy JSON-objektum.

| Kategória | Db | Mit mér |
|---|---|---|
| `simple` | 8 | egy tool elég (azonosító vagy szűrő alapján) |
| `multi_step` | 10 | keresés → azonosító → lekérdezés → összegzés / összehasonlítás |
| `unanswerable` | 6 | az adat nincs meg; a jó viselkedés ennek kimondása, nem kitalálás |
| `ambiguous` | 6 | több találat egy névre / dátumra (4 db visszakérdezős, 2 db a kérdésből feloldható) |

## Mezők

| Mező | Jelentés |
|---|---|
| `id`, `category`, `question` | azonosító, kategória, a kérdés szövege |
| `patient_ids` | a kérdésben érintett betegek FHIR-azonosítója (a „rossz találat kiválasztása” hiba kimutatásához) |
| `expected_calls` | a helyes megoldáshoz szükséges legkisebb tool-hívássorozat (tool + argumentumok) |
| `expected_behavior` | `answer` (válaszolni), `say_no_data` (a hiány kimondása), `clarify` (visszakérdezés vagy a jelöltek bemutatása, csendes választás nélkül), `disambiguate_and_answer` (a kérdés információjával feloldani, majd válaszolni) |
| `answerable` | `true`, ha van konkrét helyes válasz az adatban |
| `expected_answer` | a várt válasz szövege (a HAPI-adatból számolva, a tool-okkal ellenőrizve) |
| `must_include` | kulcstények a végső válaszhoz: minden elemnek szerepelnie kell (kis/nagybetű nem számít); ha egy elem lista, az alternatívák bármelyike elég |
| `notes` | a kérdés célja / buktatója |

## Kiértékelési szabályok (az értékelő szkript ezek szerint készül)

- **Eszközválasztás:** az `expected_calls` tool-jainak mindegyike előfordul a naplóban. A többlet hívás nem hiba, de a lépésszámban látszik.
  `say_no_data` és `clarify` kérdésnél csak a beteg azonosításához szükséges hívás számít.
- **Argumentumok:** azonosító, LOINC-kód és státusz pontosan egyezik; a név akkor jó, ha minden szava szerepel a hívásban
  (kis/nagybetű és ékezet nélkül); a dátumtartománynak le kell fednie a várt időszakot. Az üres `args` nem kér argumentumot.
- **Helyes végső válasz:** `answer` / `disambiguate_and_answer` esetén a `must_include` automatikusan, a határesetek kézzel.
  `say_no_data` esetén a válasz kimondja a hiányt és nem állít értéket; `clarify` esetén visszakérdez vagy felsorolja a jelölteket.
- **Hallucinált adat:** olyan állítás a válaszban, amely nincs a tool-ok kimenetében (kézi ellenőrzés a naplóból).

## Megjegyzések

- A „jelenleg” = `status: active`; a „legutóbbi” = legfrissebb dátumú elem.
- Ha az adatkészletet újragenerálod, a várt válaszokat is újra kell számolni.
- A szabályalapú alapvonal szabályait nem szabad ezekre a kérdésekre hangolni; ha mégis történik hangolás, azt dokumentálni kell.

## Futtatás és pontozás

```
python -m eval.run --system baseline|small|large [--repeat N] [--ids q01,q05] [--max-cost USD] [--dry-run]
python -m eval.score [--logs ...] [--review SYSTEM [--only-failed]] [--per-question]
```

- A futtató a naplót a `logs/<rendszer>.jsonl`-be írja, megszakítás után folytatható; a pontozó a (rendszer, kérdés, ismétlés) hármasok legutóbbi sorát veszi.
- Kimenet: `eval/results/` (`results.md`, `results.csv`, `runs.csv`, `per_question.csv`).
- Kézi javítás: `eval/overrides.jsonl` soraiban (`system`, `question_id`, `repeat`, majd `answer_ok` / `halluc` / `error_category` / `comment`).
  A 45 kézi javítást az MI-asszisztens (Claude) javasolta a naplók átnézése után, a szerző mindet átnézte és jóváhagyta (2026-11-05). Négy típusuk van:
  magyarított gyógyszer-/diagnózisnevek elfogadása (`answer_ok: true`), származtatott számok kivétele a hallucináció-becslésből (`halluc: false`),
  az alapvonal sablon-számainak kivétele (`halluc: false`), a helyi modell „szerencsés” hiányválaszainak hibásnak minősítése (`answer_ok: false`).
- Az automata ítélet `say_no_data` és `clarify` kérdésnél, valamint a hallucináció-becslésnél alacsony megbízhatóságú: kézi átnézés kell.
