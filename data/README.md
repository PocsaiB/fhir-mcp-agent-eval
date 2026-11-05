# Adat: Synthea + helyi HAPI

Reprodukálható felállítás (Docker Desktop-pal):

```powershell
docker compose -f data/docker-compose.yml up -d      # HAPI: http://localhost:8080/fhir
# Synthea (Java nincs a gépen, ezért konténerben); a jar helye: <JARDIR>
docker run --rm -v "<JARDIR>:/synthea:ro" -v "$PWD/data/synthea_output:/out" -w /tmp eclipse-temurin:17-jre `
  java -jar /synthea/synthea-with-dependencies.jar -s 42 -cs 42 -r 20261001 -p 40 `
  --exporter.baseDirectory=/out --exporter.fhir.export=true --exporter.csv.export=false `
  --generate.append_numbers_to_person_names=false Massachusetts
python data/load_bundles.py --dir data/synthea_output
```

Rögzített paraméterek (a mérés reprodukálhatóságához): seed 42, klinikus-seed 42, referenciadátum 2026-10-01,
population 40 (a Synthea 47 rekordot állított elő: 40 élő, 7 elhunyt), állam: Massachusetts, számok nélküli nevek.

Szabadon változó része: a Synthea jar a `master-branch-latest` kiadásból való (mozgó címke), ezért a jar
SHA-256 értéke dokumentálva van: `20098EC70C24C715EA874B19140983634711CC4986196105B5B0A9CA72A4B644`
(187,9 MB, letöltve 2026-10-26). Más jarral ugyanaz a seed eltérő adatot adhat; a mérés alapja a betöltött adatkészlet,
amely a `data/synthea_output/` mappában marad (nincs a repóban, a .gitignore kizárja; ~144 MB).

Betöltött állapot: 47 Patient a HAPI-ban.
