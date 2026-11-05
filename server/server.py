"""MCP server (FastMCP, stdio). Start: python -m server.server"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from server import tools

mcp = FastMCP("fhir-tools")


@mcp.tool()
def search_patients(name: str | None = None, birthdate: str | None = None,
                    gender: str | None = None, limit: int = 10) -> dict:
    """Betegek keresése név, születési dátum és/vagy nem alapján.

    Args:
        name: Teljes név vagy névrészletek szóközzel elválasztva, pl. "Agnes Treutel" vagy csak "Treutel".
            Minden szónak egy névrész (keresztnév vagy családnév) elejére kell illeszkednie. Kis/nagybetű és ékezet nem számít.
        birthdate: Születési dátum ISO formátumban (YYYY-MM-DD), pl. "1980-05-17".
        gender: Nem: "male", "female", "other" vagy "unknown".
        limit: Legfeljebb ennyi találatot adjon vissza (1-50, alapértelmezés 10).
    Több találat is lehet; a találatok id mezője kell a többi tool patient_id paraméteréhez.
    """
    return tools.search_patients(name, birthdate, gender, limit)


@mcp.tool()
def get_patient(patient_id: str) -> dict:
    """Egy beteg alapadatai (név, nem, születési dátum, lakhely, családi állapot) azonosító alapján.

    Args:
        patient_id: A beteg FHIR azonosítója (a search_patients id mezője).
    """
    return tools.get_patient(patient_id)


@mcp.tool()
def get_conditions(patient_id: str, clinical_status: str | None = None, limit: int = 30) -> dict:
    """Egy beteg diagnózisai / betegségei (Condition), legfrissebb kezdet elöl.

    Args:
        patient_id: A beteg FHIR azonosítója.
        clinical_status: Opcionális szűrés: "active", "resolved" vagy "inactive".
        limit: Legfeljebb ennyi találat (1-50, alapértelmezés 30).
    """
    return tools.get_conditions(patient_id, clinical_status, limit)


@mcp.tool()
def get_observations(patient_id: str, code: str | None = None, category: str | None = None,
                     date_from: str | None = None, date_to: str | None = None, limit: int = 20) -> dict:
    """Egy beteg laborértékei és vitális jelei (Observation), legfrissebb elöl.

    Args:
        patient_id: A beteg FHIR azonosítója.
        code: Opcionális LOINC kód, pl. "85354-9" (vérnyomás; a szisztolés és diasztolés érték a mérés összetevőjeként szerepel),
            "2093-3" (összkoleszterin), "18262-6" (LDL), "2085-9" (HDL), "2571-8" (trigliceridek), "4548-4" (HbA1c),
            "2339-0" (vércukor), "29463-7" (testtömeg), "8302-2" (testmagasság), "39156-5" (BMI), "8867-4" (pulzus).
        category: Opcionális kategória: "vital-signs" vagy "laboratory".
        date_from: Opcionális kezdődátum (YYYY-MM-DD), ezen a napon vagy utána.
        date_to: Opcionális végdátum (YYYY-MM-DD), ezen a napon vagy előtte.
        limit: Legfeljebb ennyi találat (1-50, alapértelmezés 20).
    """
    return tools.get_observations(patient_id, code, category, date_from, date_to, limit)


@mcp.tool()
def get_medications(patient_id: str, status: str | None = None, limit: int = 30) -> dict:
    """Egy beteg gyógyszerrendelései (MedicationRequest), legfrissebb rendelés elöl.

    Args:
        patient_id: A beteg FHIR azonosítója.
        status: Opcionális szűrés: "active", "completed", "stopped" stb.
        limit: Legfeljebb ennyi találat (1-50, alapértelmezés 30).
    """
    return tools.get_medications(patient_id, status, limit)


if __name__ == "__main__":
    mcp.run()
