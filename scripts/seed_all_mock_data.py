"""
seed_all_mock_data.py
---------------------
AgentForge — Healthcare RCM AI Agent — Full Mock Data Seeder
-------------------------------------------------------------
Seeds allergies, medical problems, medications, and insurance records
for all patients from mock_data/ into the OpenEMR Docker portal.

Patients must already exist in OpenEMR (run seed_portal.py first).
This script looks up each patient by name via FHIR, then posts their
clinical data via the OpenEMR Legacy Standard REST API.

Data seeded per patient:
  • Allergies          POST /api/patient/{uuid}/allergy
  • Medical Problems   POST /api/patient/{uuid}/encounter/{eid}/medical_problem
  • Medications        POST /api/patient/{uuid}/encounter/{eid}/medication
  • Insurance          POST /api/patient/{uuid}/insurance/{type}

Usage:
    cd openemr-agent
    python scripts/seed_all_mock_data.py
    python scripts/seed_all_mock_data.py --dry-run
    python scripts/seed_all_mock_data.py --filter P001,P012

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)

from openemr_client import OpenEMRClient, OpenEMRAPIError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seed_all")

_MOCK_DIR         = os.path.join(_REPO_ROOT, "mock_data")
_PATIENTS_JSON    = os.path.join(_MOCK_DIR, "patients.json")
_MEDICATIONS_JSON = os.path.join(_MOCK_DIR, "medications.json")

# ---------------------------------------------------------------------------
# ICD-10 condition → code mapping
# ---------------------------------------------------------------------------
_ICD10: dict[str, str] = {
    "type 2 diabetes":              "E11",
    "hypertension":                 "I10",
    "chronic kidney disease":       "N18",
    "rheumatoid arthritis":         "M05",
    "osteoporosis":                 "M81",
    "atrial fibrillation":          "I48",
    "heart failure":                "I50",
    "depression":                   "F32",
    "chronic pain":                 "R52",
    "hyperlipidemia":               "E78",
    "gout":                         "M10",
    "bacterial infection":          "A49",
    "bipolar disorder":             "F31",
    "hypothyroidism":               "E03",
    "breast cancer":                "C50",
    "her2-positive":                "C50.912",
    "post-surgical monitoring":     "Z09",
    "annual checkup":               "Z00",
    "post-discharge monitoring":    "Z09",
    "preventive care":              "Z01",
    "malignant neoplasm":           "C50",
    "invasive ductal carcinoma":    "C50.912",
}

def _icd10(condition: str) -> str:
    return _ICD10.get(condition.lower().strip(), "Z99")


# ---------------------------------------------------------------------------
# Insurance plan mapping per patient local ID
# ---------------------------------------------------------------------------
_INSURANCE: dict[str, dict[str, str]] = {
    "P001": {"provider": "Cigna",        "plan_name": "Cigna HMO Plus",       "policy_number": "CIG-456123", "group_number": "GRP-CIG-001"},
    "P002": {"provider": "UnitedHealth", "plan_name": "UHC Choice Plus PPO",  "policy_number": "UHC-321654", "group_number": "GRP-UHC-002"},
    "P003": {"provider": "Medicare",     "plan_name": "Medicare Part B",       "policy_number": "MED-654321", "group_number": "GRP-MED-003"},
    "P004": {"provider": "Cigna",        "plan_name": "Cigna PPO Select",      "policy_number": "CIG-789456", "group_number": "GRP-CIG-004"},
    "P005": {"provider": "Aetna",        "plan_name": "Aetna HMO Standard",   "policy_number": "AET-234567", "group_number": "GRP-AET-005"},
    "P006": {"provider": "Aetna",        "plan_name": "Aetna PPO Silver",      "policy_number": "AET-112233", "group_number": "GRP-AET-006"},
    "P007": {"provider": "Blue Cross",   "plan_name": "BCBS PPO Classic",      "policy_number": "BCB-345678", "group_number": "GRP-BCB-007"},
    "P008": {"provider": "Medicare",     "plan_name": "Medicare Advantage",    "policy_number": "MED-567890", "group_number": "GRP-MED-008"},
    "P009": {"provider": "UnitedHealth", "plan_name": "UHC Student Health",    "policy_number": "UHC-901234", "group_number": "GRP-UHC-009"},
    "P010": {"provider": "Cigna",        "plan_name": "Cigna Select Plus",     "policy_number": "CIG-567890", "group_number": "GRP-CIG-010"},
    "P011": {"provider": "Blue Cross",   "plan_name": "BCBS HMO Essential",   "policy_number": "BCB-123456", "group_number": "GRP-BCB-011"},
    "P012": {"provider": "Aetna",        "plan_name": "Aetna PPO Gold",        "policy_number": "AET-789012", "group_number": "GRP-AET-012"},
}

# Supplemental patient not in patients.json
_SUPPLEMENTAL: list[dict[str, Any]] = [
    {
        "id": "P012",
        "name": "Maria J. Gonzalez",
        "dob": "1978-06-15",
        "gender": "Female",
        "allergies": ["Penicillin"],
        "conditions": ["Breast Cancer", "Invasive Ductal Carcinoma", "Post-Surgical Monitoring"],
    }
]

_SUPPLEMENTAL_MEDS: dict[str, list[dict]] = {
    "P012": [
        {"name": "Palbociclib", "dose": "125mg", "frequency": "once daily (21-day cycle)", "prescribed": "2026-02-15"},
        {"name": "Letrozole",   "dose": "2.5mg",  "frequency": "once daily",               "prescribed": "2025-12-01"},
    ]
}


# ---------------------------------------------------------------------------
# Patient lookup helpers
# ---------------------------------------------------------------------------

async def _find_patient_uuid(client: OpenEMRClient, name: str) -> str | None:
    """
    Search OpenEMR FHIR for a patient by name and return their UUID.

    Args:
        client: Connected OpenEMRClient.
        name:   Full name string (e.g. "John Smith" or "Maria J. Gonzalez").

    Returns:
        FHIR UUID string, or None if not found.
    """
    parts  = name.split()
    family = parts[-1]
    given  = parts[0]
    token  = await client._ensure_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    resp = await client._http.get(
        f"{client.base_url}/apis/default/fhir/Patient",
        params={"family": family, "given": given},
        headers=headers,
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    entries = data.get("entry", [])
    if not entries:
        return None
    return entries[0]["resource"].get("id")


async def _get_patient_pid(client: OpenEMRClient, puuid: str) -> int | None:
    """Return OpenEMR integer pid for a given FHIR UUID (needed for encounter)."""
    token = await client._ensure_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    resp = await client._http.get(
        f"{client.base_url}/apis/default/fhir/Patient/{puuid}",
        headers=headers,
    )
    if resp.status_code != 200:
        return None
    resource = resp.json()
    for ident in resource.get("identifier", []):
        if "pid" in ident.get("system", "").lower() or ident.get("use") == "usual":
            try:
                return int(ident["value"])
            except (ValueError, KeyError):
                pass
    return None


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

async def _seed_allergy(
    client: OpenEMRClient,
    puuid: str,
    allergy: str,
    dry_run: bool,
) -> bool:
    """POST one allergy record for a patient."""
    payload = {
        "title":        allergy,
        "begdate":      "2020-01-01",
        "allergy_type": "allergy",
        "severity_al":  "severe" if allergy.lower() in ("penicillin", "sulfa") else "moderate",
        "reaction":     "Anaphylaxis" if allergy.lower() == "penicillin" else "Adverse reaction",
    }
    if dry_run:
        log.info("    [DRY-RUN] POST allergy: %s", allergy)
        return True
    try:
        token = await client._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        resp = await client._http.post(
            f"{client.base_url}/apis/default/api/patient/{puuid}/allergy",
            headers=headers,
            json=payload,
        )
        if resp.status_code in range(200, 300):
            log.info("    ✅ Allergy: %s", allergy)
            return True
        log.warning("    ⚠️  Allergy %s — HTTP %d: %s", allergy, resp.status_code, resp.text[:100])
        return False
    except Exception as exc:
        log.warning("    ❌ Allergy %s error: %s", allergy, exc)
        return False


async def _create_encounter_for_seeding(
    client: OpenEMRClient,
    puuid: str,
    dry_run: bool,
) -> str:
    """Create a seeding encounter and return its integer eid as string."""
    if dry_run:
        return "DRY_RUN_EID"
    try:
        enc = await client.post_encounter(
            patient_uuid=puuid,
            reason="AgentForge mock data seed — clinical history",
        )
        return str(enc.get("eid") or enc.get("encounter") or "")
    except Exception as exc:
        log.warning("    ❌ Could not create encounter: %s", exc)
        return ""


async def _seed_medical_problem(
    client: OpenEMRClient,
    puuid: str,
    eid: str,
    condition: str,
    dry_run: bool,
) -> bool:
    """POST one medical problem under an encounter."""
    dx_code = _icd10(condition)
    payload = {
        "title":   condition,
        "begdate": "2020-01-01",
        "dx_code": dx_code,
    }
    if dry_run:
        log.info("    [DRY-RUN] POST medical_problem: %s (%s)", condition, dx_code)
        return True
    try:
        token = await client._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        resp = await client._http.post(
            f"{client.base_url}/apis/default/api/patient/{puuid}/encounter/{eid}/medical_problem",
            headers=headers,
            json=payload,
        )
        if resp.status_code in range(200, 300):
            log.info("    ✅ Condition: %s (%s)", condition, dx_code)
            return True
        log.warning("    ⚠️  Condition %s — HTTP %d: %s", condition, resp.status_code, resp.text[:100])
        return False
    except Exception as exc:
        log.warning("    ❌ Condition %s error: %s", condition, exc)
        return False


async def _seed_medication(
    client: OpenEMRClient,
    puuid: str,
    eid: str,
    med: dict[str, Any],
    dry_run: bool,
) -> bool:
    """POST one medication under an encounter."""
    payload = {
        "drug":      med["name"],
        "dosage":    med.get("dose", ""),
        "route":     "oral",
        "interval":  med.get("frequency", ""),
        "start_date": med.get("prescribed", "2024-01-01"),
        "note":      f"Prescribed {med.get('prescribed', '')}",
    }
    if dry_run:
        log.info("    [DRY-RUN] POST medication: %s %s", med["name"], med.get("dose", ""))
        return True
    try:
        token = await client._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        resp = await client._http.post(
            f"{client.base_url}/apis/default/api/patient/{puuid}/encounter/{eid}/medication",
            headers=headers,
            json=payload,
        )
        if resp.status_code in range(200, 300):
            log.info("    ✅ Medication: %s %s", med["name"], med.get("dose", ""))
            return True
        log.warning("    ⚠️  Medication %s — HTTP %d: %s", med["name"], resp.status_code, resp.text[:100])
        return False
    except Exception as exc:
        log.warning("    ❌ Medication %s error: %s", med["name"], exc)
        return False


async def _seed_insurance(
    client: OpenEMRClient,
    puuid: str,
    ins: dict[str, str],
    dry_run: bool,
) -> bool:
    """POST primary insurance record for a patient."""
    payload = {
        "type":          "primary",
        "provider":      ins["provider"],
        "plan_name":     ins["plan_name"],
        "policy_number": ins["policy_number"],
        "group_number":  ins["group_number"],
        "date":          "2024-01-01",
    }
    if dry_run:
        log.info("    [DRY-RUN] POST insurance: %s — %s", ins["provider"], ins["plan_name"])
        return True
    try:
        token = await client._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        resp = await client._http.post(
            f"{client.base_url}/apis/default/api/patient/{puuid}/insurance/primary",
            headers=headers,
            json=payload,
        )
        if resp.status_code in range(200, 300):
            log.info("    ✅ Insurance: %s — %s (%s)", ins["provider"], ins["plan_name"], ins["policy_number"])
            return True
        log.warning("    ⚠️  Insurance — HTTP %d: %s", resp.status_code, resp.text[:120])
        return False
    except Exception as exc:
        log.warning("    ❌ Insurance error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main seed function
# ---------------------------------------------------------------------------

async def seed(args: argparse.Namespace) -> None:
    """Seed all mock data into OpenEMR for each patient."""
    # Load source files
    with open(_PATIENTS_JSON) as f:
        patients: list[dict] = json.load(f)["patients"]
    with open(_MEDICATIONS_JSON) as f:
        meds_by_id: dict[str, list[dict]] = json.load(f)["medications"]

    # Merge supplemental patients
    known_ids = {p["id"] for p in patients}
    for sp in _SUPPLEMENTAL:
        if sp["id"] not in known_ids:
            patients.append(sp)
    meds_by_id.update(_SUPPLEMENTAL_MEDS)

    # Apply --filter if provided
    filter_ids: set[str] = set()
    if args.filter:
        filter_ids = {x.strip().upper() for x in args.filter.split(",")}

    totals = {"allergy": 0, "condition": 0, "medication": 0, "insurance": 0, "skip": 0}

    async with OpenEMRClient() as client:
        for p in patients:
            pid   = p["id"]
            name  = p["name"]
            allergies  = p.get("allergies", [])
            conditions = p.get("conditions", [])
            meds       = meds_by_id.get(pid, [])
            ins_data   = _INSURANCE.get(pid)

            if filter_ids and pid not in filter_ids:
                totals["skip"] += 1
                continue

            log.info("━━━ %s (%s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", pid, name)

            # 1. Look up FHIR UUID
            puuid = await _find_patient_uuid(client, name)
            if not puuid:
                log.warning("  ⚠️  Patient not found in OpenEMR — skipping. Run seed_portal.py first.")
                totals["skip"] += 1
                continue
            log.info("  Found: %s → FHIR UUID %s", name, puuid)

            # 2. Allergies (no encounter needed)
            if allergies:
                log.info("  Seeding %d allerg%s…", len(allergies), "y" if len(allergies) == 1 else "ies")
                for allergy in allergies:
                    ok = await _seed_allergy(client, puuid, allergy, args.dry_run)
                    if ok:
                        totals["allergy"] += 1

            # 3. Create one encounter to anchor conditions + medications
            eid = ""
            if conditions or meds:
                log.info("  Creating seed encounter…")
                eid = await _create_encounter_for_seeding(client, puuid, args.dry_run)
                if not eid and not args.dry_run:
                    log.warning("  ⚠️  No encounter — skipping conditions and medications.")

            # 4. Medical problems
            if conditions and eid:
                log.info("  Seeding %d condition%s…", len(conditions), "" if len(conditions) == 1 else "s")
                for cond in conditions:
                    ok = await _seed_medical_problem(client, puuid, eid, cond, args.dry_run)
                    if ok:
                        totals["condition"] += 1

            # 5. Medications
            if meds and eid:
                log.info("  Seeding %d medication%s…", len(meds), "" if len(meds) == 1 else "s")
                for med in meds:
                    ok = await _seed_medication(client, puuid, eid, med, args.dry_run)
                    if ok:
                        totals["medication"] += 1

            # 6. Insurance
            if ins_data:
                log.info("  Seeding insurance…")
                ok = await _seed_insurance(client, puuid, ins_data, args.dry_run)
                if ok:
                    totals["insurance"] += 1

    # Summary
    mode = "DRY-RUN" if args.dry_run else "COMPLETE"
    log.info("")
    log.info("━━━ SEED %s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", mode)
    log.info("  Allergies seeded:    %d", totals["allergy"])
    log.info("  Conditions seeded:   %d", totals["condition"])
    log.info("  Medications seeded:  %d", totals["medication"])
    log.info("  Insurance seeded:    %d", totals["insurance"])
    log.info("  Patients skipped:    %d", totals["skip"])
    log.info("")
    log.info("  OpenEMR portal → http://localhost:8300")
    log.info("  Login: admin / pass")
    log.info("  Find Patient → [any name] → Chart → Allergies / Encounters / Insurance")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed allergies, conditions, medications, and insurance into OpenEMR.")
    parser.add_argument("--dry-run",   action="store_true", help="Preview without writing.")
    parser.add_argument("--filter",    type=str, default="", help="Comma-separated patient IDs to seed (e.g. P001,P012).")
    parser.add_argument("--base-url",  type=str, default="", help="Override OpenEMR base URL.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.base_url:
        os.environ["OPENEMR_BASE_URL"] = args.base_url
    asyncio.run(seed(args))
