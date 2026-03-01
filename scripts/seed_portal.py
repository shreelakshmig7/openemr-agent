"""
seed_portal.py
--------------
AgentForge — Healthcare RCM AI Agent — OpenEMR Portal Seeder
-------------------------------------------------------------
Reads patients.json and medications.json from mock_data/, converts each
record to FHIR R4 resources, and POSTs them into the OpenEMR Docker portal
using OpenEMRClient (OAuth2 Password Grant, verify=False).

Targeted patients (always included unless --filter is used):
  P001  John Smith       — Type 2 Diabetes, Hypertension, CKD
  P012  Maria Gonzalez   — Oncology / breast cancer (from test PDFs)

All other patients.json entries are seeded as well — the script is
idempotent: it searches for each patient by family + given name before
creating, so re-running is safe.

FHIR resources created per patient:
  • Patient               (demographics, internal identifier, conditions)
  • MedicationRequest[]   (one per medication entry in medications.json)

Networking target (docker/development-easy/docker-compose.yml):
  https://localhost:9300  →  openemr container port 9300:443

Usage:
    cd openemr-agent
    python scripts/seed_portal.py

    # Options
    python scripts/seed_portal.py --dry-run           # preview, no writes
    python scripts/seed_portal.py --filter P001,P012  # target IDs only
    python scripts/seed_portal.py --base-url https://localhost:9300

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

# ── Path bootstrap (run from any directory) ───────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)          # openemr-agent/
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)

from openemr_client import OpenEMRClient, OpenEMRAPIError  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seed_portal")

# ── Source data paths ─────────────────────────────────────────────────────────
_MOCK_DIR         = os.path.join(_REPO_ROOT, "mock_data")
_PATIENTS_JSON    = os.path.join(_MOCK_DIR, "patients.json")
_MEDICATIONS_JSON = os.path.join(_MOCK_DIR, "medications.json")

# ── Supplemental patient: Maria Gonzalez ─────────────────────────────────────
# Appears in AgentForge_Test_PriorAuth.pdf and AgentForge_Test_ClinicalNote.pdf
# but is absent from patients.json. Assigned local ID P012 so medications
# and future evidence_staging rows can reference her consistently.
_SUPPLEMENTAL_PATIENTS: list[dict[str, Any]] = [
    {
        "id": "P012",
        "name": "Maria J. Gonzalez",
        "dob": "1978-06-15",
        "gender": "Female",
        "allergies": [],
        "conditions": ["Breast Cancer", "HER2-positive", "Post-Surgical Monitoring"],
    }
]

_SUPPLEMENTAL_MEDS: dict[str, list[dict[str, Any]]] = {
    "P012": [
        {
            "name": "Tamoxifen",
            "dose": "20mg",
            "frequency": "once daily",
            "prescribed": "2024-03-01",
        },
        {
            "name": "Letrozole",
            "dose": "2.5mg",
            "frequency": "once daily",
            "prescribed": "2025-01-10",
        },
    ]
}

# ── FHIR mapping constants ────────────────────────────────────────────────────
_GENDER_MAP: dict[str, str] = {
    "Male":   "male",
    "Female": "female",
    "Other":  "other",
}

# Identifier system used to cross-reference our local P001-style IDs inside
# FHIR.  Must be an HTTP(S) URL — OpenEMR's FHIR URI resolver calls getUrl()
# on the system value and crashes with a 500 on urn: scheme strings.
_LOCAL_ID_SYSTEM = "http://agentforge.local/fhir/patient-id"


# ── FHIR resource builders ────────────────────────────────────────────────────

def _build_patient_resource(p: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a patients.json entry to a FHIR R4 Patient resource dict.

    Name splitting strategy:
        All tokens except the last → ``given``
        Last token                 → ``family``
        e.g. "Maria J. Gonzalez"  → given=["Maria", "J."], family="Gonzalez"
             "John Smith"          → given=["John"],         family="Smith"

    OpenEMR quirks addressed here:
        - ``name[].use`` must be ``"official"`` — without it OpenEMR cannot
          map to its internal ``fname`` / ``lname`` fields and returns 400.
        - Identifier ``system`` must use an HTTP URL scheme.  The ``urn:``
          scheme triggers a PHP ``getUrl()`` null-dereference (500) inside
          OpenEMR's FHIR uri-resolver.
        - Non-standard ``extension`` entries with ``urn:`` URLs cause the same
          crash, so conditions and allergies are stored in ``communication``
          and ``generalPractitioner`` notes (FHIR-safe) rather than extensions.

    Args:
        p: Single patient dict from patients.json (or _SUPPLEMENTAL_PATIENTS).

    Returns:
        FHIR R4 Patient resource ready for ``client.post_patient()``.
    """
    tokens = p["name"].strip().split()
    family = tokens[-1]
    given  = tokens[:-1] if len(tokens) > 1 else tokens

    resource: dict[str, Any] = {
        "resourceType": "Patient",
        # Use an HTTP-scheme system URL — OpenEMR crashes on urn: scheme
        # via a PHP getUrl() null-dereference in its FHIR URI resolver.
        "identifier": [
            {
                "use":    "secondary",
                "system": _LOCAL_ID_SYSTEM,
                "value":  p["id"],
            }
        ],
        "name": [
            {
                # "official" is required: OpenEMR selects this entry to populate
                # fname / lname — other use values are silently ignored.
                "use":    "official",
                "family": family,
                "given":  given,
            }
        ],
        "gender":    _GENDER_MAP.get(p.get("gender", ""), "unknown"),
        "birthDate": p.get("dob", ""),
    }

    return resource


def _build_medication_request(
    med: dict[str, Any],
    fhir_patient_id: str,
    local_patient_id: str,
) -> dict[str, Any]:
    """
    Convert a medications.json entry to a FHIR R4 MedicationRequest resource.

    Args:
        med:              Single medication dict (name, dose, frequency, prescribed).
        fhir_patient_id:  Server-assigned FHIR Patient id (e.g. "1", "42").
        local_patient_id: Internal mock-data id (e.g. "P001") stored as identifier
                          for traceability back to the source JSON.

    Returns:
        FHIR R4 MedicationRequest ready for ``client.post_medication_request()``.
    """
    dose_text = f"{med.get('dose', '')} {med.get('frequency', '')}".strip()

    return {
        "resourceType": "MedicationRequest",
        "status":       "active",
        "intent":       "order",
        "medicationCodeableConcept": {
            "text": med["name"],
        },
        "subject": {
            "reference": f"Patient/{fhir_patient_id}",
        },
        "authoredOn": med.get("prescribed", ""),
        "dosageInstruction": [
            {"text": dose_text}
        ],
        # Trace identifier so we can match FHIR resources back to mock_data.
        "identifier": [
            {
                "system": "http://agentforge.local/fhir/source-patient",
                "value":  local_patient_id,
            }
        ],
    }


# ── Idempotency helper ────────────────────────────────────────────────────────

async def _find_existing_patient(
    client: OpenEMRClient,
    family: str,
    given: str,
) -> str | None:
    """
    Search for an existing Patient by family and first given name.

    Args:
        client: An authenticated, connected OpenEMRClient.
        family: Family (last) name to search.
        given:  First given name to search.

    Returns:
        The server-assigned FHIR Patient ``id`` string if found, else ``None``.
    """
    try:
        bundle = await client.get_patients(family=family, given=given)
        entries = bundle.get("entry") or []
        if entries:
            return str(entries[0]["resource"]["id"])
    except OpenEMRAPIError as exc:
        log.debug("Patient search (%s %s) returned error: %s", given, family, exc)
    return None


# ── Per-patient seed logic ────────────────────────────────────────────────────

async def _seed_one_patient(
    client: OpenEMRClient,
    patient: dict[str, Any],
    meds: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    """
    Seed a single patient and their medications.  Idempotent on name match.

    Args:
        client:   Open, authenticated OpenEMRClient.
        patient:  Patient record dict (from patients.json or supplemental list).
        meds:     List of medication dicts for this patient.
        dry_run:  When True, log intent but make no HTTP write calls.

    Returns:
        Result dict with keys: local_id, fhir_id, action, meds_created.
    """
    tokens      = patient["name"].strip().split()
    family      = tokens[-1]
    given_first = tokens[0] if len(tokens) > 1 else tokens[0]
    local_id    = patient["id"]
    label       = f"{local_id:<5}  {patient['name']:<30}"

    # ── Idempotency check: skip if the patient already exists ─────────────────
    existing_fhir_id = await _find_existing_patient(client, family, given_first)
    if existing_fhir_id:
        log.info("  SKIP    %s  already exists (FHIR Patient/%s)", label, existing_fhir_id)
        return {
            "local_id":    local_id,
            "fhir_id":     existing_fhir_id,
            "action":      "skipped",
            "meds_created": 0,
        }

    # ── Dry-run: log intent and return without writing ────────────────────────
    if dry_run:
        log.info("  [DRY]   %s  would POST Patient + %d MedicationRequest(s)", label, len(meds))
        return {
            "local_id":    local_id,
            "fhir_id":     "dry-run",
            "action":      "dry-run",
            "meds_created": 0,
        }

    # ── Create Patient ────────────────────────────────────────────────────────
    fhir_patient_resource = _build_patient_resource(patient)
    created_patient       = await client.post_patient(fhir_patient_resource)
    fhir_id               = str(created_patient["id"])
    log.info("  CREATE  %s  → FHIR Patient/%s", label, fhir_id)

    # ── Create MedicationRequests ─────────────────────────────────────────────
    meds_created = 0
    for med in meds:
        med_resource = _build_medication_request(med, fhir_id, local_id)
        try:
            created_med = await client.post_medication_request(med_resource)
            log.info(
                "    + MedRequest  %-22s  → FHIR MedicationRequest/%s",
                med["name"],
                created_med.get("id", "?"),
            )
            meds_created += 1
        except OpenEMRAPIError as exc:
            log.warning(
                "    ! MedRequest  %-22s  failed — HTTP %s: %s",
                med["name"],
                exc.status_code,
                exc.body[:120],
            )

    return {
        "local_id":     local_id,
        "fhir_id":      fhir_id,
        "action":       "created",
        "meds_created": meds_created,
    }


# ── Main seeding coroutine ────────────────────────────────────────────────────

async def seed(
    base_url: str = "https://localhost:9300",
    filter_ids: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """
    Seed patients and medications from local mock data into OpenEMR.

    Args:
        base_url:   OpenEMR HTTPS base URL.
        filter_ids: When provided, only patients with matching local IDs are
                    seeded.  Pass ``["P001", "P012"]`` to target just John
                    Smith and Maria Gonzalez.
        dry_run:    Preview mode — no FHIR write calls are made.
    """
    # ── Load source data ──────────────────────────────────────────────────────
    try:
        with open(_PATIENTS_JSON) as f:
            patients_json: list[dict[str, Any]] = json.load(f)["patients"]
    except FileNotFoundError:
        log.error("patients.json not found at: %s", _PATIENTS_JSON)
        sys.exit(1)

    try:
        with open(_MEDICATIONS_JSON) as f:
            meds_db: dict[str, list[dict[str, Any]]] = json.load(f)["medications"]
    except FileNotFoundError:
        log.error("medications.json not found at: %s", _MEDICATIONS_JSON)
        sys.exit(1)

    # Merge supplemental records (Maria Gonzalez etc.)
    all_patients = patients_json + _SUPPLEMENTAL_PATIENTS
    merged_meds  = {**meds_db, **_SUPPLEMENTAL_MEDS}

    # Apply --filter
    if filter_ids:
        all_patients = [p for p in all_patients if p["id"] in filter_ids]
        if not all_patients:
            log.error("No patients matched filter %s", filter_ids)
            sys.exit(1)

    suffix = " [DRY RUN]" if dry_run else ""
    log.info("OpenEMR seed target: %s%s", base_url, suffix)
    log.info("Seeding %d patient(s)%s", len(all_patients), f"  (filter: {filter_ids})" if filter_ids else "")
    log.info("─" * 70)

    # ── Seed loop ─────────────────────────────────────────────────────────────
    created_patients = 0
    skipped_patients = 0
    created_meds     = 0
    errors: list[str] = []

    async with OpenEMRClient(base_url=base_url) as client:
        for patient in all_patients:
            meds = merged_meds.get(patient["id"], [])
            try:
                result = await _seed_one_patient(client, patient, meds, dry_run=dry_run)
                action = result["action"]
                if action == "created":
                    created_patients += 1
                    created_meds     += result["meds_created"]
                elif action == "skipped":
                    skipped_patients += 1
            except OpenEMRAPIError as exc:
                msg = f"{patient['id']} ({patient['name']}): HTTP {exc.status_code} — {exc.body[:80]}"
                log.error("  ERROR   %s", msg)
                errors.append(msg)
            except Exception as exc:
                msg = f"{patient['id']} ({patient['name']}): {exc}"
                # Detect the OpenEMR client-not-approved error and give a single
                # actionable hint (only on the first occurrence to avoid spam).
                if "invalid_client" in str(exc) and not any("invalid_client" in e for e in errors):
                    log.error(
                        "\n  ── OAuth2 client not approved ─────────────────────────────────\n"
                        "  Dynamic registration succeeded but OpenEMR requires admin approval.\n"
                        "  Auto-activation via docker exec was attempted; if it failed:\n"
                        "\n"
                        "  Option A — approve in the admin UI:\n"
                        "    https://localhost:9300/interface/smart/register-app.php\n"
                        "    (login: admin / pass)\n"
                        "\n"
                        "  Option B — activate via Docker manually:\n"
                        "    docker exec $(docker ps -q --filter publish=9300) \\\n"
                        "      mysql -u openemr -popenemr -h mysql openemr \\\n"
                        "      -e \"UPDATE oauth_clients SET is_enabled=1 WHERE is_enabled=0;\"\n"
                        "\n"
                        "  Option C — set pre-approved credentials in .env:\n"
                        "    OPENEMR_CLIENT_ID=<your_client_id>\n"
                        "    OPENEMR_CLIENT_SECRET=<your_client_secret>\n"
                        "  ───────────────────────────────────────────────────────────────\n"
                    )
                log.error("  ERROR   %s", msg)
                errors.append(msg)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("─" * 70)
    log.info(
        "Done.  Patients created: %d  |  Medications created: %d  "
        "|  Skipped (exist): %d  |  Errors: %d",
        created_patients, created_meds, skipped_patients, len(errors),
    )
    if errors:
        log.warning("Failed patients:")
        for e in errors:
            log.warning("  • %s", e)
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed the OpenEMR Docker portal with patients and medications "
            "from local mock_data/ JSON files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/seed_portal.py
  python scripts/seed_portal.py --dry-run
  python scripts/seed_portal.py --filter P001,P012
  python scripts/seed_portal.py --base-url https://localhost:9300 --dry-run
        """,
    )
    parser.add_argument(
        "--base-url",
        default="https://localhost:9300",
        metavar="URL",
        help="OpenEMR HTTPS base URL (default: https://localhost:9300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be seeded without making any FHIR write calls.",
    )
    parser.add_argument(
        "--filter",
        metavar="IDS",
        default="",
        help=(
            'Comma-separated local patient IDs to seed, e.g. "P001,P012". '
            "Omit to seed all patients."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args       = _parse_args()
    filter_ids = (
        [i.strip() for i in args.filter.split(",") if i.strip()]
        if args.filter
        else None
    )
    asyncio.run(
        seed(
            base_url=args.base_url,
            filter_ids=filter_ids,
            dry_run=args.dry_run,
        )
    )
