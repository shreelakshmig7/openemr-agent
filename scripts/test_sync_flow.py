#!/usr/bin/env python3
"""
test_sync_flow.py
-----------------
AgentForge — Healthcare RCM AI Agent — Evidence Staging Sync Verification
--------------------------------------------------------------------------
End-to-end integration test that exercises the full staging → sync cycle:

  Step 1  INSERT  — stage a clinical marker in evidence_staging
                    (sync_status = 'PENDING')
  Step 2  VERIFY  — confirm the row is PENDING before any network call
  Step 3  POST    — POST an Encounter to OpenEMR via OpenEMRClient
                    (OpenEMR's FHIR R4 layer is read-only for Observation;
                     the standard REST API encounter endpoint is used as the
                     sync anchor — this proves a live round-trip to OpenEMR)
  Step 4  UPDATE  — call update_sync_status() → 'SYNCED' with the encounter UUID
  Step 5  ASSERT  — re-read the row and assert sync_status == 'SYNCED'
                    and fhir_observation_id is populated

On failure at any step the row is updated to 'FAILED' so the DB reflects
the true state, and the script exits with code 1.

Target patient: John Smith (P001) — seeded by scripts/seed_portal.py.

Usage:
    cd openemr-agent
    python scripts/test_sync_flow.py [--base-url https://localhost:9300]

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

# ── Path bootstrap ────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)

import database as db                                        # noqa: E402
from openemr_client import OpenEMRClient, OpenEMRAPIError   # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_sync_flow")

# ── Test constants ────────────────────────────────────────────────────────────
_TEST_SESSION  = "test-sync-flow-session"
_TEST_PATIENT  = "P001"          # John Smith — seeded by seed_portal.py
_TEST_MARKER   = "HER2"
_TEST_VALUE    = "positive"
_TEST_RAW_TEXT = "HER2 positive, IHC 3+ — per pathology report page 2"
_TEST_SOURCE   = "AgentForge_Test_ClinicalNote.pdf"
_TEST_PAGE     = 2

_SEPARATOR = "─" * 68


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_row(label: str, row: dict[str, Any]) -> None:
    log.info("%s", label)
    for key in ("id", "marker_name", "marker_value", "sync_status",
                "fhir_observation_id", "patient_id", "session_id",
                "created_at", "updated_at"):
        log.info("    %-22s %s", f"{key}:", row.get(key, "<absent>"))


def _fail(row_id: int | None, msg: str) -> None:
    """Mark the DB row as FAILED, log the error, and exit with code 1."""
    log.error("FAIL  %s", msg)
    if row_id is not None:
        try:
            db.update_sync_status(row_id, db.SYNC_STATUS_FAILED)
            log.info("DB row %d marked FAILED.", row_id)
        except Exception as exc:
            log.warning("Could not mark row FAILED: %s", exc)
    log.info(_SEPARATOR)
    sys.exit(1)


# ── Core test coroutine ───────────────────────────────────────────────────────

async def run_test(base_url: str) -> None:
    log.info(_SEPARATOR)
    log.info("AgentForge — Evidence Staging Sync Verification")
    log.info("Target: %s", base_url)
    log.info(_SEPARATOR)

    db.init_db()
    row_id: int | None = None

    # ── Step 1: INSERT marker with sync_status='PENDING' ─────────────────────
    log.info("Step 1 / 5  INSERT clinical marker into evidence_staging …")
    try:
        row_id = db.insert_clinical_marker(
            marker_name=_TEST_MARKER,
            raw_text=_TEST_RAW_TEXT,
            session_id=_TEST_SESSION,
            patient_id=_TEST_PATIENT,
            marker_value=_TEST_VALUE,
            source_file=_TEST_SOURCE,
            page_number=_TEST_PAGE,
            element_type="NarrativeText",
            confidence=1.0,
        )
        log.info("    Inserted row_id=%d", row_id)
    except Exception as exc:
        _fail(None, f"INSERT failed: {exc}")

    # ── Step 2: Verify row is PENDING before any network call ─────────────────
    log.info("Step 2 / 5  Verify sync_status = 'PENDING' …")
    rows_before = db.get_pending_markers(session_id=_TEST_SESSION)
    target_before = next((r for r in rows_before if r["id"] == row_id), None)
    if target_before is None:
        _fail(row_id, f"Row {row_id} not found in PENDING results.")
    if target_before["sync_status"] != "PENDING":
        _fail(row_id, f"Expected PENDING, got '{target_before['sync_status']}'")
    _print_row("    Before sync:", target_before)
    log.info("    ✓  sync_status = 'PENDING'")

    # ── Step 3: POST Encounter to OpenEMR (sync anchor) ───────────────────────
    #
    # OpenEMR's FHIR R4 layer exposes Observation as read-only (GET + search
    # only; no POST route is registered in _rest_routes_fhir_r4_us_core_3_1_0).
    # We use the standard REST API encounter endpoint instead — this proves a
    # live authenticated round-trip and provides a server-issued UUID that we
    # can store as the sync reference in evidence_staging.fhir_observation_id.
    log.info("Step 3 / 5  POST Encounter to OpenEMR (sync anchor) …")
    sync_ref: str | None = None
    fhir_patient_id: str = ""
    async with OpenEMRClient(base_url=base_url) as client:

        # Resolve John Smith's FHIR Patient UUID dynamically.
        try:
            bundle = await client.get_patients(family="Smith", given="John")
            entries = bundle.get("entry") or []
            if not entries:
                _fail(row_id, "Patient 'John Smith' not found in OpenEMR. "
                               "Run scripts/seed_portal.py first.")
            fhir_patient_id = str(entries[0]["resource"]["id"])
            log.info("    Resolved FHIR Patient id = %s", fhir_patient_id)
        except OpenEMRAPIError as exc:
            _fail(row_id, f"Patient lookup failed: {exc}")

        # POST the encounter — real HTTP write, proves the sync path works.
        try:
            enc = await client.post_encounter(
                patient_uuid=fhir_patient_id,
                reason=f"AgentForge evidence sync — HER2={_TEST_VALUE} (staging row {row_id})",
            )
            enc_uuid = str(enc.get("id") or enc.get("uuid") or enc.get("encounter") or "")
            if not enc_uuid:
                _fail(row_id, f"POST encounter succeeded but no id returned: {enc}")
            sync_ref = f"Encounter/{enc_uuid}"
            log.info("    ✓  Created %s", sync_ref)
        except OpenEMRAPIError as exc:
            _fail(row_id, f"POST /encounter failed — HTTP {exc.status_code}: {exc.body[:200]}")
        except Exception as exc:
            _fail(row_id, f"POST /encounter raised: {exc}")

    # ── Step 4: Update DB → SYNCED ────────────────────────────────────────────
    log.info("Step 4 / 5  UPDATE evidence_staging row → 'SYNCED' …")
    try:
        db.update_sync_status(
            row_id=row_id,
            status=db.SYNC_STATUS_SYNCED,
            fhir_observation_id=sync_ref,   # stores Encounter UUID as sync anchor
        )
        log.info("    update_sync_status(row_id=%d, SYNCED, sync_ref=%s).", row_id, sync_ref)
    except Exception as exc:
        _fail(row_id, f"update_sync_status failed: {exc}")

    # ── Step 5: Assert final DB state ─────────────────────────────────────────
    log.info("Step 5 / 5  ASSERT final state in evidence_staging …")
    rows_after = db.get_markers_by_session(session_id=_TEST_SESSION)
    target_after = next((r for r in rows_after if r["id"] == row_id), None)
    if target_after is None:
        _fail(row_id, f"Row {row_id} missing after update.")

    _print_row("    After sync:", target_after)

    # Assertions
    failures: list[str] = []
    if target_after["sync_status"] != "SYNCED":
        failures.append(
            f"sync_status = '{target_after['sync_status']}' (expected 'SYNCED')"
        )
    if target_after["fhir_observation_id"] != sync_ref:
        failures.append(
            f"fhir_observation_id = '{target_after['fhir_observation_id']}' "
            f"(expected '{sync_ref}')"
        )
    if target_after["updated_at"] <= target_after["created_at"]:
        failures.append("updated_at not advanced past created_at")

    if failures:
        for f in failures:
            log.error("  ASSERTION FAILED: %s", f)
        _fail(row_id, "One or more assertions failed (see above).")

    # ── Result ────────────────────────────────────────────────────────────────
    log.info(_SEPARATOR)
    log.info("PASS  All 5 steps completed successfully.")
    log.info("")
    log.info("  evidence_staging row %d:", row_id)
    log.info("    sync_status         = %s", target_after["sync_status"])
    log.info("    fhir_observation_id = %s", target_after["fhir_observation_id"])
    log.info("    marker_name         = %s  (%s)", target_after["marker_name"], target_after["marker_value"])
    log.info("    patient_id          = %s  → FHIR Patient/%s", _TEST_PATIENT, fhir_patient_id)
    log.info(_SEPARATOR)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that evidence_staging sync_status transitions "
                    "PENDING → SYNCED after a successful FHIR Observation POST.",
    )
    parser.add_argument(
        "--base-url",
        default="https://localhost:9300",
        metavar="URL",
        help="OpenEMR HTTPS base URL (default: https://localhost:9300)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run_test(base_url=args.base_url))
