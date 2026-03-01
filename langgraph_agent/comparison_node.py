"""
comparison_node.py
------------------
AgentForge — Healthcare RCM AI Agent — PDF Comparison Node
-----------------------------------------------------------
Runs immediately after the Extractor Node whenever a PDF has just been
processed.  Compares the newly staged markers in ``evidence_staging``
against previously SYNCED rows to determine what is genuinely new versus
what already exists in the portal, then sets the HITL (Human-in-the-Loop)
sync confirmation flag so the workflow pauses for user approval.

Pipeline position:
    extractor → comparison_node → output
                     ↓  (sets pending_sync_confirmation = True)
    [user says "yes/sync"]
    orchestrator (detects confirmation) → sync_execution → output

Deduplication strategy
----------------------
Within the current session's PENDING rows:
  - Group by (LOINC code, normalised value) — same logic as sync_node.
  - Champion = row with the longest raw_text (richest evidence context).
  - All other rows in the group are absorbed duplicates.

"Already in portal" check
--------------------------
Uses evidence_staging (not OpenEMR FHIR) as the source of truth because:
  - OpenEMR's FHIR layer returns 404 on POST /Observation (write not
    supported in this build), so no rows are ever truly SYNCED via the
    API yet.
  - The evidence_staging table IS the authoritative audit trail for this
    demo — a SYNCED row means the agent previously confirmed that marker
    to the EHR.

Once OpenEMR supports FHIR writes, this check can be upgraded to:
    GET /fhir/Observation?patient=<uuid>&code=<loinc>

Safety guards
-------------
- If pending_sync_confirmation is already True (left from a prior turn)
  AND the current PDF belongs to a different patient, the stale flag is
  cleared before setting a new one.  This prevents Maria's staging data
  from being accidentally synced against a different patient.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from __future__ import annotations

import logging
import sys
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Path bootstrap so this module can import siblings whether run directly or
# imported as part of the langgraph_agent package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _AGENT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import database as _db
import fhir_mapper as _fhir_mapper
from langgraph_agent.state import AgentState

logger = logging.getLogger(__name__)

# Words the user can say to confirm the sync (checked in orchestrator_node).
SYNC_CONFIRM_WORDS = {"yes", "sync", "proceed", "do it", "confirm", "go ahead", "push"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_value(value: str) -> str:
    return value.strip().lower()


def _group_pending_by_champion(
    pending_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Deduplicate pending rows by (LOINC code, normalised value).

    Returns:
        Tuple of:
          * champions       — one row per unique (marker, value) pair,
                              with richest raw_text selected.
          * duplicate_count — number of rows absorbed into champions.
    """
    valid = [
        r for r in pending_rows
        if str(r.get("marker_value") or "").strip()           # skip empty values
        and _fhir_mapper.get_loinc_code(r.get("marker_name", ""))  # skip unknown markers
    ]

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in valid:
        loinc     = _fhir_mapper.get_loinc_code(row["marker_name"]) or row["marker_name"]
        value_key = _normalise_value(str(row.get("marker_value", "")))
        groups[f"{loinc}|{value_key}"].append(row)

    champions: List[Dict[str, Any]] = []
    for group in groups.values():
        best = max(group, key=lambda r: len(str(r.get("raw_text", ""))))
        champions.append(best)

    duplicate_count = len(valid) - len(champions)
    return champions, duplicate_count


def _check_already_synced(
    champions: List[Dict[str, Any]],
    patient_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split champions into 'new' (never SYNCED) and 'existing' (previously SYNCED).

    Checks evidence_staging for any SYNCED row with the same
    (patient_id, marker_name, normalised_value) across all sessions.

    Args:
        champions:  Deduplicated champion rows from the current session.
        patient_id: patient_id stored in evidence_staging rows.

    Returns:
        Tuple[new_discoveries, already_in_portal]
    """
    try:
        all_synced = _db.get_synced_markers(patient_id=patient_id)
    except Exception as exc:
        logger.warning("comparison_node: could not read SYNCED markers — %s", exc)
        # Fail open: treat all champions as new if DB read fails.
        return champions, []

    synced_keys: set = set()
    for row in all_synced:
        loinc     = _fhir_mapper.get_loinc_code(row.get("marker_name", "")) or row.get("marker_name", "")
        value_key = _normalise_value(str(row.get("marker_value", "")))
        synced_keys.add(f"{loinc}|{value_key}")

    new_discoveries:     List[Dict[str, Any]] = []
    already_in_portal:   List[Dict[str, Any]] = []

    for champion in champions:
        loinc     = _fhir_mapper.get_loinc_code(champion["marker_name"]) or champion["marker_name"]
        value_key = _normalise_value(str(champion.get("marker_value", "")))
        if f"{loinc}|{value_key}" in synced_keys:
            already_in_portal.append(champion)
        else:
            new_discoveries.append(champion)

    return new_discoveries, already_in_portal


def _format_marker_label(row: Dict[str, Any]) -> str:
    """Return a compact display label: 'ER positive' → '**ER+**', etc."""
    name  = row.get("marker_name", "?")
    value = str(row.get("marker_value", "")).strip().lower()

    # Short-form value abbreviations for display
    _ABBREV = {
        "positive": "+", "negative": "-",
        "equivocal": "±", "detected": "+",
        "not detected": "-",
    }
    short_val = _ABBREV.get(value, value)

    # Canonical short name (strip " Status" suffix for display)
    short_name = name.replace(" Status", "").replace(" status", "")
    return f"**{short_name}{short_val}**"


def _build_safety_alert_block(
    allergy_conflict: Dict[str, Any],
    denial_risk: Dict[str, Any],
) -> str:
    """
    Build a high-severity alert block to prepend to the sync prompt.

    Re-states Phase 2 findings (allergy conflicts and HIGH denial risk) so the
    human auditor sees them immediately before being offered the sync gate.
    Returns an empty string when no high-severity alerts were raised.

    Phase 4 gate rule: per the strict tool protocol, this block is surfaced
    unconditionally — the user must acknowledge the risks before confirming.

    Args:
        allergy_conflict: Result of tool_check_drug_interactions — keys:
                          conflict (bool), drug, allergy, conflict_type, severity.
        denial_risk:      Result of tool_analyze_denial_risk — keys:
                          risk_level, matched_patterns, recommendations,
                          denial_risk_score, missing_documentation.

    Returns:
        str: Formatted alert block (may be empty).
    """
    alerts: List[str] = []

    # ── Allergy / Drug-Interaction alert ─────────────────────────────────
    if allergy_conflict and allergy_conflict.get("conflict"):
        drug     = allergy_conflict.get("drug", "the requested drug")
        allergy  = allergy_conflict.get("allergy", "a known allergen")
        severity = (allergy_conflict.get("severity") or "HIGH").upper()
        ctype    = allergy_conflict.get("conflict_type", "conflict")
        alerts.append(
            f"🚨 **HIGH-SEVERITY SAFETY ALERT — {severity} {ctype.upper()}**\n"
            f"   The PDF requests **{drug}**, which conflicts with the patient's "
            f"documented allergy to **{allergy}**. "
            f"**This must be resolved with the prescribing physician before syncing.**"
        )

    # ── Denial risk alert ─────────────────────────────────────────────────
    risk_level = (denial_risk or {}).get("risk_level", "").upper()
    if risk_level in ("HIGH", "CRITICAL"):
        score    = (denial_risk or {}).get("denial_risk_score", 0.0)
        missing  = (denial_risk or {}).get("missing_documentation", [])
        missing_str = (
            ", ".join(f"**{m}**" for m in missing) if missing
            else "see recommendations above"
        )
        alerts.append(
            f"⚠️ **{risk_level} DENIAL RISK** (score: {score:.0%})\n"
            f"   Missing documentation: {missing_str}.\n"
            f"   **Resolve these gaps before submitting to the payer.**"
        )

    if not alerts:
        return ""

    header = "---\n### ⚠️ Phase 2 Safety & Compliance Alerts\n"
    footer = "\n---"
    return header + "\n\n".join(alerts) + footer


def _build_sync_prompt(
    new: List[Dict[str, Any]],
    existing: List[Dict[str, Any]],
    total_raw: int,
    duplicate_count: int,
    allergy_conflict: Optional[Dict[str, Any]] = None,
    denial_risk: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the Phase 3 reconciliation + Phase 4 HITL gate message.

    Exact output format (per 4-phase protocol):
        "I found [X] clinical markers in this PDF.
         [Y] are already recorded in the OpenEMR portal.
         [Z] are new unique discoveries: ER+, HER2-, PR+.
         Would you like me to sync the new discoveries to OpenEMR?"

    Phase 4 WARNING PROTOCOL: if Phase 2 raised a HIGH-SEVERITY alert
    (allergy conflict or CRITICAL denial risk), re-state the warning in
    bold before the sync confirmation question, per the strict gate rule.

    Args:
        new:              Champion rows not yet in the portal.
        existing:         Champion rows already SYNCED in a prior session.
        total_raw:        Total PENDING rows before deduplication.
        duplicate_count:  Rows absorbed into champions (shown for transparency).
        allergy_conflict: Result from tool_check_drug_interactions (may be None).
        denial_risk:      Result from tool_analyze_denial_risk (may be None).

    Returns:
        str: Full formatted Phase 3+4 reconciliation and sync gate message.
    """
    total_unique = len(new) + len(existing)
    new_labels   = ", ".join(_format_marker_label(r) for r in new)    if new      else "none"
    exist_labels = ", ".join(_format_marker_label(r) for r in existing) if existing else "none"

    lines: List[str] = []

    lines.append("---")
    lines.append("### 🔬 Phase 3 — Clinical Reconciliation")
    lines.append("")

    # ── Core reconciliation statement (exact required format) ─────────────
    lines.append(
        f"I found **{total_raw} clinical marker occurrences** in this PDF "
        f"({duplicate_count} duplicate{'s' if duplicate_count != 1 else ''} de-duplicated "
        f"→ **{total_unique} unique finding{'s' if total_unique != 1 else ''}**)."
    )
    lines.append("")
    lines.append(f"✅ **{len(existing)} already recorded in the OpenEMR portal:** "
                 f"{exist_labels if existing else 'none'}")
    lines.append(f"🆕 **{len(new)} new unique {'discoveries' if len(new) != 1 else 'discovery'}:** "
                 f"{new_labels if new else 'none'}")

    if not new:
        lines.append("")
        lines.append("✅ All extracted markers are already in the portal. No sync needed.")
        return "\n".join(lines)

    lines.append("")

    # ── Phase 4 WARNING PROTOCOL (mandatory re-statement before gate) ─────
    alert_block = _build_safety_alert_block(
        allergy_conflict or {},
        denial_risk or {},
    )
    if alert_block:
        lines.append(alert_block)
        lines.append("")
        lines.append(
            f"⚠️ **WARNING: High-severity alerts were found in the Phase 2 audit "
            f"(see above). Do you still wish to proceed?**"
        )
        lines.append("")

    # ── Phase 4 HITL sync gate ────────────────────────────────────────────
    lines.append("---")
    lines.append("### 🔒 Phase 4 — Sync Gate")
    lines.append("")
    lines.append(
        f"Would you like me to sync the **{len(new)} new "
        f"{'discovery' if len(new) == 1 else 'discoveries'}** "
        f"({new_labels}) to the OpenEMR portal?"
    )
    lines.append("")
    lines.append(
        "> Reply **Yes**, **Sync**, or **Proceed** to push to OpenEMR.  "
        "> Any other response will clear the sync queue."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# comparison_node
# ---------------------------------------------------------------------------

def comparison_node(state: AgentState) -> AgentState:
    """
    LangGraph node: compare newly staged PDF markers against prior SYNCED rows.

    Only runs when a PDF was processed in the current turn.  Sets
    ``pending_sync_confirmation = True`` and builds the sync prompt message
    when new discoveries are found; sets it to ``False`` when everything is
    already in the portal.

    Safety: clears a stale ``pending_sync_confirmation`` flag if the current
    patient differs from the patient stored in ``staged_patient_fhir_id``.

    Args:
        state: AgentState after extractor_node has run.

    Returns:
        AgentState with HITL fields populated and ``final_response`` set to
        the sync confirmation prompt (appended after the clinical summary).
    """
    session_id       = state.get("session_id", "")
    pdf_file         = state.get("pdf_source_file", "")
    patient          = state.get("extracted_patient") or {}
    patient_id       = patient.get("id", "") or patient.get("fhir_id", "")
    patient_name     = patient.get("name", "this patient")
    allergy_conflict = state.get("allergy_conflict_result") or {}
    denial_risk      = state.get("denial_risk") or {}

    # ── Only run when a PDF was just processed ────────────────────────────
    if not pdf_file:
        logger.debug("comparison_node: no PDF in state — skipping.")
        return state

    # ── Stale-flag safety guard ───────────────────────────────────────────
    # If the user uploads a new PDF for a different patient while a prior
    # sync confirmation is still pending, clear the old flag entirely.
    prior_fhir_id = state.get("staged_patient_fhir_id", "")
    if state.get("pending_sync_confirmation") and prior_fhir_id and prior_fhir_id != patient_id:
        logger.info(
            "comparison_node: clearing stale pending_sync_confirmation "
            "(prior patient=%s, current=%s).",
            prior_fhir_id, patient_id,
        )
        state["pending_sync_confirmation"] = False
        state["sync_summary"] = {}
        state["staged_patient_fhir_id"] = ""
        state["staged_session_id"] = ""

    # ── Pull PENDING rows for this session ───────────────────────────────
    try:
        pending = _db.get_pending_markers(session_id=session_id)
    except Exception as exc:
        logger.warning("comparison_node: DB read failed — %s", exc)
        return state

    if not pending:
        logger.info("comparison_node: no PENDING rows for session '%s'.", session_id)
        return state

    total_raw = len(pending)

    # ── Deduplicate → champions ───────────────────────────────────────────
    champions, duplicate_count = _group_pending_by_champion(pending)

    if not champions:
        logger.info(
            "comparison_node: all %d pending rows have empty values or unknown markers.",
            total_raw,
        )
        return state

    # ── Split into new vs. already-synced ────────────────────────────────
    new_discoveries, already_in_portal = _check_already_synced(champions, patient_id)

    logger.info(
        "comparison_node: %d total raw | %d champions | %d new | %d existing | %d dupes absorbed",
        total_raw, len(champions), len(new_discoveries), len(already_in_portal), duplicate_count,
    )

    # ── Build HITL state ─────────────────────────────────────────────────
    sync_summary = {
        "new":       [{"marker_name": r["marker_name"], "marker_value": r["marker_value"]} for r in new_discoveries],
        "existing":  [{"marker_name": r["marker_name"], "marker_value": r["marker_value"]} for r in already_in_portal],
        "total_raw": total_raw,
    }

    state["sync_summary"]          = sync_summary
    state["staged_session_id"]     = session_id
    state["staged_patient_fhir_id"] = patient_id

    sync_prompt = _build_sync_prompt(
        new_discoveries,
        already_in_portal,
        total_raw,
        duplicate_count,
        allergy_conflict=allergy_conflict,
        denial_risk=denial_risk,
    )

    if new_discoveries:
        state["pending_sync_confirmation"] = True
        logger.info(
            "comparison_node: pending_sync_confirmation=True for patient=%s session=%s",
            patient_id, session_id,
        )
    else:
        # Nothing new to sync — clear flag so orchestrator doesn't wait.
        state["pending_sync_confirmation"] = False

    # ── Append sync prompt to existing final_response ────────────────────
    existing_response = state.get("final_response", "")
    if existing_response:
        state["final_response"] = existing_response.rstrip() + "\n\n---\n\n" + sync_prompt
    else:
        state["final_response"] = sync_prompt

    return state
