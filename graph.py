"""
graph.py
--------
AgentForge — Healthcare RCM AI Agent — FHIR Evidence Staging Sync Graph
-----------------------------------------------------------------------
Defines the Sync_Node and the LangGraph StateGraph that drives the
evidence-staging → OpenEMR FHIR synchronisation pipeline.

Pipeline (4 steps, one pass through sync_node):

  Step 1  PULL    database.get_pending_markers()
                  Fetch all evidence_staging rows where sync_status = 'PENDING'.
                  Rows are optionally scoped to a single session_id.

  Step 2  MAP     fhir_mapper.map_to_bundle()
                  Translate resolvable rows into a FHIR R4 Transaction Bundle.
                  Rows whose marker_name has no LOINC entry are pre-screened out
                  and immediately marked FAILED with reason "unknown_marker".

  Step 3  POST    openemr_client.OpenEMRClient.post_bundle()
                  Iterate through each bundle entry and POST to the appropriate
                  FHIR endpoint.  Returns a per-entry result list.

  Step 4  UPDATE  database.update_sync_status()
                  • 200/201 success → SYNCED  (fhir_observation_id populated)
                  • API error       → FAILED
                  • Unknown marker  → FAILED  (no network call attempted)

Graph topology:
    START → sync_node → END

SyncState fields:
    Input (set by caller):
        session_id       Filter pending markers to one session; empty = all.
        patient_fhir_id  FHIR Patient UUID used as bundle subject (required).
        base_url         OpenEMR HTTPS base URL (default: https://localhost:9300).

    Output (written by sync_node):
        pending_count    Rows found with sync_status='PENDING'.
        mapped_count     Rows that had a known LOINC code (→ bundle entries).
        skipped_count    Rows with unknown marker_name (→ immediately FAILED).
        synced_count     Rows successfully POSTed (→ SYNCED).
        failed_count     Rows that failed FHIR POST (→ FAILED).
        sync_results     List of per-row result dicts for audit/logging.
        error            Top-level error string if the node aborts early.

Usage:
    from graph import run_sync

    result = run_sync(
        patient_fhir_id="a1312c03-cd3f-44b5-9d5f-1ef5751a7550",
        session_id="my-session",
    )
    print(result["synced_count"], "rows synced.")

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# ── Path bootstrap (so imports work whether run from openemr-agent/ or repo root)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import database as db                                # evidence_staging helpers
import fhir_mapper                                   # map_to_bundle, get_loinc_code
from openemr_client import OpenEMRClient, OpenEMRAPIError  # async FHIR client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SyncState — the shared state that flows through the sync graph
# ---------------------------------------------------------------------------

class SyncState(TypedDict):
    """
    LangGraph state for the FHIR sync pipeline.

    Caller-supplied input fields:
        session_id:      Filter pending markers to one session (empty = all).
        patient_fhir_id: FHIR Patient UUID (e.g. "a1312c03-...").  Required.
        base_url:        OpenEMR HTTPS base URL.

    Output fields written by sync_node:
        pending_count:   Rows with sync_status='PENDING' at call time.
        mapped_count:    Rows resolved to a LOINC code (→ bundle entries).
        skipped_count:   Rows with unknown marker_name (→ FAILED, no POST).
        synced_count:    Rows successfully POSTed to OpenEMR (→ SYNCED).
        failed_count:    Rows whose FHIR POST was rejected (→ FAILED).
        sync_results:    Ordered list of per-row result dicts for audit.
        error:           Non-None when the node aborts before completing all rows.
    """
    # ── Input ───────────────────────────────────────────────────────────────
    session_id:      str
    patient_fhir_id: str
    base_url:        str
    # ── Output ──────────────────────────────────────────────────────────────
    pending_count:    int
    mapped_count:     int   # unique (champion) rows sent to bundle
    duplicate_count:  int   # rows absorbed as duplicates of a champion
    skipped_count:    int   # rows rejected (empty value / unknown LOINC)
    synced_count:     int
    superseded_count: int   # duplicate rows marked SUPERSEDED after champion synced
    failed_count:     int
    sync_results:     List[Dict[str, Any]]
    error:            Optional[str]


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def _deduplicate_resolvable(
    resolvable: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[int, List[int]]]:
    """
    Group resolvable rows by ``(loinc_code, normalised_value)`` and elect one
    champion per group — the row with the most evidence text (longest
    ``raw_text``).

    Duplicates arise because the same marker may be mentioned multiple times
    across different PDF pages, table cells, and narrative paragraphs.  Only
    the champion is included in the FHIR bundle; duplicates are recorded so
    the caller can mark them ``SUPERSEDED`` after a successful POST.

    Args:
        resolvable: Pre-screened rows — every row has a non-empty
                    ``marker_value`` and a resolvable LOINC code.

    Returns:
        Tuple of:
          * ``champions``       — one row per unique (marker, value) pair,
                                  ordered by first occurrence.
          * ``duplicates_map``  — maps each champion's ``id`` → list of
                                  duplicate ``id`` values (may be empty list
                                  when a group has only one member).

    Grouping key
    ------------
    ``loinc_code`` normalises across short-name aliases (``"HER2"`` and
    ``"HER2 Status"`` share LOINC 85337-4 → treated as the same marker).
    ``marker_value`` is stripped and lower-cased so ``"Positive"`` and
    ``"positive"`` are treated as the same result.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in resolvable:
        loinc     = fhir_mapper.get_loinc_code(row.get("marker_name", "")) or row.get("marker_name", "")
        value_key = str(row.get("marker_value", "")).strip().lower()
        groups[f"{loinc}|{value_key}"].append(row)

    champions:     List[Dict[str, Any]] = []
    duplicates_map: Dict[int, List[int]] = {}

    for group in groups.values():
        # Champion = row with the longest raw_text (richest evidence context).
        sorted_group = sorted(
            group,
            key=lambda r: len(str(r.get("raw_text", ""))),
            reverse=True,
        )
        champion  = sorted_group[0]
        dup_ids   = [r["id"] for r in sorted_group[1:]]
        champions.append(champion)
        duplicates_map[champion["id"]] = dup_ids

    logger.info(
        "sync_node: dedup — %d resolvable → %d champion(s), %d duplicate(s) absorbed.",
        len(resolvable),
        len(champions),
        len(resolvable) - len(champions),
    )
    return champions, duplicates_map


# ---------------------------------------------------------------------------
# Sync_Node
# ---------------------------------------------------------------------------

def sync_node(state: SyncState) -> SyncState:
    """
    LangGraph node: pull → map → POST → update.

    Implements the four-step FHIR sync pipeline documented in the module
    docstring.  Bridges the synchronous LangGraph runtime to the async
    ``OpenEMRClient`` via ``asyncio.run()``.

    Args:
        state: SyncState populated by the caller (session_id, patient_fhir_id,
               base_url).

    Returns:
        SyncState with all output fields populated (pending_count, synced_count,
        failed_count, skipped_count, mapped_count, sync_results, error).

    Raises:
        Never — all errors are captured in ``state["error"]``.
    """
    session_id:      str = state.get("session_id", "")
    patient_fhir_id: str = state.get("patient_fhir_id", "").strip()
    base_url:        str = state.get("base_url", "https://localhost:9300")

    sync_results: List[Dict[str, Any]] = []

    # Guard: patient_fhir_id is required for bundle subject.
    if not patient_fhir_id:
        logger.error("sync_node: patient_fhir_id is required but was not provided.")
        return {
            **state,
            "error":         "patient_fhir_id is required but was not provided.",
            "pending_count": 0,
            "mapped_count":  0,
            "skipped_count": 0,
            "synced_count":  0,
            "failed_count":  0,
            "sync_results":  [],
        }

    # ── Step 1: Pull PENDING rows from evidence_staging ─────────────────────
    try:
        pending: List[Dict[str, Any]] = db.get_pending_markers(
            session_id=session_id
        )
    except Exception as exc:
        logger.exception("sync_node: database read failed.")
        return {
            **state,
            "error":         f"DB read failed: {exc}",
            "pending_count": 0,
            "mapped_count":  0,
            "skipped_count": 0,
            "synced_count":  0,
            "failed_count":  0,
            "sync_results":  [],
        }

    pending_count = len(pending)
    logger.info(
        "sync_node: %d PENDING rows found (session=%s).",
        pending_count, session_id or "<all>",
    )

    if not pending:
        return {
            **state,
            "error":         None,
            "pending_count": 0,
            "mapped_count":  0,
            "skipped_count": 0,
            "synced_count":  0,
            "failed_count":  0,
            "sync_results":  [],
        }

    # ── Step 2a: Pre-screen — reject empty values and unknown LOINC markers ──
    # Rejection criteria (row → unresolvable → FAILED immediately):
    #   1. marker_value is empty — extractor matched a keyword in a table
    #      header or criteria list where no result value was nearby (row 27).
    #   2. marker_name has no LOINC code in the registry / aliases.
    resolvable:   List[Dict[str, Any]] = []
    unresolvable: List[Dict[str, Any]] = []

    for row in pending:
        marker_value = str(row.get("marker_value") or "").strip()
        if not marker_value:
            row["_reject_reason"] = "empty_value — no result found near marker in source text"
            unresolvable.append(row)
        elif not fhir_mapper.get_loinc_code(row.get("marker_name", "")):
            row["_reject_reason"] = "unknown_marker — no LOINC code in registry"
            unresolvable.append(row)
        else:
            resolvable.append(row)

    skipped_count = len(unresolvable)

    logger.info(
        "sync_node: %d resolvable, %d rejected (empty value / unknown LOINC).",
        len(resolvable), skipped_count,
    )

    # Mark rejected rows as FAILED immediately.
    for row in unresolvable:
        row_id = row["id"]
        reason = row.get("_reject_reason", "unknown rejection reason")
        try:
            db.update_sync_status(row_id=row_id, status=db.SYNC_STATUS_FAILED)
        except Exception as exc:
            logger.warning("sync_node: failed to mark row %d as FAILED: %s", row_id, exc)
        sync_results.append({
            "row_id":      row_id,
            "marker_name": row.get("marker_name"),
            "status":      "FAILED",
            "reason":      reason,
            "fhir_id":     None,
            "role":        "rejected",
        })

    if not resolvable:
        return {
            **state,
            "error":            None,
            "pending_count":    pending_count,
            "mapped_count":     0,
            "duplicate_count":  0,
            "skipped_count":    skipped_count,
            "synced_count":     0,
            "superseded_count": 0,
            "failed_count":     skipped_count,
            "sync_results":     sync_results,
        }

    # ── Step 2b: Deduplicate — elect one champion per (marker, value) pair ──
    # Group all resolvable rows by (LOINC code, normalised value).  Within each
    # group the row with the longest raw_text is elected as champion and is the
    # only one sent to the FHIR bundle.  All others are tracked as duplicates
    # and will be marked SUPERSEDED (champion SYNCED) or FAILED (champion FAILED)
    # after the POST so they never resurface as PENDING.
    champions, duplicates_map = _deduplicate_resolvable(resolvable)
    mapped_count    = len(champions)
    duplicate_count = len(resolvable) - mapped_count

    # ── Step 2c: Build FHIR R4 Transaction Bundle from champions only ────────
    try:
        bundle = fhir_mapper.map_to_bundle(
            patient_id=patient_fhir_id,
            facts=champions,
        )
    except Exception as exc:
        logger.exception("sync_node: fhir_mapper.map_to_bundle() failed.")
        all_resolvable_ids = [r["id"] for r in resolvable]
        try:
            db.bulk_update_sync_status(all_resolvable_ids, db.SYNC_STATUS_FAILED)
        except Exception:
            pass
        return {
            **state,
            "error":            f"Bundle construction failed: {exc}",
            "pending_count":    pending_count,
            "mapped_count":     mapped_count,
            "duplicate_count":  duplicate_count,
            "skipped_count":    skipped_count,
            "synced_count":     0,
            "superseded_count": 0,
            "failed_count":     pending_count,
            "sync_results":     sync_results,
        }

    # Sanity: bundle entries must line up 1-to-1 with champion rows by index.
    bundle_entries: int = len(bundle.get("entry", []))
    if bundle_entries != mapped_count:
        logger.error(
            "sync_node: bundle/champion count mismatch — expected %d, got %d.",
            mapped_count, bundle_entries,
        )
        all_resolvable_ids = [r["id"] for r in resolvable]
        try:
            db.bulk_update_sync_status(all_resolvable_ids, db.SYNC_STATUS_FAILED)
        except Exception:
            pass
        return {
            **state,
            "error":            (
                f"Bundle/row count mismatch ({bundle_entries} entries vs "
                f"{mapped_count} champion rows). All rows marked FAILED."
            ),
            "pending_count":    pending_count,
            "mapped_count":     mapped_count,
            "duplicate_count":  duplicate_count,
            "skipped_count":    skipped_count,
            "synced_count":     0,
            "superseded_count": 0,
            "failed_count":     pending_count,
            "sync_results":     sync_results,
        }

    # ── Step 3: POST bundle via OpenEMRClient ────────────────────────────────
    try:
        bundle_result: Dict[str, Any] = asyncio.run(
            _post_bundle_async(base_url=base_url, bundle=bundle)
        )
    except Exception as exc:
        logger.exception("sync_node: async bundle POST raised an exception.")
        all_resolvable_ids = [r["id"] for r in resolvable]
        try:
            db.bulk_update_sync_status(all_resolvable_ids, db.SYNC_STATUS_FAILED)
        except Exception:
            pass
        return {
            **state,
            "error":            f"FHIR POST failed: {exc}",
            "pending_count":    pending_count,
            "mapped_count":     mapped_count,
            "duplicate_count":  duplicate_count,
            "skipped_count":    skipped_count,
            "synced_count":     0,
            "superseded_count": 0,
            "failed_count":     pending_count,
            "sync_results":     sync_results,
        }

    # ── Step 4: Update champion rows + mark duplicates based on POST results ─
    entry_results:    List[Dict[str, Any]] = bundle_result.get("results", [])
    synced_count    = 0
    superseded_count = 0
    failed_count    = skipped_count   # already counted above

    for idx, champion in enumerate(champions):
        champion_id  = champion["id"]
        marker_name  = champion.get("marker_name", "")
        entry_result = entry_results[idx] if idx < len(entry_results) else {}
        success      = entry_result.get("status") == "success"
        fhir_id      = entry_result.get("fhir_id")
        dup_ids      = duplicates_map.get(champion_id, [])

        # ── Update champion row ───────────────────────────────────────────
        try:
            if success:
                db.update_sync_status(
                    row_id=champion_id,
                    status=db.SYNC_STATUS_SYNCED,
                    fhir_observation_id=fhir_id,
                )
                synced_count += 1
                logger.info(
                    "sync_node: champion row %d SYNCED → %s.", champion_id, fhir_id or "<no id>"
                )
            else:
                db.update_sync_status(row_id=champion_id, status=db.SYNC_STATUS_FAILED)
                failed_count += 1
                logger.warning(
                    "sync_node: champion row %d FAILED — HTTP %s: %s",
                    champion_id,
                    entry_result.get("http_status", "?"),
                    entry_result.get("error", ""),
                )
        except Exception as db_exc:
            logger.error("sync_node: DB update failed for champion row %d: %s", champion_id, db_exc)
            failed_count += 1

        sync_results.append({
            "row_id":      champion_id,
            "marker_name": marker_name,
            "status":      "SYNCED" if success else "FAILED",
            "reason":      None if success else entry_result.get("error", "unknown"),
            "fhir_id":     fhir_id,
            "http_status": entry_result.get("http_status"),
            "role":        "champion",
            "duplicates":  dup_ids,
        })

        # ── Update duplicate rows ─────────────────────────────────────────
        if not dup_ids:
            continue

        dup_status = db.SYNC_STATUS_SUPERSEDED if success else db.SYNC_STATUS_FAILED
        try:
            db.bulk_update_sync_status(dup_ids, dup_status, fhir_observation_id=fhir_id)
            if success:
                superseded_count += len(dup_ids)
                logger.info(
                    "sync_node: %d duplicate(s) of row %d marked SUPERSEDED (fhir_id=%s).",
                    len(dup_ids), champion_id, fhir_id or "<none>",
                )
            else:
                failed_count += len(dup_ids)
        except Exception as db_exc:
            logger.error(
                "sync_node: bulk update failed for duplicates of champion %d: %s",
                champion_id, db_exc,
            )

        for dup_id in dup_ids:
            sync_results.append({
                "row_id":      dup_id,
                "marker_name": marker_name,
                "status":      "SUPERSEDED" if success else "FAILED",
                "reason":      f"duplicate of champion row {champion_id}",
                "fhir_id":     fhir_id,
                "http_status": None,
                "role":        "duplicate",
                "champion_id": champion_id,
            })

    logger.info(
        "sync_node: complete — %d SYNCED, %d SUPERSEDED, %d FAILED, %d skipped (of %d pending).",
        synced_count, superseded_count, failed_count, skipped_count, pending_count,
    )

    return {
        **state,
        "error":            None,
        "pending_count":    pending_count,
        "mapped_count":     mapped_count,
        "duplicate_count":  duplicate_count,
        "skipped_count":    skipped_count,
        "synced_count":     synced_count,
        "superseded_count": superseded_count,
        "failed_count":     failed_count,
        "sync_results":     sync_results,
    }


# ---------------------------------------------------------------------------
# Async helper (bridges sync LangGraph node → async OpenEMRClient)
# ---------------------------------------------------------------------------

async def _post_bundle_async(
    base_url: str,
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Coroutine that opens an OpenEMRClient and calls post_bundle().

    Isolated into its own coroutine so ``sync_node`` can call it via a
    single ``asyncio.run()`` without leaking the event loop.

    Args:
        base_url: OpenEMR HTTPS base URL.
        bundle:   FHIR R4 Transaction Bundle produced by fhir_mapper.

    Returns:
        The dict returned by ``OpenEMRClient.post_bundle()``.
    """
    async with OpenEMRClient(base_url=base_url) as client:
        return await client.post_bundle(bundle)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_sync_graph():
    """
    Assemble and compile the LangGraph sync StateGraph.

    Topology:
        START → sync_node → END

    The graph is intentionally minimal — one node, no conditional edges.
    The sync pipeline is designed to be idempotent: rows that fail remain
    PENDING (or become FAILED) and can be re-synced by calling run_sync()
    again.

    Returns:
        CompiledGraph: Ready for ``.invoke(initial_state)``.
    """
    graph: StateGraph = StateGraph(SyncState)
    graph.add_node("sync_node", sync_node)
    graph.set_entry_point("sync_node")
    graph.add_edge("sync_node", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sync(
    patient_fhir_id: str,
    *,
    session_id: str = "",
    base_url: str = "https://localhost:9300",
) -> Dict[str, Any]:
    """
    Run the FHIR evidence staging sync pipeline for a patient.

    Builds and invokes the sync LangGraph, then returns the final SyncState
    as a plain dict.

    Args:
        patient_fhir_id: FHIR Patient resource UUID — used as the bundle
                         ``subject.reference``.  Required.
        session_id:      When non-empty, only PENDING rows for this session
                         are synced.  Empty string syncs all PENDING rows.
        base_url:        OpenEMR HTTPS base URL.

    Returns:
        dict: Final SyncState containing synced_count, failed_count,
              skipped_count, pending_count, sync_results, and error.

    Raises:
        Never — errors are captured in the returned dict's ``error`` field.

    Example::

        from graph import run_sync

        result = run_sync(
            patient_fhir_id="a1312c03-cd3f-44b5-9d5f-1ef5751a7550",
            session_id="test-sync-flow-session",
        )
        print(f"Synced: {result['synced_count']}, Failed: {result['failed_count']}")
    """
    initial_state: SyncState = {
        "session_id":       session_id,
        "patient_fhir_id":  patient_fhir_id,
        "base_url":         base_url,
        # Output fields — initialised to zero; sync_node will populate them.
        "pending_count":    0,
        "mapped_count":     0,
        "duplicate_count":  0,
        "skipped_count":    0,
        "synced_count":     0,
        "superseded_count": 0,
        "failed_count":     0,
        "sync_results":     [],
        "error":            None,
    }

    try:
        graph = build_sync_graph()
        result = graph.invoke(initial_state)
        return dict(result)
    except Exception as exc:
        logger.exception("run_sync: graph invocation failed.")
        return {**initial_state, "error": f"Graph invocation failed: {exc}"}


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run the FHIR evidence staging sync pipeline."
    )
    parser.add_argument(
        "--patient-fhir-id",
        default="a1312c03-cd3f-44b5-9d5f-1ef5751a7550",
        help="FHIR Patient UUID (default: John Smith seeded by seed_portal.py)",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Filter PENDING rows to this session (default: all sessions)",
    )
    parser.add_argument(
        "--base-url",
        default="https://localhost:9300",
        help="OpenEMR HTTPS base URL",
    )
    args = parser.parse_args()

    result = run_sync(
        patient_fhir_id=args.patient_fhir_id,
        session_id=args.session_id,
        base_url=args.base_url,
    )

    print(json.dumps(
        {k: v for k, v in result.items() if k != "sync_results"},
        indent=2,
    ))
    print(f"\nPer-row results ({len(result.get('sync_results', []))} rows):")
    for r in result.get("sync_results", []):
        status = r["status"]
        role   = r.get("role", "")
        icon   = {"SYNCED": "✓", "SUPERSEDED": "◎", "FAILED": "✗", "rejected": "⊘"}.get(status, "?")
        print(
            f"  {icon}  row={r['row_id']:<4} "
            f"{r['marker_name']:<18} "
            f"{status:<12} "
            f"[{role:<9}]  "
            f"fhir_id={r.get('fhir_id') or '<none>'}"
        )
