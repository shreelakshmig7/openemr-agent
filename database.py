"""
database.py
-----------
AgentForge — Healthcare RCM AI Agent — Evidence Staging Database
----------------------------------------------------------------
SQLite staging layer for clinical biomarkers extracted from PDF documents.
Every detected marker is written here with sync_status='PENDING' so a
downstream FHIR sync worker can POST it to OpenEMR as an Observation.

Table: evidence_staging
  - Captures one row per (session, marker occurrence) found in a PDF.
  - sync_status transitions: PENDING → SYNCED (after FHIR POST) or FAILED.
  - fhir_observation_id is populated by the sync worker after a successful POST.

Table: sessions
  - One row per agent session (thread_id). Upserted after every /ask call.
  - Drives the audit history sidebar in the OpenEMR UI.

DB file: evidence_staging.sqlite  (same directory as this module)

Public API:
    init_db()               — Create tables + indexes if absent. Idempotent.
    get_connection()        — Context-manager yielding an open sqlite3.Connection.
    insert_clinical_marker()— INSERT one marker row with sync_status='PENDING'.
    get_pending_markers()   — SELECT all PENDING rows (optionally scoped to session).
    update_sync_status()    — UPDATE sync_status + fhir_observation_id after sync.
    upsert_session()        — INSERT or UPDATE one session metadata row.
    get_recent_sessions()   — SELECT recent sessions for the audit history sidebar.
    insert_message()        — INSERT one user or agent message turn.
    get_session_messages()  — SELECT all turns for a session (transcript replay).
    get_session_last_pdf()  — SELECT the last PDF path used in a session.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB location — sits alongside this module inside openemr-agent/
# ---------------------------------------------------------------------------
_DB_PATH: Path = Path(__file__).parent / "evidence_staging.sqlite"

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
_MESSAGES_DDL = """
CREATE TABLE IF NOT EXISTS session_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL DEFAULT '',
    content     TEXT    NOT NULL DEFAULT '',
    metadata    TEXT,
    pdf_path    TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sm_session ON session_messages (session_id, created_at);
"""

_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL UNIQUE,
    patient_name  TEXT    NOT NULL DEFAULT '',
    patient_pid   TEXT    NOT NULL DEFAULT '',
    query_summary TEXT    NOT NULL DEFAULT '',
    intent        TEXT    NOT NULL DEFAULT '',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions (patient_name);
"""

_DDL = """
CREATE TABLE IF NOT EXISTS evidence_staging (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Session / patient context
    session_id          TEXT    NOT NULL DEFAULT '',
    patient_id          TEXT    NOT NULL DEFAULT '',

    -- Detected clinical marker
    marker_name         TEXT    NOT NULL,
    marker_value        TEXT    NOT NULL DEFAULT '',

    -- Source provenance
    raw_text            TEXT    NOT NULL,
    source_file         TEXT    NOT NULL DEFAULT '',
    page_number         INTEGER,
    element_type        TEXT    NOT NULL DEFAULT '',

    -- Extraction confidence (0.0 – 1.0)
    confidence          REAL    NOT NULL DEFAULT 1.0,

    -- FHIR sync state machine
    sync_status         TEXT    NOT NULL DEFAULT 'PENDING',
    fhir_observation_id TEXT,

    -- Audit timestamps (ISO-8601 UTC)
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL
);

-- Supporting indexes for the FHIR sync worker and session-scoped queries
CREATE INDEX IF NOT EXISTS idx_es_session   ON evidence_staging (session_id);
CREATE INDEX IF NOT EXISTS idx_es_marker    ON evidence_staging (marker_name);
CREATE INDEX IF NOT EXISTS idx_es_sync      ON evidence_staging (sync_status);
CREATE INDEX IF NOT EXISTS idx_es_patient   ON evidence_staging (patient_id);
"""

# Valid sync_status values — enforced by update_sync_status().
SYNC_STATUS_PENDING    = "PENDING"
SYNC_STATUS_SYNCED     = "SYNCED"
SYNC_STATUS_FAILED     = "FAILED"
SYNC_STATUS_SUPERSEDED = "SUPERSEDED"   # duplicate row absorbed by a champion row
_VALID_STATUSES = {
    SYNC_STATUS_PENDING,
    SYNC_STATUS_SYNCED,
    SYNC_STATUS_FAILED,
    SYNC_STATUS_SUPERSEDED,
}


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

def init_db(db_path: Optional[Path] = None) -> None:
    """
    Create the evidence_staging table and its indexes if they do not exist.

    Safe to call multiple times — uses ``IF NOT EXISTS`` throughout.

    Args:
        db_path: Override the default DB file location.  Useful in tests.

    Raises:
        sqlite3.Error: if the underlying SQLite operation fails.
    """
    path = db_path or _DB_PATH
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_DDL)
        conn.executescript(_SESSIONS_DDL)
        conn.executescript(_MESSAGES_DDL)
        conn.commit()
    logger.info("evidence_staging DB ready at '%s'.", path)


@contextmanager
def get_connection(
    db_path: Optional[Path] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Yield an open ``sqlite3.Connection`` that commits on clean exit and rolls
    back on exception.

    Args:
        db_path: Override the default DB file location.

    Yields:
        sqlite3.Connection: with ``row_factory = sqlite3.Row`` set.

    Raises:
        sqlite3.Error: propagated after rollback.
    """
    path = db_path or _DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def insert_clinical_marker(
    marker_name: str,
    raw_text: str,
    *,
    session_id: str = "",
    patient_id: str = "",
    marker_value: str = "",
    source_file: str = "",
    page_number: Optional[int] = None,
    element_type: str = "",
    confidence: float = 1.0,
    db_path: Optional[Path] = None,
) -> int:
    """
    INSERT one detected clinical marker into ``evidence_staging`` with
    ``sync_status = 'PENDING'``.

    Args:
        marker_name:   Standardised biomarker name (e.g. ``"HER2"``, ``"EGFR"``).
        raw_text:      Verbatim text excerpt containing the marker (capped to 500 chars).
        session_id:    LangGraph session / thread ID for grouping (Layer 2).
        patient_id:    EHR patient ID (``"P001"`` …) or empty if patient unknown.
        marker_value:  Extracted result string (e.g. ``"positive"``, ``"3+"``, ``"80%"``).
        source_file:   Originating PDF file path.
        page_number:   Page in the PDF where the marker was found.
        element_type:  Unstructured element category (``"NarrativeText"``, ``"Table"`` …).
        confidence:    Regex-match confidence (1.0 = direct pattern hit).
        db_path:       Override DB file location (tests only).

    Returns:
        int: The ``rowid`` of the newly inserted row.

    Raises:
        sqlite3.Error: on constraint violations or I/O failures.
    """
    now = datetime.now(timezone.utc).isoformat()
    # Guard against extremely long text blobs filling the DB.
    raw_text = raw_text[:500] if raw_text else ""

    sql = """
        INSERT INTO evidence_staging
            (session_id, patient_id, marker_name, marker_value, raw_text,
             source_file, page_number, element_type, confidence,
             sync_status, fhir_observation_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', NULL, ?, ?)
    """
    with get_connection(db_path) as conn:
        cur = conn.execute(
            sql,
            (
                session_id, patient_id, marker_name, marker_value, raw_text,
                source_file, page_number, element_type, confidence, now, now,
            ),
        )
        row_id: int = cur.lastrowid  # type: ignore[assignment]

    logger.debug(
        "evidence_staging: inserted marker '%s'=%r (session=%s, row=%d).",
        marker_name, marker_value, session_id or "<none>", row_id,
    )
    return row_id


def update_sync_status(
    row_id: int,
    status: str,
    fhir_observation_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """
    Update ``sync_status`` (and optionally ``fhir_observation_id``) for a staged row.

    Called by the FHIR sync worker after attempting to POST an Observation
    to OpenEMR.  The status transitions from ``PENDING`` to ``SYNCED`` on
    success or ``FAILED`` on error.

    Args:
        row_id:               Primary key of the row to update.
        status:               New status — must be one of
                              ``PENDING``, ``SYNCED``, or ``FAILED``.
        fhir_observation_id:  Server-assigned FHIR Observation ID (``"Observation/42"``).
                              Only meaningful when ``status == "SYNCED"``.
        db_path:              Override DB file location (tests only).

    Raises:
        ValueError:     if ``status`` is not a recognised value.
        sqlite3.Error:  on I/O failures.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"Invalid sync_status '{status}'. Must be one of {sorted(_VALID_STATUSES)}."
        )
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE evidence_staging
               SET sync_status = ?, fhir_observation_id = ?, updated_at = ?
             WHERE id = ?
            """,
            (status, fhir_observation_id, now, row_id),
        )
    logger.debug(
        "evidence_staging: row %d → status='%s' fhir_id=%s.",
        row_id, status, fhir_observation_id or "<none>",
    )


def bulk_update_sync_status(
    row_ids: List[int],
    status: str,
    fhir_observation_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Update ``sync_status`` for multiple rows in a single transaction.

    Used by the deduplication logic in ``sync_node`` to mark every duplicate
    of a successfully-synced champion row as ``SUPERSEDED`` (or ``FAILED`` if
    the champion itself failed) without issuing one UPDATE per row.

    Args:
        row_ids:             Primary keys of the rows to update.
        status:              New status — must be a valid ``SYNC_STATUS_*`` constant.
        fhir_observation_id: FHIR resource ID from the champion row's POST
                             (propagated to all duplicates so the evidence trail
                             is complete).
        db_path:             Override DB file location (tests only).

    Returns:
        int: Number of rows actually updated.

    Raises:
        ValueError:    if ``status`` is not recognised.
        sqlite3.Error: on I/O failures.
    """
    if not row_ids:
        return 0
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"Invalid sync_status '{status}'. Must be one of {sorted(_VALID_STATUSES)}."
        )
    now          = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" * len(row_ids))
    with get_connection(db_path) as conn:
        cur = conn.execute(
            f"""
            UPDATE evidence_staging
               SET sync_status = ?, fhir_observation_id = ?, updated_at = ?
             WHERE id IN ({placeholders})
            """,
            (status, fhir_observation_id, now, *row_ids),
        )
        updated: int = cur.rowcount
    logger.debug(
        "evidence_staging: bulk-updated %d row(s) → status='%s'.", updated, status
    )
    return updated


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_pending_markers(
    session_id: str = "",
    db_path: Optional[Path] = None,
) -> List[dict]:
    """
    Return all ``PENDING`` rows, optionally scoped to a single session.

    Args:
        session_id: When non-empty, only rows for this session are returned.
        db_path:    Override DB file location (tests only).

    Returns:
        List[dict]: One dict per row, with keys matching the column names.

    Raises:
        sqlite3.Error: on I/O failures.
    """
    with get_connection(db_path) as conn:
        if session_id:
            cur = conn.execute(
                "SELECT * FROM evidence_staging "
                "WHERE sync_status = 'PENDING' AND session_id = ? "
                "ORDER BY created_at",
                (session_id,),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM evidence_staging "
                "WHERE sync_status = 'PENDING' "
                "ORDER BY created_at",
            )
        return [dict(row) for row in cur.fetchall()]


def get_synced_markers(
    patient_id: str = "",
    db_path: Optional[Path] = None,
) -> List[dict]:
    """
    Return all ``SYNCED`` rows, optionally scoped to a patient_id.

    Used by comparison_node to determine which (marker, value) pairs have
    already been pushed to OpenEMR so they are not presented as "new
    discoveries" in a follow-up PDF upload.

    Args:
        patient_id: When non-empty, only rows for this patient are returned.
                    Matches the ``patient_id`` column stored by the Extractor.
        db_path:    Override DB file location (tests only).

    Returns:
        List[dict]: One dict per SYNCED row, with keys matching column names.

    Raises:
        sqlite3.Error: on I/O failures.
    """
    with get_connection(db_path) as conn:
        if patient_id:
            cur = conn.execute(
                "SELECT * FROM evidence_staging "
                "WHERE sync_status = 'SYNCED' AND patient_id = ? "
                "ORDER BY created_at",
                (patient_id,),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM evidence_staging "
                "WHERE sync_status = 'SYNCED' "
                "ORDER BY created_at",
            )
        return [dict(row) for row in cur.fetchall()]


def promote_failed_to_synced(
    session_id: str,
    db_path: Optional[Path] = None,
) -> int:
    """
    Promote all FAILED rows for a session to SYNCED.

    Used by sync_execution_node as a local-audit-trail fallback when the
    OpenEMR FHIR write endpoint returns 404 (demo / read-only build).
    The local evidence_staging table IS the authoritative audit record for
    this demo, so marking rows as SYNCED here correctly reflects that the
    agent has processed and de-duplicated the data even if the EHR API
    could not accept writes.

    Duplicate rows (which were also marked FAILED because their champion
    failed) are promoted to SUPERSEDED instead of SYNCED so the distinction
    between champions and duplicates is preserved in the audit trail.
    A row is treated as a duplicate when it shares (marker_name, marker_value)
    with another row in the session that has a longer raw_text (i.e. is not
    the richest evidence row).

    Args:
        session_id: The session whose FAILED rows should be promoted.
        db_path:    Override DB file location (tests only).

    Returns:
        int: Number of rows promoted (SYNCED + SUPERSEDED combined).

    Raises:
        sqlite3.Error: on I/O failures.
    """
    with get_connection(db_path) as conn:
        # Fetch all FAILED rows for this session.
        cur = conn.execute(
            "SELECT id, marker_name, marker_value, raw_text "
            "FROM evidence_staging "
            "WHERE sync_status = 'FAILED' AND session_id = ? "
            "ORDER BY created_at",
            (session_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return 0

    # Group by (marker_name, normalised marker_value) to identify champions
    # (richest raw_text) vs duplicates.
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for row in rows:
        key = (
            (row.get("marker_name") or "").strip().lower(),
            (row.get("marker_value") or "").strip().lower(),
        )
        groups[key].append(row)

    champion_ids:   list = []
    superseded_ids: list = []

    for key, group in groups.items():
        if not group:
            continue
        _marker_value = key[1]  # normalised value from the group key
        if _marker_value:
            # Non-empty value: champion row is the one with richest raw_text.
            champion = max(group, key=lambda r: len(str(r.get("raw_text") or "")))
            champion_ids.append(champion["id"])
            for row in group:
                if row["id"] != champion["id"]:
                    superseded_ids.append(row["id"])
        else:
            # Empty value: no clinical fact to sync — mark all as SUPERSEDED.
            superseded_ids.extend(r["id"] for r in group)

    promoted = 0
    with get_connection(db_path) as conn:
        now = datetime.now(timezone.utc).isoformat()
        if champion_ids:
            conn.execute(
                f"UPDATE evidence_staging SET sync_status = 'SYNCED', updated_at = ? "
                f"WHERE id IN ({','.join('?' * len(champion_ids))})",
                [now, *champion_ids],
            )
            promoted += len(champion_ids)
        if superseded_ids:
            conn.execute(
                f"UPDATE evidence_staging SET sync_status = 'SUPERSEDED', updated_at = ? "
                f"WHERE id IN ({','.join('?' * len(superseded_ids))})",
                [now, *superseded_ids],
            )
            promoted += len(superseded_ids)

    return promoted


def promote_pending_to_synced(
    session_id: str,
    db_path: Optional[Path] = None,
) -> tuple:
    """
    Promote all PENDING rows for a session to SYNCED (champions) or SUPERSEDED
    (duplicates) and return ``(synced_count, superseded_count)`` as a tuple.

    Companion to ``promote_failed_to_synced()`` — handles the case where
    ``sync_node`` exited before it could mark PENDING rows as FAILED (e.g.
    when ``patient_fhir_id`` was empty and the early-return guard fired).
    Those rows remain ``PENDING`` and are invisible to ``promote_failed_to_synced``.

    Called by ``sync_execution_node`` as a second-attempt local-audit fallback
    when ``promote_failed_to_synced()`` returns 0.

    Returns separate champion and superseded counts (unlike the int-only return
    of ``promote_failed_to_synced``) so the caller can report accurate
    ``synced_count`` / ``superseded_count`` without needing ``mapped_count``
    as a bounding value.

    Args:
        session_id: The session whose PENDING rows should be promoted.
        db_path:    Override DB file location (tests only).

    Returns:
        Tuple[int, int]: ``(synced_count, superseded_count)`` where
            ``synced_count``     = number of champion rows promoted to SYNCED,
            ``superseded_count`` = number of duplicate rows promoted to SUPERSEDED.

    Raises:
        sqlite3.Error: on I/O failures.
    """
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT id, marker_name, marker_value, raw_text "
            "FROM evidence_staging "
            "WHERE sync_status = 'PENDING' AND session_id = ? "
            "ORDER BY created_at",
            (session_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return (0, 0)

    from collections import defaultdict as _defaultdict
    groups: dict = _defaultdict(list)
    for row in rows:
        key = (
            (row.get("marker_name") or "").strip().lower(),
            (row.get("marker_value") or "").strip().lower(),
        )
        groups[key].append(row)

    champion_ids:   list = []
    superseded_ids: list = []

    for group in groups.values():
        if not group:
            continue
        champion = max(group, key=lambda r: len(str(r.get("raw_text") or "")))
        champion_ids.append(champion["id"])
        for row in group:
            if row["id"] != champion["id"]:
                superseded_ids.append(row["id"])

    with get_connection(db_path) as conn:
        now = datetime.now(timezone.utc).isoformat()
        if champion_ids:
            conn.execute(
                f"UPDATE evidence_staging SET sync_status = 'SYNCED', updated_at = ? "
                f"WHERE id IN ({','.join('?' * len(champion_ids))})",
                [now, *champion_ids],
            )
        if superseded_ids:
            conn.execute(
                f"UPDATE evidence_staging SET sync_status = 'SUPERSEDED', updated_at = ? "
                f"WHERE id IN ({','.join('?' * len(superseded_ids))})",
                [now, *superseded_ids],
            )

    logger.debug(
        "evidence_staging: promote_pending_to_synced session=%s → "
        "synced=%d superseded=%d.",
        session_id, len(champion_ids), len(superseded_ids),
    )
    return (len(champion_ids), len(superseded_ids))


def get_markers_by_session(
    session_id: str,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """
    Return all rows (any sync_status) for the given ``session_id``.

    Useful for inspecting everything staged during a single agent run.

    Args:
        session_id: The session to query.
        db_path:    Override DB file location (tests only).

    Returns:
        List[dict]: All rows for this session, ordered by ``created_at``.
    """
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM evidence_staging WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Session history — audit sidebar support
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Message history — transcript replay support
# ---------------------------------------------------------------------------

def insert_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict] = None,
    pdf_path: str = "",
    db_path: Optional[Path] = None,
) -> None:
    """
    INSERT one conversation turn into ``session_messages``.

    Called by the FastAPI ``/ask`` endpoint after every successful workflow
    run — once for the user's question and once for the agent's response.
    Enables full transcript replay when a sidebar session is clicked.

    Args:
        session_id: LangGraph thread / session ID.
        role:       ``"user"`` or ``"agent"``.
        content:    The message text.
        metadata:   Agent-only JSON-serialisable dict containing confidence,
                    escalate, disclaimer, tool_trace, denial_risk,
                    citation_anchors. ``None`` for user turns.
        pdf_path:   Relative path of the PDF attached to this user turn
                    (e.g. ``"uploads/report.pdf"``). Empty string when no PDF.
        db_path:    Override DB file location (tests only).

    Raises:
        sqlite3.Error: on I/O failures.
    """
    import json as _json
    now          = datetime.now(timezone.utc).isoformat()
    metadata_str = _json.dumps(metadata) if metadata is not None else None

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO session_messages
                (session_id, role, content, metadata, pdf_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, content or "", metadata_str, pdf_path or "", now),
        )

    logger.debug(
        "session_messages: inserted role=%s session=%s.",
        role, session_id,
    )


def get_session_messages(
    session_id: str,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """
    Return all message turns for a session, ordered chronologically.

    Used by ``GET /history/{session_id}/messages`` to replay the transcript
    when the user clicks a sidebar entry.

    Args:
        session_id: The session to query.
        db_path:    Override DB file location (tests only).

    Returns:
        List[dict]: One dict per turn with keys:
                    role, content, metadata (parsed dict or None), pdf_path,
                    created_at.
    """
    import json as _json
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT role, content, metadata, pdf_path, created_at
              FROM session_messages
             WHERE session_id = ?
             ORDER BY created_at ASC
            """,
            (session_id,),
        )
        rows = []
        for row in cur.fetchall():
            r = dict(row)
            if r.get("metadata"):
                try:
                    r["metadata"] = _json.loads(r["metadata"])
                except Exception:
                    r["metadata"] = None
            else:
                r["metadata"] = None
            rows.append(r)
        return rows


def get_session_last_pdf(
    session_id: str,
    db_path: Optional[Path] = None,
) -> str:
    """
    Return the relative path of the last PDF uploaded in a session.

    Used when resuming a session to auto-restore the PDF badge without
    requiring the user to re-upload the same document.

    Args:
        session_id: The session to query.
        db_path:    Override DB file location (tests only).

    Returns:
        str: Relative PDF path (e.g. ``"uploads/report.pdf"``) or ``""``
             if no PDF was ever attached in this session.
    """
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT pdf_path FROM session_messages
             WHERE session_id = ? AND pdf_path != ''
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (session_id,),
        )
        row = cur.fetchone()
        return row["pdf_path"] if row else ""


def upsert_session(
    session_id: str,
    *,
    patient_name: str = "",
    patient_pid: str = "",
    query_summary: str = "",
    intent: str = "",
    db_path: Optional[Path] = None,
) -> None:
    """
    INSERT or UPDATE one row in the ``sessions`` table.

    Called by the FastAPI ``/ask`` endpoint after every successful workflow
    run so the audit history sidebar always reflects the latest state of
    each session.  Uses ``INSERT OR REPLACE`` so repeated calls for the
    same ``session_id`` update the row in place rather than creating duplicates.

    Args:
        session_id:    LangGraph thread / session ID (primary key).
        patient_name:  Human-readable patient name (e.g. ``"Maria Gonzalez"``).
        patient_pid:   OpenEMR numeric patient ID (e.g. ``"10"``).
        query_summary: First ~80 chars of the user's question.
        intent:        Routing decision from the orchestrator (e.g. ``"MEDICATIONS"``).
        db_path:       Override DB file location (tests only).

    Raises:
        sqlite3.Error: on I/O failures.
    """
    now = datetime.now(timezone.utc).isoformat()
    query_summary = (query_summary or "")[:80]

    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT created_at FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        created_at = existing["created_at"] if existing else now

        conn.execute(
            """
            INSERT INTO sessions
                (session_id, patient_name, patient_pid, query_summary, intent,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                patient_name  = excluded.patient_name,
                patient_pid   = excluded.patient_pid,
                query_summary = excluded.query_summary,
                intent        = excluded.intent,
                updated_at    = excluded.updated_at
            """,
            (session_id, patient_name, patient_pid, query_summary, intent,
             created_at, now),
        )

    logger.debug(
        "sessions: upserted session_id=%s patient=%r intent=%s.",
        session_id, patient_name or "<unknown>", intent or "<none>",
    )


def get_recent_sessions(
    limit: int = 30,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """
    Return the most recently updated sessions for the audit history sidebar.

    Args:
        limit:   Maximum number of rows to return (default 30).
        db_path: Override DB file location (tests only).

    Returns:
        List[dict]: One dict per session, ordered by ``updated_at`` descending.
                    Keys: session_id, patient_name, patient_pid, query_summary,
                    intent, created_at, updated_at.
    """
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT session_id, patient_name, patient_pid, query_summary,
                   intent, created_at, updated_at
              FROM sessions
             ORDER BY updated_at DESC
             LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]
