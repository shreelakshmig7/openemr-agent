"""
main.py
-------
AgentForge — Healthcare RCM AI Agent — FastAPI server
------------------------------------------------------
Exposes the healthcare AI agent as a REST API. Routes queries through
the LangGraph multi-agent state machine (Extractor → Auditor →
Clarification/Output). Manages sessions in-memory for pending_user_input
pause/resume. Provides endpoints for health check, natural language
queries, eval execution, and eval result retrieval.

Endpoints:
    GET  /health                        — Service health check
    POST /ask                           — Natural language query with session context
    POST /ask/stream                    — Same as /ask but streams SSE progress (node updates then done/error)
    GET  /history                       — Recent session list for the audit history sidebar
    GET  /history/{session_id}/messages — Full message transcript for a session
    POST /eval                          — Run eval suite and return results
    GET  /eval/results                  — Return latest saved eval results

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import os
import re
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Any

import logging

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

from langgraph_agent.workflow import run_workflow, stream_workflow, get_state_for_audit
from tools.pii_scrubber import scrub_pii
from healthcare_guidelines import CLINICAL_SAFETY_RULES
from eval.run_eval import run_eval, DEFAULT_GOLDEN_DATA_PATH
import database

load_dotenv()

# Enable INFO-level logging for all agent modules so staging/sync traces appear
# in the uvicorn terminal without needing --log-level debug on the server.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)

# ── Config ─────────────────────────────────────────────────────────────────────

VERSION = "2.0.0"
SERVICE_NAME = "AgentForge Healthcare RCM AI Agent"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "results")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Ensure SQLite tables exist (idempotent — safe to run on every startup).
database.init_db()

# ── In-memory session store ────────────────────────────────────────────────────
# Keyed by session_id. Each value is the last AgentState dict returned by run_workflow.
# Used to detect pending_user_input and pass clarification_response on resume.
_sessions: dict = {}

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title=SERVICE_NAME,
    version=VERSION,
    description="Healthcare RCM AI agent for medication safety review.",
)

# Allow the OpenEMR UI (served on 8300/9300) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8300",
        "https://localhost:9300",
        "http://127.0.0.1:8300",
        "http://64.225.50.120:8300",   # production DigitalOcean OpenEMR
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request body for POST /ask."""
    question: str
    session_id: Optional[str] = None
    thread_id: Optional[str] = None  # Alias for session_id; frontend sends thread_id for New Case flow
    pdf_source_file: Optional[str] = None
    payer_id: Optional[str] = None
    procedure_code: Optional[str] = None


class AskResponse(BaseModel):
    """Response body for POST /ask."""
    answer: str
    session_id: str
    thread_id: Optional[str] = None  # Echo of session_id for frontend (HIPAA New Case flow)
    timestamp: str
    escalate: bool
    confidence: float
    disclaimer: str
    tool_trace: list
    denial_risk: dict
    citation_anchors: list
    query_redacted_preview: Optional[str] = None  # PII-scrubbed version of user question (for UI verification)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_session_id(session_id: Optional[str]) -> str:
    """
    Return existing session_id or generate a new one.

    Args:
        session_id: Caller-supplied session ID or None.

    Returns:
        str: Valid session ID to use for this request.
    """
    return session_id if session_id else str(uuid.uuid4())


def _get_last_patient_from_state(prior_state: dict) -> Optional[str]:
    """
    Extract the last resolved patient name or ID from the prior workflow result.
    Used only as a last-resort fallback for the pending_user_input clarification
    resume path. Normal follow-up context is now handled by the memory system:
    prior_query_context (Layer 2) and messages (Layer 1) carried into the next
    run_workflow call via the prior_state merge.

    Args:
        prior_state: Last AgentState dict stored for this session.

    Returns:
        str | None: Patient identifier (e.g. "John Smith" or "P001"), or None.
    """
    try:
        extracted = prior_state.get("extracted_patient_identifier") or {}
        if isinstance(extracted, dict) and not extracted.get("ambiguous"):
            val = extracted.get("value")
            if val and isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        pass
    return None


def _load_latest_results(results_dir: str) -> Optional[dict]:
    """
    Load the most recent eval results JSON file from results_dir.

    Args:
        results_dir: Directory containing eval_results_*.json files.

    Returns:
        dict | None: Parsed results or None if no files found.
    """
    try:
        if not os.path.isdir(results_dir):
            return None
        files = sorted(
            [f for f in os.listdir(results_dir) if f.startswith("eval_results_") and f.endswith(".json")],
            reverse=True,
        )
        if not files:
            return None
        filepath = os.path.join(results_dir, files[0])
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _resolve_pdf_path(pdf_source_file: Optional[str]) -> Optional[str]:
    """
    Resolve pdf_source_file to an absolute path under the application root.

    The upload endpoint returns a relative path (e.g. "uploads/file.pdf"). In
    production, the process CWD may differ from the app root, so open() would
    fail. This helper resolves relative paths against the directory containing
    main.py so the extractor finds the file regardless of CWD.

    Args:
        pdf_source_file: Path from the client (relative or absolute).

    Returns:
        Absolute path under the app root, or None if invalid or path traversal.
    """
    if not pdf_source_file or not pdf_source_file.strip():
        return None
    path = pdf_source_file.strip()
    base = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(path):
        resolved = os.path.normpath(path)
    else:
        resolved = os.path.normpath(os.path.join(base, path))
    try:
        base_real = os.path.realpath(base)
        resolved_real = os.path.realpath(resolved)
        if resolved_real != base_real and not resolved_real.startswith(base_real + os.sep):
            # Path escapes the app root (e.g. a stale Docker-container path like
            # /app/uploads/foo.pdf sent from a previous environment).  Try to
            # recover by looking for the bare filename in the local uploads dir.
            uploads_real = os.path.realpath(os.path.join(base_real, "uploads"))
            filename = os.path.basename(resolved)
            if filename:
                fallback = os.path.join(uploads_real, filename)
                if os.path.isfile(fallback):
                    logger.info(
                        "_resolve_pdf_path: stale absolute path '%s' recovered via "
                        "uploads fallback → '%s'",
                        path,
                        fallback,
                    )
                    return fallback
            return None
    except Exception:
        return None
    return resolved


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    """
    Serve the chat UI.

    Returns:
        HTMLResponse: The chat interface HTML page.
    """
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/pdf")
def serve_pdf(path: str = Query(..., description="Relative path to the PDF (uploads/ or mock_data/ only)")) -> FileResponse:
    """
    Serve a PDF file for the inline viewer. Only paths under uploads/ and
    mock_data/ are allowed — all other paths return 404 to prevent path traversal.

    Args:
        path: Relative path to the PDF (e.g. "uploads/report.pdf").

    Returns:
        FileResponse: The PDF file with application/pdf content type.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    allowed_prefixes = (
        os.path.join(base, "uploads"),
        os.path.join(base, "mock_data"),
    )
    candidate = os.path.normpath(os.path.join(base, path))
    if not any(candidate.startswith(p) for p in allowed_prefixes):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="PDF not found or access denied.")
    if not os.path.isfile(candidate):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(candidate, media_type="application/pdf")


@app.get("/health")
def health_check() -> dict:
    """
    Return service health status.

    Returns:
        dict: service, version, status, timestamp.
    """
    return {
        "service": SERVICE_NAME,
        "version": VERSION,
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Accept a natural language question, route through the LangGraph multi-agent
    state machine, and return a cited, verified response.

    If the session has a pending clarification (pending_user_input=True), the
    incoming question is treated as the clarification response and the workflow
    is resumed from the Extractor Node. Uses thread_id or session_id from the
    request so the frontend can send a consistent thread ID (e.g. from New Case).

    Args:
        request: AskRequest with question and optional thread_id/session_id.

    Returns:
        AskResponse: answer, session_id, thread_id, timestamp, escalate, confidence, disclaimer.
    """
    session_id = _get_session_id(request.thread_id or request.session_id)
    prior_state = _sessions.get(session_id)

    clarification_response = None
    query = request.question

    if prior_state and prior_state.get("pending_user_input"):
        # Treat the incoming message as the clarification answer, not a new query.
        # The router must see the ORIGINAL query so it classifies intent correctly —
        # a bare patient name like "John Smith" would otherwise be OUT_OF_SCOPE.
        clarification_response = request.question
        query = prior_state.get("input_query", request.question)

    # Resolve relative PDF path to absolute so the extractor finds the file in production
    # (CWD may differ from app root when deployed).
    pdf_path = _resolve_pdf_path(request.pdf_source_file)

    result = run_workflow(
        query=query,
        session_id=session_id,
        clarification_response=clarification_response,
        pdf_source_file=pdf_path,
        prior_state=prior_state,
        payer_id=request.payer_id,
        procedure_code=request.procedure_code,
    )

    _sessions[session_id] = result

    # Resolve the agent reply fields first so they are in scope for DB persistence below.
    answer = result.get("final_response") or result.get("clarification_needed") or "Unable to process request."
    confidence = result.get("confidence_score", 0.0)
    denial_risk_level = (result.get("denial_risk") or {}).get("risk_level", "")
    # Escalate on low confidence, partial results, OR HIGH/CRITICAL denial risk.
    # A confirmed CONTRAINDICATED interaction or critical documentation gap
    # always requires physician review regardless of confidence score.
    escalate = (
        confidence < CLINICAL_SAFETY_RULES["confidence_threshold"]
        or result.get("is_partial", False)
        or denial_risk_level in ("HIGH", "CRITICAL")
    )
    disclaimer = CLINICAL_SAFETY_RULES["disclaimer"]

    # Persist session metadata and message transcript. Wrapped in try/except
    # so a DB hiccup never breaks the /ask response.
    try:
        # identified_patient_name: set by Orchestrator Haiku on every request (Optional[str]).
        # query_intent: set by Router Node — MEDICATIONS | ALLERGIES | INTERACTIONS |
        #               SAFETY_CHECK | GENERAL_CLINICAL | OUT_OF_SCOPE (always a str).
        # extracted_patient: full patient dict from EHR lookup; "id" is the patient PID.
        patient_name = (result.get("identified_patient_name") or "").strip()
        patient_pid  = str((result.get("extracted_patient") or {}).get("id", ""))
        intent       = str(result.get("query_intent", "") or "")
        database.upsert_session(
            session_id,
            patient_name=patient_name,
            patient_pid=patient_pid,
            query_summary=query[:80],
            intent=intent,
        )
        # Store user turn — use request.question (what the user typed),
        # and the original relative pdf path (not the resolved absolute path).
        database.insert_message(
            session_id,
            role="user",
            content=request.question,
            metadata=None,
            pdf_path=request.pdf_source_file or "",
        )
        # Store agent turn with full rich metadata for transcript replay.
        database.insert_message(
            session_id,
            role="agent",
            content=answer,
            metadata={
                "confidence":       confidence,
                "escalate":         escalate,
                "disclaimer":       disclaimer,
                "tool_trace":       result.get("tool_trace", []),
                "denial_risk":      result.get("denial_risk") or {},
                "citation_anchors": result.get("citation_anchors") or [],
            },
            pdf_path="",
        )
    except Exception:
        pass

    # Scrubbed version of the user's question (what was sent to the agent after PII redaction).
    # Always a non-null string so the UI can show the Privacy block (redacted text or fallback message).
    _REDACT_UNAVAILABLE = "[Redaction unavailable for this message.]"
    try:
        query_redacted_preview = scrub_pii(request.question)
        if query_redacted_preview is None or (isinstance(query_redacted_preview, str) and not query_redacted_preview.strip()):
            query_redacted_preview = _REDACT_UNAVAILABLE
    except Exception as e:
        logging.warning("PII scrubber failed, query_redacted_preview omitted: %s", e, exc_info=True)
        query_redacted_preview = _REDACT_UNAVAILABLE

    return AskResponse(
        answer=answer,
        session_id=session_id,
        thread_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        escalate=escalate,
        confidence=confidence,
        disclaimer=disclaimer,
        tool_trace=result.get("tool_trace", []),
        denial_risk=result.get("denial_risk") or {},
        citation_anchors=result.get("citation_anchors") or [],
        query_redacted_preview=query_redacted_preview or _REDACT_UNAVAILABLE,
    )


def _build_ask_response_dict(result: dict, session_id: str, request_question: str) -> dict:
    """Build the same dict shape as AskResponse for streaming 'done' event or reuse."""
    answer = result.get("final_response") or result.get("clarification_needed") or "Unable to process request."
    confidence = result.get("confidence_score", 0.0)
    denial_risk_level = (result.get("denial_risk") or {}).get("risk_level", "")
    escalate = (
        confidence < CLINICAL_SAFETY_RULES["confidence_threshold"]
        or result.get("is_partial", False)
        or denial_risk_level in ("HIGH", "CRITICAL")
    )
    disclaimer = CLINICAL_SAFETY_RULES["disclaimer"]
    _REDACT_UNAVAILABLE = "[Redaction unavailable for this message.]"
    try:
        query_redacted_preview = scrub_pii(request_question)
        if query_redacted_preview is None or (isinstance(query_redacted_preview, str) and not query_redacted_preview.strip()):
            query_redacted_preview = _REDACT_UNAVAILABLE
    except Exception as e:
        logging.warning("PII scrubber failed, query_redacted_preview omitted: %s", e, exc_info=True)
        query_redacted_preview = _REDACT_UNAVAILABLE
    return {
        "answer": answer,
        "session_id": session_id,
        "thread_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "escalate": escalate,
        "confidence": confidence,
        "disclaimer": disclaimer,
        "tool_trace": result.get("tool_trace", []),
        "denial_risk": result.get("denial_risk") or {},
        "citation_anchors": result.get("citation_anchors") or [],
        "query_redacted_preview": query_redacted_preview or _REDACT_UNAVAILABLE,
    }


async def _stream_ask_events(request: AskRequest):
    """Async generator yielding SSE event strings for POST /ask/stream."""
    session_id = _get_session_id(request.thread_id or request.session_id)
    prior_state = _sessions.get(session_id)
    clarification_response = None
    query = request.question
    if prior_state and prior_state.get("pending_user_input"):
        clarification_response = request.question
        query = prior_state.get("input_query", request.question)
    pdf_path = _resolve_pdf_path(request.pdf_source_file)

    try:
        async for event in stream_workflow(
            query=query,
            session_id=session_id,
            clarification_response=clarification_response,
            pdf_source_file=pdf_path,
            prior_state=prior_state,
            payer_id=request.payer_id,
            procedure_code=request.procedure_code,
        ):
            if event.get("event") == "node":
                yield f"event: node\ndata: {json.dumps({'summary': event.get('summary', '')})}\n\n"
            elif event.get("event") == "done":
                result = event.get("state", {})
                _sessions[session_id] = result
                done_payload = _build_ask_response_dict(result, session_id, request.question)
                try:
                    patient_name = (result.get("identified_patient_name") or "").strip()
                    patient_pid = str((result.get("extracted_patient") or {}).get("id", ""))
                    intent = str(result.get("query_intent", "") or "")
                    database.upsert_session(
                        session_id,
                        patient_name=patient_name,
                        patient_pid=patient_pid,
                        query_summary=query[:80],
                        intent=intent,
                    )
                    database.insert_message(
                        session_id,
                        role="user",
                        content=request.question,
                        metadata=None,
                        pdf_path=request.pdf_source_file or "",
                    )
                    database.insert_message(
                        session_id,
                        role="agent",
                        content=done_payload["answer"],
                        metadata={
                            "confidence": done_payload["confidence"],
                            "escalate": done_payload["escalate"],
                            "disclaimer": done_payload["disclaimer"],
                            "tool_trace": done_payload["tool_trace"],
                            "denial_risk": done_payload["denial_risk"],
                            "citation_anchors": done_payload["citation_anchors"],
                        },
                        pdf_path="",
                    )
                except Exception:
                    pass
                yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
            elif event.get("event") == "error":
                yield f"event: error\ndata: {json.dumps({'error': event.get('error', 'Unknown error')})}\n\n"
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': f'An unexpected error occurred: {str(e)}'})}\n\n"


@app.post("/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    """
    Same as POST /ask but streams progress events (SSE) as each node completes,
    then sends a final 'done' event with the full response or 'error' on failure.
    """
    return StreamingResponse(
        _stream_ask_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history")
def get_history(limit: int = Query(default=30, ge=1, le=100)) -> list:
    """
    Return recent agent sessions for the audit history sidebar.

    Each entry contains session_id, patient_name, patient_pid, query_summary,
    intent, created_at, and updated_at — enough for the sidebar to render a
    labelled entry and resume the session when clicked.

    Args:
        limit: Maximum sessions to return (1–100, default 30).

    Returns:
        list: Session metadata dicts ordered by updated_at descending.
    """
    try:
        return database.get_recent_sessions(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history/{session_id}/messages")
def get_session_messages(session_id: str) -> list:
    """
    Return the full message transcript for a session, ordered chronologically.

    Used by the OpenEMR sidebar to replay chat history when the user clicks
    a past session entry. Each entry contains role (user/agent), content,
    metadata (agent rich data), pdf_path, and created_at.

    Args:
        session_id: The session / thread ID to fetch.

    Returns:
        list: Message dicts ordered oldest-first.
    """
    try:
        return database.get_session_messages(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/audit/{thread_id}")
def get_audit_trail(
    thread_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> dict:
    """
    Return the audit trail for a given thread (messages, extractions, tool calls, etc.).
    Requires Bearer token matching AUDIT_TOKEN env var. Tries in-memory session store
    first, then falls back to SQLite checkpointer (e.g. after server restart).

    Args:
        thread_id: Session/thread identifier.
        authorization: Header "Authorization: Bearer <AUDIT_TOKEN>".

    Returns:
        dict: thread_id, messages, extractions, audit_results, confidence_score,
              tool_call_history, ehr_confidence_penalty.

    Raises:
        HTTPException 403: Missing or invalid Authorization.
        HTTPException 404: Thread not found.
    """
    audit_token = os.getenv("AUDIT_TOKEN", "").strip()
    if not audit_token:
        raise HTTPException(status_code=501, detail="Audit endpoint not configured (AUDIT_TOKEN not set).")
    if not authorization or not authorization.strip().lower().startswith("bearer "):
        raise HTTPException(status_code=403, detail="Unauthorized: missing or invalid Authorization header.")
    token_value = authorization.strip()[7:].strip()  # after "Bearer "
    if token_value != audit_token:
        raise HTTPException(status_code=403, detail="Unauthorized.")

    state = _sessions.get(thread_id)
    if not state:
        state = get_state_for_audit(thread_id)

    if not state:
        raise HTTPException(status_code=404, detail="Thread not found.")

    return {
        "thread_id": thread_id,
        "messages": state.get("messages", []),
        "extractions": state.get("extractions", []),
        "audit_results": state.get("audit_results", []),
        "confidence_score": state.get("confidence_score"),
        "tool_call_history": state.get("tool_call_history", []),
        "ehr_confidence_penalty": state.get("ehr_confidence_penalty", 0),
    }


class SaveMessageRequest(BaseModel):
    """Request body for POST /save-message (navigator.sendBeacon on tab close)."""
    session_id: str
    content: str
    role: str = "user"


@app.post("/save-message")
def save_message(request: SaveMessageRequest) -> dict:
    """
    Persist a single message without running the agent workflow.

    Called via navigator.sendBeacon when the tab is closed mid-request so
    the user's in-flight question is not silently lost. The session row is
    upserted with minimal metadata if it does not already exist.

    Args:
        request: SaveMessageRequest with session_id, content, and role.

    Returns:
        dict: {"ok": bool}
    """
    try:
        session_id = (request.session_id or "").strip()
        content    = (request.content or "").strip()
        role       = request.role if request.role in ("user", "agent") else "user"
        if not session_id or not content:
            return {"ok": False}
        database.upsert_session(
            session_id,
            patient_name="",
            patient_pid="",
            query_summary=content[:80],
            intent="",
        )
        database.insert_message(session_id, role=role, content=content, metadata=None, pdf_path="")
        return {"ok": True}
    except Exception:
        return {"ok": False}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    """
    Accept a PDF upload from the UI, save it to the uploads/ directory,
    and return the server-side path for use in subsequent /ask requests.

    Args:
        file: Uploaded PDF file (multipart/form-data).

    Returns:
        dict: {"success": bool, "path": str, "filename": str, "error": str | None}

    Raises:
        Never — all failures return a structured error dict.
    """
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            return {"success": False, "path": "", "filename": "", "error": "Only PDF files are accepted."}
        safe_name = re.sub(r"[^\w.\-]", "_", file.filename)
        save_path = os.path.join(UPLOADS_DIR, safe_name)
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
        relative_path = os.path.join("uploads", safe_name)
        return {"success": True, "path": relative_path, "filename": safe_name, "error": None}
    except Exception as e:
        return {"success": False, "path": "", "filename": "", "error": str(e)}


@app.post("/eval")
def run_eval_endpoint() -> dict:
    """
    Run the full eval suite against golden_data.yaml and return results.

    Returns:
        dict: total, passed, failed, pass_rate, results, timestamp.
    """
    try:
        result = run_eval(
            test_cases_path=DEFAULT_GOLDEN_DATA_PATH,
            save_results=True,
            results_dir=RESULTS_DIR,
        )
        return result
    except Exception as e:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "results": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }


@app.get("/eval/results")
def get_eval_results() -> dict:
    """
    Return the most recently saved eval results from tests/results/.

    Returns:
        dict: Latest eval result, or message if none exist.
    """
    results = _load_latest_results(RESULTS_DIR)
    if results is None:
        return {
            "message": "No eval results found. Run POST /eval to generate results.",
        }
    return results
