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
    GET  /health        — Service health check
    POST /ask           — Natural language query with session context
    POST /eval          — Run eval suite and return results
    GET  /eval/results  — Return latest saved eval results

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import os
import re
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from langgraph_agent.workflow import run_workflow
from healthcare_guidelines import CLINICAL_SAFETY_RULES
from eval.run_eval import run_eval, DEFAULT_GOLDEN_DATA_PATH

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

VERSION = "2.0.0"
SERVICE_NAME = "AgentForge Healthcare RCM AI Agent"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "results")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

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


# ── Request / Response models ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request body for POST /ask."""
    question: str
    session_id: Optional[str] = None
    pdf_source_file: Optional[str] = None


class AskResponse(BaseModel):
    """Response body for POST /ask."""
    answer: str
    session_id: str
    timestamp: str
    escalate: bool
    confidence: float
    disclaimer: str
    tool_trace: list
    denial_risk: dict
    citation_anchors: list


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


def _query_contains_patient_identifier(question: str) -> bool:
    """
    Return True if the question contains a patient ID (P001-style) or
    a two-word capitalized name. Used to decide whether to prepend
    session context for follow-up questions.

    Args:
        question: Raw user question.

    Returns:
        bool: True if a patient identifier appears in the question.
    """
    if not question or not question.strip():
        return False
    if re.search(r"\b[Pp]\d{3,}\b", question):
        return True
    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", question.strip()):
        return True
    return False


def _get_last_patient_from_state(prior_state: dict) -> Optional[str]:
    """
    Extract the last resolved patient name or ID from the prior workflow result.
    Prefers extracted_patient_identifier.value (full name/ID from Step 0) so
    prepend never uses a truncated first name; falls back to tool_trace.

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
    try:
        trace = prior_state.get("tool_trace") or []
        for call in trace:
            if call.get("tool") == "tool_get_patient_info":
                inp = call.get("input")
                if inp and isinstance(inp, str) and inp.strip():
                    return inp.strip()
                break
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
    is resumed from the Extractor Node.

    Args:
        request: AskRequest with question and optional session_id.

    Returns:
        AskResponse: answer, session_id, timestamp, escalate, confidence, disclaimer.
    """
    session_id = _get_session_id(request.session_id)
    prior_state = _sessions.get(session_id)

    clarification_response = None
    query_to_use = request.question

    if prior_state and prior_state.get("pending_user_input"):
        clarification_response = request.question
    else:
        if prior_state and not _query_contains_patient_identifier(request.question):
            last_patient = _get_last_patient_from_state(prior_state)
            if last_patient:
                query_to_use = f"Regarding {last_patient}: {request.question}"

    result = run_workflow(
        query=query_to_use,
        session_id=session_id,
        clarification_response=clarification_response,
        pdf_source_file=request.pdf_source_file,
    )

    _sessions[session_id] = result

    answer = result.get("final_response") or result.get("clarification_needed") or "Unable to process request."
    confidence = result.get("confidence_score", 0.0)
    escalate = confidence < CLINICAL_SAFETY_RULES["confidence_threshold"] or result.get("is_partial", False)
    disclaimer = CLINICAL_SAFETY_RULES["disclaimer"]

    return AskResponse(
        answer=answer,
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        escalate=escalate,
        confidence=confidence,
        disclaimer=disclaimer,
        tool_trace=result.get("tool_trace", []),
        denial_risk=result.get("denial_risk") or {},
        citation_anchors=result.get("citation_anchors") or [],
    )


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
