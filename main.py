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
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langgraph_agent.workflow import run_workflow
from healthcare_guidelines import CLINICAL_SAFETY_RULES
from eval.run_eval import run_eval, DEFAULT_GOLDEN_DATA_PATH

# ── Config ─────────────────────────────────────────────────────────────────────

VERSION = "2.0.0"
SERVICE_NAME = "AgentForge Healthcare RCM AI Agent"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "results")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

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


class AskResponse(BaseModel):
    """Response body for POST /ask."""
    answer: str
    session_id: str
    timestamp: str
    escalate: bool
    confidence: float
    disclaimer: str
    tool_trace: list


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
    if prior_state and prior_state.get("pending_user_input"):
        clarification_response = request.question

    result = run_workflow(
        query=request.question,
        session_id=session_id,
        clarification_response=clarification_response,
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
    )


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
