"""
main.py
-------
AgentForge — Healthcare RCM AI Agent — FastAPI server
------------------------------------------------------
Exposes the healthcare AI agent as a REST API. Manages conversation
sessions in-memory, wires conversation and verification layers, and
provides endpoints for health check, natural language queries, eval
execution, and eval result retrieval.

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

from conversation import create_conversation_agent, chat
from verification import check_allergy_conflict, calculate_confidence, should_escalate_to_human
from healthcare_guidelines import CLINICAL_SAFETY_RULES
from eval.run_eval import run_eval, DEFAULT_GOLDEN_DATA_PATH

# ── Config ─────────────────────────────────────────────────────────────────────

VERSION = "1.0.0"
SERVICE_NAME = "AgentForge Healthcare RCM AI Agent"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "results")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# ── In-memory session store ────────────────────────────────────────────────────
# Keyed by session_id. Each value is (agent, history).
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_or_create_session(session_id: Optional[str]):
    """
    Retrieve existing session or create a new one.

    Args:
        session_id: Caller-supplied session ID or None.

    Returns:
        Tuple[str, agent, List]: (session_id, agent, history)
    """
    if session_id and session_id in _sessions:
        agent, history = _sessions[session_id]
        return session_id, agent, history
    new_id = session_id or str(uuid.uuid4())
    agent, history = create_conversation_agent()
    _sessions[new_id] = (agent, history)
    return new_id, agent, history


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
    Accept a natural language question, route through the agent with
    conversation history, apply verification, and return enriched response.

    Args:
        request: AskRequest with question and optional session_id.

    Returns:
        AskResponse: answer, session_id, timestamp, escalate, confidence, disclaimer.
    """
    session_id, agent, history = _get_or_create_session(request.session_id)

    answer, new_history = chat(agent, history, request.question)

    # Persist updated history back to session store.
    _sessions[session_id] = (agent, new_history)

    # Verification: naive confidence based on whether the response is healthy.
    # tools_succeeded = 3 if response looks like real data, else 1.
    tools_total = 3
    tools_succeeded = 3 if len(answer) > 50 else 1
    interactions_found = any(word in answer.lower() for word in ["interaction", "high severity", "contraindicated"])
    allergy_conflict = any(word in answer.lower() for word in ["allergy conflict", "known allergy"])

    confidence = calculate_confidence(tools_succeeded, tools_total, interactions_found, allergy_conflict)
    escalation = should_escalate_to_human(confidence)
    disclaimer = CLINICAL_SAFETY_RULES["disclaimer"]

    return AskResponse(
        answer=answer,
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        escalate=escalation.get("escalate", False),
        confidence=confidence,
        disclaimer=disclaimer,
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
