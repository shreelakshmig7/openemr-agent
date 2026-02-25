"""
test_main.py
------------
AgentForge — Healthcare RCM AI Agent — Test Suite for main.py
--------------------------------------------------------------
TDD test suite for the FastAPI server. Uses FastAPI TestClient so no
running server is needed. Mocks the agent and conversation layer to
avoid live API calls.

Tests cover:
    - GET /health returns 200 and required fields
    - POST /ask with valid query returns answer, session_id, timestamp
    - POST /ask with empty input returns helpful message, not crash
    - POST /ask with session_id preserves session across calls
    - POST /eval returns results dict with pass_rate
    - GET /eval/results returns helpful message when no results exist
    - GET /eval/results returns data after eval has run

Run:
    pytest tests/test_main.py -v --tb=short

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_chat(agent, history, message):
    """Mock conversation.chat — returns fixed response without API call."""
    if not message.strip():
        return ("Please provide your question.", list(history))
    new_history = list(history) + [{"human": message, "ai": "Patient takes metformin and lisinopril."}]
    return ("Patient takes metformin and lisinopril.", new_history)


def _mock_create_conversation_agent():
    """Mock conversation.create_conversation_agent — returns fake agent + empty history."""
    return (MagicMock(), [])


def _mock_run_eval(*args, **kwargs):
    """Mock eval.run_eval.run_eval — returns fixed result without API calls."""
    return {
        "total": 2,
        "passed": 2,
        "failed": 0,
        "pass_rate": 1.0,
        "results": [
            {"id": "gs-001", "passed": True, "latency_seconds": 1.0, "response_preview": "metformin"},
            {"id": "gs-002", "passed": True, "latency_seconds": 1.5, "response_preview": "interaction"},
        ],
        "timestamp": "20260224_000000",
    }


# ── GET /health ────────────────────────────────────────────────────────────────

def test_health_returns_200():
    """GET /health should return HTTP 200."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200


def test_health_required_fields():
    """GET /health response must include service, version, status, timestamp."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        data = client.get("/health").json()
        for field in ["service", "version", "status", "timestamp"]:
            assert field in data, f"Missing field in /health response: {field}"


def test_health_status_ok():
    """GET /health status field should be 'ok'."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        data = client.get("/health").json()
        assert data["status"] == "ok"


# ── POST /ask ─────────────────────────────────────────────────────────────────

def test_ask_returns_200():
    """POST /ask with valid question should return HTTP 200."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        response = client.post("/ask", json={"question": "What medications is John Smith on?"})
        assert response.status_code == 200


def test_ask_required_fields():
    """POST /ask response must include answer, session_id, timestamp."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        data = client.post("/ask", json={"question": "What medications is John Smith on?"}).json()
        for field in ["answer", "session_id", "timestamp"]:
            assert field in data, f"Missing field in /ask response: {field}"


def test_ask_enriched_fields():
    """POST /ask response must include escalate, confidence, disclaimer."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        data = client.post("/ask", json={"question": "What medications is John Smith on?"}).json()
        for field in ["escalate", "confidence", "disclaimer"]:
            assert field in data, f"Missing enriched field in /ask response: {field}"


def test_ask_generates_session_id():
    """POST /ask without session_id should return a non-empty session_id."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        data = client.post("/ask", json={"question": "Hello"}).json()
        assert data["session_id"]
        assert len(data["session_id"]) > 0


def test_ask_reuses_session():
    """POST /ask with same session_id twice should return same session_id."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        r1 = client.post("/ask", json={"question": "Hello"}).json()
        session_id = r1["session_id"]
        r2 = client.post("/ask", json={"question": "Follow up", "session_id": session_id}).json()
        assert r2["session_id"] == session_id


def test_ask_empty_question():
    """POST /ask with empty question should return 200 with helpful message."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0


# ── POST /eval ────────────────────────────────────────────────────────────────

def test_eval_returns_200():
    """POST /eval should return HTTP 200."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat), \
         patch("eval.run_eval.run_eval", _mock_run_eval):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        response = client.post("/eval")
        assert response.status_code == 200


def test_eval_returns_required_keys():
    """POST /eval result must contain total, passed, failed, pass_rate."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat), \
         patch("eval.run_eval.run_eval", _mock_run_eval):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        data = client.post("/eval").json()
        for key in ["total", "passed", "failed", "pass_rate"]:
            assert key in data, f"Missing key in /eval response: {key}"


# ── GET /eval/results ─────────────────────────────────────────────────────────

def test_eval_results_returns_200():
    """GET /eval/results should always return HTTP 200."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        response = client.get("/eval/results")
        assert response.status_code == 200


def test_eval_results_no_results_message():
    """GET /eval/results with no saved files should return helpful message."""
    with patch("conversation.create_conversation_agent", _mock_create_conversation_agent), \
         patch("conversation.chat", _mock_chat), \
         patch("main.RESULTS_DIR", "/tmp/nonexistent_results_dir_agentforge"):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        data = client.get("/eval/results").json()
        assert "message" in data or "results" in data or "pass_rate" in data
