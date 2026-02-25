"""
test_langgraph_workflow.py
--------------------------
AgentForge — Healthcare RCM AI Agent — Tests for assembled LangGraph workflow
------------------------------------------------------------------------------
End-to-end tests of the full LangGraph state machine. All node functions
are mocked. Tests routing paths, review loop enforcement, clarification
pause/resume, session isolation, and error handling.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
from unittest.mock import patch
from langgraph_agent.workflow import run_workflow


# ── Mock node functions ────────────────────────────────────────────────────────

def _make_extractor(state):
    """Mock extractor: populates extractions with a valid verbatim citation."""
    state["extractions"] = [
        {
            "claim": "John Smith is prescribed Metformin 500mg twice daily.",
            "citation": "Metformin 500mg twice daily",
            "source": "mock_data/medications.json",
            "verbatim": True
        }
    ]
    state["tool_trace"] = [
        {"tool": "tool_get_patient_info", "input": "John Smith"},
        {"tool": "tool_get_medications", "input": "P001"}
    ]
    return state


def _make_passing_auditor(state):
    """Mock auditor: always passes extractions as valid."""
    state["routing_decision"] = "pass"
    state["audit_results"] = [{"validated": True}]
    state["confidence_score"] = 0.95
    state["final_response"] = "John Smith is taking Metformin 500mg twice daily. Source: mock_data/medications.json"
    return state


def _make_failing_then_passing_auditor(call_tracker):
    """Mock auditor: fails once, then passes — simulates one review loop cycle."""
    def _auditor(state):
        call_tracker["count"] += 1
        if call_tracker["count"] == 1:
            state["routing_decision"] = "missing"
            state["iteration_count"] += 1
        else:
            state["routing_decision"] = "pass"
            state["confidence_score"] = 0.92
            state["final_response"] = "Verified response with citation."
        return state
    return _auditor


def _make_always_failing_auditor(state):
    """Mock auditor: always rejects, triggering ceiling after 3 iterations."""
    max_iterations = 3
    if state["iteration_count"] >= max_iterations:
        state["routing_decision"] = "partial"
        state["is_partial"] = True
        state["insufficient_documentation_flags"] = [
            "Insufficient Documentation: citation missing for claim 1"
        ]
        state["final_response"] = "Partial results — citation could not be verified after 3 attempts."
    else:
        state["routing_decision"] = "missing"
        state["iteration_count"] += 1
    return state


def _make_ambiguous_auditor(state):
    """Mock auditor: detects ambiguity and triggers Clarification Node."""
    state["routing_decision"] = "ambiguous"
    state["pending_user_input"] = True
    state["clarification_needed"] = "Which patient — John Smith or John Doe?"
    return state


# ── Tests: happy path ─────────────────────────────────────────────────────────

class TestWorkflowHappyPath:
    def test_workflow_happy_path(self):
        """Known patient query flows Extractor → Auditor → Output with correct response shape."""
        with patch("langgraph_agent.workflow.extractor_node", side_effect=_make_extractor), \
             patch("langgraph_agent.workflow.auditor_node", side_effect=_make_passing_auditor):
            result = run_workflow("What medications is John Smith taking?", session_id="test-happy-1")
        assert "final_response" in result
        assert len(result["final_response"]) > 0
        assert "confidence_score" in result
        assert "tool_trace" in result


# ── Tests: review loop ────────────────────────────────────────────────────────

class TestWorkflowReviewLoop:
    def test_workflow_review_loop_one_cycle(self):
        """Auditor rejects once, Extractor re-runs, Auditor passes. Final iteration_count is 1."""
        call_tracker = {"count": 0}
        with patch("langgraph_agent.workflow.extractor_node", side_effect=_make_extractor), \
             patch("langgraph_agent.workflow.auditor_node", side_effect=_make_failing_then_passing_auditor(call_tracker)):
            result = run_workflow("What medications is John Smith taking?", session_id="test-loop-1")
        assert result["iteration_count"] == 1
        assert result["routing_decision"] == "pass"

    def test_workflow_caps_review_loop_at_three(self):
        """After 3 Extractor→Auditor cycles, returns partial response with Insufficient Documentation."""
        with patch("langgraph_agent.workflow.extractor_node", side_effect=_make_extractor), \
             patch("langgraph_agent.workflow.auditor_node", side_effect=_make_always_failing_auditor):
            result = run_workflow("What medications is John Smith taking?", session_id="test-cap-1")
        assert result["is_partial"] is True
        assert result["iteration_count"] == 3
        assert any(
            "Insufficient Documentation" in flag
            for flag in result["insufficient_documentation_flags"]
        )


# ── Tests: session isolation ──────────────────────────────────────────────────

class TestWorkflowSessionIsolation:
    def test_workflow_state_reset_between_sessions(self):
        """A new session starts with iteration_count=0 — no state leaks from a previous session."""
        # Session A: hit the 3-iteration ceiling
        with patch("langgraph_agent.workflow.extractor_node", side_effect=_make_extractor), \
             patch("langgraph_agent.workflow.auditor_node", side_effect=_make_always_failing_auditor):
            result_a = run_workflow("What meds is John Smith on?", session_id="session-A")
        assert result_a["iteration_count"] == 3

        # Session B: fresh session must start at 0, not inherit session A's count
        with patch("langgraph_agent.workflow.extractor_node", side_effect=_make_extractor), \
             patch("langgraph_agent.workflow.auditor_node", side_effect=_make_passing_auditor):
            result_b = run_workflow("What meds is Mary Johnson on?", session_id="session-B")
        assert result_b["iteration_count"] == 0


# ── Tests: clarification pause/resume ────────────────────────────────────────

class TestWorkflowClarification:
    def test_workflow_pauses_on_ambiguous_input(self):
        """Ambiguous patient query causes workflow to pause with pending_user_input=True."""
        with patch("langgraph_agent.workflow.extractor_node", side_effect=_make_extractor), \
             patch("langgraph_agent.workflow.auditor_node", side_effect=_make_ambiguous_auditor):
            result = run_workflow("What medications is John taking?", session_id="test-ambig-1")
        assert result["pending_user_input"] is True
        assert len(result["clarification_needed"]) > 0

    def test_workflow_resumes_after_clarification(self):
        """Providing clarification_response resumes workflow from Extractor and reaches final response.

        When clarification_response is set in run_workflow(), it is written into state
        via resume_from_clarification() before the graph runs. The Extractor re-runs
        with the clarification context available, and the Auditor passes this time.
        """
        def _passing_auditor_for_resume(state):
            # On resume, clarification_response is in state — Auditor passes cleanly.
            state["routing_decision"] = "pass"
            state["pending_user_input"] = False
            state["final_response"] = "John Smith is taking Metformin."
            state["confidence_score"] = 0.95
            return state

        with patch("langgraph_agent.workflow.extractor_node", side_effect=_make_extractor), \
             patch("langgraph_agent.workflow.auditor_node", side_effect=_passing_auditor_for_resume):
            result = run_workflow(
                "What medications is John taking?",
                session_id="test-resume-1",
                clarification_response="John Smith"
            )
        assert result["pending_user_input"] is False
        assert len(result["final_response"]) > 0


# ── Tests: error handling ─────────────────────────────────────────────────────

class TestWorkflowErrorHandling:
    def test_workflow_returns_no_raw_exceptions(self):
        """If the Extractor crashes, run_workflow returns a structured error dict — never raises."""
        def _crashing_extractor(state):
            raise RuntimeError("Simulated extractor crash")

        with patch("langgraph_agent.workflow.extractor_node", side_effect=_crashing_extractor):
            result = run_workflow("What medications is John Smith taking?", session_id="test-crash-1")
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] is not None
