"""
test_langgraph_clarification.py
--------------------------------
AgentForge — Healthcare RCM AI Agent — Tests for Clarification Node
--------------------------------------------------------------------
Tests that the Clarification Node correctly pauses the workflow,
preserves all existing state (extractions, iteration_count), scrubs
PII from the clarification question, and resumes correctly back
to the Extractor with pending_user_input cleared.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
from langgraph_agent.state import create_initial_state
from langgraph_agent.clarification_node import clarification_node, resume_from_clarification


def make_paused_state(extractions=None, iteration_count=0, clarification_needed=""):
    """Create a state that is paused awaiting user clarification."""
    state = create_initial_state("test query")
    state["pending_user_input"] = True
    state["clarification_needed"] = clarification_needed or "Which John — Smith or Doe?"
    state["extractions"] = extractions or [
        {
            "claim": "Partial extraction before ambiguity",
            "citation": "verbatim quote from source",
            "source": "mock_data/patients.json",
            "verbatim": True
        }
    ]
    state["iteration_count"] = iteration_count
    return state


# ── Tests: pause behavior ─────────────────────────────────────────────────────

class TestClarificationPause:
    def test_clarification_sets_pending_flag(self):
        """Clarification Node ensures pending_user_input remains True."""
        state = make_paused_state()
        result = clarification_node(state)
        assert result["pending_user_input"] is True

    def test_clarification_question_contains_no_pii(self):
        """Stub PII scrubber removes sensitive patterns (SSN) from clarification question."""
        state = make_paused_state(
            clarification_needed="John Smith SSN: 123-45-6789 or John Doe SSN: 987-65-4321?"
        )
        result = clarification_node(state)
        question = result["clarification_needed"]
        assert "123-45-6789" not in question
        assert "987-65-4321" not in question

    def test_clarification_preserves_existing_extractions(self):
        """Extractions already in state are not cleared when Clarification Node runs."""
        existing = [
            {
                "claim": "Already extracted data",
                "citation": "verbatim quote",
                "source": "mock_data/medications.json",
                "verbatim": True
            }
        ]
        state = make_paused_state(extractions=existing)
        result = clarification_node(state)
        assert result["extractions"] == existing

    def test_clarification_preserves_iteration_count(self):
        """iteration_count is not reset or changed when Clarification Node runs."""
        state = make_paused_state(iteration_count=1)
        result = clarification_node(state)
        assert result["iteration_count"] == 1


# ── Tests: resume behavior ────────────────────────────────────────────────────

class TestClarificationResume:
    def test_resume_clears_pending_flag(self):
        """After user responds, pending_user_input is set to False."""
        state = make_paused_state()
        result = resume_from_clarification(state, user_response="John Smith")
        assert result["pending_user_input"] is False

    def test_resume_writes_response_into_state(self):
        """User's clarification response is written into clarification_response field."""
        state = make_paused_state()
        result = resume_from_clarification(state, user_response="John Smith")
        assert result["clarification_response"] == "John Smith"

    def test_resume_routes_back_to_extractor_not_auditor(self):
        """After resume, routing_decision is 'extractor' so workflow re-runs Extractor with context."""
        state = make_paused_state()
        result = resume_from_clarification(state, user_response="John Smith")
        assert result["routing_decision"] == "extractor"
