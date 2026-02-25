"""
test_langgraph_auditor.py
-------------------------
AgentForge — Healthcare RCM AI Agent — Tests for Auditor Node
--------------------------------------------------------------
Tests that the Auditor Node correctly validates extractions, sets
routing_decision, enforces the 3-iteration ceiling, detects hallucinated
facts, and handles ambiguous patient cases. No real API calls.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
from unittest.mock import patch
from langgraph_agent.state import create_initial_state
from langgraph_agent.auditor_node import auditor_node
from healthcare_guidelines import HIPAA_RULES


# ── Fixture extractions ────────────────────────────────────────────────────────

VALID_EXTRACTION = {
    "claim": "John Smith is prescribed Metformin 500mg twice daily.",
    "citation": "Metformin 500mg twice daily",
    "source": "mock_data/medications.json",
    "verbatim": True
}

MISSING_CITATION_EXTRACTION = {
    "claim": "John Smith takes Metformin.",
    "citation": "",
    "source": "",
    "verbatim": False
}

NON_VERBATIM_EXTRACTION = {
    "claim": "John Smith takes a diabetes medication twice a day.",
    "citation": "Patient is on diabetes management",
    "source": "mock_data/medications.json",
    "verbatim": False
}

HALLUCINATED_EXTRACTION = {
    "claim": "John Smith has failed conservative therapy.",
    "citation": "patient showed resistance to initial treatment",
    "source": "mock_data/medications.json",
    "verbatim": True
}

AMBIGUOUS_EXTRACTION = {
    "claim": "AMBIGUOUS: Multiple patients found matching 'John'",
    "citation": "",
    "source": "",
    "verbatim": False,
    "ambiguous": True
}


def make_state(extractions, iteration_count=0):
    """Create a test state with given extractions and iteration_count."""
    state = create_initial_state("test query")
    state["extractions"] = extractions
    state["iteration_count"] = iteration_count
    return state


# ── Tests: pass path ──────────────────────────────────────────────────────────

class TestAuditorRoutingPass:
    def test_auditor_passes_valid_extractions(self):
        """Auditor sets routing_decision='pass' when all claims have verbatim citations."""
        state = make_state([VALID_EXTRACTION])
        with patch("langgraph_agent.auditor_node._verify_citation_exists_in_source", return_value=True):
            result = auditor_node(state)
        assert result["routing_decision"] == "pass"


# ── Tests: review loop ────────────────────────────────────────────────────────

class TestAuditorRoutingReviewLoop:
    def test_auditor_sends_back_when_missing_citations(self):
        """Auditor sets routing_decision='missing' when a claim has no citation."""
        state = make_state([MISSING_CITATION_EXTRACTION])
        result = auditor_node(state)
        assert result["routing_decision"] == "missing"

    def test_auditor_sends_back_on_non_verbatim_citation(self):
        """Auditor sets routing_decision='missing' when citation is a paraphrase, not verbatim."""
        state = make_state([NON_VERBATIM_EXTRACTION])
        result = auditor_node(state)
        assert result["routing_decision"] == "missing"

    def test_auditor_detects_hallucinated_fact(self):
        """Auditor flags a claim whose citation does not exist verbatim in the source."""
        state = make_state([HALLUCINATED_EXTRACTION])
        with patch("langgraph_agent.auditor_node._verify_citation_exists_in_source", return_value=False):
            result = auditor_node(state)
        assert result["routing_decision"] == "missing"

    def test_auditor_increments_iteration_count_on_sendback(self):
        """iteration_count increases by 1 each time Auditor sends back to Extractor."""
        state = make_state([MISSING_CITATION_EXTRACTION], iteration_count=0)
        result = auditor_node(state)
        assert result["iteration_count"] == 1

    def test_auditor_recovery_on_second_try(self):
        """After one loop back, Extractor fixes citations, Auditor passes with iteration_count=1."""
        state = make_state([VALID_EXTRACTION], iteration_count=1)
        with patch("langgraph_agent.auditor_node._verify_citation_exists_in_source", return_value=True):
            result = auditor_node(state)
        assert result["routing_decision"] == "pass"
        assert result["iteration_count"] == 1


# ── Tests: iteration ceiling ──────────────────────────────────────────────────

class TestAuditorIterationCeiling:
    def test_auditor_caps_at_three_iterations(self):
        """When iteration_count==3, Auditor routes to 'partial' and does not loop again."""
        state = make_state([MISSING_CITATION_EXTRACTION], iteration_count=3)
        result = auditor_node(state)
        assert result["routing_decision"] == "partial"
        assert result["iteration_count"] == 3

    def test_auditor_ceiling_flags_insufficient_documentation(self):
        """At iteration ceiling, output includes is_partial=True and 'Insufficient Documentation' flags."""
        state = make_state([MISSING_CITATION_EXTRACTION], iteration_count=3)
        result = auditor_node(state)
        assert result["is_partial"] is True
        assert len(result["insufficient_documentation_flags"]) > 0
        assert any(
            "Insufficient Documentation" in flag
            for flag in result["insufficient_documentation_flags"]
        )

    def test_auditor_ceiling_with_forced_count_four(self):
        """If iteration_count is somehow 4 (state corruption), system exits with partial — no crash."""
        state = make_state([MISSING_CITATION_EXTRACTION], iteration_count=4)
        result = auditor_node(state)
        assert isinstance(result, dict)
        assert result["routing_decision"] == "partial"
        assert result["is_partial"] is True


# ── Tests: ambiguity ──────────────────────────────────────────────────────────

class TestAuditorAmbiguity:
    def test_auditor_detects_ambiguous_patient_sets_clarification(self):
        """Auditor sets pending_user_input=True and a non-empty clarification_needed on ambiguity."""
        state = make_state([AMBIGUOUS_EXTRACTION])
        result = auditor_node(state)
        assert result["routing_decision"] == "ambiguous"
        assert result["pending_user_input"] is True
        assert len(result["clarification_needed"]) > 0

    def test_auditor_routes_ambiguity_to_clarification_not_loop(self):
        """Ambiguity does not increment iteration_count — it is a separate path from the review loop."""
        state = make_state([AMBIGUOUS_EXTRACTION], iteration_count=0)
        result = auditor_node(state)
        assert result["routing_decision"] == "ambiguous"
        assert result["iteration_count"] == 0

    def test_auditor_clarification_question_contains_no_pii(self):
        """The clarification question must not expose raw PII (SSN, MRN) to the user."""
        state = make_state([AMBIGUOUS_EXTRACTION])
        result = auditor_node(state)
        question = result.get("clarification_needed", "").lower()
        for field in ["ssn", "mrn"]:
            assert field not in question, f"PII field '{field}' found in clarification question"
