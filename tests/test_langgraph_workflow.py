"""
test_langgraph_workflow.py
--------------------------
AgentForge — Healthcare RCM AI Agent — Tests for assembled LangGraph workflow
------------------------------------------------------------------------------
End-to-end tests of the full LangGraph state machine. All node functions
are mocked. Tests routing paths, review loop enforcement, clarification
pause/resume, session isolation, error handling, and memory system behaviour.

Memory system tests (added for memory system implementation):
    - prior_state Layer 2 cache fields are merged into new initial state
    - prior_state=None on first call (no cache to merge)
    - session_id is written into initial state before graph runs
    - Layer 2 cache fields not in prior_state are not overwritten

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
    state["extracted_patient_identifier"] = {"type": "name", "value": "John Smith", "ambiguous": False}
    return state


def _make_extractor_no_patient(state):
    """Mock extractor: no patient identifier — routes to clarification (Extractor → clarification)."""
    state["extracted_patient_identifier"] = {"type": "none", "ambiguous": True, "reason": "no patient name or ID found"}
    state["clarification_needed"] = "Which patient are you referring to?"
    state["pending_user_input"] = True
    state["extractions"] = []
    state["tool_trace"] = []
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
    def test_workflow_extractor_routes_to_clarification_when_no_identifier(self):
        """When Extractor Step 0 finds no patient identifier, workflow routes Extractor → clarification → END."""
        with patch("langgraph_agent.workflow.extractor_node", side_effect=_make_extractor_no_patient):
            result = run_workflow("Does he have any allergies?", session_id="test-no-id-1")
        assert result["pending_user_input"] is True
        assert "Which patient" in (result.get("clarification_needed") or "")
        # Should not have run auditor (clarification node runs then END)
        assert result.get("extractions") == []

    def test_workflow_pauses_on_ambiguous_input(self):
        """Ambiguous patient query (from Auditor) causes workflow to pause with pending_user_input=True."""
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


# ── Tests: memory system (Layer 2 prior_state merge) ─────────────────────────

class TestWorkflowMemorySystem:
    def test_no_prior_state_first_call_has_empty_caches(self):
        """On a first call (prior_state=None), the initial state has empty Layer 2 caches."""
        captured_states = []

        def _capture_extractor(state):
            captured_states.append(dict(state))
            state["extractions"] = []
            state["tool_trace"] = []
            state["extracted_patient_identifier"] = {"type": "none", "ambiguous": True, "reason": "test"}
            state["clarification_needed"] = "Which patient?"
            state["pending_user_input"] = True
            return state

        with patch("langgraph_agent.workflow.extractor_node", side_effect=_capture_extractor), \
             patch("langgraph_agent.workflow.orchestrator_node", side_effect=lambda s: s), \
             patch("langgraph_agent.workflow.router_node", side_effect=lambda s: {**s, "routing_decision": "clinical", "query_intent": "MEDICATIONS"}):
            run_workflow("What medications is John Smith taking?", session_id="test-mem-first-1", prior_state=None)

        assert len(captured_states) == 1
        assert captured_states[0]["extracted_patient"] == {}
        assert captured_states[0]["extracted_pdf_hash"] == ""
        assert captured_states[0]["prior_query_context"] == {}

    def test_prior_state_cache_fields_merged_into_new_state(self):
        """When prior_state contains Layer 2 cache fields, they are merged into
        the new initial state before the graph runs — so Orchestrator sees the cache."""
        captured_states = []
        prior = {
            "extracted_patient": {"id": "P001", "name": "John Smith"},
            "extracted_pdf_hash": "abc123",
            "extracted_pdf_pages": {"1": "some text"},
            "payer_policy_cache": {"cigna": {"criteria": "met"}},
            "prior_query_context": {"patient": "John Smith", "intent": "MEDICATIONS"},
            "tool_call_history": [{"tool": "patient_lookup", "status": "success"}],
            "messages": [],
        }

        def _capture_extractor(state):
            captured_states.append(dict(state))
            state["extractions"] = []
            state["tool_trace"] = []
            state["extracted_patient_identifier"] = {"type": "none", "ambiguous": True, "reason": "test"}
            state["clarification_needed"] = "Which patient?"
            state["pending_user_input"] = True
            return state

        with patch("langgraph_agent.workflow.extractor_node", side_effect=_capture_extractor), \
             patch("langgraph_agent.workflow.orchestrator_node", side_effect=lambda s: s), \
             patch("langgraph_agent.workflow.router_node", side_effect=lambda s: {**s, "routing_decision": "clinical", "query_intent": "MEDICATIONS"}):
            run_workflow("Does he have any allergies?", session_id="test-mem-merge-1", prior_state=prior)

        assert len(captured_states) == 1
        merged = captured_states[0]
        assert merged["extracted_patient"] == {"id": "P001", "name": "John Smith"}
        assert merged["extracted_pdf_hash"] == "abc123"
        assert merged["prior_query_context"]["patient"] == "John Smith"

    def test_session_id_written_into_initial_state(self):
        """session_id passed to run_workflow must appear in the state seen by nodes."""
        captured_states = []

        def _capture_extractor(state):
            captured_states.append(dict(state))
            state["extractions"] = []
            state["tool_trace"] = []
            state["extracted_patient_identifier"] = {"type": "none", "ambiguous": True, "reason": "test"}
            state["clarification_needed"] = "Which patient?"
            state["pending_user_input"] = True
            return state

        with patch("langgraph_agent.workflow.extractor_node", side_effect=_capture_extractor), \
             patch("langgraph_agent.workflow.orchestrator_node", side_effect=lambda s: s), \
             patch("langgraph_agent.workflow.router_node", side_effect=lambda s: {**s, "routing_decision": "clinical", "query_intent": "MEDICATIONS"}):
            run_workflow("Any query", session_id="my-session-id-42", prior_state=None)

        assert captured_states[0]["session_id"] == "my-session-id-42"

    def test_prior_state_empty_cache_fields_not_overwritten(self):
        """If a cache field is empty in prior_state (falsy), it should not overwrite
        the fresh initial state default — only populated cache fields are merged."""
        prior = {
            "extracted_patient": {},
            "extracted_pdf_hash": "",
            "prior_query_context": {},
        }
        captured_states = []

        def _capture_extractor(state):
            captured_states.append(dict(state))
            state["extractions"] = []
            state["tool_trace"] = []
            state["extracted_patient_identifier"] = {"type": "none", "ambiguous": True, "reason": "test"}
            state["clarification_needed"] = "Which patient?"
            state["pending_user_input"] = True
            return state

        with patch("langgraph_agent.workflow.extractor_node", side_effect=_capture_extractor), \
             patch("langgraph_agent.workflow.orchestrator_node", side_effect=lambda s: s), \
             patch("langgraph_agent.workflow.router_node", side_effect=lambda s: {**s, "routing_decision": "clinical", "query_intent": "MEDICATIONS"}):
            run_workflow("Any query", session_id="test-mem-empty-1", prior_state=prior)

        assert captured_states[0]["extracted_patient"] == {}
        assert captured_states[0]["extracted_pdf_hash"] == ""

    def test_pdf_source_file_carried_forward_when_not_re_attached(self):
        """Turn 2 follow-up with no PDF re-upload must still have pdf_source_file from Turn 1.
        This is the core fix: the user shouldn't need to re-attach the same PDF every turn."""
        captured_states = []
        prior = {
            "extracted_pdf_pages": {"1": "PT duration: 6 weeks"},
            "extracted_pdf_hash": "abc123",
            "pdf_source_file": "/uploads/AgentForge_Test_ClinicalNote.pdf",
        }

        def _capture_extractor(state):
            captured_states.append(dict(state))
            state["extractions"] = []
            state["tool_trace"] = []
            state["extracted_patient_identifier"] = {"type": "none", "ambiguous": True, "reason": "test"}
            state["clarification_needed"] = "Which patient?"
            state["pending_user_input"] = True
            return state

        with patch("langgraph_agent.workflow.extractor_node", side_effect=_capture_extractor), \
             patch("langgraph_agent.workflow.orchestrator_node", side_effect=lambda s: s), \
             patch("langgraph_agent.workflow.router_node", side_effect=lambda s: {**s, "routing_decision": "clinical", "query_intent": "CHART_REVIEW"}):
            run_workflow(
                "Find the physical therapy duration. Scan the whole chart.",
                session_id="test-pdf-carryforward",
                pdf_source_file=None,
                prior_state=prior,
            )

        merged = captured_states[0]
        assert merged["pdf_source_file"] == "/uploads/AgentForge_Test_ClinicalNote.pdf"
        assert merged["extracted_pdf_pages"] == {"1": "PT duration: 6 weeks"}
        assert merged["extracted_pdf_hash"] == "abc123"

    def test_new_pdf_upload_wins_over_carried_forward_path(self):
        """If the user attaches a new PDF on a follow-up turn, it must win over the
        cached path — explicit upload always takes priority over carryforward."""
        captured_states = []
        prior = {
            "extracted_pdf_pages": {"1": "old content"},
            "extracted_pdf_hash": "oldhash",
            "pdf_source_file": "/uploads/OldDocument.pdf",
        }

        def _capture_extractor(state):
            captured_states.append(dict(state))
            state["extractions"] = []
            state["tool_trace"] = []
            state["extracted_patient_identifier"] = {"type": "none", "ambiguous": True, "reason": "test"}
            state["clarification_needed"] = "Which patient?"
            state["pending_user_input"] = True
            return state

        with patch("langgraph_agent.workflow.extractor_node", side_effect=_capture_extractor), \
             patch("langgraph_agent.workflow.orchestrator_node", side_effect=lambda s: s), \
             patch("langgraph_agent.workflow.router_node", side_effect=lambda s: {**s, "routing_decision": "clinical", "query_intent": "CHART_REVIEW"}):
            run_workflow(
                "Review this new document.",
                session_id="test-new-pdf-wins",
                pdf_source_file="/uploads/NewDocument.pdf",
                prior_state=prior,
            )

        merged = captured_states[0]
        assert merged["pdf_source_file"] == "/uploads/NewDocument.pdf"

    def test_no_prior_state_pdf_source_file_stays_empty(self):
        """On a first-turn call with no prior_state and no PDF, pdf_source_file stays empty.
        No cross-session bleed."""
        captured_states = []

        def _capture_extractor(state):
            captured_states.append(dict(state))
            state["extractions"] = []
            state["tool_trace"] = []
            state["extracted_patient_identifier"] = {"type": "none", "ambiguous": True, "reason": "test"}
            state["clarification_needed"] = "Which patient?"
            state["pending_user_input"] = True
            return state

        with patch("langgraph_agent.workflow.extractor_node", side_effect=_capture_extractor), \
             patch("langgraph_agent.workflow.orchestrator_node", side_effect=lambda s: s), \
             patch("langgraph_agent.workflow.router_node", side_effect=lambda s: {**s, "routing_decision": "clinical", "query_intent": "MEDICATIONS"}):
            run_workflow(
                "What medications is John on?",
                session_id="test-no-prior",
                pdf_source_file=None,
                prior_state=None,
            )

        assert captured_states[0]["pdf_source_file"] == ""
