"""
test_langgraph_orchestrator.py
------------------------------
AgentForge — Healthcare RCM AI Agent — Tests for Orchestrator Node
-------------------------------------------------------------------
Tests that the Orchestrator Node correctly classifies queries via the
mocked Haiku LLM, builds the right tool_plan, respects Layer 2 session
caches, handles general knowledge bypass, and falls back safely on
classification failure. All LLM calls are mocked — no real API calls.

Test coverage:
    - General knowledge query sets tool_plan=[] and orchestrator_ran=True
    - Patient-specific query without cache adds patient_lookup + med_retrieval
    - Patient-specific query with cache hit skips patient_lookup + med_retrieval
    - Document evidence query without PDF cache adds pdf_extractor
    - Document evidence query with matching hash skips pdf_extractor
    - Document evidence query with hash mismatch re-adds pdf_extractor
    - Policy check query adds policy_search
    - Denial analysis query with audit_results adds denial_analyzer
    - Denial analysis skipped when no audit_results exist
    - Haiku parse failure falls back to safe defaults (patient_lookup + med_retrieval)
    - orchestrator_ran is always True after node runs

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import hashlib
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from langgraph_agent.orchestrator_node import (
    orchestrator_node, _classify_query, _orchestrator_fallback,
    _names_match, _invalidate_patient_cache,
)
from langgraph_agent.state import create_initial_state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_haiku_response(flags: dict) -> MagicMock:
    """Build a mock ChatAnthropic response returning the given classification flags."""
    mock_response = MagicMock()
    mock_response.content = json.dumps(flags)
    return mock_response


def _state_with(**kwargs) -> dict:
    """Return a fresh initial state with extra fields overlaid."""
    state = create_initial_state("test query")
    state.update(kwargs)
    return state


# ── Shared classification flag sets ──────────────────────────────────────────
# All sets include patient_name — Haiku now returns it in the same call.

_GENERAL_KNOWLEDGE = {
    "needs_specific_patient": False,
    "needs_document_evidence": False,
    "needs_policy_check": False,
    "needs_denial_analysis": False,
    "is_general_knowledge": True,
    "patient_name": None,
    "data_source_required": "NONE",
    "pdf_required": False,
}

_PATIENT_SPECIFIC = {
    "needs_specific_patient": True,
    "needs_document_evidence": False,
    "needs_policy_check": False,
    "needs_denial_analysis": False,
    "is_general_knowledge": False,
    "patient_name": "John Smith",
    "data_source_required": "EHR",
    "pdf_required": False,
}

_PATIENT_PRONOUN_ONLY = {
    "needs_specific_patient": True,
    "needs_document_evidence": False,
    "needs_policy_check": False,
    "needs_denial_analysis": False,
    "is_general_knowledge": False,
    "patient_name": None,
    "data_source_required": "EHR",
    "pdf_required": False,
}

_DOCUMENT_EVIDENCE = {
    "needs_specific_patient": False,
    "needs_document_evidence": True,
    "needs_policy_check": False,
    "needs_denial_analysis": False,
    "is_general_knowledge": False,
    "patient_name": None,
    "data_source_required": "PDF",
    "pdf_required": True,
}

_RESIDENT_NOTE = {
    "needs_specific_patient": True,
    "needs_document_evidence": True,
    "needs_policy_check": False,
    "needs_denial_analysis": False,
    "is_general_knowledge": False,
    "patient_name": "Maria",
    "data_source_required": "RESIDENT_NOTE",
    "pdf_required": True,
}

_POLICY_CHECK = {
    "needs_specific_patient": False,
    "needs_document_evidence": False,
    "needs_policy_check": True,
    "needs_denial_analysis": False,
    "is_general_knowledge": False,
    "patient_name": None,
    "data_source_required": "EHR",
    "pdf_required": False,
}

_DENIAL_ANALYSIS = {
    "needs_specific_patient": False,
    "needs_document_evidence": False,
    "needs_policy_check": False,
    "needs_denial_analysis": True,
    "is_general_knowledge": False,
    "patient_name": None,
    "data_source_required": "EHR",
    "pdf_required": False,
}


# ── General knowledge bypass ──────────────────────────────────────────────────

class TestGeneralKnowledgeBypass:
    def test_general_knowledge_sets_empty_tool_plan(self):
        """General knowledge query must produce tool_plan=[] — no tools called."""
        state = _state_with(input_query="What are the contraindications of Warfarin in general?")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_GENERAL_KNOWLEDGE)
            result = orchestrator_node(state)
        assert result["tool_plan"] == []
        assert result["is_general_knowledge"] is True

    def test_general_knowledge_sets_orchestrator_ran(self):
        """orchestrator_ran must be True even for general knowledge queries."""
        state = _state_with(input_query="What is the mechanism of action of Metformin?")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_GENERAL_KNOWLEDGE)
            result = orchestrator_node(state)
        assert result["orchestrator_ran"] is True

    def test_general_knowledge_enforces_all_other_flags_false(self):
        """Even if Haiku returns other flags as True alongside is_general_knowledge,
        the invariant enforcer must set them all to False."""
        contaminated_flags = {
            "needs_specific_patient": True,
            "needs_document_evidence": True,
            "needs_policy_check": False,
            "needs_denial_analysis": False,
            "is_general_knowledge": True,
        }
        state = _state_with(input_query="General pharmacology question")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(contaminated_flags)
            result = orchestrator_node(state)
        assert result["tool_plan"] == []


# ── Patient lookup decisions ──────────────────────────────────────────────────

class TestPatientLookupDecisions:
    def test_patient_query_without_cache_adds_lookup_and_retrieval(self):
        """New patient query with no cached patient adds patient_lookup and med_retrieval."""
        state = _state_with(
            input_query="What medications is John Smith taking?",
            extracted_patient={},
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert "patient_lookup" in result["tool_plan"]
        assert "med_retrieval" in result["tool_plan"]
        assert result["tool_plan"].index("patient_lookup") < result["tool_plan"].index("med_retrieval")

    def test_patient_query_with_cache_hit_skips_lookup(self):
        """Follow-up query with extracted_patient already populated skips patient_lookup."""
        cached_patient = {"id": "P001", "name": "John Smith"}
        state = _state_with(
            input_query="Does he have any allergies?",
            extracted_patient=cached_patient,
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert "patient_lookup" not in result["tool_plan"]
        assert "med_retrieval" not in result["tool_plan"]

    def test_orchestrator_ran_is_true_for_patient_query(self):
        """orchestrator_ran must be True after any patient query."""
        state = _state_with(input_query="What medications is Mary Johnson on?")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert result["orchestrator_ran"] is True


# ── PDF extraction cache decisions ────────────────────────────────────────────

class TestPdfExtractionCacheDecisions:
    def test_document_query_without_cache_adds_pdf_extractor(self):
        """Query needing document evidence with no cached PDF adds pdf_extractor."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf content")
            pdf_path = f.name
        try:
            state = _state_with(
                input_query="What does the chart say about the diagnosis?",
                pdf_source_file=pdf_path,
                extracted_pdf_hash="",
                extracted_pdf_pages={},
            )
            with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
                mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_DOCUMENT_EVIDENCE)
                result = orchestrator_node(state)
            assert "pdf_extractor" in result["tool_plan"]
        finally:
            os.unlink(pdf_path)

    def test_document_query_with_matching_hash_skips_pdf_extractor(self):
        """PDF cache hit (content hash matches) must skip pdf_extractor."""
        content = b"stable clinical note content"
        current_hash = hashlib.md5(content).hexdigest()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(content)
            pdf_path = f.name
        try:
            state = _state_with(
                input_query="What does the chart say?",
                pdf_source_file=pdf_path,
                extracted_pdf_hash=current_hash,
                extracted_pdf_pages={"1": "some extracted text"},
            )
            with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
                mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_DOCUMENT_EVIDENCE)
                result = orchestrator_node(state)
            assert "pdf_extractor" not in result["tool_plan"]
        finally:
            os.unlink(pdf_path)

    def test_document_query_with_hash_mismatch_adds_pdf_extractor(self):
        """PDF hash mismatch (different document uploaded) must add pdf_extractor."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"new document content")
            pdf_path = f.name
        try:
            state = _state_with(
                input_query="What does this chart say?",
                pdf_source_file=pdf_path,
                extracted_pdf_hash="000000deadbeef",
                extracted_pdf_pages={"1": "old document text"},
            )
            with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
                mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_DOCUMENT_EVIDENCE)
                result = orchestrator_node(state)
            assert "pdf_extractor" in result["tool_plan"]
        finally:
            os.unlink(pdf_path)

    def test_document_query_without_pdf_path_does_not_add_extractor(self):
        """If needs_document_evidence=True but no pdf_source_file, pdf_extractor is not added."""
        state = _state_with(
            input_query="What does the chart say?",
            pdf_source_file="",
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_DOCUMENT_EVIDENCE)
            result = orchestrator_node(state)
        assert "pdf_extractor" not in result["tool_plan"]


# ── Policy search decisions ───────────────────────────────────────────────────

class TestPolicySearchDecisions:
    def test_policy_query_without_cache_adds_policy_search(self):
        """Policy check query with uncached payer adds policy_search to plan."""
        state = _state_with(
            input_query="Does Cigna cover this procedure?",
            payer_policy_cache={},
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_POLICY_CHECK)
            result = orchestrator_node(state)
        assert "policy_search" in result["tool_plan"]

    def test_policy_query_with_cached_payer_skips_policy_search(self):
        """Policy check query when payer_id already cached skips policy_search."""
        state = _state_with(
            input_query="Does Cigna cover this?",
            payer_policy_cache={"cigna": {"criteria": "..."}},
        )
        state["payer_id"] = "cigna"
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_POLICY_CHECK)
            result = orchestrator_node(state)
        assert "policy_search" not in result["tool_plan"]


# ── Denial analysis decisions ─────────────────────────────────────────────────

class TestDenialAnalysisDecisions:
    def test_denial_query_with_audit_results_adds_denial_analyzer(self):
        """Denial analysis query with existing audit_results adds denial_analyzer."""
        state = _state_with(
            input_query="What is the denial risk for this claim?",
            audit_results=[{"claim": "some claim", "valid": True}],
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_DENIAL_ANALYSIS)
            result = orchestrator_node(state)
        assert "denial_analyzer" in result["tool_plan"]

    def test_denial_query_without_audit_results_skips_denial_analyzer(self):
        """Denial analysis query with no audit_results must not add denial_analyzer
        — there is nothing to analyze yet."""
        state = _state_with(
            input_query="What is the denial risk?",
            audit_results=[],
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_DENIAL_ANALYSIS)
            result = orchestrator_node(state)
        assert "denial_analyzer" not in result["tool_plan"]

    def test_denial_query_with_cached_result_skips_denial_analyzer(self):
        """When denial risk for this (payer, cpt) pair is already cached, skip denial_analyzer."""
        state = _state_with(
            input_query="What is the denial risk?",
            audit_results=[{"claim": "some claim", "valid": True}],
            denial_risk_cache={"cigna:99213": {"risk_level": "LOW"}},
        )
        state["payer_id"] = "cigna"
        state["procedure_code"] = "99213"
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_DENIAL_ANALYSIS)
            result = orchestrator_node(state)
        assert "denial_analyzer" not in result["tool_plan"]


# ── Failure and safety defaults ────────────────────────────────────────────────

class TestFailureAndSafetyDefaults:
    def test_haiku_parse_failure_falls_back_to_safe_defaults(self):
        """If Haiku returns invalid JSON, orchestrator falls back to patient_lookup + med_retrieval."""
        state = _state_with(input_query="Some query")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.side_effect = Exception("API timeout")
            result = orchestrator_node(state)
        assert "patient_lookup" in result["tool_plan"]
        assert "med_retrieval" in result["tool_plan"]
        assert result["orchestrator_ran"] is True

    def test_haiku_malformed_json_falls_back_to_safe_defaults(self):
        """Malformed JSON from Haiku (not parseable) falls back to safe defaults."""
        state = _state_with(input_query="What medications is John on?")
        mock_response = MagicMock()
        mock_response.content = "not valid json {{{"
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = mock_response
            result = orchestrator_node(state)
        assert "patient_lookup" in result["tool_plan"]
        assert result["orchestrator_ran"] is True

    def test_orchestrator_ran_always_true_on_exception(self):
        """orchestrator_ran must be True even when the node catches an exception."""
        state = _state_with(input_query="Any query")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.side_effect = RuntimeError("crash")
            result = orchestrator_node(state)
        assert result["orchestrator_ran"] is True

    def test_missing_haiku_keys_filled_with_safe_defaults(self):
        """Partial JSON from Haiku (missing some keys) gets missing keys filled with safe defaults."""
        partial_flags = {"needs_specific_patient": True}
        state = _state_with(input_query="Look up John Smith")
        mock_response = MagicMock()
        mock_response.content = json.dumps(partial_flags)
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = mock_response
            result = orchestrator_node(state)
        assert "patient_lookup" in result["tool_plan"]
        assert result["orchestrator_ran"] is True


# ── _classify_query unit tests ────────────────────────────────────────────────

class TestClassifyQuery:
    def test_classify_query_returns_dict_with_all_fields(self):
        """_classify_query returns a dict with all 5 boolean flags plus patient_name."""
        flags = _PATIENT_SPECIFIC.copy()
        mock_response = MagicMock()
        mock_response.content = json.dumps(flags)
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = mock_response
            result = _classify_query("What meds is John on?", {})
        for key in ["needs_specific_patient", "needs_document_evidence",
                    "needs_policy_check", "needs_denial_analysis", "is_general_knowledge",
                    "patient_name"]:
            assert key in result

    def test_classify_query_returns_patient_name(self):
        """_classify_query returns the patient_name field from Haiku's JSON."""
        flags = _PATIENT_SPECIFIC.copy()  # patient_name: "John Smith"
        mock_response = MagicMock()
        mock_response.content = json.dumps(flags)
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = mock_response
            result = _classify_query("What meds is John Smith on?", {})
        assert result["patient_name"] == "John Smith"

    def test_classify_query_strips_markdown_code_fences(self):
        """_classify_query handles Haiku wrapping JSON in markdown code fences."""
        flags = _PATIENT_SPECIFIC.copy()
        mock_response = MagicMock()
        mock_response.content = f"```json\n{json.dumps(flags)}\n```"
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = mock_response
            result = _classify_query("What meds is John on?", {})
        assert result["needs_specific_patient"] is True

    def test_classify_query_prior_context_included_in_prompt(self):
        """prior_query_context patient name is passed to the Haiku classification call."""
        prior_context = {"patient": "John Smith", "intent": "MEDICATIONS"}
        mock_response = MagicMock()
        mock_response.content = json.dumps(_PATIENT_SPECIFIC)
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_instance = mock_llm_cls.return_value
            mock_instance.invoke.return_value = mock_response
            _classify_query("Does he have allergies?", prior_context)
            call_args = mock_instance.invoke.call_args[0][0]
            human_message_content = call_args[1].content
        assert "John Smith" in human_message_content


# ── identified_patient_name in state ─────────────────────────────────────────

class TestIdentifiedPatientNameInState:
    def test_patient_name_stored_in_state_when_haiku_returns_it(self):
        """Orchestrator stores Haiku's patient_name in state['identified_patient_name']."""
        state = _state_with(input_query="What medications is John Smith taking?")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert result["identified_patient_name"] == "John Smith"

    def test_patient_name_none_when_pronoun_only_and_no_prior_context(self):
        """Orchestrator stores None when Haiku finds no name AND prior_query_context has no patient."""
        state = _state_with(input_query="What are his allergies?")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_PRONOUN_ONLY)
            result = orchestrator_node(state)
        assert result["identified_patient_name"] is None

    def test_pronoun_resolved_from_prior_context_patient(self):
        """When Haiku returns patient_name=None (pronoun) but prior_query_context has a
        patient, the Orchestrator resolves incoming_name from context so the cache
        collision check fires correctly and the right patient data is served."""
        state = _state_with(input_query="Does he have any allergies?")
        state["prior_query_context"] = {"patient": "John Smith", "intent": "MEDICATIONS"}
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_PRONOUN_ONLY)
            result = orchestrator_node(state)
        assert result["identified_patient_name"] == "John Smith"

    def test_pronoun_resolution_keeps_same_patient_cache(self):
        """After pronoun resolution, if the resolved name matches the cached patient,
        the cache is NOT invalidated — no unnecessary patient_lookup added."""
        cached = {"id": "P001", "name": "John Smith", "allergies": ["Penicillin"], "conditions": []}
        state = _state_with(input_query="Does he have any allergies?")
        state["extracted_patient"] = cached
        state["prior_query_context"] = {"patient": "John Smith", "intent": "MEDICATIONS"}
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_PRONOUN_ONLY)
            result = orchestrator_node(state)
        # Cache should not be invalidated — same patient
        assert result.get("extracted_patient") == cached

    def test_pronoun_resolution_invalidates_cache_if_prior_context_differs_from_cached(self):
        """If prior context resolves the pronoun to a different patient than the cache,
        the cache must be invalidated to prevent cross-patient data leakage."""
        cached = {"id": "P002", "name": "Maria Gonzalez", "allergies": [], "conditions": []}
        state = _state_with(input_query="Does he have any allergies?")
        state["extracted_patient"] = cached
        state["prior_query_context"] = {"patient": "John Smith", "intent": "MEDICATIONS"}
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_PRONOUN_ONLY)
            result = orchestrator_node(state)
        # Cache must be cleared — pronoun resolved to different patient
        # _invalidate_patient_cache sets extracted_patient to {} (empty dict sentinel)
        assert not result.get("extracted_patient")

    def test_patient_name_none_for_general_knowledge_query(self):
        """General knowledge queries always result in identified_patient_name=None."""
        state = _state_with(input_query="What are the contraindications of Warfarin?")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_GENERAL_KNOWLEDGE)
            result = orchestrator_node(state)
        assert result["identified_patient_name"] is None

    def test_orchestrator_fallback_false_on_success(self):
        """orchestrator_fallback is False when the Haiku call succeeds."""
        state = _state_with(input_query="What medications is John Smith taking?")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert result["orchestrator_fallback"] is False


# ── 529 retry and fallback ────────────────────────────────────────────────────

class TestRetryAndFallback:
    def test_529_triggers_retry_then_fallback(self):
        """HTTP 529 on all 3 attempts invokes _orchestrator_fallback, sets orchestrator_fallback=True."""
        state = _state_with(input_query="What medications is John Smith taking?")
        exc = Exception("Error code: 529 - overloaded")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls, \
             patch("langgraph_agent.orchestrator_node.time.sleep"):  # skip real waits
            mock_llm_cls.return_value.invoke.side_effect = exc
            result = orchestrator_node(state)
        assert result["orchestrator_ran"] is True
        assert result["orchestrator_fallback"] is True

    def test_529_fallback_extracts_name_via_regex(self):
        """Fallback extracts patient name via regex when Haiku is unavailable."""
        state = _state_with(input_query="What medications is John Smith taking?")
        exc = Exception("Error code: 529 - overloaded")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls, \
             patch("langgraph_agent.orchestrator_node.time.sleep"):
            mock_llm_cls.return_value.invoke.side_effect = exc
            result = orchestrator_node(state)
        assert result["identified_patient_name"] == "John Smith"

    def test_529_fallback_sets_default_tool_plan(self):
        """Fallback always sets tool_plan to the full patient pipeline."""
        state = _state_with(input_query="What medications is John Smith taking?")
        exc = Exception("Error code: 529 - overloaded")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls, \
             patch("langgraph_agent.orchestrator_node.time.sleep"):
            mock_llm_cls.return_value.invoke.side_effect = exc
            result = orchestrator_node(state)
        assert "patient_lookup" in result["tool_plan"]
        assert "med_retrieval" in result["tool_plan"]

    def test_529_fallback_no_regex_match_sets_none(self):
        """Fallback sets identified_patient_name=None when regex finds no capitalized name."""
        state = _state_with(input_query="check the denial risk")
        exc = Exception("Error code: 529 - overloaded")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls, \
             patch("langgraph_agent.orchestrator_node.time.sleep"):
            mock_llm_cls.return_value.invoke.side_effect = exc
            result = orchestrator_node(state)
        assert result["identified_patient_name"] is None

    def test_non_529_error_uses_safe_defaults_without_retry(self):
        """Non-529 errors (parse failure, auth) use safe defaults immediately, no retry."""
        state = _state_with(input_query="Some query")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls, \
             patch("langgraph_agent.orchestrator_node.time.sleep") as mock_sleep:
            mock_llm_cls.return_value.invoke.side_effect = ValueError("unexpected error")
            result = orchestrator_node(state)
        mock_sleep.assert_not_called()
        assert "patient_lookup" in result["tool_plan"]
        assert result["orchestrator_ran"] is True

    def test_orchestrator_fallback_function_sets_flag(self):
        """_orchestrator_fallback directly sets orchestrator_fallback=True in state."""
        state = _state_with(input_query="Any query")
        result = _orchestrator_fallback(state, "What medications is Emily Rodriguez on?")
        assert result["orchestrator_fallback"] is True
        assert result["orchestrator_ran"] is True
        assert result["identified_patient_name"] == "Emily Rodriguez"


# ── Bug 1: _names_match helper ────────────────────────────────────────────────

class TestNamesMatch:
    def test_exact_match(self):
        assert _names_match("John Smith", "John Smith") is True

    def test_middle_initial_tolerated(self):
        assert _names_match("Maria Gonzalez", "Maria J. Gonzalez") is True

    def test_suffix_tolerated(self):
        assert _names_match("John Smith", "John Smith Jr") is True

    def test_different_patients_no_match(self):
        assert _names_match("John Smith", "Maria Gonzalez") is False

    def test_case_insensitive(self):
        assert _names_match("john smith", "John Smith") is True

    def test_single_name_no_match(self):
        """Single token names never match (require 2 common tokens)."""
        assert _names_match("Maria", "Maria") is False


# ── Bug 1: _invalidate_patient_cache helper ───────────────────────────────────

class TestInvalidatePatientCache:
    def test_clears_extracted_patient(self):
        state = _state_with(extracted_patient={"id": "P001", "name": "Maria"})
        _invalidate_patient_cache(state)
        assert state["extracted_patient"] == {}

    def test_clears_pdf_pages_and_hash(self):
        state = _state_with(
            extracted_pdf_pages={"1": "old text"},
            extracted_pdf_hash="abc123",
        )
        _invalidate_patient_cache(state)
        assert state["extracted_pdf_pages"] == {}
        assert state["extracted_pdf_hash"] == ""

    def test_leaves_payer_cache_intact(self):
        state = _state_with(payer_policy_cache={"cigna": {"criteria": "..."}})
        _invalidate_patient_cache(state)
        assert state["payer_policy_cache"] == {"cigna": {"criteria": "..."}}

    def test_preserves_pdf_source_file_on_patient_switch(self):
        """_invalidate_patient_cache must preserve pdf_source_file (request-level
        input) while clearing the PDF cache (pages + hash). The user may have
        attached a new PDF for the new patient — wiping the path would block
        Scenario A (PDF-only fallback when patient is not in EHR)."""
        state = create_initial_state("test")
        state["pdf_source_file"] = "/uploads/MariaGonzalez.pdf"
        state["extracted_pdf_pages"] = {"1": "gonzalez page one"}
        state["extracted_pdf_hash"] = "gonzalez_hash"
        _invalidate_patient_cache(state)
        assert state["pdf_source_file"] == "/uploads/MariaGonzalez.pdf"
        assert state["extracted_pdf_pages"] == {}
        assert state["extracted_pdf_hash"] == ""

    def test_patient_switch_preserves_pdf_source_file_end_to_end(self):
        """Full orchestrator run: switching from Maria Gonzalez to Maria Santos must
        preserve pdf_source_file (request-level input) while clearing the PDF cache
        so the extractor re-extracts if the same path is reused with new content."""
        gonzalez_record = {"id": "P002", "name": "Maria Gonzalez", "allergies": [], "conditions": []}
        state = _state_with(input_query="What medications is Maria Santos on?")
        state["extracted_patient"] = gonzalez_record
        state["pdf_source_file"] = "/uploads/MariaGonzalez.pdf"
        state["extracted_pdf_pages"] = {"1": "Gonzalez chart content"}
        state["extracted_pdf_hash"] = "gonzalez_hash"
        santos_flags = {
            "needs_specific_patient": True,
            "needs_document_evidence": False,
            "needs_policy_check": False,
            "needs_denial_analysis": False,
            "is_general_knowledge": False,
            "patient_name": "Maria Santos",
            "data_source_required": "EHR",
            "pdf_required": False,
        }
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(santos_flags)
            result = orchestrator_node(state)
        assert result["pdf_source_file"] == "/uploads/MariaGonzalez.pdf"
        assert not result.get("extracted_pdf_pages")


# ── Bug 1: patient identity cache invalidation in orchestrator_node ───────────

class TestPatientIdentityCacheInvalidation:
    def test_different_patient_invalidates_cache(self):
        """When a new patient is named and cached patient differs, cache is cleared."""
        state = _state_with(
            input_query="What medications is John Smith taking?",
            extracted_patient={"id": "P002", "name": "Maria Gonzalez"},
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert result["extracted_patient"] == {}

    def test_same_patient_keeps_cache(self):
        """When the same patient is queried again, the cache is preserved."""
        cached_patient = {"id": "P001", "name": "John Smith", "allergies": []}
        state = _state_with(
            input_query="Does he have any allergies?",
            extracted_patient=cached_patient,
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert result["extracted_patient"] == cached_patient

    def test_pronoun_query_keeps_cache(self):
        """Pronoun-only query (patient_name=None) never invalidates the cache."""
        cached_patient = {"id": "P002", "name": "Maria Gonzalez", "allergies": []}
        state = _state_with(
            input_query="What are her allergies?",
            extracted_patient=cached_patient,
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_PRONOUN_ONLY)
            result = orchestrator_node(state)
        assert result["extracted_patient"] == cached_patient

    def test_patient_switch_adds_patient_lookup_to_plan(self):
        """After cache invalidation, patient_lookup must appear in tool_plan."""
        state = _state_with(
            input_query="What medications is John Smith taking?",
            extracted_patient={"id": "P002", "name": "Maria Gonzalez"},
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert "patient_lookup" in result["tool_plan"]


# ── Bug 2: pdf_required short-circuit ────────────────────────────────────────

class TestPdfRequiredShortCircuit:
    def test_resident_note_query_without_pdf_sets_source_unavailable(self):
        """Query requiring resident note with no PDF attached sets source_unavailable=True."""
        state = _state_with(
            input_query="Is there an ECOG score in the resident note for Maria?",
            pdf_source_file="",
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_RESIDENT_NOTE)
            result = orchestrator_node(state)
        assert result["source_unavailable"] is True
        assert result["source_unavailable_reason"] == "RESIDENT_NOTE"
        assert result["tool_plan"] == []

    def test_resident_note_query_sets_orchestrator_ran(self):
        """source_unavailable short-circuit still sets orchestrator_ran=True."""
        state = _state_with(input_query="Check the resident note for Maria.", pdf_source_file="")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_RESIDENT_NOTE)
            result = orchestrator_node(state)
        assert result["orchestrator_ran"] is True

    def test_pdf_required_with_pdf_attached_proceeds_normally(self):
        """When pdf_required=True and a PDF is attached, pipeline continues normally."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"clinical note content")
            pdf_path = f.name
        try:
            state = _state_with(
                input_query="What does the chart say?",
                pdf_source_file=pdf_path,
            )
            with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
                mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_DOCUMENT_EVIDENCE)
                result = orchestrator_node(state)
            assert result.get("source_unavailable") is not True
        finally:
            os.unlink(pdf_path)

    def test_ehr_query_never_sets_source_unavailable(self):
        """EHR queries (pdf_required=False) never set source_unavailable."""
        state = _state_with(
            input_query="What medications is John Smith taking?",
            pdf_source_file="",
        )
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert result.get("source_unavailable") is not True

    def test_data_source_required_stored_in_state(self):
        """data_source_required from Haiku classification is written to state."""
        state = _state_with(input_query="What medications is John Smith taking?")
        with patch("langgraph_agent.orchestrator_node.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = _make_haiku_response(_PATIENT_SPECIFIC)
            result = orchestrator_node(state)
        assert result["data_source_required"] == "EHR"
