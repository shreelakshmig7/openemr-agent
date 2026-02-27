"""
test_langgraph_extractor.py
---------------------------
AgentForge — Healthcare RCM AI Agent — Tests for Extractor Node
---------------------------------------------------------------
Tests that the Extractor Node correctly calls tools in sequence,
populates extractions[] in AgentState, and includes verbatim citations
for every claim. All tool calls are mocked — no real API calls.

Memory system tests (added for memory system implementation):
    - General knowledge bypass: orchestrator_ran=True + tool_plan=[] skips all tools
    - Patient cache hit: extracted_patient populated skips Step 0 + patient_lookup
    - PDF content-hash cache hit: matching hash skips tool_extract_pdf
    - PDF content-hash cache miss: hash mismatch forces re-extraction
    - prior_query_context is written with resolved patient after each run

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import hashlib
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from langgraph_agent.state import create_initial_state
from langgraph_agent.extractor_node import extractor_node


MOCK_PATIENT = {
    "success": True,
    "patient": {
        "id": "P001",
        "name": "John Smith",
        "age": 58,
        "gender": "Male",
        "allergies": ["Penicillin", "Sulfa"],
        "conditions": ["Type 2 Diabetes", "Hypertension"]
    }
}

MOCK_MEDICATIONS = {
    "success": True,
    "patient_id": "P001",
    "medications": [
        {"name": "Metformin", "dose": "500mg", "frequency": "twice daily"},
        {"name": "Lisinopril", "dose": "10mg", "frequency": "once daily"}
    ]
}

MOCK_INTERACTIONS = {
    "success": True,
    "interactions_found": False,
    "message": "No known dangerous interactions found.",
    "interactions": []
}


# Step 0 (LLM) returns this for unambiguous full-name queries in tests
MOCK_EXTRACTED_PATIENT = {"type": "name", "value": "John Smith", "ambiguous": False}


class TestExtractorNode:
    def test_extractor_returns_updated_state(self):
        """Extractor returns a state dict with extractions[] populated."""
        state = create_initial_state("What medications is John Smith taking?")
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            result = extractor_node(state)
        assert "extractions" in result
        assert isinstance(result["extractions"], list)
        assert len(result["extractions"]) > 0
        assert result.get("extracted_patient_identifier") == MOCK_EXTRACTED_PATIENT

    def test_extractor_calls_patient_tool(self):
        """tool_get_patient_info is called exactly once when query contains a patient name."""
        state = create_initial_state("What medications is John Smith taking?")
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT) as mock_patient, \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            extractor_node(state)
        mock_patient.assert_called_once_with("John Smith")

    def test_extractor_calls_medications_tool(self):
        """tool_get_medications is called with the patient ID retrieved from patient lookup."""
        state = create_initial_state("What medications is John Smith taking?")
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS) as mock_meds, \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            extractor_node(state)
        mock_meds.assert_called_once_with("P001")

    def test_extractor_calls_interactions_tool(self):
        """tool_check_drug_interactions is called when medications are successfully retrieved."""
        state = create_initial_state("Check drug interactions for John Smith.")
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS) as mock_interactions:
            extractor_node(state)
        mock_interactions.assert_called_once()

    def test_extractor_citations_are_verbatim(self):
        """Every extraction includes a non-empty verbatim citation and a source reference."""
        state = create_initial_state("What medications is John Smith taking?")
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            result = extractor_node(state)
        for extraction in result["extractions"]:
            assert "citation" in extraction, f"Missing citation in extraction: {extraction}"
            assert "source" in extraction, f"Missing source in extraction: {extraction}"
            assert len(extraction["citation"]) > 0, "Citation must not be empty"
            assert len(extraction["source"]) > 0, "Source must not be empty"

    def test_extractor_routes_to_clarification_when_no_identifier(self):
        """When Step 0 returns type 'none' or ambiguous, extractor sets pending_user_input and clarification_needed."""
        state = create_initial_state("Does he have any allergies?")
        no_id = {"type": "none", "ambiguous": True, "reason": "no patient name or ID found in query"}
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=no_id):
            result = extractor_node(state)
        assert result["pending_user_input"] is True
        assert result["clarification_needed"]
        assert result["extractions"] == []
        assert result["tool_trace"] == []
        assert result["extracted_patient_identifier"] == no_id

    def test_extractor_routes_to_clarification_when_first_name_only(self):
        """When Step 0 returns ambiguous first name only, extractor routes to clarification."""
        state = create_initial_state("What about John?")
        ambiguous_name = {"type": "name", "value": "John", "ambiguous": True, "reason": "first name only — multiple patients possible"}
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=ambiguous_name):
            result = extractor_node(state)
        assert result["pending_user_input"] is True
        assert result["clarification_needed"]
        assert result["extractions"] == []
        assert result["tool_trace"] == []


# ── Memory System: general knowledge bypass ───────────────────────────────────

class TestGeneralKnowledgeBypass:
    def test_is_general_knowledge_true_skips_all_tools(self):
        """is_general_knowledge=True is the authoritative bypass signal — tools not called."""
        state = create_initial_state("What are the contraindications of Warfarin?")
        state["orchestrator_ran"] = True
        state["tool_plan"] = []
        state["is_general_knowledge"] = True
        with patch("langgraph_agent.extractor_node.tool_get_patient_info") as mock_patient, \
             patch("langgraph_agent.extractor_node.tool_get_medications") as mock_meds, \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions") as mock_ix:
            result = extractor_node(state)
        mock_patient.assert_not_called()
        mock_meds.assert_not_called()
        mock_ix.assert_not_called()
        assert result["extractions"] == []

    def test_orchestrator_ran_true_empty_plan_without_flag_does_not_bypass(self):
        """orchestrator_ran=True + tool_plan=[] alone must NOT trigger bypass (cache-hit path).
        is_general_knowledge=False (default) — extractor continues to patient cache resolution."""
        cached_patient = {"id": "P001", "name": "John Smith",
                          "allergies": ["Penicillin"], "conditions": []}
        state = create_initial_state("Does he have any allergies?")
        state["orchestrator_ran"] = True
        state["tool_plan"] = []
        state["is_general_knowledge"] = False
        state["extracted_patient"] = cached_patient
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm") as mock_step0, \
             patch("langgraph_agent.extractor_node.tool_get_patient_info") as mock_patient:
            result = extractor_node(state)
        # Neither Step 0 nor patient_lookup should fire — cache-hit path, not bypass
        mock_step0.assert_not_called()
        mock_patient.assert_not_called()

    def test_orchestrator_not_ran_empty_plan_runs_full_suite(self):
        """When orchestrator_ran=False (legacy/direct test path) and tool_plan=[],
        extractor uses full tool suite — backward compatible."""
        state = create_initial_state("What medications is John Smith taking?")
        state["orchestrator_ran"] = False
        state["tool_plan"] = []
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT) as mock_patient, \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            result = extractor_node(state)
        mock_patient.assert_called_once()
        assert len(result["extractions"]) > 0

    def test_general_knowledge_bypass_sets_zero_ehr_confidence_penalty(self):
        """General knowledge bypass must not set ehr_confidence_penalty (no phantom Scenario A)."""
        state = create_initial_state("What is the half-life of Metformin?")
        state["orchestrator_ran"] = True
        state["tool_plan"] = []
        state["is_general_knowledge"] = True
        with patch("langgraph_agent.extractor_node.tool_get_patient_info"), \
             patch("langgraph_agent.extractor_node.tool_get_medications"), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions"):
            result = extractor_node(state)
        assert result.get("ehr_confidence_penalty", 0) == 0


# ── Memory System: patient cache hit ─────────────────────────────────────────

class TestPatientCacheHit:
    def test_patient_cache_hit_skips_step0_and_patient_lookup(self):
        """When orchestrator_ran=True, patient_lookup not in tool_plan, and
        extracted_patient is populated, Step 0 and tool_get_patient_info are skipped."""
        cached_patient = {"id": "P001", "name": "John Smith",
                          "allergies": ["Penicillin"], "conditions": []}
        state = create_initial_state("Does he have any allergies?")
        state["orchestrator_ran"] = True
        state["tool_plan"] = []
        state["extracted_patient"] = cached_patient
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm") as mock_step0, \
             patch("langgraph_agent.extractor_node.tool_get_patient_info") as mock_patient:
            result = extractor_node(state)
        mock_step0.assert_not_called()
        mock_patient.assert_not_called()

    def test_patient_cache_populated_after_fresh_lookup(self):
        """After a successful patient_lookup, extracted_patient is written to state
        so the next turn's Orchestrator can use the cache."""
        state = create_initial_state("What medications is John Smith taking?")
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            result = extractor_node(state)
        assert result["extracted_patient"] == MOCK_PATIENT["patient"]

    def test_prior_query_context_written_after_run(self):
        """prior_query_context is written with resolved patient name and intent
        so the next turn's Orchestrator can resolve follow-up pronouns."""
        state = create_initial_state("What medications is John Smith taking?")
        state["query_intent"] = "MEDICATIONS"
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            result = extractor_node(state)
        ctx = result.get("prior_query_context", {})
        assert ctx.get("patient") == "John Smith"
        assert ctx.get("intent") == "MEDICATIONS"


# ── Memory System: orchestrated path (orchestrator_ran=True) ─────────────────

class TestOrchestratedPath:
    def test_orchestrated_path_skips_step0_uses_identified_patient_name(self):
        """When orchestrator_ran=True, extractor reads identified_patient_name instead of
        calling _extract_patient_identifier_llm (Step 0 is deleted from the orchestrated path)."""
        state = create_initial_state("What medications is John Smith taking?")
        state["orchestrator_ran"] = True
        state["tool_plan"] = ["patient_lookup", "med_retrieval"]
        state["identified_patient_name"] = "John Smith"
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm") as mock_step0, \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            extractor_node(state)
        mock_step0.assert_not_called()

    def test_orchestrated_path_calls_patient_lookup_with_identified_name(self):
        """Extractor passes identified_patient_name to tool_get_patient_info when
        orchestrator_ran=True and patient_lookup is in tool_plan."""
        state = create_initial_state("What medications is John Smith taking?")
        state["orchestrator_ran"] = True
        state["tool_plan"] = ["patient_lookup", "med_retrieval"]
        state["identified_patient_name"] = "John Smith"
        with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm"), \
             patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT) as mock_patient, \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            extractor_node(state)
        mock_patient.assert_called_once_with("John Smith")

    def test_orchestrated_path_no_name_asks_clarification(self):
        """When orchestrator_ran=True but identified_patient_name is None/empty,
        extractor asks for clarification instead of crashing."""
        state = create_initial_state("What are their medications?")
        state["orchestrator_ran"] = True
        state["tool_plan"] = ["patient_lookup", "med_retrieval"]
        state["identified_patient_name"] = None
        result = extractor_node(state)
        assert result["pending_user_input"] is True
        assert "patient" in result["clarification_needed"].lower()

    def test_orchestrated_path_returns_extractions(self):
        """Full orchestrated path with a valid patient name produces extractions."""
        state = create_initial_state("What medications is John Smith taking?")
        state["orchestrator_ran"] = True
        state["tool_plan"] = ["patient_lookup", "med_retrieval"]
        state["identified_patient_name"] = "John Smith"
        with patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            result = extractor_node(state)
        assert len(result["extractions"]) > 0
        assert result.get("extracted_patient") == MOCK_PATIENT["patient"]


# ── Memory System: PDF content-hash cache ────────────────────────────────────

class TestPdfContentHashCache:
    def _make_pdf(self, content: bytes) -> str:
        """Write content to a temp PDF file and return its path."""
        f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_pdf_cache_hit_skips_tool_extract_pdf(self):
        """When extracted_pdf_hash matches current file bytes, tool_extract_pdf is not called.
        Uses orchestrated path with cached patient so the extractor reaches the PDF section."""
        content = b"stable clinical note - no changes"
        pdf_path = self._make_pdf(content)
        current_hash = hashlib.md5(content).hexdigest()
        cached_patient = {"id": "P001", "name": "John Smith", "allergies": [], "conditions": []}
        try:
            state = create_initial_state("What does the chart say?")
            state["orchestrator_ran"] = True
            state["tool_plan"] = ["patient_lookup", "med_retrieval"]
            state["identified_patient_name"] = "John Smith"
            state["pdf_source_file"] = pdf_path
            state["extracted_pdf_hash"] = current_hash
            state["extracted_pdf_pages"] = {"1": "stable clinical note text"}
            with patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
                 patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
                 patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS), \
                 patch("langgraph_agent.extractor_node.tool_extract_pdf") as mock_pdf:
                extractor_node(state)
            mock_pdf.assert_not_called()
        finally:
            os.unlink(pdf_path)

    def test_pdf_cache_miss_calls_tool_extract_pdf(self):
        """When extracted_pdf_hash does not match current file bytes, tool_extract_pdf is called.
        orchestrator_ran=False so the extractor runs the full suite and reaches the PDF section."""
        content = b"new document content - different from cached"
        pdf_path = self._make_pdf(content)
        mock_pdf_result = {
            "success": True,
            "extractions": [{"verbatim_quote": "new text", "page_number": 1,
                             "source_file": pdf_path, "element_type": "NarrativeText"}],
            "element_count": 1,
            "source_file": pdf_path,
            "error": None,
        }
        try:
            state = create_initial_state("What does the chart say?")
            state["orchestrator_ran"] = False
            state["pdf_source_file"] = pdf_path
            state["extracted_pdf_hash"] = "000000stale_hash"
            state["extracted_pdf_pages"] = {"1": "old document text"}
            with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
                 patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
                 patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
                 patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS), \
                 patch("langgraph_agent.extractor_node.tool_extract_pdf", return_value=mock_pdf_result) as mock_pdf:
                extractor_node(state)
            mock_pdf.assert_called_once_with(pdf_path)
        finally:
            os.unlink(pdf_path)

    def test_pdf_hash_updated_in_state_after_extraction(self):
        """After a successful extraction, extracted_pdf_hash is updated to the new file's hash.
        orchestrator_ran=False so the extractor runs the full suite and reaches the PDF section."""
        content = b"fresh document content"
        pdf_path = self._make_pdf(content)
        expected_hash = hashlib.md5(content).hexdigest()
        mock_pdf_result = {
            "success": True,
            "extractions": [{"verbatim_quote": "fresh text", "page_number": 1,
                             "source_file": pdf_path, "element_type": "NarrativeText"}],
            "element_count": 1,
            "source_file": pdf_path,
            "error": None,
        }
        try:
            state = create_initial_state("What does the chart say?")
            state["orchestrator_ran"] = False
            state["pdf_source_file"] = pdf_path
            state["extracted_pdf_hash"] = ""
            state["extracted_pdf_pages"] = {}
            with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
                 patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
                 patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
                 patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS), \
                 patch("langgraph_agent.extractor_node.tool_extract_pdf", return_value=mock_pdf_result):
                result = extractor_node(state)
            assert result["extracted_pdf_hash"] == expected_hash
        finally:
            os.unlink(pdf_path)

    def test_pdf_cache_hit_preserves_page_numbers_from_dict_keys(self):
        """Cache hit path must restore page_number from the dict key.
        Before fix: all pages got page_number=None → collapsed to p.1 in UI.
        After fix: each page gets its original page number back."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"cached pdf content")
            pdf_path = f.name
        try:
            import hashlib
            content_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
            cached_pages = {"1": "page one text", "2": "page two text", "3": "page three text"}
            state = create_initial_state("Review the chart.")
            state["orchestrator_ran"] = False
            state["pdf_source_file"] = pdf_path
            state["extracted_pdf_hash"] = content_hash   # hash matches → cache hit
            state["extracted_pdf_pages"] = cached_pages
            with patch("langgraph_agent.extractor_node._extract_patient_identifier_llm", return_value=MOCK_EXTRACTED_PATIENT), \
                 patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
                 patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
                 patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS), \
                 patch("langgraph_agent.extractor_node.tool_extract_pdf") as mock_extract:
                result = extractor_node(state)
            mock_extract.assert_not_called()  # cache hit — no re-extraction
            pdf_extractions = [e for e in result["extractions"] if e.get("source", "").endswith(".pdf")]
            assert len(pdf_extractions) == 3
            page_numbers = {e["page_number"] for e in pdf_extractions}
            assert page_numbers == {1, 2, 3}   # not {None}
        finally:
            os.unlink(pdf_path)
