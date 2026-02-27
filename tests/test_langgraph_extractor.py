"""
test_langgraph_extractor.py
---------------------------
AgentForge — Healthcare RCM AI Agent — Tests for Extractor Node
---------------------------------------------------------------
Tests that the Extractor Node correctly calls tools in sequence,
populates extractions[] in AgentState, and includes verbatim citations
for every claim. All tool calls are mocked — no real API calls.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
from unittest.mock import patch
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
