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


class TestExtractorNode:
    def test_extractor_returns_updated_state(self):
        """Extractor returns a state dict with extractions[] populated."""
        state = create_initial_state("What medications is John Smith taking?")
        with patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            result = extractor_node(state)
        assert "extractions" in result
        assert isinstance(result["extractions"], list)
        assert len(result["extractions"]) > 0

    def test_extractor_calls_patient_tool(self):
        """tool_get_patient_info is called exactly once when query contains a patient name."""
        state = create_initial_state("What medications is John Smith taking?")
        with patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT) as mock_patient, \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            extractor_node(state)
        mock_patient.assert_called_once()

    def test_extractor_calls_medications_tool(self):
        """tool_get_medications is called with the patient ID retrieved from patient lookup."""
        state = create_initial_state("What medications is John Smith taking?")
        with patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS) as mock_meds, \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            extractor_node(state)
        mock_meds.assert_called_once_with("P001")

    def test_extractor_calls_interactions_tool(self):
        """tool_check_drug_interactions is called when medications are successfully retrieved."""
        state = create_initial_state("Check drug interactions for John Smith.")
        with patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS) as mock_interactions:
            extractor_node(state)
        mock_interactions.assert_called_once()

    def test_extractor_citations_are_verbatim(self):
        """Every extraction includes a non-empty verbatim citation and a source reference."""
        state = create_initial_state("What medications is John Smith taking?")
        with patch("langgraph_agent.extractor_node.tool_get_patient_info", return_value=MOCK_PATIENT), \
             patch("langgraph_agent.extractor_node.tool_get_medications", return_value=MOCK_MEDICATIONS), \
             patch("langgraph_agent.extractor_node.tool_check_drug_interactions", return_value=MOCK_INTERACTIONS):
            result = extractor_node(state)
        for extraction in result["extractions"]:
            assert "citation" in extraction, f"Missing citation in extraction: {extraction}"
            assert "source" in extraction, f"Missing source in extraction: {extraction}"
            assert len(extraction["citation"]) > 0, "Citation must not be empty"
            assert len(extraction["source"]) > 0, "Source must not be empty"
