"""
test_langgraph_state.py
-----------------------
AgentForge — Healthcare RCM AI Agent — Tests for AgentState schema
-------------------------------------------------------------------
Tests that AgentState TypedDict initializes correctly with proper
default values and accepts the correct types for all fields.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
from langgraph_agent.state import AgentState, create_initial_state


class TestAgentStateDefaults:
    def test_state_default_values(self):
        """All fields initialize to correct defaults when create_initial_state is called."""
        state = create_initial_state("test query")
        assert state["iteration_count"] == 0
        assert state["pending_user_input"] is False
        assert state["extractions"] == []
        assert state["audit_results"] == []
        assert state["documents_processed"] == []
        assert state["clarification_needed"] == ""
        assert state["clarification_response"] == ""
        assert state["final_response"] == ""
        assert state["error"] is None
        assert state["routing_decision"] == ""
        assert state["is_partial"] is False
        assert state["insufficient_documentation_flags"] == []
        assert state["tool_trace"] == []
        assert state["extracted_patient_identifier"] == {}

    def test_state_fields_accept_correct_types(self):
        """State fields accept and store the correct Python types."""
        state = create_initial_state("test query")
        state["input_query"] = "What meds is John Smith on?"
        state["confidence_score"] = 0.95
        state["final_response"] = "John Smith is on Metformin."
        assert isinstance(state["input_query"], str)
        assert isinstance(state["confidence_score"], float)
        assert isinstance(state["final_response"], str)

    def test_iteration_count_can_reach_three(self):
        """iteration_count can be set to 3 — the maximum allowed value."""
        state = create_initial_state("test query")
        state["iteration_count"] = 3
        assert state["iteration_count"] == 3

    def test_pending_user_input_is_bool(self):
        """pending_user_input is a boolean that can be toggled True and False."""
        state = create_initial_state("test query")
        assert isinstance(state["pending_user_input"], bool)
        state["pending_user_input"] = True
        assert state["pending_user_input"] is True
        state["pending_user_input"] = False
        assert state["pending_user_input"] is False
