"""
test_agent.py
-------------
Healthcare AI Agent — Test Suite for agent.py
----------------------------------------------
TDD test suite for the LangChain agent.
Tests are written BEFORE agent.py is implemented — they will fail first,
then pass once agent.py is built correctly.

Tests cover:
    - Agent creation and initialization
    - Basic natural language query handling
    - Tool invocation through natural language
    - Unknown patient graceful handling

Run:
    pytest tests/test_agent.py -v --tb=short

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_agent_imports():
    """Agent module should import without errors"""
    from agent import create_agent
    assert create_agent is not None

def test_agent_creation():
    """Agent should be created successfully"""
    from agent import create_agent
    agent = create_agent()
    assert agent is not None

def test_agent_simple_query():
    """Agent should respond to a basic patient query"""
    from agent import create_agent
    agent = create_agent()
    response = agent.invoke({
        "input": "What medications is John Smith on?"
    })
    assert response is not None
    assert "output" in response
    assert len(response["output"]) > 0

def test_agent_response_contains_medication():
    """Agent response should mention at least one medication"""
    from agent import create_agent
    agent = create_agent()
    response = agent.invoke({
        "input": "What medications is John Smith on?"
    })
    # Handle both string and list output formats
    output = response["output"]
    if isinstance(output, list):
        output = " ".join([o.get("text", "") if isinstance(o, dict) else str(o) for o in output])
    output = output.lower()
    assert any(med in output for med in ["metformin", "lisinopril", "atorvastatin"])

def test_agent_unknown_patient():
    """Agent should handle unknown patient gracefully"""
    from agent import create_agent
    agent = create_agent()
    response = agent.invoke({
        "input": "What medications is Unknown Person on?"
    })
    assert response is not None
    assert "output" in response