"""
test_conversation.py
--------------------
AgentForge — Healthcare RCM AI Agent — Test Suite for conversation.py
----------------------------------------------------------------------
TDD test suite for the multi-turn conversation manager. Tests are
written BEFORE conversation.py is implemented — they will fail first,
then pass once conversation.py is built correctly.

Tests cover:
    - create_conversation_agent returns agent and empty history
    - chat() handles empty input without crashing
    - chat() single turn returns response and updated history
    - chat() multi-turn maintains context (e.g. "his medications" refers to prior patient)
    - chat() returns error message on failure, never raises

Run:
    pytest tests/test_conversation.py -v --tb=short

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_conversation_imports():
    """Conversation module should expose create_conversation_agent and chat."""
    from conversation import create_conversation_agent, chat
    assert create_conversation_agent is not None
    assert chat is not None


def test_create_conversation_agent_returns_agent_and_list():
    """create_conversation_agent() should return (agent, history) with empty history."""
    from conversation import create_conversation_agent
    agent, history = create_conversation_agent()
    assert agent is not None
    assert isinstance(history, list)
    assert len(history) == 0


def test_chat_empty_input():
    """chat() with empty or whitespace message should return helpful message, not crash."""
    from conversation import create_conversation_agent, chat
    agent, history = create_conversation_agent()
    response, new_history = chat(agent, history, "")
    assert isinstance(response, str)
    assert len(response) > 0
    assert "error" in response.lower() or "please" in response.lower() or "provide" in response.lower()
    assert isinstance(new_history, list)


def test_chat_empty_input_whitespace():
    """chat() with only whitespace should behave like empty input."""
    from conversation import create_conversation_agent, chat
    agent, history = create_conversation_agent()
    response, new_history = chat(agent, history, "   \n\t  ")
    assert isinstance(response, str)
    assert isinstance(new_history, list)


def test_chat_single_turn():
    """chat() with one message should return agent response and history with one turn."""
    from conversation import create_conversation_agent, chat
    agent, history = create_conversation_agent()
    response, new_history = chat(agent, history, "What medications is John Smith on?")
    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(new_history, list)
    assert len(new_history) >= 1
    assert "metformin" in response.lower() or "lisinopril" in response.lower() or "medication" in response.lower()


def test_chat_multi_turn_context():
    """Second turn should use context from first (e.g. 'his' refers to previously discussed patient)."""
    from conversation import create_conversation_agent, chat
    agent, history = create_conversation_agent()
    _, history = chat(agent, history, "What medications is John Smith on?")
    assert len(history) >= 1
    response2, history2 = chat(agent, history, "Does he have any drug interactions?")
    assert isinstance(response2, str)
    assert len(response2) > 0
    assert isinstance(history2, list)
    assert len(history2) >= 2
    # Response should reflect John's data (interactions or none), not "who?"
    assert "don't know" not in response2.lower() or "john" in response2.lower()


def test_chat_returns_error_on_failure_no_exception():
    """chat() with invalid agent should return (error_message, history) and never raise."""
    from conversation import chat
    response, new_history = chat(None, [], "Hello")
    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(new_history, list)
    assert "error" in response.lower() or "consult" in response.lower() or "professional" in response.lower()
