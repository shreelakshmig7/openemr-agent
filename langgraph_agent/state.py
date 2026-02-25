"""
state.py
--------
AgentForge — Healthcare RCM AI Agent — LangGraph AgentState schema
-------------------------------------------------------------------
Defines the AgentState TypedDict that flows through every node in the
LangGraph state machine. All node functions read from and write to this
shared state. Provides create_initial_state() to ensure consistent
initialization with correct defaults.

Key fields:
    pending_user_input: Pauses entire workflow without discarding work.
    iteration_count: Enforces 3-iteration ceiling on Auditor review loop.
    routing_decision: Set by Auditor to control conditional edge routing.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from typing import List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state passed through every node in the LangGraph state machine.

    Args:
        input_query: Original natural language query from the user.
        documents_processed: List of source documents/sections processed.
        extractions: List of extracted claims, each with citation and source.
        audit_results: Validation results from the Auditor Node.
        pending_user_input: When True, workflow is paused awaiting user clarification.
        clarification_needed: The specific question to surface to the user.
        clarification_response: The user's answer to the clarification question.
        iteration_count: Number of Extractor→Auditor review cycles completed.
        confidence_score: Confidence in the final response (0.0 to 1.0).
        final_response: The formatted response returned to the user.
        error: Structured error message if something went wrong.
        routing_decision: Set by Auditor to control graph edge routing.
        is_partial: True when iteration ceiling was hit and response is incomplete.
        insufficient_documentation_flags: Gaps explicitly listed when is_partial=True.
        tool_trace: Record of all tool calls made during extraction.

    Returns:
        TypedDict compatible with LangGraph StateGraph.
    """
    input_query: str
    documents_processed: List[str]
    extractions: List[dict]
    audit_results: List[dict]
    pending_user_input: bool
    clarification_needed: str
    clarification_response: str
    iteration_count: int
    confidence_score: float
    final_response: str
    error: Optional[str]
    routing_decision: str
    is_partial: bool
    insufficient_documentation_flags: List[str]
    tool_trace: List[dict]


def create_initial_state(query: str) -> AgentState:
    """
    Create a fresh AgentState with correct defaults for a new workflow run.

    Args:
        query: The natural language query to process.

    Returns:
        AgentState: Initialized state dict ready for graph invocation.

    Raises:
        Never — always returns a valid state dict.
    """
    return {
        "input_query": query,
        "documents_processed": [],
        "extractions": [],
        "audit_results": [],
        "pending_user_input": False,
        "clarification_needed": "",
        "clarification_response": "",
        "iteration_count": 0,
        "confidence_score": 0.0,
        "final_response": "",
        "error": None,
        "routing_decision": "",
        "is_partial": False,
        "insufficient_documentation_flags": [],
        "tool_trace": [],
    }
