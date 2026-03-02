"""
clarification_node.py
---------------------
AgentForge — Healthcare RCM AI Agent — LangGraph Clarification Node
--------------------------------------------------------------------
Implements the Clarification Node in the LangGraph state machine.
When the Auditor detects ambiguous input, this node pauses the entire
workflow by ensuring pending_user_input=True. All existing state
(extractions, iteration_count, documents_processed) is preserved —
no work is lost.

Provides resume_from_clarification() to incorporate the user's response
and route the workflow back to the Extractor Node.

Key functions:
    clarification_node: Pauses workflow, scrubs PII from question, preserves state.
    resume_from_clarification: Writes user response into state, routes to Extractor.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from langgraph_agent.state import AgentState
from tools.pii_scrubber import scrub_pii


def _scrub_pii_from_question(text: str) -> str:
    """
    Remove HIPAA PII from the clarification question before surfacing to the user.
    Delegates to Presidio-based scrub_pii() (NLP + custom pattern recognizers).

    Args:
        text: Raw clarification question that may contain PII.

    Returns:
        str: Question with PII replaced by typed placeholders (e.g. <PERSON>).

    Raises:
        Never — scrub_pii() itself never raises; returns original text on failure.
    """
    return scrub_pii(text)


# ── Clarification Node ────────────────────────────────────────────────────────

def clarification_node(state: AgentState) -> AgentState:
    """
    Clarification Node — pauses the workflow and preserves all existing state.
    Scrubs PII from the clarification question before it is surfaced to the user.
    Does not clear extractions, iteration_count, or any other state fields.

    Args:
        state: Current AgentState with pending_user_input=True set by Auditor.

    Returns:
        AgentState: State unchanged except clarification_needed is PII-scrubbed.

    Raises:
        Never — errors are caught and state is returned as-is.
    """
    try:
        raw_question = state.get("clarification_needed", "")
        state["clarification_needed"] = _scrub_pii_from_question(raw_question)
        state["pending_user_input"] = True
        return state
    except Exception:
        return state


# ── Resume from clarification ─────────────────────────────────────────────────

def resume_from_clarification(state: AgentState, user_response: str) -> AgentState:
    """
    Incorporate the user's clarification response into state and route back to Extractor.
    Clears pending_user_input, writes user response into clarification_response,
    and sets routing_decision to 'extractor' so the workflow resumes at Extractor Node.

    Args:
        state: Paused AgentState with pending_user_input=True.
        user_response: The user's answer to the clarification question.

    Returns:
        AgentState: Updated state ready to re-enter the Extractor Node.

    Raises:
        Never — returns state unchanged on any failure.
    """
    try:
        state["pending_user_input"] = False
        state["clarification_response"] = user_response
        state["clarification_needed"] = ""
        state["routing_decision"] = "extractor"
        return state
    except Exception:
        return state
