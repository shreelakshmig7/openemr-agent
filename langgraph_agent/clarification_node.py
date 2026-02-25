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

import re

from langgraph_agent.state import AgentState


# ── PII scrubber (same stub as extractor_node) ────────────────────────────────

# TODO: Replace with Microsoft Presidio in PII infrastructure PR
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN_PATTERN = re.compile(r"\bMRN[:\s]*\w+\b", re.IGNORECASE)
_DOB_PATTERN = re.compile(r"\b(DOB|Date of Birth)[:\s]*[\d/\-]+\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"\b\d{3}[.\-]\d{3}[.\-]\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")


def _scrub_pii_from_question(text: str) -> str:
    """
    Remove HIPAA PII patterns from the clarification question before surfacing to user.

    Args:
        text: Raw clarification question that may contain PII patterns.

    Returns:
        str: Question with PII replaced by [REDACTED] placeholders.

    Raises:
        Never — returns original text unchanged if scrubbing fails.
    """
    try:
        text = _SSN_PATTERN.sub("[REDACTED-SSN]", text)
        text = _MRN_PATTERN.sub("[REDACTED-MRN]", text)
        text = _DOB_PATTERN.sub("[REDACTED-DOB]", text)
        text = _PHONE_PATTERN.sub("[REDACTED-PHONE]", text)
        text = _EMAIL_PATTERN.sub("[REDACTED-EMAIL]", text)
        return text
    except Exception:
        return text


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
