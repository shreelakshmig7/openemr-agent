"""
conversation.py
---------------
AgentForge — Healthcare RCM AI Agent — Multi-turn conversation manager
-----------------------------------------------------------------------
This module manages conversation history across multiple turns so the
agent can resolve references like "his medications" or "she" to the
previously discussed patient. History is prepended to each new message
when invoking the agent.

Key functions:
    - create_conversation_agent: returns a new agent and empty history list
    - chat: sends a message with history context, returns (response, updated_history)

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from typing import List, Tuple, Any
from langsmith import traceable
from agent import create_agent


# Message shown when user sends empty or whitespace-only input (PRD 6.5).
EMPTY_INPUT_MESSAGE = (
    "Please provide your question or request. I can help you look up patient "
    "medications, check for drug interactions, and verify allergy conflicts."
)

# Safe error message when agent call fails (PRD 6.5 — no raw exceptions to user).
AGENT_ERROR_MESSAGE = (
    "I encountered an issue processing your request. Please try again or "
    "consult your clinical staff for immediate assistance."
)

# When agent is None or invalid.
INVALID_AGENT_MESSAGE = (
    "The assistant is not available. Please try again later or consult a "
    "qualified healthcare professional."
)


def create_conversation_agent() -> Tuple[Any, List[dict]]:
    """
    Create a new agent instance and an empty conversation history list.

    Returns:
        Tuple[Any, List[dict]]: (AgentExecutor, history). history is a list of
            {"human": str, "ai": str} dicts, initially empty.
    """
    agent = create_agent()
    history = []
    return agent, history


def _normalize_output(raw_output: Any) -> str:
    """
    Convert agent output (string or list of message blocks) to a single string.

    Args:
        raw_output: The "output" value from agent.invoke(); may be str or list.

    Returns:
        str: Plain text of the agent response.
    """
    if raw_output is None:
        return ""
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, list):
        parts = []
        for item in raw_output:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(raw_output)

@traceable
def chat(agent: Any, history: List[dict], message: str) -> Tuple[str, List[dict]]:
    """
    Send a message to the agent with conversation history prepended; return
    response and updated history. Handles empty input and failures without
    raising exceptions.

    Args:
        agent: AgentExecutor from create_conversation_agent (or None to test failure path).
        history: List of {"human": str, "ai": str} for previous turns.
        message: Current user message.

    Returns:
        Tuple[str, List[dict]]: (response_text, updated_history). On failure,
            response_text is a safe error message and history is unchanged.
    """
    if agent is None:
        return (INVALID_AGENT_MESSAGE, list(history))

    if not isinstance(message, str):
        message = str(message) if message is not None else ""

    stripped = message.strip()
    if not stripped:
        return (EMPTY_INPUT_MESSAGE, list(history))

    try:
        # Build full input: previous turns + current message (PRD 6.4 — context).
        parts = []
        for turn in history:
            human = turn.get("human", "")
            ai = turn.get("ai", "")
            parts.append(f"User: {human}\nAssistant: {ai}")
        parts.append(f"User: {stripped}")
        full_input = "\n\n".join(parts)

        response = agent.invoke({"input": full_input})
        output = response.get("output")
        response_text = _normalize_output(output)

        # Append this turn to history.
        new_history = list(history) + [{"human": stripped, "ai": response_text}]
        return (response_text, new_history)

    except Exception as e:
        return (AGENT_ERROR_MESSAGE, list(history))
