"""
router_node.py
--------------
AgentForge — Healthcare RCM AI Agent — LangGraph Supervisory Router Node
-------------------------------------------------------------------------
Implements the Supervisory Router Node — the first node in the LangGraph
state machine. Classifies every incoming query into one of six intents
before any patient data or tools are touched.

Clinical intents (MEDICATIONS, ALLERGIES, INTERACTIONS, SAFETY_CHECK,
GENERAL_CLINICAL) are routed to the Extractor Node. OUT_OF_SCOPE queries
receive an immediate professional refusal and exit without touching any
patient data.

Why this node exists (not in PRD — see agent_reference.md Part 14):
    Before this node, queries like "talk like a pirate" were returning full
    patient medical records because session context prepended patient names
    to every follow-up. This node intercepts all queries first and refuses
    non-healthcare queries before any patient data is accessed.

Key functions:
    router_node: Main node function — classifies intent, sets state fields.
    _classify_intent_llm: Calls Haiku to return one of 6 intent labels.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_agent.state import AgentState
from healthcare_guidelines import CLINICAL_SAFETY_RULES


# ── Intent constants ──────────────────────────────────────────────────────────

INTENT_MEDICATIONS = "MEDICATIONS"
INTENT_ALLERGIES = "ALLERGIES"
INTENT_INTERACTIONS = "INTERACTIONS"
INTENT_SAFETY_CHECK = "SAFETY_CHECK"
INTENT_GENERAL_CLINICAL = "GENERAL_CLINICAL"
INTENT_OUT_OF_SCOPE = "OUT_OF_SCOPE"

CLINICAL_INTENTS = {
    INTENT_MEDICATIONS,
    INTENT_ALLERGIES,
    INTENT_INTERACTIONS,
    INTENT_SAFETY_CHECK,
    INTENT_GENERAL_CLINICAL,
}

OUT_OF_SCOPE_REFUSAL = (
    "I am a specialized Healthcare RCM agent. "
    "I can only assist with clinical documentation, patient medications, "
    "drug interactions, allergy checks, and insurance verification."
)

# ── Router system prompt ───────────────────────────────────────────────────────

_ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a Healthcare RCM AI agent.
Classify the user query into exactly ONE of these six labels. Return only the label — no explanation.

Labels:
- MEDICATIONS     : Query asks what medications a patient is taking or prescribed.
- ALLERGIES       : Query asks about a patient's known allergies or allergy review for a procedure.
- INTERACTIONS    : Query asks about drug-drug interactions for a patient.
- SAFETY_CHECK    : Query asks whether it is safe to give/prescribe a specific drug to a patient.
- GENERAL_CLINICAL: Query is about healthcare, clinical data, patient information, insurance/payer
                    policy verification, prior authorization, CPT code criteria, clinical note review,
                    PDF document review, denial risk, or any other RCM/clinical topic not in the above.
- OUT_OF_SCOPE    : Query is completely unrelated to healthcare, clinical data, or patient information.

Examples:
  "What medications is John Smith on?"                              → MEDICATIONS
  "Does she have any allergies?"                                    → ALLERGIES
  "Review John Smith's allergies for surgical prophylaxis"          → ALLERGIES
  "Check drug interactions for Mary Johnson"                        → INTERACTIONS
  "Is it safe to give Robert Davis Aspirin?"                        → SAFETY_CHECK
  "Can I give penicillin to him?"                                   → SAFETY_CHECK
  "What conditions does John have?"                                 → GENERAL_CLINICAL
  "Verify if John Smith meets Cigna Medical Policy #012 for CPT 27447" → GENERAL_CLINICAL
  "Does this patient meet Aetna criteria for knee replacement?"     → GENERAL_CLINICAL
  "Review this clinical note for prior authorization"               → GENERAL_CLINICAL
  "What is the denial risk for this claim?"                         → GENERAL_CLINICAL
  "Analyze this PDF for insurance coverage"                         → GENERAL_CLINICAL
  "Talk like a pirate"                                              → OUT_OF_SCOPE
  "What is the weather in Austin?"                                  → OUT_OF_SCOPE
  "Write me a poem"                                                 → OUT_OF_SCOPE
"""


# ── LLM intent classifier ─────────────────────────────────────────────────────

def _classify_intent_llm(query: str) -> str:
    """
    Call Claude Haiku to classify the query into one of six intent labels.
    Returns the label as a plain uppercase string.

    Args:
        query: The raw user query (may include session context prepend).

    Returns:
        str: One of MEDICATIONS | ALLERGIES | INTERACTIONS | SAFETY_CHECK |
             GENERAL_CLINICAL | OUT_OF_SCOPE.
             Falls back to GENERAL_CLINICAL on any LLM or parse failure
             to avoid falsely refusing clinical queries.

    Raises:
        Never — all failures return the GENERAL_CLINICAL fallback.
    """
    try:
        llm = ChatAnthropic(
            model="claude-haiku-4-5",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
            max_tokens=16,
        )
        response = llm.invoke([
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        label = content.strip().upper().split()[0] if content.strip() else ""
        if label in CLINICAL_INTENTS or label == INTENT_OUT_OF_SCOPE:
            return label
        logger.warning("Router LLM returned unrecognized label '%s' — defaulting to GENERAL_CLINICAL", label)
        return INTENT_GENERAL_CLINICAL
    except Exception as e:
        logger.exception("Router intent classification failed — defaulting to GENERAL_CLINICAL: %s", e)
        return INTENT_GENERAL_CLINICAL


# ── Router Node ───────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """
    Supervisory Router Node — entry point of the LangGraph graph.
    Classifies query intent and either routes to Extractor (clinical)
    or sets a professional refusal response (out-of-scope).

    For OUT_OF_SCOPE: sets final_response to the standard refusal string,
    sets routing_decision to 'out_of_scope', and sets confidence_score to
    1.0 (the refusal is certain — no escalation needed).

    For clinical intents: sets query_intent in state and returns — the
    conditional edge in workflow.py routes to the Extractor Node.

    Args:
        state: Current AgentState at graph entry.

    Returns:
        AgentState: Updated with query_intent and, if out-of-scope,
            final_response and routing_decision set.

    Raises:
        Never — errors default to GENERAL_CLINICAL to avoid blocking clinical queries.
    """
    try:
        query = state.get("input_query", "")
        intent = _classify_intent_llm(query)

        state["query_intent"] = intent

        if intent == INTENT_OUT_OF_SCOPE:
            state["routing_decision"] = "out_of_scope"
            state["final_response"] = OUT_OF_SCOPE_REFUSAL
            state["confidence_score"] = 1.0
            state["extractions"] = []
            state["tool_trace"] = []
            logger.info("Router: OUT_OF_SCOPE query refused — '%s'", query[:80])
        else:
            logger.info("Router: classified as %s — '%s'", intent, query[:80])

        return state

    except Exception as e:
        logger.exception("Router node error — defaulting to GENERAL_CLINICAL: %s", e)
        state["query_intent"] = INTENT_GENERAL_CLINICAL
        return state
