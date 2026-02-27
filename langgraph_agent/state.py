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
    query_intent: Set by Router Node — classifies query type before Extractor runs.
    pending_user_input: Pauses entire workflow without discarding work.
    iteration_count: Enforces 3-iteration ceiling on Auditor review loop.
    routing_decision: Set by Router/Auditor to control conditional edge routing.
    pdf_source_file: Optional PDF path to process alongside the text query.
    denial_risk: Set by Extractor Node — result of analyze_denial_risk() on all extractions.

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
        query_intent: Set by Router Node — one of MEDICATIONS | ALLERGIES | INTERACTIONS |
            SAFETY_CHECK | GENERAL_CLINICAL | OUT_OF_SCOPE. Controls Output Node filtering.
        proposed_drug: Set by Extractor Node when query_intent is SAFETY_CHECK. The drug
            name extracted from the query (e.g. "Penicillin" from "Can I give him penicillin?").
            Used by Output Node to check allergy and interaction extractions for a verdict.
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
        routing_decision: Set by Router/Auditor to control graph edge routing.
            Values: "pass" | "missing" | "ambiguous" | "partial" | "out_of_scope"
        is_partial: True when iteration ceiling was hit and response is incomplete.
        insufficient_documentation_flags: Gaps explicitly listed when is_partial=True.
        tool_trace: Record of all tool calls made during extraction.
        extracted_patient_identifier: Result of Step 0 (Haiku) — type, value, ambiguous, reason.
        pdf_source_file: Optional path to a clinical PDF to process alongside the text query.
            When set, the Extractor Node runs extract_pdf() and merges results into extractions.
        denial_risk: Set by Extractor Node after all extractions are collected. Contains the
            result of analyze_denial_risk() — risk_level, matched_patterns, recommendations,
            and denial_risk_score (0.0–1.0).
        allergy_conflict_result: Set by Extractor Node for SAFETY_CHECK queries. Contains the
            result of check_allergy_conflict() — conflict, drug, allergy, conflict_type, severity.
            Passed to the Auditor Node for use in response synthesis.
        ehr_confidence_penalty: Confidence penalty (0–100) applied when EHR data is unavailable.
            Set to 45 when patient is None (Scenario A: PDF + unknown patient). Subtracted from
            the base confidence score in the Auditor Node to guarantee escalation below 90%.
        citation_anchors: List of PDF citation anchor dicts built by Output Node. Each entry
            contains label, file, and page so the UI can open the PDF viewer and jump to the
            exact page. Only populated for PDF-sourced extractions — EHR/mock sources are excluded.
            Schema: [{"label": str, "file": str, "page": int}]

    Returns:
        TypedDict compatible with LangGraph StateGraph.
    """
    input_query: str
    query_intent: str
    proposed_drug: str
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
    extracted_patient_identifier: dict
    pdf_source_file: str
    denial_risk: dict
    allergy_conflict_result: dict
    ehr_confidence_penalty: int
    citation_anchors: List[dict]


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
        "query_intent": "",
        "proposed_drug": "",
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
        "extracted_patient_identifier": {},
        "pdf_source_file": "",
        "denial_risk": {},
        "allergy_conflict_result": {},
        "ehr_confidence_penalty": 0,
        "citation_anchors": [],
    }
