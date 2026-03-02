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

Memory System fields (Layer 1 — Conversation History):
    messages: Full conversation history managed by LangGraph add_messages reducer.
        Nodes append new messages; the reducer merges without overwriting prior turns.
        Enables pronoun resolution ("his", "her") across turns without re-running Step 0.

Memory System fields (Layer 2 — Session Cache):
    session_id: Ties graph invocations to a single user thread for checkpointer scoping.
    tool_call_history: Ordered log of every tool called this session with status/timestamp.
    prior_query_context: Resolved patient name and query intent from the previous turn.
        Passed to Orchestrator so follow-up pronouns resolve without re-running Step 0 Haiku.
    extracted_patient: Cached patient dict. Orchestrator skips patient_lookup when populated.
    extracted_pdf_pages: Page-keyed dict of extracted PDF text. Valid only when
        extracted_pdf_hash matches the MD5 of the current pdf_source_file content.
    extracted_pdf_hash: MD5 hex digest of the raw bytes of the last extracted PDF.
        Hashes file content (not path) — catches same-path replacement of uploaded files.
    payer_policy_cache: Policy search results keyed by payer_id.
    denial_risk_cache: Denial analysis results keyed by "payer_id:cpt_code".

Orchestrator fields:
    tool_plan: Ordered list of tool name strings set by Orchestrator Node. Extractor
        executes only these tools in order. Empty list = general knowledge query — Extractor
        skips all tools with no Scenario A trigger or phantom confidence penalty.
    orchestrator_ran: True once Orchestrator has intentionally set tool_plan. Distinguishes
        orchestrator-set empty plan (general knowledge) from legacy/direct-test empty plan
        (backward compatible — runs full tool suite when False).
    identified_patient_name: Full patient name (or None) extracted by the Orchestrator's
        single Haiku call. Extractor reads this instead of calling _extract_patient_identifier_llm
        (Step 0), eliminating the second Haiku call per request and halving API load.
    orchestrator_fallback: Set to True when all Haiku retries are exhausted and the
        Orchestrator falls back to regex-based name extraction. Captured in state so
        LangSmith traces can surface 529 frequency over time.
    data_source_required: Classified by Orchestrator Haiku — the source the query needs:
        "EHR", "PDF", "RESIDENT_NOTE", "IMAGING", or "NONE". Output Node verifies this
        source was actually checked before synthesizing a response.
    source_unavailable: True when pdf_required=True but no PDF is attached. Short-circuits
        tool execution; Output Node Gate 1 emits "I don't have access to..." response.
    source_unavailable_reason: The data_source_required value that triggered the flag
        (e.g. "RESIDENT_NOTE"). Used by Output Node to name the missing source clearly.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


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

        --- Memory System Fields ---
        messages: Full conversation history (Layer 1). Uses LangGraph add_messages reducer —
            new messages are appended automatically; prior turns are never overwritten.
        session_id: Caller-supplied session identifier threaded through to SQLite checkpointer
            as thread_id for cross-restart state persistence (Layer 3).
        tool_call_history: Ordered log of every tool called this session with status and
            timestamp. Used by Orchestrator to detect redundant calls (Layer 2).
        prior_query_context: Lightweight summary of the previous turn — intent + resolved
            patient identifier — so follow-up pronouns resolve without re-running Step 0.
        extracted_patient: Cached patient dict from the current session (Layer 2 cache).
            Orchestrator skips patient_lookup when this is populated.
        extracted_pdf_pages: Page-keyed dict of extracted PDF text (Layer 2 cache). Only
            valid when extracted_pdf_hash matches the MD5 of the current pdf_source_file.
        extracted_pdf_hash: MD5 hex digest of the raw bytes of the last extracted PDF.
            Compared against the hash of the current pdf_source_file before using the cache.
        payer_policy_cache: Policy search results keyed by payer_id (Layer 2 cache).
        denial_risk_cache: Denial analysis results keyed by "payer_id:cpt_code" (Layer 2 cache).
        tool_plan: Ordered list of tool name strings produced by Orchestrator Node. The
            Extractor executes only these tools, in order. Empty list = general knowledge
            query — Extractor skips all tools and passes directly to Auditor.

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
    payer_id: str
    procedure_code: str
    denial_risk: dict
    allergy_conflict_result: dict
    ehr_confidence_penalty: int
    citation_anchors: List[dict]

    # Memory System — Layer 1: Conversation History
    messages: Annotated[list, add_messages]

    # Memory System — Layer 2: Session Context Cache
    session_id: str
    tool_call_history: List[dict]
    prior_query_context: dict
    extracted_patient: dict
    extracted_pdf_pages: dict
    extracted_pdf_hash: str
    payer_policy_cache: dict
    denial_risk_cache: dict

    # Orchestrator
    tool_plan: List[str]
    orchestrator_ran: bool      # True once Orchestrator Node has set tool_plan intentionally.
    identified_patient_name: Optional[str]  # Patient name extracted by Orchestrator's single Haiku call.
    identified_patient_dob: Optional[str]   # Patient DOB (ISO YYYY-MM-DD) when stated in query or from PDF; for composite-key identity resolution.
    orchestrator_fallback: bool  # True when all Haiku retries failed and regex fallback was used.
    data_source_required: str   # Data source the query needs: EHR / PDF / RESIDENT_NOTE / IMAGING / NONE.
    source_unavailable: bool    # True when required source (PDF/note) was not attached.
    source_unavailable_reason: str  # The data_source_required value that triggered the unavailable flag.
    is_general_knowledge: bool  # True ONLY for factual/pharmacology queries with no patient or doc needed.
                                # Extractor reads this — not tool_plan=[] — to trigger the bypass,
                                # so patient cache-hit paths (also tool_plan=[]) are not incorrectly skipped.

    # ── Human-in-the-Loop Sync Confirmation (HITL) ──────────────────────────
    # Set by comparison_node after a PDF is processed.  Cleared by
    # sync_execution_node on completion, or by orchestrator_node when the
    # user uploads a new PDF or responds with anything other than a sync
    # confirmation — preventing stale flags from bleeding across patients.
    pending_sync_confirmation: bool   # True = waiting for user "yes/sync" before posting to OpenEMR.
    sync_summary: dict                # {"new": [...], "existing": [...], "total_raw": int}
                                      #   new      — unique (marker, value) pairs not yet SYNCED
                                      #   existing — pairs already SYNCED in a prior session
                                      #   total_raw — total PENDING rows before dedup
    staged_patient_fhir_id: str       # FHIR UUID for run_sync(); sourced from patient["fhir_id"].
    staged_session_id: str            # evidence_staging session_id to sync when user confirms.


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
        "payer_id": "",
        "procedure_code": "",
        "denial_risk": {},
        "allergy_conflict_result": {},
        "ehr_confidence_penalty": 0,
        "citation_anchors": [],
        # Memory System defaults
        "messages": [],
        "session_id": "",
        "tool_call_history": [],
        "prior_query_context": {},
        "extracted_patient": {},
        "extracted_pdf_pages": {},
        "extracted_pdf_hash": "",
        "payer_policy_cache": {},
        "denial_risk_cache": {},
        "tool_plan": [],
        "orchestrator_ran": False,
        "identified_patient_name": None,
        "identified_patient_dob": None,
        "orchestrator_fallback": False,
        "data_source_required": "NONE",
        "source_unavailable": False,
        "source_unavailable_reason": "",
        "is_general_knowledge": False,
        # HITL sync fields — always initialised to safe defaults
        "pending_sync_confirmation": False,
        "sync_summary": {},
        "staged_patient_fhir_id": "",
        "staged_session_id": "",
    }
