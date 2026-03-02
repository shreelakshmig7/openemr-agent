"""
workflow.py
-----------
AgentForge — Healthcare RCM AI Agent — LangGraph workflow assembler
--------------------------------------------------------------------
Assembles the full LangGraph state machine and exposes run_workflow()
as the single entry point for main.py.

Graph topology (Human-in-the-Loop HITL sync):
    START → router → (OUT_OF_SCOPE → output → END; else → orchestrator)
    orchestrator → (if pending_sync_confirmation confirmed → sync_execution → output → END)
    orchestrator → extractor
    extractor → (if pending_user_input → clarification → END; else → auditor)
    auditor → output (always — synthesises clinical response first)
    output  → comparison (PDF path: appends sync prompt to synthesised response → END)
            → END        (non-PDF path)
    comparison → END  (sync prompt already appended; never loops back to output)

Memory System:
    Layer 1 (Conversation History): messages field with add_messages reducer.
    Layer 2 (Session Cache): extracted_patient, extracted_pdf_pages/hash,
        payer_policy_cache, denial_risk_cache — checked by Orchestrator before
        adding tools to plan; written by Extractor after tool calls.
    Layer 3 (Persistent State): SqliteSaver checkpointer writes full AgentState
        to agent_checkpoints.sqlite after every node. Survives server restarts.
        thread_id = session_id passed from main.py.

Key functions:
    run_workflow: Entry point — creates fresh state, builds graph, invokes it.
    _build_graph: Assembles and compiles the LangGraph StateGraph with checkpointer.
    _route_from_router: Conditional edge — routes OUT_OF_SCOPE to output, clinical to orchestrator.
    _route_from_auditor: Conditional edge function reading routing_decision.
    _output_node: Formats the final state for API response, filters by query_intent.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import os
from typing import Any, Optional

try:
    from langchain.callbacks.tracers import LangChainTracer
except ImportError:
    LangChainTracer = None  # optional: run without LangSmith when langchain not installed
from langgraph.graph import StateGraph, END

from langgraph_agent.state import AgentState, create_initial_state
from langgraph_agent.router_node import router_node, INTENT_OUT_OF_SCOPE
from langgraph_agent.orchestrator_node import orchestrator_node
from langgraph_agent.extractor_node import extractor_node
from langgraph_agent.auditor_node import auditor_node, _synthesize_response, _build_citations_part
from langgraph_agent.clarification_node import clarification_node, resume_from_clarification
from langgraph_agent.comparison_node import comparison_node
from langgraph_agent.sync_execution_node import sync_execution_node
from healthcare_guidelines import CLINICAL_SAFETY_RULES

# ── SQLite checkpointer (Layer 3 — Persistent Memory) ─────────────────────────
# Writes the full AgentState to SQLite after every node so state survives
# server restarts and session interruptions. Falls back gracefully if the
# langgraph-checkpoint-sqlite package is not installed.
_CHECKPOINT_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "agent_checkpoints.sqlite",
)

def _get_checkpointer():
    """
    Return a SqliteSaver instance for persistent state checkpointing, or None
    if the package is unavailable. SqliteSaver.from_conn_string() returns a
    context manager — callers must use it inside a `with` block (see run_workflow
    and get_state_for_audit). Do not pass the return value of from_conn_string
    directly to graph.compile(); it causes AttributeError on get_next_version.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver.from_conn_string(_CHECKPOINT_DB)
    except Exception:
        return None


# ── Output Node ───────────────────────────────────────────────────────────────

def _filter_extractions_by_intent(extractions: list, query_intent: str) -> list:
    """
    Filter the full extractions list to only what the query asked for,
    using the query_intent set by the Router Node.

    Why this exists: before intent filtering, every query returned the full
    data dump. "Does he have any allergies?" returned medications + allergies
    + interactions. See agent_reference.md Part 14 Hotfix 3.

    Args:
        extractions: Full list of extraction dicts from the Extractor Node.
        query_intent: Intent label set by router_node (e.g. "ALLERGIES").

    Returns:
        list: Filtered subset of extractions matching the intent.
              Returns all extractions if intent is GENERAL_CLINICAL or unrecognized.

    Raises:
        Never — returns all extractions on any failure.
    """
    try:
        if not extractions or not query_intent:
            return extractions

        def _kind(e: dict) -> str:
            """Return semantic kind tag; fall back to source-string heuristic."""
            k = e.get("kind", "")
            if k:
                return k
            src = e.get("source", "").lower()
            # PDF sources always get their own kind so they survive every filter.
            if src.endswith(".pdf"):
                return "pdf_content"
            if "interaction" in src:
                return "interaction"
            if "medication" in src:
                return "medication"
            if "patient" in src or "allerg" in src:
                return "allergy"
            if "openemr_fhir" in src or "ehr" in src:
                # FHIR extractions — use claim text to disambiguate
                claim = e.get("claim", "").lower()
                if "prescribed" in claim or "is on" in claim or "taking" in claim:
                    return "medication"
                if "allerg" in claim:
                    return "allergy"
            return "unknown"

        if query_intent == "MEDICATIONS":
            # Include medications, interactions, AND any PDF content (the PDF may
            # list medications or contraindications relevant to the question).
            return [e for e in extractions if _kind(e) in ("medication", "interaction", "pdf_content")]

        if query_intent == "ALLERGIES":
            # Always include PDF content — the allergy answer may be in the attached note.
            return [e for e in extractions if _kind(e) in ("allergy", "pdf_content")]

        if query_intent == "INTERACTIONS":
            return [e for e in extractions if _kind(e) in ("interaction", "medication", "pdf_content")]

        if query_intent == "SAFETY_CHECK":
            # Include everything — allergy, meds, interactions, and PDF evidence.
            return list(extractions)

        # GENERAL_CLINICAL or OUT_OF_SCOPE (refusal already set) — return all
        return extractions
    except Exception:
        return extractions


def _output_node(state: AgentState) -> AgentState:
    """
    Output Node — three-gate source integrity stack, then filters and synthesizes.

    Gate 1 — Source unavailable: required PDF/note was not attached.
        Returns "I don't have access to [source]..." with no synthesis.
    Gate 2 — Source not checked: query asked about PDF/note but no PDF
        extraction exists in the checked sources set.
        Returns "I did not have access to that document..." with no synthesis.
    Gate 3 — LLM synthesis with SOURCE_INTEGRITY hard rule injected into prompt:
        Tells Sonnet which sources were actually checked so it cannot fabricate
        absence from an unchecked source.

    For OUT_OF_SCOPE queries, final_response is already set by the Router Node.
    For partial results, final_response is already set by the Auditor Node.

    Args:
        state: Final AgentState from Router (out_of_scope) or Auditor (pass/partial).

    Returns:
        AgentState: State with synthesized final_response and disclaimer appended.

    Raises:
        Never — returns state unchanged on any failure.
    """
    try:
        query_intent = state.get("query_intent", "")

        # OUT_OF_SCOPE and partial paths already have final_response set — just add disclaimer.
        if query_intent == INTENT_OUT_OF_SCOPE or state.get("is_partial"):
            if state.get("final_response") and not state["final_response"].endswith(CLINICAL_SAFETY_RULES["disclaimer"]):
                state["final_response"] += "\n\n" + CLINICAL_SAFETY_RULES["disclaimer"]
            if not state.get("confidence_score"):
                state["confidence_score"] = 0.0
            return state

        # ── Gate 1: required source was not attached at all ────────────────
        if state.get("source_unavailable"):
            source_label = (
                state.get("source_unavailable_reason", "that document")
                .lower()
                .replace("_", " ")
            )
            state["final_response"] = (
                f"I don't have access to {source_label} for this patient. "
                "To answer this question, please attach the relevant clinical document."
                "\n\n" + CLINICAL_SAFETY_RULES["disclaimer"]
            )
            state["confidence_score"] = 0.0
            return state

        # Filter extractions to what the query actually asked for.
        extractions = state.get("extractions", [])
        relevant = _filter_extractions_by_intent(extractions, query_intent)
        if not relevant:
            relevant = extractions

        # ── Gate 2: required source was not in what was actually checked ───
        # Catches the case where the orchestrator didn't short-circuit (e.g. a
        # PDF was once attached but the query now asks about a note/imaging source
        # and no PDF extraction exists in the current extractions set.
        query_source = state.get("data_source_required", "EHR")
        if query_source in ("PDF", "RESIDENT_NOTE", "IMAGING"):
            checked_sources = {e.get("source", "") for e in relevant}
            has_pdf_extraction = any(s.endswith(".pdf") for s in checked_sources)
            if not has_pdf_extraction:
                state["final_response"] = (
                    "I did not have access to that document for this query. "
                    "I cannot confirm the absence of information from a source I did not check."
                    "\n\n" + CLINICAL_SAFETY_RULES["disclaimer"]
                )
                state["confidence_score"] = 0.0
                return state

        # Build the checked_sources string for Gate 3 prompt injection.
        all_checked = {e.get("source", "") for e in relevant if e.get("source")}
        checked_sources_str = ", ".join(sorted(all_checked)) if all_checked else ""

        citations_part = _build_citations_part(relevant)

        # Build PDF-only citation anchors for the UI viewer.
        # EHR/mock sources (patients.json, medications.json, etc.) are excluded —
        # only PDF extractions carry page numbers and deserve a viewer link.
        # Anchors are sorted so the page most relevant to the query comes first —
        # the UI auto-jumps to anchors[0], so this ensures it opens the answer page.
        query_text = state.get("input_query", "").lower()
        _stop_words = {"the", "a", "an", "is", "are", "was", "were", "what",
                       "does", "do", "has", "have", "for", "in", "of", "to",
                       "and", "or", "with", "any", "this", "that", "be", "it"}
        query_words = {w for w in query_text.split() if w not in _stop_words}

        seen_anchors: set = set()
        anchor_scores: dict = {}
        citation_anchors = []
        for e in relevant:
            src = e.get("source", "")
            page = e.get("page_number")
            if not src.endswith(".pdf"):
                continue
            page_num = int(page) if page is not None else 1
            anchor_key = (src, page_num)
            if anchor_key in seen_anchors:
                continue
            seen_anchors.add(anchor_key)
            file_label = src.split("/")[-1]
            citation_anchors.append({
                "label": f"{file_label} p.{page_num}",
                "file": src,
                "page": page_num,
            })
            # Score by keyword overlap between the query and this extraction's claim.
            claim = e.get("claim", "").lower()
            anchor_scores[anchor_key] = (
                sum(1 for w in query_words if w in claim) if query_words else 0
            )

        # Put the highest-scoring (most query-relevant) page first so the UI
        # auto-jumps straight to the answer page.
        citation_anchors.sort(
            key=lambda a: anchor_scores.get((a["file"], a["page"]), 0),
            reverse=True,
        )
        state["citation_anchors"] = citation_anchors

        # ── Gate 3: synthesize with SOURCE_INTEGRITY hard rule injected ────
        # ehr_unavailable=True when Scenario A fired (patient not in EHR, PDF is
        # the sole source of truth). Directs synthesis to answer from PDF facts
        # instead of applying the EHR-gated SAFETY_CHECK "no conflict" template.
        ehr_unavailable = state.get("ehr_confidence_penalty", 0) > 0
        synthesized = _synthesize_response(
            query=state.get("input_query", ""),
            query_intent=query_intent,
            extractions=relevant,
            allergy_conflict=state.get("allergy_conflict_result"),
            denial_risk=state.get("denial_risk"),
            checked_sources_str=checked_sources_str,
            ehr_unavailable=ehr_unavailable,
        )
        if synthesized:
            body = synthesized
        else:
            body = " ".join(e["claim"] for e in relevant if e.get("claim"))

        citations_block = f" [{citations_part}]" if citations_part else ""
        state["final_response"] = body + citations_block + "\n\n" + CLINICAL_SAFETY_RULES["disclaimer"]

        if not state.get("confidence_score"):
            state["confidence_score"] = 0.0
        return state
    except Exception:
        return state


# ── Conditional edge routers ──────────────────────────────────────────────────

def _route_from_router(state: AgentState) -> str:
    """
    After Router Node: if query is OUT_OF_SCOPE route directly to output
    (refusal already set in final_response); otherwise route to orchestrator.

    HITL OVERRIDE: if a sync confirmation is pending, always route to
    orchestrator regardless of how the router classified the message.
    The router sees "Yes" / "Sync" as OUT_OF_SCOPE (no clinical keywords),
    but the orchestrator's HITL pre-check is the correct handler for those
    one-word confirmations.

    Args:
        state: Current AgentState after Router Node has run.

    Returns:
        str: "output" for out-of-scope queries, "orchestrator" for all clinical intents.

    Raises:
        Never — returns "orchestrator" as safe fallback.
    """
    # HITL override: a pending sync confirmation must reach the orchestrator's
    # pre-check, which is the only node that knows how to handle "yes"/"sync".
    if state.get("pending_sync_confirmation"):
        return "orchestrator"

    if state.get("routing_decision") == "out_of_scope":
        return "output"
    return "orchestrator"


def _route_from_extractor(state: AgentState) -> str:
    """
    After Extractor: if Step 0 set pending_user_input (ambiguous or no identifier),
    route to clarification; otherwise to auditor.
    """
    if state.get("pending_user_input") and state.get("clarification_needed"):
        return "clarification"
    return "auditor"


def _pdf_was_processed(state: AgentState) -> bool:
    """
    Return True if a clinical PDF was processed (freshly or from cache) in
    the current session and at least one element was scanned.

    Three independent signals — any one being truthy is enough:
      1. pdf_source_file — PDF path was set in this turn's initial state.
      2. extracted_pdf_pages — PDF text cache is populated (cache-hit or
         fresh extraction on a follow-up turn).
      3. tool_trace contains a 'clinical_marker_scan' entry — markers were
         actually staged to evidence_staging this session, which is the
         most reliable signal that elements_scanned > 0.

    Args:
        state: Current AgentState.

    Returns:
        bool: True when any signal indicates a PDF was processed.
    """
    if state.get("pdf_source_file"):
        return True
    if state.get("extracted_pdf_pages"):
        return True
    for entry in state.get("tool_trace", []):
        if entry.get("tool") == "clinical_marker_scan":
            elements = (entry.get("input") or {}).get("elements_scanned", 0)
            if elements and int(elements) > 0:
                return True
    return False


def _route_from_auditor(state: AgentState) -> str:
    """
    Conditional edge after the Auditor Node.

    Always routes to output first so the clinical response is synthesised.
    The output node itself then conditionally hands off to comparison_node
    (see _route_from_output) so the sync prompt is appended AFTER the
    clinical summary — not before, which is what caused the silent discard.

    Args:
        state: Current AgentState after Auditor Node has run.

    Returns:
        str: Next node name — "output", "extractor", or "clarification".

    Raises:
        Never — returns "output" as safe fallback on any unexpected value.
    """
    decision = state.get("routing_decision", "")
    if decision == "pass":
        return "output"
    if decision == "missing":
        return "extractor"
    if decision == "ambiguous":
        return "clarification"
    if decision == "partial":
        return "output"
    return "output"


def _route_from_output(state: AgentState) -> str:
    """
    Conditional edge after the Output Node.

    Normal PDF path:
        auditor → output (synthesise) → comparison (append sync prompt) → END

    Post-sync guard:
        sync_execution sets routing_decision = "sync_complete" before writing
        final_response and exiting to END directly — it never passes through
        _output_node at all.  This guard exists as a safety net in case the
        graph wiring ever routes an unexpected state through output after sync,
        preventing comparison from re-showing the sync prompt.

    Args:
        state: AgentState after Output Node has written final_response.

    Returns:
        str: "comparison" for PDF turns, "end" for everything else.

    Raises:
        Never — returns "end" as safe fallback.
    """
    # Never run comparison after a sync turn (sentinel set by sync_execution_node).
    if state.get("routing_decision") == "sync_complete":
        return "end"
    if _pdf_was_processed(state):
        return "comparison"
    return "end"


def _route_from_orchestrator(state: AgentState) -> str:
    """
    Conditional edge after orchestrator_node.

    If the user's message is a sync confirmation (pending_sync_confirmation
    was True and orchestrator cleared it, setting routing_decision="sync"),
    route to sync_execution_node instead of extractor.

    In all other cases, route to extractor as usual.

    Args:
        state: Current AgentState after Orchestrator Node has run.

    Returns:
        str: "sync_execution" or "extractor".
    """
    if state.get("routing_decision") == "sync":
        return "sync_execution"
    return "extractor"


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph(checkpointer: Optional[Any] = None):
    """
    Assemble and compile the LangGraph StateGraph.
    Graph is built fresh on each run_workflow call to ensure mocks in tests
    are applied at call time, not at module import time.

    Graph topology:
        router → orchestrator → extractor → auditor → output
        (with clarification pause/resume loop)

    Args:
        checkpointer: Optional SqliteSaver instance for Layer 3 persistence.
            When provided, full AgentState is written to SQLite after every node.

    Returns:
        CompiledGraph: Ready-to-invoke LangGraph graph.

    Raises:
        Never — propagates LangGraph compilation errors to run_workflow.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("auditor", auditor_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("comparison", comparison_node)
    graph.add_node("sync_execution", sync_execution_node)
    graph.add_node("output", _output_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_from_router,
        {"output": "output", "orchestrator": "orchestrator"},
    )
    # Orchestrator routes to sync_execution when user confirmed a pending sync,
    # otherwise proceeds normally to extractor.
    graph.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        {"sync_execution": "sync_execution", "extractor": "extractor"},
    )
    graph.add_conditional_edges(
        "extractor",
        _route_from_extractor,
        {"clarification": "clarification", "auditor": "auditor"},
    )
    graph.add_conditional_edges(
        "auditor",
        _route_from_auditor,
        {
            "output":        "output",
            "extractor":     "extractor",
            "clarification": "clarification",
        },
    )
    # output → comparison (PDF path) or END (non-PDF / post-sync path).
    # comparison appends the sync prompt to the already-synthesised response,
    # then exits to END — it never loops back through output again.
    graph.add_conditional_edges(
        "output",
        _route_from_output,
        {"comparison": "comparison", "end": END},
    )
    # comparison always exits to END — response is already fully built.
    graph.add_edge("comparison", END)
    # sync_execution exits to END directly — it builds its own complete
    # final_response (with disclaimer) and sets routing_decision="sync_complete"
    # so _route_from_output's guard never fires even on unexpected re-entry.
    # Bypassing _output_node prevents the sync result from being overwritten
    # by a fresh LLM synthesis and stops comparison from re-running.
    graph.add_edge("sync_execution", END)
    # Clarification pauses here — workflow ends and waits for user input.
    graph.add_edge("clarification", END)

    return graph.compile(checkpointer=checkpointer)


def get_state_for_audit(thread_id: str) -> Optional[dict]:
    """
    Retrieve persisted state for a thread from the SQLite checkpointer.
    Used by GET /api/audit/{thread_id} when the in-memory session store does not
    have the state (e.g. after server restart). Uses SqliteSaver inside a
    context manager (from_conn_string returns a context manager, not the saver).

    Args:
        thread_id: Session/thread identifier used as thread_id in checkpointer config.

    Returns:
        dict | None: AgentState as a plain dict, or None if not found or on error.
    """
    if not thread_id:
        return None
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        with SqliteSaver.from_conn_string(_CHECKPOINT_DB) as checkpointer:
            graph = _build_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = graph.get_state(config)
            if snapshot and getattr(snapshot, "values", None):
                return dict(snapshot.values)
        return None
    except Exception:
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def run_workflow(
    query: str,
    session_id: Optional[str] = None,
    clarification_response: Optional[str] = None,
    pdf_source_file: Optional[str] = None,
    prior_state: Optional[dict] = None,
    payer_id: Optional[str] = None,
    procedure_code: Optional[str] = None,
) -> dict:
    """
    Run the full LangGraph multi-agent workflow for a given query.

    Creates a fresh AgentState, merges any session cache carried forward from
    prior_state (Layer 2), optionally incorporates a clarification response,
    builds the graph with the SQLite checkpointer (Layer 3), and invokes it.

    Memory System integration:
        - Layer 2 (Session Cache): extracted_patient, extracted_pdf_pages/hash,
          payer_policy_cache, denial_risk_cache, tool_call_history, and
          prior_query_context are carried forward from prior_state so the
          Orchestrator can skip redundant tool calls on follow-up questions.
        - Layer 3 (Persistent State): graph is compiled with SqliteSaver.
          thread_id = session_id so all turns in a session share the same
          SQLite thread. State survives server restarts.

    Args:
        query: Natural language query from the user.
        session_id: Session identifier used as thread_id for checkpointing.
            A new UUID is used if not provided.
        clarification_response: Optional user response to a prior clarification question.
        pdf_source_file: Optional path to a clinical PDF to process alongside the query.
            When set, the Extractor Node runs extract_pdf() and merges results.
        prior_state: Optional previous AgentState dict from the caller (main.py).
            Layer 2 cache fields are merged into the new initial state.
        payer_id: Optional payer identifier for policy_search (e.g. "cigna", "aetna").
        procedure_code: Optional CPT/procedure code for policy_search (e.g. "27447").

    Returns:
        dict: Final AgentState as a plain dict, always includes 'final_response',
            'confidence_score', 'tool_trace', 'pending_user_input', 'error'.

    Raises:
        Never — all errors are captured in the returned dict's 'error' field.
    """
    try:
        # ── Clarification resume path ─────────────────────────────────────
        # When the prior turn paused for clarification, the full state from
        # that turn (Router intent, Orchestrator classification, pdf_source_file,
        # etc.) is preserved in prior_state. Use it directly as the base instead
        # of create_initial_state — which would wipe all prior classifications
        # and cause the Router/Orchestrator to re-run from scratch, losing the
        # original query context.
        if clarification_response and prior_state and isinstance(prior_state, dict):
            initial_state = dict(prior_state)
            initial_state = resume_from_clarification(initial_state, clarification_response)
            if session_id:
                initial_state["session_id"] = session_id
        else:
            # Normal path — fresh state for a new query.
            initial_state = create_initial_state(query)

            if session_id:
                initial_state["session_id"] = session_id

            if pdf_source_file and pdf_source_file.strip():
                initial_state["pdf_source_file"] = pdf_source_file.strip()

            if payer_id:
                initial_state["payer_id"] = payer_id
            if procedure_code:
                initial_state["procedure_code"] = procedure_code

            # Layer 2 — merge session cache from previous turn.
            # pdf_source_file is included so follow-up queries can use the
            # previously extracted PDF without re-attaching it.
            # The `not initial_state.get(field)` guard ensures explicit new
            # uploads win over carried-forward values.
            #
            # HITL fields (pending_sync_confirmation, sync_summary,
            # staged_patient_fhir_id, staged_session_id) MUST be carried
            # forward so the orchestrator pre-check can detect a pending sync
            # confirmation on the next user turn (e.g. user replies "Yes").
            # Without these, create_initial_state() resets the flag to False
            # and "Yes" gets classified as an out-of-scope query.
            if prior_state and isinstance(prior_state, dict):
                _cache_fields = [
                    "extracted_patient",
                    "extracted_pdf_pages",
                    "extracted_pdf_hash",
                    "pdf_source_file",
                    "payer_policy_cache",
                    "denial_risk_cache",
                    "tool_call_history",
                    "prior_query_context",
                    "messages",
                    # HITL sync state — carried so "Yes" fires sync_execution
                    "pending_sync_confirmation",
                    "sync_summary",
                    "staged_patient_fhir_id",
                    "staged_session_id",
                ]
                for field in _cache_fields:
                    value = prior_state.get(field)
                    # Boolean fields need a special guard: `if value` would
                    # skip False, but pending_sync_confirmation=True must pass.
                    if field == "pending_sync_confirmation":
                        if value is True:
                            initial_state[field] = True
                    elif value and not initial_state.get(field):
                        initial_state[field] = value

        # Layer 3 — SqliteSaver.from_conn_string() returns a context manager; we must
        # use it inside a `with` block so the graph gets the actual checkpointer,
        # not the context manager (else: AttributeError on get_next_version).
        invoke_config = {}
        if session_id:
            invoke_config["configurable"] = {"thread_id": session_id}

        # LangSmith tracing when API key is set — node-level spans with latency per hop.
        if LangChainTracer and (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
            try:
                tracer = LangChainTracer(project_name="agentforge-rcm")
                invoke_config["callbacks"] = [tracer]
            except Exception:
                pass  # tracing is optional — don't block the pipeline

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            with SqliteSaver.from_conn_string(_CHECKPOINT_DB) as checkpointer:
                graph = _build_graph(checkpointer=checkpointer)
                result = graph.invoke(
                    initial_state,
                    config=invoke_config if invoke_config else None,
                )
        except Exception:
            # Fallback: run without persistence (e.g. package missing or DB error)
            graph = _build_graph(checkpointer=None)
            result = graph.invoke(
                initial_state,
                config=invoke_config if invoke_config else None,
            )

        result_dict = dict(result)

        if "tool_trace" not in result_dict:
            result_dict["tool_trace"] = []
        if "error" not in result_dict:
            result_dict["error"] = None

        return result_dict

    except Exception as e:
        return {
            "input_query": query,
            "query_intent": "",
            "proposed_drug": "",
            "final_response": f"An unexpected error occurred: {str(e)}",
            "confidence_score": 0.0,
            "tool_trace": [],
            "pending_user_input": False,
            "clarification_needed": "",
            "clarification_response": "",
            "iteration_count": 0,
            "extractions": [],
            "audit_results": [],
            "documents_processed": [],
            "routing_decision": "error",
            "is_partial": False,
            "insufficient_documentation_flags": [],
            "extracted_patient_identifier": {},
            "error": str(e),
        }
