"""
workflow.py
-----------
AgentForge — Healthcare RCM AI Agent — LangGraph workflow assembler
--------------------------------------------------------------------
Assembles the full LangGraph state machine from the five nodes and
exposes run_workflow() as the single entry point for main.py.

Graph topology:
    START → router → (OUT_OF_SCOPE → output → END; else → extractor)
    extractor → (if pending_user_input → clarification → END; else → auditor)
    auditor:
        "pass"                  → output → END
        "missing" (count < 3)   → extractor (review loop)
        "ambiguous"             → clarification → extractor (after user responds)
        "partial" (count >= 3)  → output → END

Key functions:
    run_workflow: Entry point — creates fresh state, builds graph, invokes it.
    _build_graph: Assembles and compiles the LangGraph StateGraph.
    _route_from_router: Conditional edge — routes OUT_OF_SCOPE to output, clinical to extractor.
    _route_from_auditor: Conditional edge function reading routing_decision.
    _output_node: Formats the final state for API response, filters by query_intent.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from typing import Optional

from langgraph.graph import StateGraph, END

from langgraph_agent.state import AgentState, create_initial_state
from langgraph_agent.router_node import router_node, INTENT_OUT_OF_SCOPE
from langgraph_agent.extractor_node import extractor_node
from langgraph_agent.auditor_node import auditor_node, _synthesize_response, _build_citations_part
from langgraph_agent.clarification_node import clarification_node, resume_from_clarification
from healthcare_guidelines import CLINICAL_SAFETY_RULES


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

        if query_intent == "MEDICATIONS":
            return [e for e in extractions if "medications" in e.get("source", "").lower()]

        if query_intent == "ALLERGIES":
            return [e for e in extractions if "patients" in e.get("source", "").lower()]

        if query_intent == "INTERACTIONS":
            return [e for e in extractions if "interactions" in e.get("source", "").lower()]

        if query_intent == "SAFETY_CHECK":
            # Include all three sources — allergy, meds, and interactions —
            # so the Output Node has full context to build a clinical verdict.
            return list(extractions)

        # GENERAL_CLINICAL or OUT_OF_SCOPE (refusal already set) — return all
        return extractions
    except Exception:
        return extractions


def _output_node(state: AgentState) -> AgentState:
    """
    Output Node — filters extractions by query_intent, synthesizes a structured
    clinical response using Claude Sonnet, appends citations and the disclaimer.

    For OUT_OF_SCOPE queries, final_response is already set by the Router Node.
    For partial results, final_response is already set by the Auditor Node.
    For all other clinical queries: filter → synthesize → citations → disclaimer.

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

        # Filter extractions to what the query actually asked for.
        extractions = state.get("extractions", [])
        relevant = _filter_extractions_by_intent(extractions, query_intent)
        if not relevant:
            relevant = extractions

        citations_part = _build_citations_part(relevant)

        # Build PDF-only citation anchors for the UI viewer.
        # EHR/mock sources (patients.json, medications.json, etc.) are excluded —
        # only PDF extractions carry page numbers and deserve a viewer link.
        seen_anchors: set = set()
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
        state["citation_anchors"] = citation_anchors

        # Synthesize a structured clinical response via Claude Sonnet.
        # Falls back to raw claim join if synthesis fails.
        synthesized = _synthesize_response(
            query=state.get("input_query", ""),
            query_intent=query_intent,
            extractions=relevant,
            allergy_conflict=state.get("allergy_conflict_result"),
            denial_risk=state.get("denial_risk"),
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
    (refusal already set in final_response); otherwise route to extractor.

    Args:
        state: Current AgentState after Router Node has run.

    Returns:
        str: "output" for out-of-scope queries, "extractor" for all clinical intents.

    Raises:
        Never — returns "extractor" as safe fallback.
    """
    if state.get("routing_decision") == "out_of_scope":
        return "output"
    return "extractor"


def _route_from_extractor(state: AgentState) -> str:
    """
    After Extractor: if Step 0 set pending_user_input (ambiguous or no identifier),
    route to clarification; otherwise to auditor.
    """
    if state.get("pending_user_input") and state.get("clarification_needed"):
        return "clarification"
    return "auditor"


def _route_from_auditor(state: AgentState) -> str:
    """
    Conditional edge function — reads routing_decision from state and returns
    the name of the next node for LangGraph to route to.

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


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph():
    """
    Assemble and compile the LangGraph StateGraph.
    Graph is built fresh on each run_workflow call to ensure mocks in tests
    are applied at call time, not at module import time.

    Returns:
        CompiledGraph: Ready-to-invoke LangGraph graph.

    Raises:
        Never — propagates LangGraph compilation errors to run_workflow.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("auditor", auditor_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("output", _output_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_from_router,
        {"output": "output", "extractor": "extractor"},
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
            "output": "output",
            "extractor": "extractor",
            "clarification": "clarification",
        },
    )
    # Clarification pauses here — workflow ends and waits for user input.
    # Resume is handled by run_workflow() injecting clarification_response
    # into a fresh invocation, bypassing the clarification node entirely.
    graph.add_edge("clarification", END)
    graph.add_edge("output", END)

    return graph.compile()


# ── Public entry point ────────────────────────────────────────────────────────

def run_workflow(
    query: str,
    session_id: Optional[str] = None,
    clarification_response: Optional[str] = None,
    pdf_source_file: Optional[str] = None,
) -> dict:
    """
    Run the full LangGraph multi-agent workflow for a given query.
    Creates a fresh AgentState, optionally incorporates a clarification
    response, builds the graph, and invokes it.

    Args:
        query: Natural language query from the user.
        session_id: Optional session identifier (used by main.py for tracking).
        clarification_response: Optional user response to a prior clarification question.
        pdf_source_file: Optional path to a clinical PDF to process alongside the query.
            When set, the Extractor Node runs extract_pdf() and merges results into extractions.

    Returns:
        dict: Final AgentState as a plain dict, always includes 'final_response',
            'confidence_score', 'tool_trace', 'pending_user_input', 'error'.

    Raises:
        Never — all errors are captured in the returned dict's 'error' field.
    """
    try:
        initial_state = create_initial_state(query)

        if pdf_source_file and pdf_source_file.strip():
            initial_state["pdf_source_file"] = pdf_source_file.strip()

        if clarification_response:
            initial_state = resume_from_clarification(initial_state, clarification_response)

        graph = _build_graph()
        result = graph.invoke(initial_state)

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
