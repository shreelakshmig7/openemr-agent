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
from langgraph_agent.auditor_node import auditor_node
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


def _build_safety_check_response(state: AgentState) -> AgentState:
    """
    Build a direct clinical verdict for SAFETY_CHECK queries.
    Checks the proposed drug against allergy and interaction extractions,
    leads with a verdict line, then appends supporting evidence.

    Args:
        state: AgentState with proposed_drug, extractions, and input_query set.

    Returns:
        AgentState: State with final_response set to a direct clinical verdict.

    Raises:
        Never — returns state unchanged on any failure.
    """
    try:
        proposed_drug = (state.get("proposed_drug") or "").strip()
        extractions = state.get("extractions", [])

        allergy_extractions = [e for e in extractions if "patients" in e.get("source", "").lower()]
        interaction_extractions = [e for e in extractions if "interactions" in e.get("source", "").lower()]
        med_extractions = [e for e in extractions if "medications" in e.get("source", "").lower()]

        verdict_lines = []

        if proposed_drug:
            drug_lower = proposed_drug.lower()

            # Check allergy conflict
            allergy_conflict = any(
                drug_lower in e.get("claim", "").lower()
                for e in allergy_extractions
            )
            if allergy_conflict:
                verdict_lines.append(
                    f"⚠️ ALLERGY CONFLICT: {proposed_drug} is listed in this patient's known allergies. "
                    f"Do NOT administer."
                )

            # Check drug interaction involving proposed drug
            drug_interactions = [
                e for e in interaction_extractions
                if drug_lower in e.get("claim", "").lower()
            ]
            for interaction in drug_interactions:
                verdict_lines.append(f"⚠️ DRUG INTERACTION: {interaction.get('claim', '')}")

            if not verdict_lines:
                verdict_lines.append(
                    f"No known allergy conflict or drug interaction found for {proposed_drug} "
                    f"based on current records. Always verify with clinical judgment."
                )
        else:
            verdict_lines.append(
                "Safety check requested. Review the patient's allergies and current medications below."
            )

        # Append supporting evidence
        supporting = []
        supporting.extend(e.get("claim", "") for e in allergy_extractions if e.get("claim"))
        supporting.extend(e.get("claim", "") for e in med_extractions if e.get("claim"))
        supporting.extend(e.get("claim", "") for e in interaction_extractions if e.get("claim"))

        citations_part = " | ".join(
            f"Source: {e['source']}" for e in extractions if e.get("source")
        )
        supporting_text = " ".join(supporting)
        evidence_block = supporting_text + (f" [{citations_part}]" if citations_part else "")

        state["final_response"] = "\n\n".join(verdict_lines) + "\n\n" + evidence_block
        return state
    except Exception:
        return state


def _output_node(state: AgentState) -> AgentState:
    """
    Output Node — filters extractions by query_intent, finalizes final_response,
    appends the disclaimer, and sets confidence_score default if not already set.

    For OUT_OF_SCOPE queries, final_response is already set by the Router Node
    to the standard refusal string — this node only appends the disclaimer.

    For clinical queries, extractions are filtered by query_intent so the
    response only contains what the user actually asked about.

    Args:
        state: Final AgentState from Router (out_of_scope) or Auditor (pass/partial).

    Returns:
        AgentState: State with intent-filtered final_response and disclaimer appended.

    Raises:
        Never — returns state unchanged on any failure.
    """
    try:
        query_intent = state.get("query_intent", "")

        if query_intent == "SAFETY_CHECK":
            state = _build_safety_check_response(state)
        elif query_intent and query_intent != INTENT_OUT_OF_SCOPE:
            # For other clinical queries, filter extractions to what was asked.
            extractions = state.get("extractions", [])
            relevant = _filter_extractions_by_intent(extractions, query_intent)

            if relevant and extractions and relevant != extractions:
                citations_part = " | ".join(
                    f"Source: {e['source']}" for e in relevant if e.get("source")
                )
                state["final_response"] = (
                    " ".join(e["claim"] for e in relevant if e.get("claim"))
                    + (f" [{citations_part}]" if citations_part else "")
                )

        if state.get("final_response") and not state["final_response"].endswith(CLINICAL_SAFETY_RULES["disclaimer"]):
            state["final_response"] = (
                state["final_response"]
                + "\n\n"
                + CLINICAL_SAFETY_RULES["disclaimer"]
            )
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
) -> dict:
    """
    Run the full LangGraph multi-agent workflow for a given query.
    Creates a fresh AgentState, optionally incorporates a clarification
    response, builds the graph, and invokes it.

    Args:
        query: Natural language query from the user.
        session_id: Optional session identifier (used by main.py for tracking).
        clarification_response: Optional user response to a prior clarification question.

    Returns:
        dict: Final AgentState as a plain dict, always includes 'final_response',
            'confidence_score', 'tool_trace', 'pending_user_input', 'error'.

    Raises:
        Never — all errors are captured in the returned dict's 'error' field.
    """
    try:
        initial_state = create_initial_state(query)

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
