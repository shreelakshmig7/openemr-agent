"""
workflow.py
-----------
AgentForge — Healthcare RCM AI Agent — LangGraph workflow assembler
--------------------------------------------------------------------
Assembles the full LangGraph state machine from the four nodes and
exposes run_workflow() as the single entry point for main.py.

Graph topology:
    START → extractor → auditor
    auditor:
        "pass"                  → output → END
        "missing" (count < 3)   → extractor (review loop)
        "ambiguous"             → clarification → extractor (after user responds)
        "partial" (count >= 3)  → output → END

Key functions:
    run_workflow: Entry point — creates fresh state, builds graph, invokes it.
    _build_graph: Assembles and compiles the LangGraph StateGraph.
    _route_from_auditor: Conditional edge function reading routing_decision.
    _output_node: Formats the final state for API response.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from typing import Optional

from langgraph.graph import StateGraph, END

from langgraph_agent.state import AgentState, create_initial_state
from langgraph_agent.extractor_node import extractor_node
from langgraph_agent.auditor_node import auditor_node
from langgraph_agent.clarification_node import clarification_node, resume_from_clarification
from healthcare_guidelines import CLINICAL_SAFETY_RULES


# ── Output Node ───────────────────────────────────────────────────────────────

def _output_node(state: AgentState) -> AgentState:
    """
    Output Node — finalizes the state before returning to the caller.
    Appends the disclaimer and sets confidence_score if not already set.

    Args:
        state: Final AgentState from Auditor (pass or partial).

    Returns:
        AgentState: State with disclaimer appended to final_response.

    Raises:
        Never — returns state unchanged on any failure.
    """
    try:
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


# ── Conditional edge router ───────────────────────────────────────────────────

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

    graph.add_node("extractor", extractor_node)
    graph.add_node("auditor", auditor_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("output", _output_node)

    graph.set_entry_point("extractor")
    graph.add_edge("extractor", "auditor")
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
            "error": str(e),
        }
