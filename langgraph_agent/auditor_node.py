"""
auditor_node.py
---------------
AgentForge — Healthcare RCM AI Agent — LangGraph Auditor Node
--------------------------------------------------------------
Implements the Auditor Node in the LangGraph state machine. Validates
every extraction from the Extractor Node against four criteria: citation
exists, citation is verbatim (not paraphrased), claim is not ambiguous,
and citation text can be verified in source data.

Makes ALL routing decisions for the graph:
    "pass"      → Output Node (all extractions valid)
    "missing"   → Extractor Node (re-extract with iteration_count < 3)
    "ambiguous" → Clarification Node (pause workflow, ask user)
    "partial"   → Output Node (iteration ceiling hit, return partial results)

Key functions:
    auditor_node: Main node function called by the LangGraph graph.
    _verify_citation_exists_in_source: Checks citation verbatim in source data.
    _build_clarification_question: Generates a PII-safe clarification question.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import json
import os
from typing import List

from healthcare_guidelines import CLINICAL_SAFETY_RULES
from langgraph_agent.state import AgentState


# ── Load source data for citation verification ────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SOURCE_DATA_CACHE: dict = {}


def _load_source_data(source_path: str) -> str:
    """
    Load source file content for citation verification. Cached after first load.

    Args:
        source_path: Relative path to the source file (e.g. mock_data/medications.json).

    Returns:
        str: Raw file content as string, or empty string if file not found.

    Raises:
        Never — returns empty string on any failure.
    """
    try:
        if source_path in _SOURCE_DATA_CACHE:
            return _SOURCE_DATA_CACHE[source_path]
        full_path = os.path.join(BASE_DIR, source_path)
        with open(full_path, "r") as f:
            content = f.read()
        _SOURCE_DATA_CACHE[source_path] = content
        return content
    except Exception:
        return ""


# ── Citation verifier ─────────────────────────────────────────────────────────

def _verify_citation_exists_in_source(claim: str, citation: str, source: str) -> bool:
    """
    Check whether the citation text exists verbatim in the source file.
    Used to detect hallucinated facts — claims whose citations cannot be
    found in the actual source data.

    Args:
        claim: The claim made by the Extractor.
        citation: The verbatim quote the Extractor cited.
        source: Path to the source file to check against.

    Returns:
        bool: True if citation text appears in source, False otherwise.

    Raises:
        Never — returns False on any failure.
    """
    try:
        if not citation or not source:
            return False
        source_content = _load_source_data(source)
        if not source_content:
            return False
        return citation.lower() in source_content.lower()
    except Exception:
        return False


# ── Clarification question builder ────────────────────────────────────────────

def _build_clarification_question(extractions: List[dict]) -> str:
    """
    Build a PII-safe clarification question for the user when ambiguity is detected.
    Does not include SSN, MRN, or other HIPAA-restricted identifiers.

    Args:
        extractions: Current extractions list from state (may be partial or ambiguous).

    Returns:
        str: A non-PII clarification question to surface to the user.

    Raises:
        Never — returns a generic fallback question on any failure.
    """
    try:
        for extraction in extractions:
            if extraction.get("ambiguous"):
                claim = extraction.get("claim", "")
                if "multiple patients" in claim.lower() or "ambiguous" in claim.lower():
                    return (
                        "Multiple patients match your query. "
                        "Please provide the full name or patient ID to continue."
                    )
        return "Your query is ambiguous. Please clarify which patient you are asking about."
    except Exception:
        return "Please clarify your query to continue."


# ── Auditor Node ──────────────────────────────────────────────────────────────

def auditor_node(state: AgentState) -> AgentState:
    """
    Auditor Node — validates all extractions and sets routing_decision.
    This is the sole routing decision-maker in the graph. It has four
    exit paths: pass, missing, ambiguous, partial.

    Args:
        state: Current AgentState containing extractions from Extractor Node.

    Returns:
        AgentState: Updated state with routing_decision, audit_results, and
            confidence_score set. May also set pending_user_input, is_partial,
            insufficient_documentation_flags depending on routing path.

    Raises:
        Never — all errors result in a partial routing decision.
    """
    try:
        max_iterations = CLINICAL_SAFETY_RULES["max_review_loop_iterations"]
        extractions = state.get("extractions", [])
        iteration_count = state.get("iteration_count", 0)

        # ── Guard: iteration count at or above ceiling ─────────────────────
        if iteration_count >= max_iterations:
            missing_flags = [
                f"Insufficient Documentation: citation could not be verified for claim — '{e.get('claim', 'unknown')}'"
                for e in extractions
                if not e.get("verbatim") or not e.get("citation")
            ]
            if not missing_flags:
                missing_flags = ["Insufficient Documentation: maximum review iterations reached without resolution."]

            state["routing_decision"] = "partial"
            state["is_partial"] = True
            state["insufficient_documentation_flags"] = missing_flags
            state["final_response"] = (
                "Partial results returned after maximum review attempts. "
                "The following gaps were identified: " + "; ".join(missing_flags)
            )
            state["confidence_score"] = 0.5
            state["audit_results"] = [{"validated": False, "reason": "iteration ceiling reached"}]
            return state

        # ── Check for ambiguity ────────────────────────────────────────────
        for extraction in extractions:
            if extraction.get("ambiguous"):
                state["routing_decision"] = "ambiguous"
                state["pending_user_input"] = True
                state["clarification_needed"] = _build_clarification_question(extractions)
                state["audit_results"] = [{"validated": False, "reason": "ambiguous input"}]
                return state

        # ── Validate each extraction ───────────────────────────────────────
        failed_extractions = []
        for extraction in extractions:
            citation = extraction.get("citation", "")
            source = extraction.get("source", "")
            verbatim = extraction.get("verbatim", False)

            if not citation:
                failed_extractions.append(extraction)
                continue

            if not verbatim:
                failed_extractions.append(extraction)
                continue

            if not _verify_citation_exists_in_source(extraction.get("claim", ""), citation, source):
                failed_extractions.append(extraction)
                continue

        if failed_extractions:
            state["routing_decision"] = "missing"
            state["iteration_count"] = iteration_count + 1
            state["audit_results"] = [{
                "validated": False,
                "reason": "citations missing or not verbatim",
                "failed_count": len(failed_extractions),
            }]
            return state

        # ── All extractions pass ───────────────────────────────────────────
        confidence = max(0.0, 0.95 - (iteration_count * 0.05))
        final_response_parts = [e["claim"] for e in extractions if e.get("claim")]
        citations_part = " | ".join(
            f"Source: {e['source']}" for e in extractions if e.get("source")
        )

        state["routing_decision"] = "pass"
        state["audit_results"] = [{"validated": True, "count": len(extractions)}]
        state["confidence_score"] = confidence
        state["final_response"] = (
            " ".join(final_response_parts) + f" [{citations_part}]"
        )
        return state

    except Exception as e:
        state["routing_decision"] = "partial"
        state["is_partial"] = True
        state["error"] = f"Auditor error: {str(e)}"
        state["insufficient_documentation_flags"] = [f"Insufficient Documentation: auditor encountered an error — {str(e)}"]
        return state
