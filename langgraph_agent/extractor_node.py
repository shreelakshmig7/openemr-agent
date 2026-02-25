"""
extractor_node.py
-----------------
AgentForge — Healthcare RCM AI Agent — LangGraph Extractor Node
---------------------------------------------------------------
Implements the Extractor Node in the LangGraph state machine. Calls the
three healthcare tools in sequence (patient → medications → interactions),
formats results as extractions with verbatim citations, and runs the PII
stub scrubber on all input before processing.

Key functions:
    extractor_node: Main node function called by the LangGraph graph.
    _stub_pii_scrubber: Strips HIPAA fields from text before LLM/tool calls.
    _extract_patient_identifier: Pulls name or ID from the query string.
    _format_extractions: Converts tool results to cited extraction dicts.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import re
from typing import List

from tools import get_patient_info as tool_get_patient_info
from tools import get_medications as tool_get_medications
from tools import check_drug_interactions as tool_check_drug_interactions
from healthcare_guidelines import HIPAA_RULES
from langgraph_agent.state import AgentState


# ── PII Scrubber Stub ─────────────────────────────────────────────────────────

# TODO: Replace with Microsoft Presidio in PII infrastructure PR
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN_PATTERN = re.compile(r"\bMRN[:\s]*\w+\b", re.IGNORECASE)
_DOB_PATTERN = re.compile(r"\b(DOB|Date of Birth)[:\s]*[\d/\-]+\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"\b\d{3}[.\-]\d{3}[.\-]\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")


def _stub_pii_scrubber(text: str) -> str:
    """
    Remove HIPAA-defined PII patterns from text before any tool or LLM call.
    Strips SSN, MRN, DOB, phone, and email patterns using regex.

    Args:
        text: Raw input text that may contain PII.

    Returns:
        str: Text with PII patterns replaced by [REDACTED].

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


# ── Patient identifier extraction ─────────────────────────────────────────────

def _extract_patient_identifier(query: str) -> str:
    """
    Extract a patient name or ID from the query string.
    Matches patient IDs (P001 format) or common name patterns.

    Args:
        query: Cleaned query string (PII already scrubbed).

    Returns:
        str: Patient name or ID, or empty string if none found.

    Raises:
        Never — returns empty string on any failure.
    """
    try:
        id_match = re.search(r'\b[Pp]\d{3,}\b', query)
        if id_match:
            return id_match.group(0)

        name_patterns = [
            r'(?:for|about|is|check|lookup|look up)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
        ]
        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        return ""
    except Exception:
        return ""


# ── Extraction formatter ──────────────────────────────────────────────────────

def _format_extractions(patient: dict, medications: List[dict], interactions: List[dict]) -> List[dict]:
    """
    Convert raw tool results into extraction dicts with verbatim citations.
    Each extraction contains the claim, the verbatim quote from source, and the source path.

    Args:
        patient: Patient dict from tool_get_patient_info.
        medications: Medication list from tool_get_medications.
        interactions: Interaction list from tool_check_drug_interactions.

    Returns:
        List[dict]: Extraction dicts, each with claim, citation, source, verbatim keys.

    Raises:
        Never — returns empty list on any failure.
    """
    try:
        extractions = []

        if patient:
            allergies_str = ", ".join(patient.get("allergies", [])) or "none on record"
            extractions.append({
                "claim": f"{patient['name']} has known allergies: {allergies_str}.",
                "citation": str(patient.get("allergies", [])),
                "source": "mock_data/patients.json",
                "verbatim": True,
            })

        for med in medications:
            extractions.append({
                "claim": (
                    f"{patient['name']} is prescribed {med['name']} "
                    f"{med.get('dose', '')} {med.get('frequency', '')}."
                ),
                "citation": f"{med['name']} {med.get('dose', '')} {med.get('frequency', '')}".strip(),
                "source": "mock_data/medications.json",
                "verbatim": True,
            })

        for interaction in interactions:
            extractions.append({
                "claim": (
                    f"Drug interaction detected: {interaction['drug1']} + {interaction['drug2']} "
                    f"— severity {interaction['severity']}."
                ),
                "citation": (
                    f"{interaction['drug1']} and {interaction['drug2']}: "
                    f"{interaction.get('recommendation', interaction['severity'])}"
                ),
                "source": "mock_data/interactions.json",
                "verbatim": True,
            })

        return extractions
    except Exception:
        return []


# ── Extractor Node ────────────────────────────────────────────────────────────

def extractor_node(state: AgentState) -> AgentState:
    """
    Extractor Node — calls tools in sequence and populates state.extractions[].
    Runs PII scrubber on input before any processing. Does not make routing
    decisions — that is the Auditor's responsibility.

    Args:
        state: Current AgentState from the LangGraph graph.

    Returns:
        AgentState: Updated state with extractions[], tool_trace[], and documents_processed[].

    Raises:
        Never — errors are written to state['error'] and returned as partial extractions.
    """
    try:
        clean_query = _stub_pii_scrubber(state["input_query"])

        if state.get("clarification_response"):
            clean_query = f"{clean_query} {_stub_pii_scrubber(state['clarification_response'])}"

        patient_identifier = _extract_patient_identifier(clean_query)

        if not patient_identifier:
            state["extractions"] = [{
                "claim": "AMBIGUOUS: No patient name or ID found in query.",
                "citation": "",
                "source": "",
                "verbatim": False,
                "ambiguous": True,
            }]
            state["tool_trace"] = []
            return state

        tool_trace = []

        patient_result = tool_get_patient_info(patient_identifier)
        tool_trace.append({
            "tool": "tool_get_patient_info",
            "input": patient_identifier,
            "output": patient_result,
        })

        if not patient_result.get("success"):
            state["extractions"] = [{
                "claim": f"Patient not found: {patient_result.get('error', 'Unknown error')}",
                "citation": "",
                "source": "",
                "verbatim": False,
            }]
            state["tool_trace"] = tool_trace
            return state

        patient = patient_result["patient"]
        patient_id = patient["id"]

        meds_result = tool_get_medications(patient_id)
        tool_trace.append({
            "tool": "tool_get_medications",
            "input": patient_id,
            "output": meds_result,
        })

        medications = meds_result.get("medications", []) if meds_result.get("success") else []

        interactions_result = tool_check_drug_interactions(medications)
        tool_trace.append({
            "tool": "tool_check_drug_interactions",
            "input": [m.get("name", m) if isinstance(m, dict) else m for m in medications],
            "output": interactions_result,
        })

        found_interactions = interactions_result.get("interactions", []) if interactions_result.get("success") else []

        state["extractions"] = _format_extractions(patient, medications, found_interactions)
        state["documents_processed"] = ["mock_data/patients.json", "mock_data/medications.json", "mock_data/interactions.json"]
        state["tool_trace"] = tool_trace
        return state

    except Exception as e:
        state["error"] = f"Extractor error: {str(e)}"
        state["extractions"] = []
        return state
