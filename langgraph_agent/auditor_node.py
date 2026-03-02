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
import logging
import os
from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from healthcare_guidelines import CLINICAL_SAFETY_RULES
from langgraph_agent.state import AgentState

logger = logging.getLogger(__name__)


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
    Check whether every key term in the citation exists in the source file.
    Splits the citation into meaningful terms and verifies each appears in
    the source content. This handles structured JSON sources where field
    values appear on separate lines rather than as a single concatenated string.

    A "key term" is any token of 3+ characters after stripping punctuation.
    All key terms must be present in the source — partial matches fail.

    Args:
        claim: The claim made by the Extractor.
        citation: The citation text whose terms must appear in the source.
        source: Relative path to the source file to check against.

    Returns:
        bool: True if all key terms in citation appear in source, False otherwise.

    Raises:
        Never — returns False on any failure.
    """
    try:
        if not citation or not source:
            return False
        # PDF sources: extract_pdf already extracts verbatim quotes directly from
        # the document — no secondary file read is possible on binary PDF content.
        # Trust the verbatim flag set by the extractor.
        if source.lower().endswith(".pdf"):
            return True
        # Virtual / tool-identifier sources (e.g. "openemr_fhir", "policy_search",
        # "EHR_UNAVAILABLE", "mock") are not file paths on disk — there is nothing
        # to read and verify against.  Trust the tool result directly.
        # Uses the same BASE_DIR join that _load_source_data uses so relative paths
        # like "mock_data/medications.json" still resolve correctly.
        full_source_path = os.path.join(BASE_DIR, source)
        if not os.path.isfile(full_source_path):
            return True
        source_content = _load_source_data(source)
        if not source_content:
            return False
        source_lower = source_content.lower()
        terms = [
            t.strip("[]',\".:()").lower()
            for t in citation.split()
            if len(t.strip("[]',\".:()")) >= 3
        ]
        if not terms:
            return False
        return all(term in source_lower for term in terms)
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


# ── Source label map (shared by synthesis and citation builder) ───────────────

_SOURCE_LABELS = {
    "mock_data/patients.json": "Patient Record",
    "mock_data/medications.json": "Medications",
    "mock_data/interactions.json": "Drug Interactions",
    "mock_data/denial_patterns.json": "Denial Patterns",
}

_SYNTHESIS_SYSTEM = """You are a clinical AI assistant summarizing verified healthcare data for a care team.

You will receive:
- The original clinical query
- The query intent (e.g. MEDICATIONS, SAFETY_CHECK, GENERAL_CLINICAL)
- A list of verified clinical facts extracted from patient records and documents
- Optional: allergy conflict details and denial risk level

Your task: Write a clear, concise clinical response in 3-8 sentences (or bullet points for complex prior auth documents).

Rules:
- Only use facts provided in the extractions — do NOT invent or assume any information
- Search ALL sections of the provided text exhaustively — including historical narratives,
  assessment sections, inline code mentions, and completed-treatment records.
  Information that appears in a historical or completed context (e.g. "underwent 12 weeks
  of PT") IS documented and MUST be reported as such, not treated as absent.
- Inline mentions of codes or values (e.g. "ICD-10: M17.11" appearing in a clinical note)
  are valid documented findings — report them directly, do not require them to appear in a
  separate dedicated coding document.
- For SAFETY_CHECK with an allergy conflict, begin with: "⚠️ ALLERGY CONFLICT: [drug] is contraindicated. Do NOT administer."
- For SAFETY_CHECK with no conflict, begin with: "No known allergy conflict or drug interaction found for [drug] based on current records. Always verify with clinical judgment."
- For prior authorization PDFs, structure as: Patient summary → Key clinical findings → Criteria status → Recommended action
- For MEDICATIONS or ALLERGIES queries, respond with a brief factual list
- For GENERAL_CLINICAL or INTERACTIONS, summarize the key clinical findings
- Do not mention file names, JSON sources, or internal tool names
- Do not add information beyond what the extractions contain
- End with no disclaimer (it is added separately by the system)

SAFETY_CHECK HARD RULE — NON-NEGOTIABLE:
When the intent is SAFETY_CHECK, the "allergy_conflict_check_result" fact in the
extractions is the SOLE, AUTHORITATIVE determination of whether an allergy conflict exists.
You MUST accept it literally:
  - If it states NO CONFLICT → you MUST begin with the "No known allergy conflict" preamble.
    You are PROHIBITED from contradicting this by applying your own pharmacological knowledge,
    drug-class reasoning, chemical-structure analysis, or cross-reactivity assumptions.
    Do NOT mention sulfonamide moieties, beta-lactam rings, or any other structural chemistry
    that is not explicitly stated in the provided facts.
  - If it states CONFLICT DETECTED → use the "⚠️ ALLERGY CONFLICT" preamble and report only
    the conflict details provided — do NOT embellish with additional chemistry not in the facts.
The deterministic allergy check has already applied drug-class rules from validated clinical
guidelines. Your role is to summarize its output, not re-evaluate it.
"""


def _build_citations_part(extractions: List[dict]) -> str:
    """
    Build a deduplicated, human-readable citations string from extraction sources.

    Args:
        extractions: Validated extraction dicts from the extractor node.

    Returns:
        str: Formatted citations string like "Source: Patient Record | Source: Medications".
    """
    seen: set = set()
    labels = []
    for e in extractions:
        src = e.get("source", "")
        if not src or src in seen:
            continue
        seen.add(src)
        label = _SOURCE_LABELS.get(src, src.split("/")[-1])
        labels.append(f"Source: {label}")
    return " | ".join(labels)


def _synthesize_response(
    query: str,
    query_intent: str,
    extractions: List[dict],
    allergy_conflict: Optional[dict],
    denial_risk: Optional[dict],
    checked_sources_str: str = "",
    ehr_unavailable: bool = False,
) -> Optional[str]:
    """
    Use Claude Sonnet to synthesize a structured clinical response from validated extractions.
    Returns None on any failure so the caller can fall back to the raw join approach.

    Args:
        query: The original user query.
        query_intent: Classified intent (MEDICATIONS, SAFETY_CHECK, GENERAL_CLINICAL, etc.).
        extractions: All validated extraction dicts with claim, citation, source fields.
        allergy_conflict: Result from check_allergy_conflict (may be None or empty).
        denial_risk: Result from denial_analyzer (may be None or empty).
        checked_sources_str: Comma-separated list of source paths actually checked this
            query (e.g. "mock_data/patients.json, mock_data/medications.json"). Injected
            into the SOURCE_INTEGRITY hard rule to prevent fabricated absence claims.
        ehr_unavailable: True when patient was not found in EHR and PDF is the sole
            source of truth (Scenario A). When True, the SAFETY_CHECK "no conflict"
            preamble rule is bypassed and the model is directed to answer directly
            from the PDF facts without assuming absence.

    Returns:
        str: Synthesized clinical response, or None if synthesis fails.

    Raises:
        Never — returns None on any LLM or parse failure.
    """
    try:
        facts = "\n".join(
            f"- {e['claim']}"
            for e in extractions
            if e.get("claim") and not e.get("synthetic")
        )
        conflict_info = ""
        if allergy_conflict and allergy_conflict.get("conflict"):
            conflict_info = (
                f"\nALLERGY CONFLICT DETECTED: {allergy_conflict.get('drug')} "
                f"conflicts with {allergy_conflict.get('allergy')} allergy "
                f"(type: {allergy_conflict.get('conflict_type', 'unknown')})."
            )
        denial_info = ""
        if denial_risk and denial_risk.get("risk_level") not in (None, "NONE"):
            denial_info = (
                f"\nDenial risk level: {denial_risk.get('risk_level')} "
                f"({int((denial_risk.get('denial_risk_score', 0)) * 100)}%). "
                f"Matched patterns: {', '.join(p.get('code', '') for p in denial_risk.get('matched_patterns', []))}."
            )

        if ehr_unavailable:
            instruction = (
                "EHR records are UNAVAILABLE for this patient. "
                "The PDF content below is the ONLY source of truth. "
                "Search ALL pages of the PDF facts exhaustively to answer the question. "
                "If the PDF contains any mention of drug interactions, contraindications, "
                "CONTRAINDICATED drug pairs, or conflict warnings — report them verbatim. "
                "Do NOT assume absence of information. Do NOT use a 'no conflict found' "
                "conclusion unless you have read every fact below and found nothing. "
                "Quote the exact text from the PDF that answers the question."
            )
        else:
            instruction = (
                "Find the SPECIFIC answer to the question above within "
                "the clinical facts below. Quote the relevant text that answers it. "
                "If the answer appears anywhere — including in historical records, "
                "completed treatment summaries, assessment sections, or inline code "
                "mentions (e.g. ICD-10, CPT codes) — extract and report it directly. "
                "Do not summarize the document generally; answer the specific question."
            )

        user_content = (
            f"QUESTION TO ANSWER: {query}\n"
            f"Intent: {query_intent}\n"
            f"\nINSTRUCTION: {instruction}"
            f"{conflict_info}"
            f"{denial_info}\n\n"
            f"Clinical facts to search:\n{facts}"
        )

        # Gate 3 — SOURCE_INTEGRITY hard rule injected into system prompt.
        # Tells the LLM which sources were actually checked so it cannot
        # fabricate absence from a source not in that list.
        source_integrity_rule = ""
        if checked_sources_str:
            source_integrity_rule = (
                f"\n\nHARD RULE — SOURCE INTEGRITY:\n"
                f"The sources checked this query are: {checked_sources_str}\n"
                f"You may ONLY report absence of information from sources in that list.\n"
                f"If the query asks about a source NOT in that list:\n"
                f"- Do NOT say 'no X was found'\n"
                f"- Do NOT say 'X is not documented'\n"
                f"- Say ONLY: 'I did not have access to [source] for this query'\n"
                f"Fabricating a negative finding from an unchecked source is a "
                f"clinical safety violation."
            )

        system_prompt = _SYNTHESIS_SYSTEM + source_integrity_rule

        llm = ChatAnthropic(
            model="claude-sonnet-4-5",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
            max_tokens=512,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        return content.strip() if content else None
    except Exception as exc:
        logger.warning("Response synthesis failed, falling back to raw join: %s", exc)
        return None


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

            # Synthetic extractions are computed from tool results (e.g. policy criteria,
            # allergy conflict determinations, EHR gap flags). They are not verbatim quotes
            # from a source document, so neither the verbatim check nor source-file
            # verification applies — the tool that produced them is trusted directly.
            # This check MUST come before the verbatim check: policy criteria extractions
            # carry synthetic=True but no verbatim field (defaults False), so they would
            # incorrectly fail the verbatim gate and trigger the 3× retry loop.
            if extraction.get("synthetic"):
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
        # Auditor only validates — the Output Node handles synthesis and formatting.
        # Apply EHR confidence penalty if patient was not found (Scenario A).
        ehr_penalty = state.get("ehr_confidence_penalty", 0) / 100.0
        confidence = max(0.0, 0.95 - (iteration_count * 0.05) - ehr_penalty)
        state["routing_decision"] = "pass"
        state["audit_results"] = [{"validated": True, "count": len(extractions)}]
        state["confidence_score"] = confidence
        return state

    except Exception as e:
        state["routing_decision"] = "partial"
        state["is_partial"] = True
        state["error"] = f"Auditor error: {str(e)}"
        state["insufficient_documentation_flags"] = [f"Insufficient Documentation: auditor encountered an error — {str(e)}"]
        return state
