"""
extractor_node.py
-----------------
AgentForge — Healthcare RCM AI Agent — LangGraph Extractor Node
---------------------------------------------------------------
Implements the Extractor Node in the LangGraph state machine. When the
Orchestrator Node has run, patient name extraction is read from
state["identified_patient_name"] — no Step 0 Haiku call is made. This halves
API calls per request and eliminates the 529-overloaded regression where the
second Haiku call failed after the Orchestrator's first call succeeded.

Step 0 (_extract_patient_identifier_llm) is retained only for the legacy path:
when orchestrator_ran=False (direct unit tests, old callers).

Tool call order:
    1. tool_get_patient_info — resolves patient identifier to structured record.
    2. tool_get_medications — retrieves current medication list.
    3. tool_check_drug_interactions — checks all medications for dangerous pairs.
    4. extract_pdf — if state.pdf_source_file is set, extracts clinical text from PDF.
    5. analyze_denial_risk — scores all extractions against historical denial patterns.

Key functions:
    extractor_node: Main node function — reads identified_patient_name (orchestrated)
        or runs Step 0 (legacy), then calls tools in sequence.
    _extract_patient_identifier_llm: Step 0 — Haiku returns JSON (type, value, ambiguous).
        Only called when orchestrator_ran=False.
    _stub_pii_scrubber: Strips HIPAA fields from text before LLM/tool calls.
    _format_extractions: Converts tool results to cited extraction dicts.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from tools import get_patient_info as tool_get_patient_info
from tools import get_medications as tool_get_medications
from tools import check_drug_interactions as tool_check_drug_interactions
from tools.policy_search import search_policy as tool_search_policy
from pdf_extractor import extract_pdf as tool_extract_pdf
from denial_analyzer import analyze_denial_risk as tool_analyze_denial_risk
from verification import check_allergy_conflict
from langgraph_agent.state import AgentState

# Intents where denial risk analysis adds RCM value.
# SAFETY_CHECK is excluded — allergy/interaction checks already answer that question,
# and showing a billing denial score alongside "Do NOT administer" confuses the signal.
_DENIAL_RISK_INTENTS = {"INTERACTIONS", "GENERAL_CLINICAL"}


# ── Safety check: proposed drug extraction ───────────────────────────────────

_EXTRACT_DRUG_SYSTEM = """You are a clinical data extractor.
Extract the name of the drug being proposed or asked about in this safety check query.
Return only the drug name as a single word or phrase. No explanation, no punctuation.

Examples:
  "Can I give him penicillin?"          → Penicillin
  "Is it safe to give Mary Aspirin?"    → Aspirin
  "Can I administer ibuprofen?"         → Ibuprofen
  "Is Warfarin safe for this patient?"  → Warfarin

If no drug name is found, return empty string.
"""


def _extract_proposed_drug_llm(query: str) -> str:
    """
    Extract the proposed drug name from a SAFETY_CHECK query using Claude Haiku.

    Args:
        query: The user's safety check query (e.g. "Can I give him penicillin?").

    Returns:
        str: The drug name (e.g. "Penicillin"), or empty string if not found or on error.

    Raises:
        Never — returns empty string on any LLM or parse failure.
    """
    try:
        llm = ChatAnthropic(
            model="claude-haiku-4-5",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
            max_tokens=16,
        )
        response = llm.invoke([
            SystemMessage(content=_EXTRACT_DRUG_SYSTEM),
            HumanMessage(content=query),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        drug = content.strip().strip(".,;:\"'").split()[0] if content.strip() else ""
        return drug
    except Exception as e:
        logger.warning("Proposed drug extraction failed: %s", e)
        return ""


# ── Step 0: LLM patient extraction ─────────────────────────────────────────────

EXTRACT_PATIENT_SYSTEM = """You are a clinical data extractor.
Extract the patient identifier from the query.
Return JSON only. No explanation.

Rules:
- If you find a full name (first + last), return:
  {"type": "name", "value": "John Smith", "ambiguous": false}

- If you find only a first name, return:
  {"type": "name", "value": "John", "ambiguous": true, "reason": "first name only — multiple patients possible"}

- If you find a patient ID (format P001, P002, etc.), return:
  {"type": "id", "value": "P001", "ambiguous": false}

- If you find no patient identifier (e.g. pronoun only, or empty), return:
  {"type": "none", "ambiguous": true, "reason": "no patient name or ID found in query"}
"""


def _extract_patient_identifier_llm(query: str) -> Dict[str, Any]:
    """
    Step 0 — Haiku extracts patient identifier from natural language.
    Returns dict with type (name|id|none), value, ambiguous (bool), reason (if ambiguous).

    Args:
        query: PII-scrubbed user query (may include "Regarding X: ..." from session).

    Returns:
        dict: {"type": str, "value": str, "ambiguous": bool, "reason": str (optional)}.
              On LLM/parse failure, returns {"type": "none", "ambiguous": True, "reason": "extraction failed"}.
    """
    try:
        llm = ChatAnthropic(
            model="claude-haiku-4-5",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
        )
        response = llm.invoke([
            SystemMessage(content=EXTRACT_PATIENT_SYSTEM),
            HumanMessage(content=query),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        if not content:
            return {"type": "none", "ambiguous": True, "reason": "no response from extractor"}
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        out = json.loads(text)
        if not isinstance(out, dict):
            return {"type": "none", "ambiguous": True, "reason": "invalid extraction format"}
        out.setdefault("ambiguous", True)
        out.setdefault("type", "none")
        out.setdefault("value", "")
        return out
    except json.JSONDecodeError as e:
        logger.warning("Extractor LLM response was not valid JSON: %s", e)
        return {"type": "none", "ambiguous": True, "reason": "extraction parse error"}
    except Exception as e:
        logger.exception("Extractor patient-identifier LLM call failed: %s", e)
        return {"type": "none", "ambiguous": True, "reason": "extraction failed"}


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
            allergies = patient.get("allergies", [])
            allergies_str = ", ".join(allergies) if allergies else "none on record"
            extractions.append({
                "claim": f"{patient['name']} has known allergies: {allergies_str}.",
                "citation": allergies_str if allergies else "no known allergies",
                "source": "mock_data/patients.json",
                "verbatim": True,
                # When the allergy list is empty, "no known allergies" is a derived
                # statement — not a verbatim string in the source file. Mark synthetic
                # so the auditor skips source-file citation verification for this case.
                "synthetic": not bool(allergies),
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


# ── PDF content-hash cache ────────────────────────────────────────────────────

def _get_pdf_content_hash(pdf_path: str) -> str:
    """
    Compute the MD5 hex digest of a PDF file's raw bytes.

    Hashes file content (not path) so that two different files uploaded to the
    same path are not incorrectly treated as identical. Used by the Extractor to
    validate the Layer 2 PDF cache before skipping pdf_extractor.

    Args:
        pdf_path: Filesystem path to the PDF file.

    Returns:
        str: MD5 hex digest, or empty string if the file cannot be read.

    Raises:
        Never — returns empty string on any I/O failure.
    """
    try:
        with open(pdf_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


# ── Extractor Node ────────────────────────────────────────────────────────────

def extractor_node(state: AgentState) -> AgentState:
    """
    Extractor Node — calls tools in sequence and populates state.extractions[].

    When the Orchestrator has run (orchestrator_ran=True), executes only the
    tools listed in state["tool_plan"] in order. If tool_plan is empty and the
    Orchestrator ran, this is a general knowledge query — all tools are skipped
    and the node passes through to the Auditor immediately.

    When the Orchestrator has not run (orchestrator_ran=False, e.g. in direct
    tests), falls back to the original behavior: all EHR tools are called.

    Layer 2 cache writes:
        - extracted_patient: written after a successful patient_lookup.
        - extracted_pdf_pages / extracted_pdf_hash: written after pdf_extractor
          runs. Hash is derived from file content bytes, not path.

    Layer 1 memory write:
        - prior_query_context: updated at the end of every run with the
          resolved patient name/ID and intent, for the next turn's Orchestrator.

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

        tool_trace = list(state.get("tool_trace") or [])
        tool_call_history = list(state.get("tool_call_history") or [])
        _ts = datetime.now(timezone.utc).isoformat()

        # ── General knowledge bypass ──────────────────────────────────────────
        # Use is_general_knowledge (set explicitly by Orchestrator) — NOT tool_plan=[],
        # which is also empty for patient cache-hit paths and would incorrectly bypass
        # them. is_general_knowledge=True is the unambiguous signal for bypass.
        orchestrator_ran = state.get("orchestrator_ran", False)
        tool_plan = state.get("tool_plan") or []
        if state.get("is_general_knowledge", False):
            logger.info("Extractor: general knowledge query — skipping all tools.")
            state["extractions"] = []
            state["documents_processed"] = []
            state["denial_risk"] = {
                "success": True,
                "risk_level": "NONE",
                "matched_patterns": [],
                "recommendations": [],
                "denial_risk_score": 0.0,
                "source": "none",
                "error": None,
            }
            state["tool_trace"] = tool_trace
            state["tool_call_history"] = tool_call_history
            return state

        # ── Patient cache check ───────────────────────────────────────────────
        # If Orchestrator already cached the patient this session, skip Step 0
        # and patient_lookup — use the cached record directly.
        cached_patient = state.get("extracted_patient") or {}
        use_patient_cache = (
            orchestrator_ran
            and "patient_lookup" not in tool_plan
            and bool(cached_patient)
        )

        patient = None
        patient_id = None
        patient_identifier = ""
        ehr_gap_extractions: List[dict] = []
        patient_result: Optional[dict] = None

        if use_patient_cache:
            # Layer 2 cache hit — patient already resolved this session.
            patient = cached_patient
            patient_id = patient.get("id", "")
            patient_identifier = patient.get("name", patient_id)
            logger.info("Extractor: patient cache hit — using cached record for '%s'", patient_identifier)
        elif orchestrator_ran:
            # Orchestrator already ran a single Haiku call that extracted the patient
            # name alongside intent classification — read from state, skip Step 0.
            patient_identifier = (state.get("identified_patient_name") or "").strip()
            if not patient_identifier:
                state["clarification_needed"] = (
                    "Which patient are you referring to? "
                    "(no patient name found in your query)"
                )
                state["pending_user_input"] = True
                state["extractions"] = []
                state["tool_trace"] = tool_trace
                state["tool_call_history"] = tool_call_history
                return state

            patient_result = tool_get_patient_info(patient_identifier)
            tool_trace.append({
                "tool": "tool_get_patient_info",
                "input": patient_identifier,
                "output": patient_result,
            })
            tool_call_history.append({"tool": "patient_lookup", "status": "success" if patient_result.get("success") else "miss", "ts": _ts})
        else:
            # Legacy path: Orchestrator has not run (direct unit tests or old callers).
            # Fall back to Step 0 — Haiku extracts patient identifier independently.
            extracted = _extract_patient_identifier_llm(clean_query)
            state["extracted_patient_identifier"] = extracted

            if extracted.get("ambiguous") or extracted.get("type") == "none":
                reason = extracted.get("reason") or "No patient identifier found."
                state["clarification_needed"] = f"Which patient are you referring to? ({reason})"
                state["pending_user_input"] = True
                state["extractions"] = []
                state["tool_trace"] = tool_trace
                state["tool_call_history"] = tool_call_history
                return state

            patient_identifier = (extracted.get("value") or "").strip()
            if not patient_identifier:
                state["clarification_needed"] = "Which patient are you referring to? Please provide a full name or patient ID."
                state["pending_user_input"] = True
                state["extractions"] = []
                state["tool_trace"] = tool_trace
                state["tool_call_history"] = tool_call_history
                return state

            patient_result = tool_get_patient_info(patient_identifier)
            tool_trace.append({
                "tool": "tool_get_patient_info",
                "input": patient_identifier,
                "output": patient_result,
            })
            tool_call_history.append({"tool": "patient_lookup", "status": "success" if patient_result.get("success") else "miss", "ts": _ts})

        # ── Patient lookup result handling (shared by orchestrator and legacy paths) ─
        # use_patient_cache skips this block — patient is already resolved above.
        if patient_result is not None:
            if not patient_result.get("success"):
                pdf_source_file_check = (state.get("pdf_source_file") or "").strip()
                if not pdf_source_file_check:
                    # Scenario B — no PDF and patient not found: hard stop.
                    state["clarification_needed"] = (
                        f"Patient not found: {patient_result.get('error', 'Unknown error')}. "
                        "Please verify the patient name or ID, or attach a clinical PDF."
                    )
                    state["pending_user_input"] = True
                    state["extractions"] = []
                    state["tool_trace"] = tool_trace
                    state["tool_call_history"] = tool_call_history
                    return state

                # Scenario A — PDF provided but patient not in EHR.
                logger.warning(
                    "Patient '%s' not found in EHR — proceeding from PDF only (Scenario A).",
                    patient_identifier,
                )
                ehr_gap_extractions = [
                    {
                        "claim": (
                            "Allergy status could not be verified — no EHR record found for this patient. "
                            "Manual allergy verification required before proceeding."
                        ),
                        "citation": "EHR_UNAVAILABLE",
                        "source": "EHR_UNAVAILABLE",
                        "verbatim": True,
                        "synthetic": True,
                        "flag": "NO_EHR_ALLERGY_DATA",
                    },
                    {
                        "claim": (
                            "Medication list could not be retrieved from EHR — no patient record found. "
                            "Drug interaction check is incomplete."
                        ),
                        "citation": "EHR_UNAVAILABLE",
                        "source": "EHR_UNAVAILABLE",
                        "verbatim": True,
                        "synthetic": True,
                        "flag": "NO_EHR_MEDICATION_DATA",
                    },
                ]
                tool_trace.append({
                    "tool": "ehr_gap_check",
                    "input": patient_identifier,
                    "output": {
                        "patient_found": False,
                        "pdf_fallback": True,
                        "penalty": 45,
                        "flags": ["NO_EHR_ALLERGY_DATA", "NO_EHR_MEDICATION_DATA"],
                    },
                })
                state["ehr_confidence_penalty"] = 45
                patient = None
                patient_id = None
            else:
                patient = patient_result["patient"]
                patient_id = patient["id"]
                # Write to Layer 2 session cache so follow-up turns skip patient_lookup.
                state["extracted_patient"] = patient

        # ── EHR tool suite — only runs when patient was found in the database ─
        medications: List[dict] = []
        found_interactions: List[dict] = []
        proposed_drug = ""
        allergy_conflict_result: dict = {}

        if patient is not None:
            meds_result = tool_get_medications(patient_id)
            tool_trace.append({
                "tool": "tool_get_medications",
                "input": patient_id,
                "output": meds_result,
            })
            tool_call_history.append({"tool": "med_retrieval", "status": "success" if meds_result.get("success") else "fail", "ts": _ts})
            medications = meds_result.get("medications", []) if meds_result.get("success") else []

            # For SAFETY_CHECK: extract proposed drug, check allergy conflict (name + class),
            # and include it in the interaction check for drug-drug conflict detection.
            meds_for_interaction_check = medications
            if state.get("query_intent") == "SAFETY_CHECK":
                proposed_drug = _extract_proposed_drug_llm(clean_query)
                state["proposed_drug"] = proposed_drug
                if proposed_drug:
                    meds_for_interaction_check = medications + [{"name": proposed_drug}]
                    logger.info("SAFETY_CHECK: proposed drug '%s' added to interaction check", proposed_drug)

                    allergy_conflict_result = check_allergy_conflict(
                        proposed_drug,
                        patient.get("allergies", []),
                    )
                    state["allergy_conflict_result"] = allergy_conflict_result
                    tool_trace.append({
                        "tool": "check_allergy_conflict",
                        "input": {
                            "drug": proposed_drug,
                            "allergies": patient.get("allergies", []),
                        },
                        "output": allergy_conflict_result,
                    })

            interactions_result = tool_check_drug_interactions(meds_for_interaction_check)
            tool_trace.append({
                "tool": "tool_check_drug_interactions",
                "input": [m.get("name", m) if isinstance(m, dict) else m for m in meds_for_interaction_check],
                "output": interactions_result,
            })
            tool_call_history.append({"tool": "interaction_check", "status": "success" if interactions_result.get("success") else "fail", "ts": _ts})
            found_interactions = interactions_result.get("interactions", []) if interactions_result.get("success") else []

        all_extractions = _format_extractions(patient, medications, found_interactions)

        # Scenario A: patient not in EHR but PDF is attached — prepend EHR gap flags.
        if patient is None and ehr_gap_extractions:
            all_extractions = ehr_gap_extractions + all_extractions

        # If a drug-class or exact allergy conflict was found, inject an explicit extraction
        # so the denial_analyzer keyword match ("allergy conflict") fires correctly.
        if allergy_conflict_result.get("conflict"):
            conflict_drug = allergy_conflict_result.get("drug", proposed_drug)
            conflict_allergy = allergy_conflict_result.get("allergy", "documented allergy")
            conflict_type = allergy_conflict_result.get("conflict_type", "allergy_conflict")
            all_extractions.append({
                "claim": (
                    f"ALLERGY CONFLICT DETECTED: {conflict_drug} is contraindicated — "
                    f"patient has a documented {conflict_allergy} allergy "
                    f"({conflict_type.replace('_', ' ')})."
                ),
                "citation": f"allergy conflict: {conflict_drug} contraindicated — {conflict_allergy} allergy",
                "source": "mock_data/patients.json",
                "verbatim": True,
                "synthetic": True,
            })
            logger.warning(
                "ALLERGY CONFLICT: %s vs %s (%s)",
                conflict_drug,
                conflict_allergy,
                conflict_type,
            )

        documents_processed = [
            "mock_data/patients.json",
            "mock_data/medications.json",
            "mock_data/interactions.json",
        ]

        # Step 4 — PDF extraction.
        # Uses content-hash cache: re-extracts only when the document bytes change.
        # This prevents stale extractions when a user uploads a different PDF in
        # the same session (a hash mismatch forces re-extraction).
        pdf_source_file = (state.get("pdf_source_file") or "").strip()
        if pdf_source_file:
            current_hash = _get_pdf_content_hash(pdf_source_file)
            cached_hash = state.get("extracted_pdf_hash", "")
            cached_pages = state.get("extracted_pdf_pages") or {}

            if current_hash and current_hash == cached_hash and cached_pages:
                # Cache hit — same document bytes, skip extraction.
                # cached_pages is a dict of page_key → list[str] (all elements on that page).
                # Each list entry is restored as a separate extraction so no element is lost
                # on follow-up turns (the previous single-string-per-page scheme silently
                # dropped every element after the first on any given page).
                logger.info("Extractor: PDF cache hit (hash=%s) — skipping pdf_extractor.", current_hash[:8])
                for page_key, page_entries in cached_pages.items():
                    page_num = int(page_key) if str(page_key).isdigit() else None
                    # Support both old (str) and new (list[str]) cache formats.
                    if isinstance(page_entries, str):
                        page_entries = [page_entries]
                    for page_text in page_entries:
                        if not page_text:
                            continue
                        all_extractions.append({
                            "claim": page_text,
                            "citation": page_text,
                            "source": pdf_source_file,
                            "verbatim": True,
                            "page_number": page_num,
                            "element_type": "cached",
                        })
                documents_processed.append(pdf_source_file)
                tool_call_history.append({"tool": "pdf_extractor", "status": "cache_hit", "ts": _ts})
            else:
                # Cache miss or hash mismatch — extract and update cache.
                if cached_hash and current_hash != cached_hash:
                    logger.info("Extractor: PDF hash mismatch — re-extracting new document.")
                pdf_result = tool_extract_pdf(pdf_source_file)
                tool_trace.append({
                    "tool": "tool_extract_pdf",
                    "input": pdf_source_file,
                    "output": {
                        "success": pdf_result.get("success"),
                        "element_count": pdf_result.get("element_count", 0),
                        "source_file": pdf_result.get("source_file"),
                        "error": pdf_result.get("error"),
                    },
                })
                tool_call_history.append({"tool": "pdf_extractor", "status": "success" if pdf_result.get("success") else "fail", "ts": _ts})
                if pdf_result.get("success"):
                    # Store ALL elements per page as a list so multi-element pages
                    # (e.g. a page with multiple tables and paragraphs) survive cache
                    # round-trips without losing any element after the first.
                    new_pages: dict = {}
                    for pdf_ext in pdf_result.get("extractions", []):
                        page_num = pdf_ext.get("page_number") or 0
                        page_key = str(page_num)
                        verbatim = pdf_ext.get("verbatim_quote", "")
                        new_pages.setdefault(page_key, []).append(verbatim)
                        all_extractions.append({
                            "claim": verbatim,
                            "citation": verbatim,
                            "source": pdf_ext.get("source_file", pdf_source_file),
                            "verbatim": True,
                            "page_number": pdf_ext.get("page_number"),
                            "element_type": pdf_ext.get("element_type"),
                        })
                    # Write to Layer 2 cache.
                    state["extracted_pdf_pages"] = new_pages
                    state["extracted_pdf_hash"] = current_hash
                    documents_processed.append(pdf_source_file)
                else:
                    logger.warning(
                        "PDF extraction failed for '%s': %s",
                        pdf_source_file,
                        pdf_result.get("error"),
                    )

        # Step 4b — Policy search (if in tool_plan).
        if "policy_search" in tool_plan:
            policy_result = tool_search_policy(
                payer_id=state.get("payer_id", ""),
                procedure_code=state.get("procedure_code", ""),
                extractions=all_extractions,
            )
            tool_trace.append({
                "tool": "tool_search_policy",
                "input": {
                    "payer_id": state.get("payer_id"),
                    "procedure_code": state.get("procedure_code"),
                },
                "output": {
                    "success": policy_result.get("success"),
                    "policy_id": policy_result.get("policy_id"),
                    "criteria_met_count": len(policy_result.get("criteria_met", [])),
                    "criteria_unmet_count": len(policy_result.get("criteria_unmet", [])),
                    "source": policy_result.get("source"),
                },
            })
            tool_call_history.append({
                "tool": "policy_search",
                "status": "success" if policy_result.get("success") else "fail",
                "ts": _ts,
            })
            if policy_result.get("no_policy_found"):
                _payer_id = state.get("payer_id", "")
                all_extractions.append({
                    "claim": policy_result["message"],
                    "citation": f"policy_search:{_payer_id}",
                    "source": "policy_search",
                    "verbatim": True,
                    "synthetic": True,
                    "flag": "NO_POLICY_FOUND",
                })
            else:
                # Surface each criterion (met or unmet) so the LLM can name them.
                # Without this, a 0/5 criteria_met result gives the LLM nothing to
                # reason about and it falls back to generic "I cannot determine."
                _payer_id = state.get("payer_id", "")
                _policy_id = policy_result.get("policy_id", "")
                for criterion in policy_result.get("criteria_met", []):
                    all_extractions.append({
                        "claim": (
                            f"[{_policy_id}] Criteria {criterion.get('id')} MET: "
                            f"{criterion.get('description', '')}"
                        ),
                        "citation": f"policy_search:{_payer_id}",
                        "source": "policy_search",
                        "synthetic": True,
                        "flag": "CRITERIA_MET",
                    })
                for criterion in policy_result.get("criteria_unmet", []):
                    all_extractions.append({
                        "claim": (
                            f"[{_policy_id}] Criteria {criterion.get('id')} NOT MET: "
                            f"{criterion.get('description', '')}"
                        ),
                        "citation": f"policy_search:{_payer_id}",
                        "source": "policy_search",
                        "synthetic": True,
                        "flag": "CRITERIA_UNMET",
                    })
            if isinstance(state.get("payer_policy_cache"), dict):
                state["payer_policy_cache"][state.get("payer_id", "")] = policy_result

        # Step 5 — Denial risk analysis.
        # Only runs for clinically meaningful intents or when a PDF is attached.
        # Skipping for simple list queries (MEDICATIONS, ALLERGIES) avoids false positives.
        query_intent = state.get("query_intent", "")
        run_denial_analysis = query_intent in _DENIAL_RISK_INTENTS or bool(pdf_source_file)
        denial_result: dict = {
            "success": True,
            "risk_level": "NONE",
            "matched_patterns": [],
            "recommendations": [],
            "denial_risk_score": 0.0,
            "source": "mock_data/denial_patterns.json",
            "error": None,
        }
        if run_denial_analysis:
            denial_result = tool_analyze_denial_risk(all_extractions)
            tool_trace.append({
                "tool": "tool_analyze_denial_risk",
                "input": {"extraction_count": len(all_extractions)},
                "output": {
                    "risk_level": denial_result.get("risk_level"),
                    "denial_risk_score": denial_result.get("denial_risk_score"),
                    "matched_pattern_count": len(denial_result.get("matched_patterns", [])),
                },
            })
            tool_call_history.append({"tool": "denial_analyzer", "status": "success" if denial_result.get("success") else "fail", "ts": _ts})

        # Update Layer 1 memory: prior_query_context for the next turn's Orchestrator.
        # Stores the resolved patient name/ID and query intent so follow-up pronouns
        # ("his", "her") resolve without re-running Step 0.
        resolved_patient_name = ""
        if patient and isinstance(patient, dict):
            resolved_patient_name = patient.get("name") or patient.get("id") or ""
        elif patient_identifier:
            resolved_patient_name = patient_identifier
        state["prior_query_context"] = {
            "patient": resolved_patient_name,
            "intent": state.get("query_intent", ""),
            "turn_ts": _ts,
        }

        state["extractions"] = all_extractions
        state["documents_processed"] = documents_processed
        state["denial_risk"] = denial_result
        state["tool_trace"] = tool_trace
        state["tool_call_history"] = tool_call_history
        return state

    except Exception as e:
        state["error"] = f"Extractor error: {str(e)}"
        state["extractions"] = []
        return state
