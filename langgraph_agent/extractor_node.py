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
    scrub_pii: Strips HIPAA PII from text before any LLM/tool call (Presidio-based).
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
from tools import _normalize_dob as _normalize_dob_tools
from tools import get_medications as tool_get_medications
from tools import get_allergies as tool_get_allergies
from tools import check_drug_interactions as tool_check_drug_interactions
from tools.policy_search import search_policy as tool_search_policy
from pdf_extractor import extract_pdf as tool_extract_pdf
from pdf_extractor import get_dob_from_pdf as tool_get_dob_from_pdf
from denial_analyzer import analyze_denial_risk as tool_analyze_denial_risk
from verification import check_allergy_conflict
from langgraph_agent.state import AgentState
import database as _db
from pydantic import ValidationError as _PydanticValidationError
import schemas as _schemas

# Initialise the evidence_staging DB at module load.  Wrapped in try/except so
# a transient I/O error does not prevent the rest of the module from loading —
# the staging layer is asynchronous with respect to the main pipeline.
try:
    _db.init_db()
except Exception as _db_init_exc:
    logger.warning("evidence_staging DB init failed — marker staging disabled: %s", _db_init_exc)

# Intents where denial risk analysis adds RCM value.
# SAFETY_CHECK is excluded — allergy/interaction checks already answer that question,
# and showing a billing denial score alongside "Do NOT administer" confuses the signal.
_DENIAL_RISK_INTENTS = {"INTERACTIONS", "GENERAL_CLINICAL"}


# ── Clinical marker scanner ───────────────────────────────────────────────────
#
# Compiled regex patterns for each oncology / RCM biomarker.
# On a cache-miss PDF extraction the scanner runs over every verbatim element
# and INSERTs found markers into evidence_staging (sync_status='PENDING') so
# the FHIR sync worker can POST them as Observations to OpenEMR.
#
# Pattern design notes:
#   - HER2 / BRCA / ROS1 / Ki-67 / PD-L1 / CA-125: hyphen variants covered.
#   - ER / PR: require a qualifier (status, positive, negative, +/-) to suppress
#     common false-positives ("ER visit", "PR manager").
#   - ALK, KRAS, NRAS, BRAF, EGFR: 4+ letter gene names — bare word-boundary
#     match is sufficient in a clinical document context.
#   - MSI: covers MSI, MSI-H, MSI-L, MSS, and full "microsatellite instability".
#   - TMB: covers both abbreviation and the spelled-out phrase.
#   - Serum markers (PSA, CEA, CA-125, AFP): cover common aliases.

_CLINICAL_MARKERS: dict[str, re.Pattern[str]] = {
    # Breast / general oncology
    "HER2":   re.compile(r"\bHER[-\s]?2(?:/neu)?\b", re.IGNORECASE),
    "ER":     re.compile(
        r"\b(?:ER\s*[+\-]"
        r"|(?:ER|estrogen\s+receptor)\s+"
        r"(?:positive|negative|status|expression|staining|score|result))\b",
        re.IGNORECASE,
    ),
    "PR":     re.compile(
        r"\b(?:PR\s*[+\-]"
        r"|(?:PR|progesterone\s+receptor)\s+"
        r"(?:positive|negative|status|expression|staining|score|result))\b",
        re.IGNORECASE,
    ),
    "Ki-67":  re.compile(r"\bKi[-\s]?67\b", re.IGNORECASE),
    # Immunotherapy
    "PD-L1":  re.compile(r"\bPD[-\s]?L[-\s]?1\b", re.IGNORECASE),
    # Hereditary cancer
    "BRCA1":  re.compile(r"\bBRCA[-\s]?1\b", re.IGNORECASE),
    "BRCA2":  re.compile(r"\bBRCA[-\s]?2\b", re.IGNORECASE),
    # Lung cancer drivers
    "EGFR":   re.compile(r"\bEGFR\b", re.IGNORECASE),
    "ALK":    re.compile(r"\bALK\b", re.IGNORECASE),
    "ROS1":   re.compile(r"\bROS[-\s]?1\b", re.IGNORECASE),
    # Colorectal / pan-cancer RAS/RAF
    "KRAS":   re.compile(r"\bKRAS\b", re.IGNORECASE),
    "NRAS":   re.compile(r"\bNRAS\b", re.IGNORECASE),
    "BRAF":   re.compile(r"\bBRAF\b", re.IGNORECASE),
    # Genomic instability / tumour burden
    "MSI":    re.compile(
        r"\bMSI(?:-[HL])?\b|\bMSS\b|\bmicrosatellite\s+instab\w+\b",
        re.IGNORECASE,
    ),
    "TMB":    re.compile(
        r"\bTMB\b|\btumor\s+mutational\s+burden\b",
        re.IGNORECASE,
    ),
    # Serum tumour markers
    "PSA":    re.compile(
        r"\bPSA\b|\bprostate[-\s]specific\s+antigen\b",
        re.IGNORECASE,
    ),
    "CEA":    re.compile(
        r"\bCEA\b|\bcarcinoembryonic\s+antigen\b",
        re.IGNORECASE,
    ),
    "CA-125": re.compile(r"\bCA[-\s]?125\b", re.IGNORECASE),
    "AFP":    re.compile(
        r"\bAFP\b|\balpha[-\s]fetoprotein\b",
        re.IGNORECASE,
    ),
}

# Searches the 150-character window immediately after a marker hit for a
# result value (IHC score, qualitative result, numeric, or percentage).
_MARKER_VALUE_PATTERN = re.compile(
    r"(?P<value>"
    r"positive|negative|equivocal"
    r"|amplif\w+"
    r"|non[-\s]amplif\w+"
    r"|overexpress\w+"
    r"|wild[-\s]?type"
    r"|mutant?"
    r"|high|low|intermediate"
    r"|MSI-H|MSI-L|MSS"
    r"|[0-3]\+"
    r"|\d+(?:\.\d+)?\s*%"
    r"|\d+(?:\.\d+)?\s*(?:ng/mL|U/mL|mIU/mL|muts/Mb)"
    r")",
    re.IGNORECASE,
)

# Character window after the marker match start to search for a result value.
_VALUE_WINDOW = 150


def _scan_and_stage_markers(
    pdf_extractions: List[dict],
    session_id: str,
    patient_id: str,
    source_file: str,
) -> int:
    """
    Multi-fact scan: search every freshly-extracted PDF element for clinical
    biomarkers and INSERT each occurrence into ``evidence_staging`` with
    ``sync_status='PENDING'`` before the Extractor Node returns its results.

    Only called on a cache-miss (fresh) PDF extraction — cache-hit paths
    already have their markers staged from a prior run in this session.

    For each element the function:
      1. Tests every pattern in ``_CLINICAL_MARKERS`` against the verbatim text.
      2. For each hit, inspects the 150-character window after the match start
         for a result value (e.g. ``"positive"``, ``"3+"``, ``"80%"``).
      3. Calls ``database.insert_clinical_marker()`` — errors are caught per-row
         so a single DB failure never aborts the remaining scan.

    Args:
        pdf_extractions: Raw extraction dicts from ``tool_extract_pdf()``; each
                         must carry ``verbatim_quote``, ``page_number``, and
                         ``element_type``.
        session_id:      LangGraph session ID (``state["session_id"]``).
        patient_id:      Resolved EHR patient ID, or empty string.
        source_file:     Path of the source PDF (propagated to each DB row).

    Returns:
        int: Total number of marker rows successfully inserted.
    """
    staged = 0
    rejected = 0
    for ext in pdf_extractions:
        text = ext.get("verbatim_quote", "")
        if not text:
            continue
        page_number = ext.get("page_number")
        element_type = ext.get("element_type", "")

        for marker_name, pattern in _CLINICAL_MARKERS.items():
            for match in pattern.finditer(text):
                # Extract a result value from the immediate context after the match.
                window_start = match.start()
                window_end = min(match.end() + _VALUE_WINDOW, len(text))
                window = text[window_start:window_end]
                value_match = _MARKER_VALUE_PATTERN.search(window)
                marker_value = value_match.group("value") if value_match else ""

                # ── Validate against the LOINC registry before inserting ──
                # ClinicalObservation ensures fact_type is LOINC-mapped,
                # fact_value is a clean string, and raw_text has provenance.
                # Findings that fail validation are logged and skipped — they
                # will never enter evidence_staging or reach the FHIR sync worker.
                try:
                    obs = _schemas.ClinicalObservation(
                        fact_type=marker_name,
                        fact_value=marker_value,
                        raw_text=text,
                        session_id=session_id,
                        patient_id=patient_id,
                        source_file=source_file,
                        page_number=page_number,
                        element_type=element_type,
                        confidence=1.0,
                    )
                except _PydanticValidationError as val_exc:
                    rejected += 1
                    first_err = val_exc.errors()[0]["msg"] if val_exc.errors() else str(val_exc)
                    logger.debug(
                        "evidence_staging: skipping marker '%s' (validation rejected) — %s",
                        marker_name,
                        first_err,
                    )
                    continue

                # ── Insert validated, cleaned values into SQLite ──────────
                try:
                    _db.insert_clinical_marker(**obs.to_db_kwargs())
                    staged += 1
                except Exception as db_exc:
                    logger.warning(
                        "evidence_staging INSERT failed for marker '%s' in '%s': %s",
                        marker_name,
                        os.path.basename(source_file),
                        db_exc,
                    )

    if staged or rejected:
        logger.info(
            "evidence_staging: staged %d, rejected %d clinical marker(s) from '%s'.",
            staged,
            rejected,
            os.path.basename(source_file),
        )
    return staged


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


# ── PII Scrubber ──────────────────────────────────────────────────────────────

from tools.pii_scrubber import scrub_pii as _scrub_pii


# ── Extraction formatter ──────────────────────────────────────────────────────

def _format_extractions(
    patient: dict,
    medications: List[dict],
    interactions: List[dict],
    med_source: str = "",
) -> List[dict]:
    """
    Convert raw tool results into extraction dicts with verbatim citations.
    Each extraction contains the claim, the verbatim quote from source, and the source path.

    Args:
        patient: Patient dict from tool_get_patient_info.
        medications: Medication list from tool_get_medications.
        interactions: Interaction list from tool_check_drug_interactions.
        med_source: Source label returned by get_medications (e.g. "Live EHR (OpenEMR FHIR)").
                    When this is a live EHR source the extractions are marked ``synthetic=True``
                    so the auditor skips source-file citation verification — FHIR responses
                    cannot be verified against a local JSON file.

    Returns:
        List[dict]: Extraction dicts, each with claim, citation, source, verbatim keys.

    Raises:
        Never — returns empty list on any failure.
    """
    try:
        extractions = []

        # Determine whether patient data came from live EHR (FHIR) or local cache.
        patient_from_fhir = patient and patient.get("source") == "openemr_fhir"
        # Medications from live EHR cannot be verified against a local JSON file.
        meds_from_fhir = bool(med_source and ("fhir" in med_source.lower() or "ehr" in med_source.lower()))

        if patient:
            allergies = patient.get("allergies", [])
            allergies_str = ", ".join(allergies) if allergies else "none on record"
            # synthetic=True ONLY for the empty-allergy placeholder — it is a derived
            # statement ("no known allergies") not present verbatim in any source file
            # AND we don't want the LLM repeating it in medication-intent responses.
            # When allergies ARE populated (from FHIR or local cache) the claim is real
            # data; synthetic=False so the LLM includes it in its synthesis.
            extractions.append({
                "kind":     "allergy",
                "claim":    f"{patient['name']} has known allergies: {allergies_str}.",
                "citation": allergies_str if allergies else "no known allergies",
                "source":   "openemr_fhir" if patient_from_fhir else "mock_data/patients.json",
                "verbatim": True,
                "synthetic": not bool(allergies),
            })

        for med in medications:
            # FHIR medication data is real EHR data — synthetic=False so the LLM
            # synthesis includes it in the facts list.  The auditor's
            # _verify_citation_exists_in_source already short-circuits to True for
            # "openemr_fhir" sources, so no file-verification is attempted.
            extractions.append({
                "kind":     "medication",
                "claim":    (
                    f"{patient['name']} is prescribed {med['name']} "
                    f"{med.get('dose', '')} {med.get('frequency', '')}."
                ),
                "citation": f"{med['name']} {med.get('dose', '')} {med.get('frequency', '')}".strip(),
                "source":   "openemr_fhir" if meds_from_fhir else "mock_data/medications.json",
                "verbatim": True,
                "synthetic": False,
            })

        for interaction in interactions:
            extractions.append({
                "kind":     "interaction",
                "claim":    (
                    f"Drug interaction detected: {interaction['drug1']} + {interaction['drug2']} "
                    f"— severity {interaction['severity']}."
                ),
                "citation": (
                    f"{interaction['drug1']} and {interaction['drug2']}: "
                    f"{interaction.get('recommendation', interaction['severity'])}"
                ),
                "source":   "mock_data/interactions.json",
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
        clean_query = _scrub_pii(state["input_query"])

        if state.get("clarification_response"):
            clean_query = f"{clean_query} {_scrub_pii(state['clarification_response'])}"

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

        # ── DOB for identity resolution (composite key: name + DOB) ─────────────
        # When we have a PDF, extract DOB so same-name patients are not merged.
        pdf_source_file_for_dob = (state.get("pdf_source_file") or "").strip()
        lookup_dob: Optional[str] = _normalize_dob_tools(state.get("identified_patient_dob"))
        if not lookup_dob and pdf_source_file_for_dob:
            lookup_dob = tool_get_dob_from_pdf(pdf_source_file_for_dob)

        # ── Patient cache check ───────────────────────────────────────────────
        # If Orchestrator already cached the patient this session, skip Step 0
        # and patient_lookup — unless we have a DOB that disagrees with cache
        # (e.g. new PDF for a different same-name patient).
        cached_patient = state.get("extracted_patient") or {}
        use_patient_cache = (
            orchestrator_ran
            and "patient_lookup" not in tool_plan
            and bool(cached_patient)
        )
        if use_patient_cache and lookup_dob:
            cached_dob = _normalize_dob_tools(cached_patient.get("dob"))
            if cached_dob and cached_dob != lookup_dob:
                logger.info(
                    "Extractor: cached patient DOB %s != requested DOB %s — invalidating cache (different identity).",
                    cached_dob, lookup_dob,
                )
                state["extracted_patient"] = {}
                cached_patient = {}
                use_patient_cache = False

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

            patient_result = tool_get_patient_info(patient_identifier, dob=lookup_dob)
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

            patient_result = tool_get_patient_info(patient_identifier, dob=lookup_dob)
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

                # Scenario A — PDF provided but patient not in EHR (or name match, DOB different).
                # Clear extracted_patient so we do not bleed prior cached patient into this identity.
                state["extracted_patient"] = {}
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
        med_source: str = ""
        found_interactions: List[dict] = []
        proposed_drug = ""
        allergy_conflict_result: dict = {}

        if patient is not None:
            # ── Allergy refresh — runs every turn (cached patient may have stale [] allergies) ─
            # FHIR allergy prefetch only happens during patient_lookup. For follow-up turns the
            # patient comes from the session cache where allergies may be empty. Calling
            # tool_get_allergies with the patient's FHIR UUID ensures live data every time.
            allergy_result = tool_get_allergies(patient_id)
            tool_trace.append({
                "tool": "tool_get_allergies",
                "input": patient_id,
                "output": allergy_result,
            })
            if allergy_result.get("success") and allergy_result.get("allergies") is not None:
                fresh_allergies = allergy_result["allergies"]
                if fresh_allergies != patient.get("allergies"):
                    patient = {**patient, "allergies": fresh_allergies}
                    state["extracted_patient"] = patient
                    logger.info(
                        "Extractor: refreshed allergies for %s → %s (source: %s)",
                        patient_id, fresh_allergies, allergy_result.get("source"),
                    )

            meds_result = tool_get_medications(patient_id)
            tool_trace.append({
                "tool": "tool_get_medications",
                "input": patient_id,
                "output": meds_result,
            })
            tool_call_history.append({"tool": "med_retrieval", "status": "success" if meds_result.get("success") else "fail", "ts": _ts})
            medications = meds_result.get("medications", []) if meds_result.get("success") else []
            med_source = meds_result.get("source", "")

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

        all_extractions = _format_extractions(patient, medications, found_interactions, med_source=med_source)

        # Scenario A: patient not in EHR but PDF is attached — prepend EHR gap flags.
        if patient is None and ehr_gap_extractions:
            all_extractions = ehr_gap_extractions + all_extractions

        # Always inject the allergy check result as an explicit grounding fact so the
        # synthesis LLM has an authoritative anchor regardless of the outcome.
        # Conflict=True: fires the CONFLICT DETECTED path + denial_analyzer keyword match.
        # Conflict=False: gives the LLM an explicit "no conflict" statement to quote,
        # preventing it from filling the silence with its own pharmacological reasoning
        # (the root cause of the Fenofibrate/sulfa hallucination).
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
        elif proposed_drug:
            # Inject a grounding fact when the check returns NO conflict.
            # The synthesis prompt's SAFETY_CHECK HARD RULE requires this fact to be
            # present so the LLM can cite it directly instead of reasoning from scratch.
            all_extractions.append({
                "claim": (
                    f"allergy_conflict_check_result: NO CONFLICT — {proposed_drug} does not "
                    f"match any of the patient's documented allergies "
                    f"({allergy_conflict_result.get('source_citation', '')}). "
                    f"This is the authoritative allergy check result from validated drug-class "
                    f"guidelines. Do NOT add drug-chemistry reasoning beyond this result."
                ),
                "citation": allergy_conflict_result.get(
                    "source_citation",
                    f"Allergy check: no conflict found for {proposed_drug}.",
                ),
                "source": "mock_data/patients.json",
                "verbatim": True,
                "synthetic": True,
            })
            logger.info(
                "SAFETY_CHECK no-conflict grounding fact injected for drug '%s'.",
                proposed_drug,
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
                            "kind":         "pdf_content",
                            "claim":        page_text,
                            "citation":     page_text,
                            "source":       pdf_source_file,
                            "verbatim":     True,
                            "page_number":  page_num,
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
                        # Scrub PII before the text enters extractions or the
                        # cache — checkpoints must store scrubbed data only.
                        verbatim = _scrub_pii(pdf_ext.get("verbatim_quote", ""))
                        new_pages.setdefault(page_key, []).append(verbatim)
                        all_extractions.append({
                            "kind":         "pdf_content",
                            "claim":        verbatim,
                            "citation":     verbatim,
                            "source":       pdf_ext.get("source_file", pdf_source_file),
                            "verbatim":     True,
                            "page_number":  pdf_ext.get("page_number"),
                            "element_type": pdf_ext.get("element_type"),
                        })
                    # Write to Layer 2 cache.
                    state["extracted_pdf_pages"] = new_pages
                    state["extracted_pdf_hash"] = current_hash
                    documents_processed.append(pdf_source_file)

                    # Multi-fact clinical marker scan — INSERT every detected
                    # biomarker into evidence_staging before returning results.
                    # Runs only on fresh extraction (cache miss) so markers are
                    # not double-staged on subsequent cache-hit turns.
                    _staged = _scan_and_stage_markers(
                        pdf_extractions=pdf_result.get("extractions", []),
                        session_id=state.get("session_id", ""),
                        patient_id=patient_id or "",
                        source_file=pdf_source_file,
                    )
                    if _staged:
                        tool_trace.append({
                            "tool": "clinical_marker_scan",
                            "input": {
                                "source_file": pdf_source_file,
                                "elements_scanned": len(pdf_result.get("extractions", [])),
                            },
                            "output": {"markers_staged": _staged},
                        })
                        tool_call_history.append({
                            "tool": "clinical_marker_scan",
                            "status": "success",
                            "ts": _ts,
                        })
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
