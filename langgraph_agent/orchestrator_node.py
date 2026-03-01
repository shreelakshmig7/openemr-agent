"""
orchestrator_node.py
--------------------
AgentForge — Healthcare RCM AI Agent — Orchestrator Node
---------------------------------------------------------
Decides which tools the Extractor Node should call and in what order,
based on semantic intent classification via Claude Haiku. Sits between
the Router Node and the Extractor Node.

Responsibilities:
    1. Classify the query using a SINGLE Haiku call → structured JSON with intent
       flags AND the extracted patient name. This eliminates the former Step 0
       Haiku call in the Extractor, halving API load per request.
    2. Check Layer 2 session caches to skip redundant tool calls.
    3. Build an ordered tool_plan list consumed by the Extractor Node.
    4. For general knowledge queries (is_general_knowledge=True), set
       tool_plan=[] so the Extractor skips all tools — preventing false
       Scenario A triggers and phantom confidence penalties.

Design notes:
    - The Orchestrator NEVER calls tools itself. It only plans.
    - A single Haiku call returns BOTH intent classification AND patient_name,
      eliminating the former extractor Step 0 call (double-call regression fix).
    - Keyword matching is intentionally avoided — Haiku classification handles
      semantic intent, eliminating the false-trigger hallucination risk.
    - Cache checks use Layer 2 state fields set by prior Extractor runs in
      the same session. The Orchestrator reads but never writes to caches.
    - Haiku calls are retried up to 3 times with exponential backoff on HTTP 529
      (Anthropic overloaded). After all retries fail, _orchestrator_fallback()
      uses regex extraction so the pipeline always continues.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import json
import logging
import os
import re
import time
from typing import Any, Optional, Set

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_agent.state import AgentState

logger = logging.getLogger(__name__)

# ── 529 detection helper ──────────────────────────────────────────────────────

try:
    from anthropic import APIStatusError as _AnthropicAPIStatusError
except ImportError:
    _AnthropicAPIStatusError = None  # type: ignore[assignment,misc]


def _is_529(exc: Exception) -> bool:
    """Return True if the exception represents an Anthropic HTTP 529 (overloaded)."""
    if _AnthropicAPIStatusError and isinstance(exc, _AnthropicAPIStatusError):
        return getattr(exc, "status_code", None) == 529
    return "529" in str(exc) or "overloaded" in str(exc).lower()


# ── Haiku classification + extraction prompt ──────────────────────────────────

_ORCHESTRATOR_SYSTEM = """You are an RCM (Revenue Cycle Management) agent orchestrator.
Your job is to classify a clinical query, identify the patient (if any), and decide
which data sources are needed.
Return JSON only. No explanation, no markdown, no code fences.

Fields to return:
{
    "needs_specific_patient": true/false,
    "needs_document_evidence": true/false,
    "needs_policy_check": true/false,
    "needs_denial_analysis": true/false,
    "is_general_knowledge": true/false,
    "patient_name": "Full Name" or null,
    "payer_name": "payer name or ID as stated in query (lowercase)" or null,
    "procedure_identifier": "CPT code or procedure description as stated in query" or null,
    "data_source_required": "EHR" or "PDF" or "RESIDENT_NOTE" or "IMAGING" or "NONE",
    "pdf_required": true/false
}

Rules:
- needs_specific_patient = true ONLY if query refers to a named patient,
  a patient ID (P001-style), or a pronoun that refers to a previously
  identified patient (e.g. "his medications", "her allergies").
- needs_document_evidence = true if query asks about findings, evidence,
  or details that would come from a clinical note, PDF, or chart.
- needs_policy_check = true if query mentions insurance, coverage, payer,
  authorization, criteria, or a specific payer name.
- needs_denial_analysis = true if query asks about denial risk, submission
  likelihood, claim approval, or rejection chance.
- is_general_knowledge = true if the query is about general medical concepts,
  pharmacology, or clinical guidelines NOT tied to a specific patient or document.
  When is_general_knowledge = true, ALL other fields must be false.
- patient_name: Extract the full patient name if explicitly stated in the query.
  Preserve original casing (e.g. "John Smith" not "john smith").
  If the query references a patient by ID (e.g. "P001", "P999", "MRN-00123456")
  rather than by name, return that identifier verbatim as patient_name.
  Return null if no name or identifier is stated — do NOT resolve identifiers
  to names from prior context.
- payer_name: Only populate when needs_policy_check is true.
  Extract the payer/insurer name or ID exactly as stated in the query.
  Return lowercase (e.g. "Cigna" → "cigna", "Aetna" → "aetna", "BCB001" → "bcb001").
  Accept both name ("Cigna") and payer ID ("BCB001") — return whichever is stated.
  Return null if no payer is mentioned or needs_policy_check is false.
- procedure_identifier: Only populate when needs_policy_check is true.
  Extract the CPT code if explicitly stated (e.g. "27447", "99213"),
  or the procedure description if no code is given (e.g. "knee replacement",
  "total hip arthroplasty"). Return the value exactly as stated — do NOT
  translate descriptions to CPT codes. Return null if no procedure is mentioned
  or needs_policy_check is false.
- data_source_required: The primary data source the query needs.
  "EHR" — medications, allergies, patient demographics.
  "PDF" — clinical notes, charts, or attached documents (generic).
  "RESIDENT_NOTE" — specifically mentions resident note, attending note,
    nursing note, physician note, or any authored clinical note.
  "IMAGING" — X-ray, MRI, CT scan, PET scan, ultrasound results.
    If the query refers to an imaging section or imaging findings WITHIN
    a clinical note or attached PDF, use "PDF" not "IMAGING".
    Use "IMAGING" only when the query asks for a standalone radiology
    report or separate imaging file, not a section of a clinical document.
  "NONE" — general knowledge questions with no specific source needed.
  IMPORTANT — policy check override: When needs_policy_check is true,
    ALWAYS set data_source_required to "EHR" and pdf_required to false,
    UNLESS the query explicitly references an attached document using phrases
    such as: "this chart", "this document", "this PDF", "this note",
    "the attached note", "the attached document", "the attached chart",
    "attached doc", "attached chart", "above doc", "above attachment",
    "uploaded doc", "uploaded chart", "the note above".
    The procedure name alone (MRI, knee replacement, Palbociclib, CT scan)
    is NOT a signal to require a document — it is the subject of the criteria
    check, not a data source. Do NOT set pdf_required: true just because
    a procedure or imaging modality is mentioned in a policy/criteria query.
- pdf_required: true if data_source_required is PDF, RESIDENT_NOTE, or IMAGING
  AND the above policy check override does not apply.
  false otherwise.

Examples:
  "What medications is John Smith taking?"
    → needs_specific_patient: true, patient_name: "John Smith",
      data_source_required: "EHR", pdf_required: false

  "What are the contraindications of Warfarin in general?"
    → is_general_knowledge: true, patient_name: null,
      data_source_required: "NONE", pdf_required: false, all others: false

  "Is there an ECOG score in the resident note for Maria?"
    → needs_specific_patient: true, patient_name: "Maria",
      data_source_required: "RESIDENT_NOTE", pdf_required: true

  "Does this chart support Cigna's criteria for knee replacement?"
    → needs_document_evidence: true, needs_policy_check: true,
      data_source_required: "PDF", pdf_required: true, patient_name: null,
      payer_name: "cigna", procedure_identifier: "knee replacement"
    (pdf_required: true because query explicitly says "this chart")

  "Does John Smith meet Aetna's criteria for MRI?"
    → needs_specific_patient: true, needs_policy_check: true,
      patient_name: "John Smith", payer_name: "aetna",
      procedure_identifier: "mri", data_source_required: "EHR",
      pdf_required: false
    (pdf_required: false — "MRI" is the procedure being authorized,
     NOT a request to read an MRI scan report)

  "What does John Smith's MRI show?"
    → needs_specific_patient: true, needs_policy_check: false,
      patient_name: "John Smith", data_source_required: "IMAGING",
      pdf_required: true
    (pdf_required: true — query asks to READ imaging results)

  "Does John Smith meet criteria for procedure 27447 with BCB001?"
    → needs_specific_patient: true, needs_policy_check: true,
      patient_name: "John Smith", payer_name: "bcb001",
      procedure_identifier: "27447", data_source_required: "EHR",
      pdf_required: false

  "What are his allergies?"
    → needs_specific_patient: true, patient_name: null,
      data_source_required: "EHR", pdf_required: false
      (pronoun only — Extractor will use session context)

════════════════════════════════════════════════════════════════════
STRICT 4-PHASE CLINICAL AUDIT PROTOCOL — PDF UPLOAD (MANDATORY)
════════════════════════════════════════════════════════════════════
You are a Clinical Audit Orchestrator. When a PDF is attached OR the query
references a biopsy, pathology report, prior authorisation, clinical note,
or any attached document, this protocol is MANDATORY. No phase may be skipped.

CLASSIFICATION (set ALL of these for any PDF query):
  needs_specific_patient: true   — EHR baseline is always required
  needs_document_evidence: true  — PDF extraction is mandatory
  needs_policy_check: true       — payer compliance audit always runs
  needs_denial_analysis: true    — denial risk analysis always runs
  data_source_required: "PDF"
  pdf_required: true
  payer_name: extract from query or PDF header (e.g. "aetna", "cigna")
  procedure_identifier: extract CPT or drug name (e.g. "Palbociclib", "J9999")

────────────────────────────────────────────────────────────────────
PHASE 1 — DATA INGESTION (MANDATORY)
────────────────────────────────────────────────────────────────────
1. IDENTIFY  → tool_get_patient_info — establish Live EHR FHIR baseline.
2. EXTRACT   → tool_extract_pdf + clinical_marker_scan — populate SQLite
               evidence_staging with every biomarker (ER, PR, HER2, Ki-67,
               ECOG, etc.) found in the document.

────────────────────────────────────────────────────────────────────
PHASE 2 — CLINICAL GATEKEEPERS / AUDIT SUITE (NEVER SKIP)
────────────────────────────────────────────────────────────────────
3. SAFETY AUDIT
   → tool_get_medications + tool_check_drug_interactions
   Cross-reference patient allergies (from Phase 1 EHR) against every drug
   named in the PDF.  Any HIGH or CRITICAL conflict is a blocking alert.

4. COMPLIANCE AUDIT
   → tool_analyze_denial_risk + policy_search
   Map PDF findings against Aetna/Cigna CPB criteria. Flag MISSING items
   (ECOG Performance Status, CBC/CMP labs, etc.) as documentation gaps.

5. INTEGRITY AUDIT
   → Compare attending-physician notes against resident/nursing notes in the
   PDF for contradictions (e.g. Attending: "start Palbociclib" vs Resident:
   "pause all treatment"). Surface any contradiction as a provider conflict
   in the synthesis — this is a patient-safety signal.

────────────────────────────────────────────────────────────────────
PHASE 3 — RECONCILIATION & SYNTHESIS (automatic via comparison_node)
────────────────────────────────────────────────────────────────────
6. RECONCILE → comparison_node deduplicates evidence_staging into unique
   clinical champions and compares against the Live EHR portal.
   Required output format (exact):
     "I found [X] clinical markers in this PDF.
      [Y] are already recorded in the OpenEMR portal.
      [Z] are new unique discoveries: [ER+, HER2-, PR+, ...].
      Would you like me to sync the new discoveries to OpenEMR?"

   Combine Phase 2 audit results (denial patterns, interactions) into the
   body of the response BEFORE the reconciliation line.

────────────────────────────────────────────────────────────────────
PHASE 4 — HUMAN-IN-THE-LOOP EXECUTION GATE (STRICT LOCK)
────────────────────────────────────────────────────────────────────
7. You are PROHIBITED from calling sync_execution_node until the user
   provides an explicit confirmation matching: yes / sync / proceed /
   do it / confirm / go ahead / push.
   Any other response clears the sync flag. Do NOT infer approval.

8. WARNING PROTOCOL — If Phase 2 raised a HIGH-SEVERITY alert (allergy
   conflict or CRITICAL denial risk), you MUST re-state the warning in
   bold before displaying the sync confirmation question.
   Format: "⚠️ WARNING: [alert text]. Do you still wish to proceed?"

════════════════════════════════════════════════════════════════════
"""


def _classify_query(query: str, prior_context: dict) -> dict:
    """
    Use a single Claude Haiku call to classify query intent AND extract the patient name.

    Combines what was previously two separate Haiku calls (orchestrator classification +
    extractor Step 0 patient extraction) into one, halving API calls per request.

    Retries up to 3 times with exponential backoff on HTTP 529 (Anthropic overloaded).
    Raises the exception after the final retry so orchestrator_node can invoke
    _orchestrator_fallback() instead of returning 0% confidence.

    Args:
        query: The user's natural language query for this turn.
        prior_context: dict from state["prior_query_context"] — contains resolved
            patient name/ID and intent from the previous turn for pronoun resolution.

    Returns:
        dict: Classification with 5 boolean flags plus patient_name (str | None).
              Returns safe defaults on non-529 parse/network failures.

    Raises:
        Exception: Re-raises after max retries so caller can trigger fallback.
    """
    safe_default = {
        "needs_specific_patient": True,
        "needs_document_evidence": False,
        "needs_policy_check": False,
        "needs_denial_analysis": False,
        "is_general_knowledge": False,
        "patient_name": None,
        "payer_name": None,
        "procedure_identifier": None,
        "data_source_required": "EHR",
        "pdf_required": False,
    }

    context_note = ""
    if prior_context.get("patient"):
        context_note = (
            f"\nPrior turn context: patient = {prior_context['patient']}, "
            f"intent = {prior_context.get('intent', 'unknown')}."
        )

    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
        max_tokens=256,
    )

    max_attempts = 3
    last_exc: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            response = llm.invoke([
                SystemMessage(content=_ORCHESTRATOR_SYSTEM),
                HumanMessage(content=f"{context_note}\nQuery: {query}"),
            ])
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip()

            # Strip markdown code fences if Haiku wraps the JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)

            # Fill in any missing keys with safe defaults
            for key, default_val in safe_default.items():
                if key not in parsed:
                    parsed[key] = default_val

            # Enforce invariant: is_general_knowledge → all others False
            if parsed.get("is_general_knowledge"):
                parsed["needs_specific_patient"] = False
                parsed["needs_document_evidence"] = False
                parsed["needs_policy_check"] = False
                parsed["needs_denial_analysis"] = False
                parsed["patient_name"] = None
                parsed["data_source_required"] = "NONE"
                parsed["pdf_required"] = False

            return parsed

        except Exception as exc:
            last_exc = exc
            if _is_529(exc):
                if attempt < max_attempts - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "Orchestrator Haiku call got 529 (attempt %d/%d) — retrying in %ds",
                        attempt + 1, max_attempts, wait,
                    )
                    time.sleep(wait)
                    continue
                else:
                    logger.error(
                        "Orchestrator Haiku call failed after %d attempts (529 overloaded) — "
                        "triggering fallback",
                        max_attempts,
                    )
                    raise
            else:
                # Non-529 error (parse failure, auth, etc.) — use safe defaults immediately
                logger.warning("Orchestrator classification failed: %s — using safe defaults", exc)
                return safe_default

    # Should not reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]


# ── Patient identity helpers (Bug 1 — cross-patient cache collision) ──────────

def _names_match(name_a: str, name_b: str) -> bool:
    """
    Fuzzy patient name comparison tolerating middle initials and minor formatting.

    Splits each name into tokens, drops single-character tokens (initials), and
    checks whether at least 2 tokens are shared (first + last name minimum).

    Examples:
        "Maria Gonzalez"    vs "Maria J. Gonzalez" → True  (first + last match)
        "John Smith"        vs "John Smith Jr"      → True  (first + last match)
        "John Smith"        vs "Maria Gonzalez"     → False (no overlap)

    Args:
        name_a: First patient name string.
        name_b: Second patient name string.

    Returns:
        bool: True if names refer to the same patient, False otherwise.

    Raises:
        Never — returns False on any exception.
    """
    try:
        def _normalize(name: str) -> Set[str]:
            parts = name.lower().replace(".", "").split()
            return {p for p in parts if len(p) > 1}

        return len(_normalize(name_a) & _normalize(name_b)) >= 2
    except Exception:
        return False


def _invalidate_patient_cache(state: AgentState) -> None:
    """
    Clear all patient-specific cache fields when the Orchestrator detects that
    the incoming query is for a different patient than the one currently cached.

    Fields cleared:
        extracted_patient      — patient record (patient-specific)
        extracted_pdf_pages    — prior patient's clinical document cache
        extracted_pdf_hash     — hash tied to prior document cache

    Fields NOT cleared:
        pdf_source_file        — request-level input, NOT a cache value. The
                                 user may have attached a PDF for the new patient
                                 in this request. Clearing it would wipe their
                                 attachment and prevent Scenario A (PDF-only
                                 fallback when patient is not in EHR). The
                                 content-hash check in the Extractor already
                                 handles stale PDF re-extraction when the
                                 underlying file changes.
        payer_policy_cache, denial_risk_cache — patient-agnostic.

    Args:
        state: Current AgentState — mutated in place.

    Returns:
        None

    Raises:
        Never.
    """
    state["extracted_patient"] = {}
    state["extracted_pdf_pages"] = {}
    state["extracted_pdf_hash"] = ""


# ── Regex fallback (used only when all Haiku retries are exhausted) ───────────

_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

_EXPLICIT_PATIENT_ID_RE = re.compile(
    r'\b(?:[Pp]\d+|MRN-?\d+)\b',
    re.IGNORECASE,
)


def _query_has_explicit_patient_id(query: str) -> Optional[str]:
    """Return the first explicit patient ID (P###, MRN-###) in the query, or None."""
    m = _EXPLICIT_PATIENT_ID_RE.search(query)
    return m.group(0) if m else None


def _orchestrator_fallback(state: AgentState, query: str) -> AgentState:
    """
    Called when all Haiku retries fail (HTTP 529 exhausted).

    Extracts the patient name via simple regex rather than returning 0% confidence.
    Defaults to the full pipeline so no data is silently dropped. Sets
    orchestrator_fallback=True in state so LangSmith traces capture 529 frequency.

    Args:
        state: Current AgentState.
        query: The user's query for this turn (used for regex name extraction).

    Returns:
        AgentState: State with tool_plan, identified_patient_name, orchestrator_ran,
            and orchestrator_fallback set.
    """
    name_match = _NAME_PATTERN.search(query)
    extracted_name: Optional[str] = name_match.group(0) if name_match else None

    logger.warning(
        "Orchestrator fallback active — regex extracted name: %r — defaulting to full pipeline",
        extracted_name,
    )

    state["identified_patient_name"] = extracted_name
    state["tool_plan"] = ["patient_lookup", "med_retrieval"]
    state["orchestrator_ran"] = True
    state["orchestrator_fallback"] = True
    return state


# ── Main orchestrator node ────────────────────────────────────────────────────

def orchestrator_node(state: AgentState) -> AgentState:
    """
    Orchestrator Node — single Haiku call for both intent classification and
    patient name extraction, then builds an ordered tool_plan for the Extractor.

    A single Haiku call now returns both intent flags AND the patient name.
    This eliminates the former Step 0 call in the Extractor, reducing API calls
    from 2 per request to 1. The Extractor reads identified_patient_name from
    state instead of calling _extract_patient_identifier_llm.

    Tool plan order:
        1. patient_lookup   — resolve patient identifier to structured record
        2. med_retrieval    — get medication list (requires patient_lookup first)
        3. pdf_extractor    — extract clinical PDF (hash-validated cache)
        4. policy_search    — match against payer criteria
        5. denial_analyzer  — score denial risk (requires audit_results)

    Cache skip logic:
        - patient_lookup   skipped if extracted_patient is already populated
        - pdf_extractor    skipped if extracted_pdf_hash matches current PDF content
        - policy_search    skipped if payer_id already in payer_policy_cache
        - denial_analyzer  skipped if (payer_id:cpt) already in denial_risk_cache

    Retry / fallback:
        - Haiku call is retried up to 3 times with exponential backoff on 529.
        - After all retries fail, _orchestrator_fallback() keeps the pipeline alive.

    Args:
        state: Current AgentState after Router Node has classified intent.

    Returns:
        AgentState: State with tool_plan, identified_patient_name, and
            orchestrator_ran set. All other fields unchanged.

    Raises:
        Never — returns state with fallback tool_plan on any unrecoverable failure.
    """
    # ── HITL sync-confirmation pre-check ─────────────────────────────────────
    # This runs BEFORE any Haiku call.  If the user previously uploaded a PDF
    # and we set pending_sync_confirmation=True, check whether the new message
    # is a sync approval.  If yes, set routing_decision="sync" so
    # _route_from_orchestrator() in workflow.py diverts to sync_execution_node
    # instead of extractor, skipping the full extraction pipeline entirely.
    #
    # Safety: if the user's message is NOT a confirmation (they changed topic or
    # uploaded a new PDF), clear the stale flag so it never accidentally fires.
    if state.get("pending_sync_confirmation"):
        raw_msg = (state.get("input_query") or "").lower().strip()
        # Import here to avoid circular import at module load time.
        from langgraph_agent.comparison_node import SYNC_CONFIRM_WORDS  # noqa: PLC0415
        is_sync_confirm = any(word in raw_msg for word in SYNC_CONFIRM_WORDS)

        if is_sync_confirm:
            logger.info(
                "Orchestrator: sync confirmation detected — routing to sync_execution_node."
            )
            # Clear the flag; sync_execution_node will handle the actual clear.
            state["pending_sync_confirmation"] = False
            state["routing_decision"] = "sync"
            state["orchestrator_ran"] = True
            return state
        else:
            # User changed topic or uploaded a new PDF — clear stale sync state.
            logger.info(
                "Orchestrator: pending_sync_confirmation was True but user message "
                "is not a sync confirmation — clearing stale HITL flag."
            )
            state["pending_sync_confirmation"] = False
            state["sync_summary"] = {}
            state["staged_patient_fhir_id"] = ""
            state["staged_session_id"] = ""

    query = state.get("input_query", "")
    prior_context = state.get("prior_query_context") or {}

    # When resuming from a clarification pause, the user's answer (e.g. "John Smith")
    # is in clarification_response but NOT in input_query. Append it here so Haiku
    # sees the combined context and can extract the patient name correctly.
    # Without this, Haiku returns patient_name=null and the Extractor asks again.
    clarification = (state.get("clarification_response") or "").strip()
    if clarification:
        query = f"{query} {clarification}"

    try:
        intent = _classify_query(query, prior_context)
        logger.info("Orchestrator intent: %s", intent)

        # Store classification flags and data source in state.
        incoming_name: Optional[str] = intent.get("patient_name")
        state["orchestrator_fallback"] = False
        state["data_source_required"] = intent.get("data_source_required", "EHR")
        state["is_general_knowledge"] = bool(intent.get("is_general_knowledge", False))

        # ── Payer and procedure extraction ─────────────────────────────────
        # Mirror the patient_name pattern: Haiku extracts payer/procedure from
        # the query (or the combined query+clarification_response text built
        # above), and we write it to state exactly as stated.
        # Guard: only when needs_policy_check=true, and only when the caller
        # has not already supplied a value (e.g. eval runner sends payer_id
        # explicitly — we must not overwrite it).
        if intent.get("needs_policy_check"):
            payer_name = (intent.get("payer_name") or "").strip().lower()
            if payer_name and not state.get("payer_id"):
                state["payer_id"] = payer_name
                logger.info("Orchestrator: extracted payer_id '%s' from query", payer_name)

            proc_id = (intent.get("procedure_identifier") or "").strip()
            if proc_id and not state.get("procedure_code"):
                state["procedure_code"] = proc_id
                logger.info("Orchestrator: extracted procedure_code '%s' from query", proc_id)

        # ── Pronoun resolution ─────────────────────────────────────────────
        # When Haiku returns patient_name=null (pronoun like "he", "she", "his"),
        # resolve to the patient from the prior turn so the cache collision check
        # fires correctly. Without this, incoming_name=None skips the check and
        # a cached different patient's data is silently served.
        #
        # Guard: if the query contains an explicit patient ID (P###, MRN-###),
        # this is NOT a pronoun — the user is referencing a specific patient by
        # ID. Use the raw ID instead of falling back to the prior patient.
        if not incoming_name and intent.get("needs_specific_patient"):
            explicit_id = _query_has_explicit_patient_id(query)
            if explicit_id:
                incoming_name = explicit_id
                logger.info(
                    "Orchestrator: explicit patient ID '%s' in query — "
                    "using as identifier (not a pronoun)",
                    explicit_id,
                )
            else:
                prior_patient = prior_context.get("patient", "")
                if prior_patient:
                    incoming_name = prior_patient
                    logger.info(
                        "Orchestrator: pronoun in '%s' resolved to prior patient '%s'",
                        query[:40],
                        incoming_name,
                    )

        state["identified_patient_name"] = incoming_name

        # ── Patient identity cache validation ──────────────────────────────
        # Compare the resolved incoming name against the cached patient name
        # AND cached patient ID. If neither matches, the user switched patients
        # — invalidate patient-specific caches so stale data is never served.
        cached_patient = state.get("extracted_patient") or {}
        cached_name = cached_patient.get("name", "").strip()
        cached_id = cached_patient.get("id", "").strip()
        if cached_patient and incoming_name and cached_name:
            name_matches = _names_match(cached_name, incoming_name)
            id_matches = bool(
                cached_id
                and incoming_name.strip().upper() == cached_id.upper()
            )
            if not name_matches and not id_matches:
                logger.info(
                    "Orchestrator: patient switched '%s' → '%s' — invalidating patient cache",
                    cached_name, incoming_name,
                )
                _invalidate_patient_cache(state)

        # Safety net: if the query explicitly mentions a patient ID that differs
        # from the cached patient's ID, invalidate the cache even if Haiku
        # incorrectly resolved the name to the cached patient (prior-context
        # leakage). This prevents stale data from being served for unknown IDs.
        explicit_query_id = _query_has_explicit_patient_id(query)
        if explicit_query_id and cached_patient and cached_id:
            if explicit_query_id.upper() != cached_id.upper():
                logger.info(
                    "Orchestrator: query ID '%s' ≠ cached ID '%s' — "
                    "forcing cache invalidation",
                    explicit_query_id, cached_id,
                )
                _invalidate_patient_cache(state)
                incoming_name = explicit_query_id
                state["identified_patient_name"] = incoming_name

        # General knowledge — no tools needed. is_general_knowledge=True is the
        # authoritative signal the Extractor reads for the bypass (not tool_plan=[],
        # which is also set by cache-hit paths and would incorrectly trigger bypass).
        if state["is_general_knowledge"]:
            logger.info("Orchestrator: general knowledge query — tool_plan = []")
            state["tool_plan"] = []
            state["orchestrator_ran"] = True
            return state

        # ── Bug 2: Source availability gate (Layer 1) ─────────────────────
        # If the query requires a PDF/clinical note/imaging source but nothing
        # is attached, short-circuit immediately. The Output Node Gate 1 will
        # return "I don't have access to..." rather than fabricating an absence.
        pdf_required = intent.get("pdf_required", False)
        pdf_attached = bool((state.get("pdf_source_file") or "").strip())
        if pdf_required and not pdf_attached:
            logger.info(
                "Orchestrator: pdf_required=True for source '%s' but no PDF attached — "
                "setting source_unavailable",
                state["data_source_required"],
            )
            state["tool_plan"] = []
            state["orchestrator_ran"] = True
            state["source_unavailable"] = True
            state["source_unavailable_reason"] = state["data_source_required"]
            return state

        tool_plan = []

        # ── PDF UPLOAD: 4-Phase Protocol enforcement ───────────────────────
        # When a PDF is attached, the strict 4-phase protocol requires the full
        # Phase 1 + Phase 2 tool suite to run unconditionally, regardless of
        # what Haiku classified.  This overrides the per-intent checks below
        # and guarantees patient safety (allergy/interaction) and payer compliance
        # (denial risk) are never skipped because of a narrow Haiku classification.
        pdf_attached = bool((state.get("pdf_source_file") or "").strip())
        pdf_cache_hit = False  # will be set below if PDF bytes are unchanged

        if pdf_attached:
            logger.info(
                "Orchestrator: PDF attached — enforcing full Phase 1+2 tool plan."
            )

            # Phase 1a: patient_lookup + med_retrieval (EHR baseline).
            # Skip patient_lookup if cache is populated (same-patient guard already ran).
            if not state.get("extracted_patient"):
                tool_plan.append("patient_lookup")
                tool_plan.append("med_retrieval")
            else:
                logger.info(
                    "Orchestrator (PDF): patient cache hit — skipping patient_lookup+med_retrieval."
                )

            # Phase 1b: pdf_extractor — skip only on exact content hash match.
            pdf_path = state["pdf_source_file"]
            cached_hash = state.get("extracted_pdf_hash", "")
            if cached_hash and state.get("extracted_pdf_pages"):
                try:
                    import hashlib
                    current_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
                    if current_hash == cached_hash:
                        logger.info("Orchestrator (PDF): PDF cache hit — skipping pdf_extractor.")
                        pdf_cache_hit = True
                    else:
                        logger.info("Orchestrator (PDF): PDF hash mismatch — adding pdf_extractor.")
                        tool_plan.append("pdf_extractor")
                except Exception:
                    tool_plan.append("pdf_extractor")
            else:
                tool_plan.append("pdf_extractor")

            # Phase 2: policy_search — skip only if this payer is already cached.
            payer_id = state.get("payer_id", "")
            if payer_id and payer_id in (state.get("payer_policy_cache") or {}):
                logger.info(
                    "Orchestrator (PDF): policy cache hit for payer %s — skipping policy_search.",
                    payer_id,
                )
            else:
                tool_plan.append("policy_search")

            # Phase 2: denial_analyzer — always run for PDF uploads.
            # The denial_analyzer surfaces MISSING documentation gaps (ECOG, labs)
            # that cause RCM rejections and MUST appear in the reconciliation summary.
            payer_id = state.get("payer_id", "")
            cpt = state.get("procedure_code", "")
            cache_key = f"{payer_id}:{cpt}"
            if cache_key in (state.get("denial_risk_cache") or {}):
                logger.info(
                    "Orchestrator (PDF): denial risk cache hit — skipping denial_analyzer."
                )
            else:
                tool_plan.append("denial_analyzer")

        else:
            # ── Non-PDF path: standard per-intent tool selection ──────────────
            # Patient data — skip if already cached (and same patient, validated above)
            if intent.get("needs_specific_patient"):
                if not state.get("extracted_patient"):
                    tool_plan.append("patient_lookup")
                    tool_plan.append("med_retrieval")
                else:
                    logger.info("Orchestrator: patient cache hit — skipping patient_lookup + med_retrieval")

            # PDF evidence — skip only if hash matches current document bytes
            if intent.get("needs_document_evidence") and state.get("pdf_source_file"):
                pdf_path = state["pdf_source_file"]
                cached_hash = state.get("extracted_pdf_hash", "")
                if cached_hash and state.get("extracted_pdf_pages"):
                    try:
                        import hashlib
                        current_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
                        if current_hash == cached_hash:
                            logger.info("Orchestrator: PDF cache hit — skipping pdf_extractor")
                        else:
                            logger.info("Orchestrator: PDF hash mismatch — adding pdf_extractor")
                            tool_plan.append("pdf_extractor")
                    except Exception:
                        tool_plan.append("pdf_extractor")
                else:
                    tool_plan.append("pdf_extractor")

            # Policy check — skip if this payer is already cached
            if intent.get("needs_policy_check"):
                payer_id = state.get("payer_id", "")
                if payer_id and payer_id in (state.get("payer_policy_cache") or {}):
                    logger.info("Orchestrator: policy cache hit for payer %s — skipping policy_search", payer_id)
                else:
                    tool_plan.append("policy_search")

            # Denial analysis — only useful when audit_results exist; skip if cached
            if intent.get("needs_denial_analysis") and state.get("audit_results"):
                payer_id = state.get("payer_id", "")
                cpt = state.get("procedure_code", "")
                cache_key = f"{payer_id}:{cpt}"
                if cache_key in (state.get("denial_risk_cache") or {}):
                    logger.info("Orchestrator: denial risk cache hit — skipping denial_analyzer")
                else:
                    tool_plan.append("denial_analyzer")

        # Reset source_unavailable — this query has an available source.
        state["source_unavailable"] = False
        state["source_unavailable_reason"] = ""

        logger.info("Orchestrator tool_plan: %s", tool_plan)
        state["tool_plan"] = tool_plan
        state["orchestrator_ran"] = True
        return state

    except Exception as e:
        # All retries exhausted (529) or unexpected error — invoke regex fallback
        # rather than returning 0% confidence.
        logger.error("Orchestrator node failed after retries: %s — invoking fallback", e)
        return _orchestrator_fallback(state, query)
