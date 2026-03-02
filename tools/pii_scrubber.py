"""
pii_scrubber.py
---------------
AgentForge — Healthcare RCM AI Agent — PII Scrubber
----------------------------------------------------
Presidio-based PII scrubber.  Runs locally — no data leaves this host.

Detected entity types (HIPAA-relevant):
    PERSON          — patient names, provider names
    DATE_TIME       — dates of birth, appointment dates
    PHONE_NUMBER    — US phone numbers
    EMAIL_ADDRESS   — email addresses
    US_SSN          — Social Security Numbers
    MEDICAL_LICENSE — MRN-style identifiers (Presidio built-in)
    ACCOUNT_NUMBER  — ACC-YYYY-NNNNN format (custom recognizer)

All entities are replaced with a typed placeholder, e.g. <PERSON>, <DATE_TIME>,
so downstream audit logs are PII-free while preserving clinical meaning.

Graceful fallback: if Presidio is unavailable (import error), falls back to the
five-regex stub from the original implementation — the pipeline never hard-crashes.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import logging
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Presidio import with graceful fallback ────────────────────────────────────

try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False
    logger.warning(
        "presidio-analyzer/presidio-anonymizer not installed — "
        "falling back to regex PII stub.  Install with: "
        "pip install presidio-analyzer presidio-anonymizer && "
        "python -m spacy download en_core_web_lg"
    )

# ── Fallback regex patterns (used when Presidio is unavailable) ───────────────

_FALLBACK_SSN_PATTERN     = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_FALLBACK_MRN_PATTERN     = re.compile(r"\bMRN[:\s]*\w+\b", re.IGNORECASE)
_FALLBACK_DOB_PATTERN     = re.compile(r"\b(DOB|Date of Birth)[:\s]*[\d/\-]+\b", re.IGNORECASE)
_FALLBACK_PHONE_PATTERN   = re.compile(r"\b\d{3}[.\-]\d{3}[.\-]\d{4}\b")
_FALLBACK_EMAIL_PATTERN   = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")


def _fallback_scrub(text: str) -> str:
    """Five-regex stub — used only when Presidio is not installed."""
    try:
        text = _FALLBACK_SSN_PATTERN.sub("[REDACTED-SSN]", text)
        text = _FALLBACK_MRN_PATTERN.sub("[REDACTED-MRN]", text)
        text = _FALLBACK_DOB_PATTERN.sub("[REDACTED-DOB]", text)
        text = _FALLBACK_PHONE_PATTERN.sub("[REDACTED-PHONE]", text)
        text = _FALLBACK_EMAIL_PATTERN.sub("[REDACTED-EMAIL]", text)
        return text
    except Exception:
        return text


# ── Presidio engine initialisation (module-level singleton) ───────────────────
#
# AnalyzerEngine is expensive to construct (~1 s for model load).
# Built once at module import and reused across all calls.

_analyzer: Optional["AnalyzerEngine"] = None
_anonymizer: Optional["AnonymizerEngine"] = None


def _build_engines() -> None:
    """
    Initialise the Presidio AnalyzerEngine and AnonymizerEngine.

    Custom recognizer registered:
        ACCOUNT_NUMBER — matches ACC-YYYY-NNNNN (e.g. ACC-2026-01145).
        Presidio has no built-in recognizer for this format.

    Entities enabled (subset of all Presidio built-ins relevant to HIPAA):
        PERSON, DATE_TIME, PHONE_NUMBER, EMAIL_ADDRESS, US_SSN,
        MEDICAL_LICENSE, ACCOUNT_NUMBER.

    Raises:
        Exception: propagated to caller; _PRESIDIO_AVAILABLE stays True so the
                   next call retries.  On repeated failure the caller should fall
                   back to the regex stub.
    """
    global _analyzer, _anonymizer

    # spaCy large model provides better name recall than the small model.
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # Custom recognizer for AgentForge account number format: ACC-YYYY-NNNNN
    acc_pattern = Pattern(
        name="acc_account_number",
        regex=r"\bACC-\d{4}-\d{5}\b",
        score=0.95,
    )
    acc_recognizer = PatternRecognizer(
        supported_entity="ACCOUNT_NUMBER",
        patterns=[acc_pattern],
        name="AccAccountNumberRecognizer",
    )

    # Custom recognizer for MRN patterns (Presidio's MEDICAL_LICENSE does not
    # cover the "MRN: XXXXX" prefix format common in clinical documents).
    mrn_pattern = Pattern(
        name="mrn_pattern",
        regex=r"\bMRN[:\s]+[\w\-]+\b",
        score=0.90,
    )
    mrn_recognizer = PatternRecognizer(
        supported_entity="MEDICAL_LICENSE",
        patterns=[mrn_pattern],
        name="MrnRecognizer",
    )

    # Custom DOB recognizer — matches only DOB-prefixed date strings so that
    # clinical frequency terms ("daily", "twice weekly") are never scrubbed.
    dob_pattern = Pattern(
        name="dob_date",
        regex=r"\b(?:DOB|Date of Birth)[:\s]+[\d]{1,2}[/\-][\d]{1,2}[/\-][\d]{2,4}\b",
        score=0.95,
    )
    # Also match ISO dates immediately after DOB/Date of Birth prefix.
    dob_iso_pattern = Pattern(
        name="dob_date_iso",
        regex=r"\b(?:DOB|Date of Birth)[:\s]+\d{4}-\d{2}-\d{2}\b",
        score=0.95,
    )
    dob_recognizer = PatternRecognizer(
        supported_entity="DOB_DATE",
        patterns=[dob_pattern, dob_iso_pattern],
        name="DobDateRecognizer",
    )

    # SSN fallback — Presidio's US_SSN uses NLP context scoring which can
    # miss bare SSN patterns without surrounding context words.  This pattern
    # recognizer fires at high confidence on the bare NNN-NN-NNNN format.
    ssn_pattern = Pattern(
        name="ssn_bare",
        regex=r"\b\d{3}-\d{2}-\d{4}\b",
        score=0.85,
    )
    ssn_recognizer = PatternRecognizer(
        supported_entity="US_SSN",
        patterns=[ssn_pattern],
        name="SsnBareRecognizer",
    )

    _analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    _analyzer.registry.add_recognizer(acc_recognizer)
    _analyzer.registry.add_recognizer(mrn_recognizer)
    _analyzer.registry.add_recognizer(dob_recognizer)
    _analyzer.registry.add_recognizer(ssn_recognizer)

    _anonymizer = AnonymizerEngine()
    logger.info("PII scrubber: Presidio engines initialised (spacy en_core_web_lg).")


def _get_engines() -> Tuple[Optional["AnalyzerEngine"], Optional["AnonymizerEngine"]]:
    """Return (analyzer, anonymizer), initialising on first call."""
    global _analyzer, _anonymizer
    if _analyzer is None or _anonymizer is None:
        _build_engines()
    return _analyzer, _anonymizer


# ── Public API ────────────────────────────────────────────────────────────────

#: Entity types that will be detected and replaced.
#
# DATE_TIME is intentionally excluded: Presidio's DATE_TIME entity matches
# clinical frequency words ("daily", "twice weekly", "every 8 hours") which
# are not PHI and must not be scrubbed from medication or treatment text.
# Date of birth is covered by the custom DOB_DATE recognizer below, which
# only fires on "DOB:" / "Date of Birth:" prefixed patterns.
_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "MEDICAL_LICENSE",   # covers MRN patterns via custom MrnRecognizer
    "ACCOUNT_NUMBER",    # ACC-YYYY-NNNNN via custom AccAccountNumberRecognizer
    "DOB_DATE",          # DOB-prefixed dates via custom DobDateRecognizer
]

#: Replacement template — typed placeholders preserve clinical context.
def _operator_config() -> Dict[str, "OperatorConfig"]:
    return {
        entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
        for entity in _ENTITIES
    }


def scrub_pii(text: str) -> str:
    """
    Remove HIPAA-defined PII from text using Presidio (NLP-based detection).

    Detects and replaces: names, dates, phone numbers, email addresses, SSNs,
    MRN-style medical licence numbers, and ACC-YYYY-NNNNN account numbers.

    Falls back to the regex stub if Presidio is unavailable or raises.

    Args:
        text: Raw input text that may contain PII.

    Returns:
        str: Text with PII replaced by typed placeholders (e.g. <PERSON>).

    Raises:
        Never — returns original text unchanged on any failure.
    """
    if not text or not text.strip():
        return text

    if not _PRESIDIO_AVAILABLE:
        return _fallback_scrub(text)

    try:
        analyzer, anonymizer = _get_engines()
        if analyzer is None or anonymizer is None:
            return _fallback_scrub(text)

        results = analyzer.analyze(
            text=text,
            entities=_ENTITIES,
            language="en",
        )

        if not results:
            return text

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=_operator_config(),
        )
        return anonymized.text

    except Exception as exc:
        logger.warning("PII scrubber: Presidio error — falling back to regex stub. %s", exc)
        return _fallback_scrub(text)


def scrub_pii_with_map(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Remove PII and return both the scrubbed text and a replacement map.

    The replacement map is required for audit trail compliance (HIPAA audit
    logs must be retained 7 years). It records what was replaced and with
    what placeholder, keyed by the original PII string.

    Args:
        text: Raw input text that may contain PII.

    Returns:
        Tuple[str, Dict[str, str]]:
            - Scrubbed text with PII replaced by typed placeholders.
            - Dict mapping original PII value → placeholder used.
              Empty dict if no PII was detected or on fallback.

    Raises:
        Never — returns (original text, {}) on any failure.
    """
    if not text or not text.strip():
        return text, {}

    if not _PRESIDIO_AVAILABLE:
        return _fallback_scrub(text), {}

    try:
        analyzer, anonymizer = _get_engines()
        if analyzer is None or anonymizer is None:
            return _fallback_scrub(text), {}

        results = analyzer.analyze(
            text=text,
            entities=_ENTITIES,
            language="en",
        )

        if not results:
            return text, {}

        replacement_map: Dict[str, str] = {}
        for result in results:
            original_span = text[result.start:result.end]
            placeholder = f"<{result.entity_type}>"
            replacement_map[original_span] = placeholder

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=_operator_config(),
        )
        return anonymized.text, replacement_map

    except Exception as exc:
        logger.warning(
            "PII scrubber (with_map): Presidio error — falling back to regex stub. %s", exc
        )
        return _fallback_scrub(text), {}
