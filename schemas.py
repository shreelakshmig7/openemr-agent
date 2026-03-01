"""
schemas.py
----------
AgentForge — Healthcare RCM AI Agent — Pydantic Data Contracts
--------------------------------------------------------------
Pydantic v2 models that act as the data contract between the clinical
extraction layer and the evidence_staging SQLite database.

Validation policy
-----------------
ClinicalObservation is the single gate between raw regex extraction output
and the database.  A finding that fails validation is logged and silently
dropped — it never reaches SQLite (and therefore never reaches the FHIR sync
worker).  This ensures:

  1. fact_type integrity — only markers with a mapped LOINC code in
     fhir_mapper.LOINC_REGISTRY (or its _ALIASES) can enter the pipeline.
     Markers like "Ki-67", "BRCA1", "EGFR" that have no LOINC entry are
     rejected at the Extractor Node boundary.

  2. fact_value cleanliness — the extracted value string is stripped of
     leading/trailing whitespace, ASCII control characters, and capped at
     500 characters (matching the evidence_staging DB column limit).

  3. raw_text integrity — same sanitisation as fact_value; guaranteed to
     be a non-None, non-empty string before insertion.

  4. confidence bounds — clamped to [0.0, 1.0] regardless of extractor
     output; never raises a ValueError for out-of-range floats.

  5. Derived loinc_code — the resolved LOINC code string is stored on the
     model instance after validation so downstream consumers (graph.py,
     fhir_mapper.py) can read it without a second lookup.

Public API
----------
    ClinicalObservation     Pydantic BaseModel — the validated staging record.
    ValidationSummary       Lightweight counter returned by validate_batch().
    validate_batch()        Validate a list of raw fact dicts; return
                            (valid_observations, summary).

Usage (Extractor Node)::

    from schemas import ClinicalObservation
    from pydantic import ValidationError

    try:
        obs = ClinicalObservation(
            fact_type=marker_name,     # validated against LOINC registry
            fact_value=marker_value,   # cleaned string
            raw_text=text,
            session_id=session_id,
            patient_id=patient_id,
            source_file=source_file,
            page_number=page_number,
            element_type=element_type,
            confidence=1.0,
        )
    except ValidationError as exc:
        logger.warning("Rejected: %s", exc.errors()[0]["msg"])
        continue   # skip this finding — do NOT insert into SQLite

    _db.insert_clinical_marker(
        marker_name=obs.fact_type,
        raw_text=obs.raw_text,
        marker_value=obs.fact_value,
        session_id=obs.session_id,
        patient_id=obs.patient_id,
        source_file=obs.source_file,
        page_number=obs.page_number,
        element_type=obs.element_type,
        confidence=obs.confidence,
    )

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import guard — fhir_mapper is a sibling module; import at class level
# rather than module level to avoid circular imports if schemas is loaded
# early in the startup chain.
# ---------------------------------------------------------------------------

def _get_loinc_code(fact_type: str) -> Optional[str]:
    """Thin wrapper around fhir_mapper.get_loinc_code() with lazy import."""
    try:
        from fhir_mapper import get_loinc_code  # noqa: PLC0415 — intentional lazy import
        return get_loinc_code(fact_type)
    except ImportError as exc:
        logger.error("schemas: cannot import fhir_mapper — LOINC validation disabled: %s", exc)
        return None  # fail-open so schema doesn't block startup on import errors


# ---------------------------------------------------------------------------
# Sanitisation helpers
# ---------------------------------------------------------------------------

# ASCII control characters to strip: \x00–\x08, \x0b–\x0c, \x0e–\x1f, \x7f
# Preserved: \x09 (tab), \x0a (newline), \x0d (carriage-return).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Maximum character length that matches the evidence_staging column limit.
_MAX_STR_LEN = 500


def _sanitise_string(value: Any, *, max_len: int = _MAX_STR_LEN) -> str:
    """
    Coerce *value* to str, strip control characters and whitespace, and
    truncate to *max_len* characters.

    Args:
        value:   Raw value — may be None, int, float, or str.
        max_len: Maximum character length after sanitisation.

    Returns:
        Sanitised, stripped, truncated string.  Never raises.
    """
    if value is None:
        return ""
    cleaned = _CONTROL_CHAR_RE.sub("", str(value))
    return cleaned.strip()[:max_len]


# ---------------------------------------------------------------------------
# ClinicalObservation — the validated staging record
# ---------------------------------------------------------------------------

class ClinicalObservation(BaseModel):
    """
    Validated representation of a single clinical finding extracted from a PDF.

    All instances that pass validation are safe to insert into evidence_staging
    via ``database.insert_clinical_marker()``.  Validation raises
    ``pydantic.ValidationError`` for findings that should be rejected.

    Fields
    ------
    fact_type:    Marker name validated against the LOINC registry (and its
                  aliases).  Maps to ``evidence_staging.marker_name``.
    fact_value:   Extracted result string — stripped, de-controlled,
                  truncated.  Maps to ``evidence_staging.marker_value``.
                  May be empty string when no result value was found.
    loinc_code:   LOINC code resolved from *fact_type* (derived, read-only).
    raw_text:     Verbatim text excerpt containing the marker.  Maps to
                  ``evidence_staging.raw_text``.
    session_id:   LangGraph session / thread ID.
    patient_id:   EHR patient ID (``"P001"`` …) or empty string.
    source_file:  Source PDF file path.
    page_number:  Page in the PDF where the marker was found.
    element_type: Unstructured element type (e.g. ``"NarrativeText"``).
    confidence:   Regex-match confidence, clamped to [0.0, 1.0].
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,   # auto-strip str fields before validators
        frozen=False,                # allow model_validator(mode="after") to write loinc_code
        validate_assignment=False,   # no re-validation on attribute assignment
        extra="ignore",              # silently drop unexpected fields from callers
    )

    # ── Core validated fields ────────────────────────────────────────────────
    fact_type:    str
    fact_value:   str = ""
    raw_text:     str

    # ── Derived — populated by model_validator, not by caller ────────────────
    loinc_code:   str = Field(default="", repr=True)

    # ── Provenance / context ─────────────────────────────────────────────────
    session_id:   str            = ""
    patient_id:   str            = ""
    source_file:  str            = ""
    page_number:  Optional[int]  = None
    element_type: str            = ""
    confidence:   float          = Field(default=1.0, ge=0.0, le=1.0)

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("fact_type", mode="before")
    @classmethod
    def validate_fact_type(cls, v: Any) -> str:
        """
        Reject *fact_type* values that have no LOINC mapping.

        Delegates to ``fhir_mapper.get_loinc_code()`` which covers both the
        18 canonical LOINC_REGISTRY keys and every entry in ``_ALIASES``
        (e.g. ``"HER2"`` resolves to ``"HER2 Status"`` → LOINC 85337-4).

        Raises:
            ValueError: when the marker name is unknown or empty.
        """
        raw = str(v).strip() if v is not None else ""
        if not raw:
            raise ValueError("fact_type must not be empty.")

        loinc = _get_loinc_code(raw)
        if not loinc:
            raise ValueError(
                f"'{raw}' is not registered in the LOINC registry "
                "(fhir_mapper.LOINC_REGISTRY / _ALIASES). "
                "Add a LOINC mapping before inserting into evidence_staging."
            )
        return raw

    @field_validator("fact_value", mode="before")
    @classmethod
    def clean_fact_value(cls, v: Any) -> str:
        """
        Coerce and sanitise the extracted result value.

        Strips ASCII control characters, surrounding whitespace, and truncates
        to 500 characters.  Never raises — empty string is a valid state when
        no result value was found in the source text.
        """
        return _sanitise_string(v)

    @field_validator("raw_text", mode="before")
    @classmethod
    def clean_raw_text(cls, v: Any) -> str:
        """
        Ensure raw_text is a sanitised, non-None string.

        Raises:
            ValueError: if the result after sanitisation is empty (raw_text is
                        required — a finding without a verbatim excerpt has no
                        provenance and must not be staged).
        """
        cleaned = _sanitise_string(v)
        if not cleaned:
            raise ValueError(
                "raw_text must not be empty — every staged finding requires "
                "a verbatim source excerpt for provenance."
            )
        return cleaned

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        """
        Coerce confidence to float and clamp to [0.0, 1.0].

        Never raises — out-of-range values are clamped silently.
        """
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 1.0

    @field_validator("page_number", mode="before")
    @classmethod
    def coerce_page_number(cls, v: Any) -> Optional[int]:
        """Accept int, str-digits, or None; reject anything else."""
        if v is None:
            return None
        try:
            n = int(v)
            return n if n >= 0 else None
        except (TypeError, ValueError):
            return None

    @model_validator(mode="after")
    def derive_loinc_code(self) -> "ClinicalObservation":
        """Populate loinc_code from the already-validated fact_type."""
        self.loinc_code = _get_loinc_code(self.fact_type) or ""
        return self

    # ── Convenience method ───────────────────────────────────────────────────

    def to_db_kwargs(self) -> Dict[str, Any]:
        """
        Return a dict of keyword arguments ready to pass to
        ``database.insert_clinical_marker()``.

        Example::

            obs = ClinicalObservation(fact_type="HER2", ...)
            row_id = db.insert_clinical_marker(**obs.to_db_kwargs())
        """
        return {
            "marker_name":  self.fact_type,
            "raw_text":     self.raw_text,
            "marker_value": self.fact_value,
            "session_id":   self.session_id,
            "patient_id":   self.patient_id,
            "source_file":  self.source_file,
            "page_number":  self.page_number,
            "element_type": self.element_type,
            "confidence":   self.confidence,
        }


# ---------------------------------------------------------------------------
# ValidationSummary
# ---------------------------------------------------------------------------

class ValidationSummary(BaseModel):
    """
    Lightweight counter returned by ``validate_batch()``.

    Attributes:
        total:    Total number of raw fact dicts attempted.
        accepted: Facts that passed validation → inserted into SQLite.
        rejected: Facts that failed validation → logged and dropped.
        rejection_reasons: Mapping of fact_type → first error message for
            each rejected fact.  Useful for monitoring.
    """
    model_config = ConfigDict(frozen=True)

    total:            int
    accepted:         int
    rejected:         int
    rejection_reasons: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def validate_batch(
    raw_facts: List[Dict[str, Any]],
) -> Tuple[List[ClinicalObservation], ValidationSummary]:
    """
    Validate a list of raw fact dicts and return accepted observations with
    a summary of any rejections.

    Invalid facts are logged at WARNING level and excluded from the returned
    list.  Valid observations are returned in the same order as the input.

    Args:
        raw_facts: List of dicts with keys matching ClinicalObservation fields.
                   ``fact_type`` and ``raw_text`` are the only required keys.
                   Additional keys are silently ignored.

    Returns:
        Tuple of:
          * ``List[ClinicalObservation]``   — validated observations, safe for DB insert.
          * ``ValidationSummary``           — counts and rejection reasons.

    Example::

        from schemas import validate_batch

        raw = [
            {"fact_type": "HER2", "fact_value": "positive", "raw_text": "HER2 IHC 3+"},
            {"fact_type": "Ki-67", "fact_value": "30%",     "raw_text": "Ki-67 30%"},  # rejected
        ]
        obs_list, summary = validate_batch(raw)
        # obs_list has 1 entry; summary.rejected == 1
    """
    from pydantic import ValidationError

    accepted: List[ClinicalObservation] = []
    rejection_reasons: Dict[str, str]   = {}

    for raw in raw_facts:
        fact_type_raw = str(raw.get("fact_type", raw.get("marker_name", "?"))).strip()
        try:
            # Support both "fact_type" (schema API) and "marker_name" (DB API).
            normalised = {
                "fact_type":    raw.get("fact_type") or raw.get("marker_name", ""),
                "fact_value":   raw.get("fact_value") or raw.get("marker_value", ""),
                "raw_text":     raw.get("raw_text", ""),
                "session_id":   raw.get("session_id", ""),
                "patient_id":   raw.get("patient_id", ""),
                "source_file":  raw.get("source_file", ""),
                "page_number":  raw.get("page_number"),
                "element_type": raw.get("element_type", ""),
                "confidence":   raw.get("confidence", 1.0),
            }
            obs = ClinicalObservation(**normalised)
            accepted.append(obs)
        except ValidationError as exc:
            first_msg = exc.errors()[0]["msg"] if exc.errors() else str(exc)
            rejection_reasons[fact_type_raw] = first_msg
            logger.warning(
                "schemas.validate_batch: rejected '%s' — %s",
                fact_type_raw, first_msg,
            )

    summary = ValidationSummary(
        total=len(raw_facts),
        accepted=len(accepted),
        rejected=len(raw_facts) - len(accepted),
        rejection_reasons=rejection_reasons,
    )
    return accepted, summary
