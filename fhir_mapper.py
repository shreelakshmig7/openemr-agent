"""
fhir_mapper.py
--------------
AgentForge — Healthcare RCM AI Agent — FHIR R4 Mapping Layer
-------------------------------------------------------------
Translates extracted clinical facts (from evidence_staging / extractor_node)
into a FHIR R4 Transaction Bundle ready to be POSTed to OpenEMR.

Two FHIR resource types are produced:
  • Observation        — for all lab results, vitals, clinical notes, and
                         social-history facts.
  • AllergyIntolerance — specifically for Drug Allergy facts (LOINC 48765-2).

LOINC codes follow the project .cursorrules standard (LOINC for all clinical
coding).  UCUM unit codes are used for vital-sign Observations.  SNOMED CT
value codings are applied when the extracted value string is a recognised
clinical status term (positive / negative / detected / not detected).

Public API:
    LOINC_REGISTRY      — dict mapping 18 marker names → LOINC + metadata.
    map_to_bundle()     — Convert a patient_id + list of fact dicts into a
                          FHIR R4 Transaction Bundle (dict, ready to json.dumps).

Fact dict shape (compatible with evidence_staging rows):
    {
        "marker_name":  str,   # e.g. "HER2 Status", "HER2", "Drug Allergy"
        "marker_value": str,   # e.g. "positive", "3+", "penicillin", "98.6"
        "raw_text":     str,   # verbatim excerpt (stored in Observation.note)
        # optional:
        "session_id":   str,
        "source_file":  str,
        "confidence":   float,
    }

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LOINC Registry — 18 clinical codes
# ---------------------------------------------------------------------------
# Each entry carries:
#   code            LOINC code string
#   display         Human-readable LOINC display name
#   resource_type   "Observation" or "AllergyIntolerance"
#   category_code   FHIR observation-category code (unused for AllergyIntolerance)
#   category_display Human label for the category
#   unit            None, or {"ucum": str, "system": str, "display": str}
#                   — present only for vital-sign Observations with a known unit

_LoincEntry = Dict[str, Any]   # internal type alias

LOINC_REGISTRY: Dict[str, _LoincEntry] = {

    # ── Physical Therapy ─────────────────────────────────────────────────────
    "PT Duration": {
        "code":             "61473-5",
        "display":          "Physical therapy episode of care [Duration]",
        "resource_type":    "Observation",
        "category_code":    "therapy",
        "category_display": "Therapy",
        "unit":             None,
    },
    "PT Note": {
        "code":             "11508-9",
        "display":          "Physical therapy Consult note",
        "resource_type":    "Observation",
        "category_code":    "clinical-note",
        "category_display": "Clinical Note",
        "unit":             None,
    },

    # ── Breast Cancer Receptor Panels ─────────────────────────────────────
    "ER Status": {
        "code":             "85336-6",
        "display":          "Estrogen receptor Ag [Presence] in Breast cancer specimen by Immune stain",
        "resource_type":    "Observation",
        "category_code":    "laboratory",
        "category_display": "Laboratory",
        "unit":             None,
    },
    "PR Status": {
        "code":             "85339-0",
        "display":          "Progesterone receptor Ag [Presence] in Breast cancer specimen by Immune stain",
        "resource_type":    "Observation",
        "category_code":    "laboratory",
        "category_display": "Laboratory",
        "unit":             None,
    },
    "HER2 Status": {
        "code":             "85337-4",
        "display":          "HER2 [Presence] in Breast cancer specimen by Immune stain",
        "resource_type":    "Observation",
        "category_code":    "laboratory",
        "category_display": "Laboratory",
        "unit":             None,
    },

    # ── Pathology ────────────────────────────────────────────────────────────
    "Biopsy Report": {
        "code":             "11526-1",
        "display":          "Pathology study observation",
        "resource_type":    "Observation",
        "category_code":    "laboratory",
        "category_display": "Laboratory",
        "unit":             None,
    },

    # ── Allergy (produces AllergyIntolerance, not Observation) ───────────────
    "Drug Allergy": {
        "code":             "48765-2",
        "display":          "Allergies and adverse drug reactions",
        "resource_type":    "AllergyIntolerance",
        "category_code":    "medication",
        "category_display": "Medication",
        "unit":             None,
    },

    # ── Clinical History ────────────────────────────────────────────────────
    "Med History": {
        "code":             "10160-0",
        "display":          "History of Medication use Narrative",
        "resource_type":    "Observation",
        "category_code":    "social-history",
        "category_display": "Social History",
        "unit":             None,
    },
    "Problem List": {
        "code":             "11450-4",
        "display":          "Problem list - Reported",
        "resource_type":    "Observation",
        "category_code":    "clinical-note",
        "category_display": "Clinical Note",
        "unit":             None,
    },

    # ── Vital Signs ─────────────────────────────────────────────────────────
    "Temp": {
        "code":             "8310-5",
        "display":          "Body temperature",
        "resource_type":    "Observation",
        "category_code":    "vital-signs",
        "category_display": "Vital Signs",
        "unit": {
            "ucum":    "Cel",
            "system":  "http://unitsofmeasure.org",
            "display": "°C",
        },
    },
    "Heart Rate": {
        "code":             "8867-4",
        "display":          "Heart rate",
        "resource_type":    "Observation",
        "category_code":    "vital-signs",
        "category_display": "Vital Signs",
        "unit": {
            "ucum":    "/min",
            "system":  "http://unitsofmeasure.org",
            "display": "beats/min",
        },
    },
    "BP Sys": {
        "code":             "8480-6",
        "display":          "Systolic blood pressure",
        "resource_type":    "Observation",
        "category_code":    "vital-signs",
        "category_display": "Vital Signs",
        "unit": {
            "ucum":    "mm[Hg]",
            "system":  "http://unitsofmeasure.org",
            "display": "mmHg",
        },
    },
    "BP Dia": {
        "code":             "8462-4",
        "display":          "Diastolic blood pressure",
        "resource_type":    "Observation",
        "category_code":    "vital-signs",
        "category_display": "Vital Signs",
        "unit": {
            "ucum":    "mm[Hg]",
            "system":  "http://unitsofmeasure.org",
            "display": "mmHg",
        },
    },
    "O2 Sat": {
        "code":             "2708-6",
        "display":          "Oxygen saturation in Arterial blood",
        "resource_type":    "Observation",
        "category_code":    "vital-signs",
        "category_display": "Vital Signs",
        "unit": {
            "ucum":    "%",
            "system":  "http://unitsofmeasure.org",
            "display": "%",
        },
    },

    # ── Social / Lifestyle ───────────────────────────────────────────────────
    "Tobacco": {
        "code":             "72166-2",
        "display":          "Tobacco smoking status",
        "resource_type":    "Observation",
        "category_code":    "social-history",
        "category_display": "Social History",
        "unit":             None,
    },
    "Social History": {
        "code":             "29762-2",
        "display":          "Social history Narrative",
        "resource_type":    "Observation",
        "category_code":    "social-history",
        "category_display": "Social History",
        "unit":             None,
    },

    # ── Encounter / Surgical Notes ───────────────────────────────────────────
    "Encounter": {
        "code":             "46240-8",
        "display":          "History of Hospitalizations+Outpatient visits Narrative",
        "resource_type":    "Observation",
        "category_code":    "clinical-note",
        "category_display": "Clinical Note",
        "unit":             None,
    },
    "Surgical Note": {
        "code":             "11504-8",
        "display":          "Surgical operation note",
        "resource_type":    "Observation",
        "category_code":    "clinical-note",
        "category_display": "Clinical Note",
        "unit":             None,
    },
}

# ---------------------------------------------------------------------------
# Short-name aliases (extractor_node.py uses abbreviated marker_names)
# Maps extracted short names → canonical LOINC_REGISTRY keys.
# ---------------------------------------------------------------------------
_ALIASES: Dict[str, str] = {
    # Receptor panel — extractor writes "ER", "PR", "HER2"
    "ER":           "ER Status",
    "PR":           "PR Status",
    "HER2":         "HER2 Status",
    # Vital sign shortcuts
    "TEMPERATURE":  "Temp",
    "HEARTRATE":    "Heart Rate",
    "HEART_RATE":   "Heart Rate",
    "BPS":          "BP Sys",
    "BPD":          "BP Dia",
    "O2SAT":        "O2 Sat",
    "SPO2":         "O2 Sat",
    # Social / clinical shortcuts
    "TOBACCO":      "Tobacco",
    "SMOKING":      "Tobacco",
    # Allergy shortcuts
    "ALLERGY":      "Drug Allergy",
    "DRUG_ALLERGY": "Drug Allergy",
}

# ---------------------------------------------------------------------------
# SNOMED CT codings for common clinical status values
# ---------------------------------------------------------------------------
_SNOMED_STATUS: Dict[str, Dict[str, str]] = {
    "positive":     {"code": "10828004",  "display": "Positive"},
    "negative":     {"code": "260385009", "display": "Negative"},
    "detected":     {"code": "260373001", "display": "Detected"},
    "not detected": {"code": "260415000", "display": "Not detected"},
    "present":      {"code": "52101004",  "display": "Present"},
    "absent":       {"code": "2667000",   "display": "Absent"},
    "high":         {"code": "75540009",  "display": "High"},
    "low":          {"code": "62482003",  "display": "Low"},
    "normal":       {"code": "17621005",  "display": "Normal"},
    "equivocal":    {"code": "42425007",  "display": "Equivocal"},
    "unknown":      {"code": "261665006", "display": "Unknown"},
}

_CATEGORY_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/observation-category"
)
_LOINC_SYSTEM    = "http://loinc.org"
_SNOMED_SYSTEM   = "http://snomed.info/sct"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_marker(raw_name: str) -> Optional[_LoincEntry]:
    """
    Return the LOINC_REGISTRY entry for *raw_name*, tolerating aliases and
    case variations.  Returns ``None`` if no match is found.
    """
    # 1. Exact match
    if raw_name in LOINC_REGISTRY:
        return LOINC_REGISTRY[raw_name]

    # 2. Case-insensitive match against registry keys
    lower = raw_name.strip().lower()
    for key, entry in LOINC_REGISTRY.items():
        if key.lower() == lower:
            return entry

    # 3. Alias lookup (upper-case, strip spaces/hyphens for normalisation)
    normalised = raw_name.upper().replace(" ", "_").replace("-", "_")
    alias_key = _ALIASES.get(normalised) or _ALIASES.get(raw_name.upper())
    if alias_key and alias_key in LOINC_REGISTRY:
        return LOINC_REGISTRY[alias_key]

    return None


def _try_float(value: str) -> Optional[float]:
    """Extract the first numeric token from *value*, or return ``None``."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def _build_value(entry: _LoincEntry, value_str: str) -> Dict[str, Any]:
    """
    Return the appropriate FHIR value[x] field for an Observation.

    Decision tree:
      1. Vital-sign entry with unit defined → try ``valueQuantity``
         (falls back to ``valueString`` if value_str is not numeric).
      2. Value string matches a known clinical status term
         → ``valueCodeableConcept`` with SNOMED CT.
      3. Default → ``valueString``.
    """
    # Numeric vitals
    if entry.get("unit"):
        numeric = _try_float(value_str)
        if numeric is not None:
            unit = entry["unit"]
            return {
                "valueQuantity": {
                    "value":  numeric,
                    "unit":   unit["display"],
                    "system": unit["system"],
                    "code":   unit["ucum"],
                }
            }

    # Clinical status terms
    snomed = _SNOMED_STATUS.get(value_str.strip().lower())
    if snomed:
        return {
            "valueCodeableConcept": {
                "coding": [{
                    "system":  _SNOMED_SYSTEM,
                    "code":    snomed["code"],
                    "display": snomed["display"],
                }],
                "text": value_str,
            }
        }

    # Default: plain string
    return {"valueString": value_str}


def _build_observation(
    patient_id: str,
    entry: _LoincEntry,
    fact: Dict[str, Any],
    effective_dt: str,
) -> Dict[str, Any]:
    """
    Construct a minimal-but-valid FHIR R4 Observation resource.

    Args:
        patient_id:   FHIR Patient resource ID (UUID string).
        entry:        LOINC_REGISTRY entry for the marker.
        fact:         Raw fact dict from evidence_staging or the extractor.
        effective_dt: ISO-8601 UTC datetime string for ``effectiveDateTime``.

    Returns:
        dict: A FHIR R4 Observation resource (no ``id`` — server will assign).
    """
    value_str = str(fact.get("marker_value", "")).strip()
    raw_text  = str(fact.get("raw_text", "")).strip()

    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "status":       "final",
        "category": [{
            "coding": [{
                "system":  _CATEGORY_SYSTEM,
                "code":    entry["category_code"],
                "display": entry["category_display"],
            }]
        }],
        "code": {
            "coding": [{
                "system":  _LOINC_SYSTEM,
                "code":    entry["code"],
                "display": entry["display"],
            }],
            "text": entry["display"],
        },
        "subject":           {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": effective_dt,
    }

    # Attach the extracted value
    if value_str:
        obs.update(_build_value(entry, value_str))

    # Provenance note — verbatim text excerpt from the source PDF
    if raw_text:
        obs["note"] = [{"text": raw_text[:500]}]   # cap at 500 chars (DB limit)

    # Optional: traceability extension back to evidence_staging
    _maybe_add_staging_extension(obs, fact)

    return obs


def _build_allergy_intolerance(
    patient_id: str,
    entry: _LoincEntry,
    fact: Dict[str, Any],
    effective_dt: str,
) -> Dict[str, Any]:
    """
    Construct a minimal-but-valid FHIR R4 AllergyIntolerance resource.

    The LOINC code 48765-2 is stored in ``code`` so that round-tripped data
    can be mapped back to the LOINC_REGISTRY on ingest.  The extracted
    ``marker_value`` (e.g. ``"penicillin"``) is treated as the substance text.

    Args:
        patient_id:   FHIR Patient resource ID.
        entry:        LOINC_REGISTRY entry for "Drug Allergy".
        fact:         Raw fact dict.
        effective_dt: ISO-8601 UTC datetime string.

    Returns:
        dict: A FHIR R4 AllergyIntolerance resource.
    """
    substance_text = str(fact.get("marker_value", "")).strip() or "Unknown"
    raw_text       = str(fact.get("raw_text", "")).strip()

    allergy: Dict[str, Any] = {
        "resourceType": "AllergyIntolerance",
        "clinicalStatus": {
            "coding": [{
                "system":  "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                "code":    "active",
                "display": "Active",
            }]
        },
        "verificationStatus": {
            "coding": [{
                "system":  "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
                "code":    "unconfirmed",
                "display": "Unconfirmed",
            }]
        },
        "type":     "allergy",
        "category": ["medication"],
        # LOINC code on 'code' so the registry mapping is preserved
        "code": {
            "coding": [{
                "system":  _LOINC_SYSTEM,
                "code":    entry["code"],       # 48765-2
                "display": entry["display"],
            }],
            "text": substance_text,
        },
        # FHIR R4 AllergyIntolerance canonical field is "patient"; "subject" is
        # added here as a parallel reference so every bundle entry exposes a
        # uniform "subject" field, matching the FHIR Transaction conventions
        # used by Observation resources and expected by the sync worker.
        "patient":      {"reference": f"Patient/{patient_id}"},
        "subject":      {"reference": f"Patient/{patient_id}"},
        "recordedDate": effective_dt,
        "reaction": [{
            "substance": {
                "coding": [{
                    "system":  "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "display": substance_text,
                }],
                "text": substance_text,
            },
            "manifestation": [{
                "coding": [{
                    "system":  _SNOMED_SYSTEM,
                    "code":    "418038007",
                    "display": "Propensity to adverse reactions to drug",
                }],
                "text": "Adverse drug reaction",
            }],
            "severity": "mild",
        }],
    }

    if raw_text:
        allergy["note"] = [{"text": raw_text[:500]}]

    _maybe_add_staging_extension(allergy, fact)
    return allergy


def _maybe_add_staging_extension(
    resource: Dict[str, Any],
    fact: Dict[str, Any],
) -> None:
    """
    Attach a non-breaking extension that links the FHIR resource back to
    the originating ``evidence_staging`` row / session.

    Only added when ``session_id`` or ``source_file`` are present in *fact*,
    so the mapper stays valid for facts that don't come from the DB.
    """
    ext_base = "http://agentforge.local/fhir/extension"
    extensions: List[Dict[str, Any]] = []

    if fact.get("session_id"):
        extensions.append({
            "url":         f"{ext_base}/evidence-session-id",
            "valueString": str(fact["session_id"]),
        })
    if fact.get("source_file"):
        extensions.append({
            "url":         f"{ext_base}/source-file",
            "valueString": str(fact["source_file"]),
        })
    if fact.get("confidence") is not None:
        extensions.append({
            "url":          f"{ext_base}/extraction-confidence",
            "valueDecimal": float(fact["confidence"]),
        })

    if extensions:
        resource["extension"] = extensions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_to_bundle(
    patient_id: str,
    facts: List[Dict[str, Any]],
    *,
    effective_dt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a list of clinical fact dicts into a FHIR R4 Transaction Bundle.

    Each fact is resolved against ``LOINC_REGISTRY`` (with alias tolerance)
    and converted to either an ``Observation`` or ``AllergyIntolerance``
    resource.  Unrecognised markers are logged and skipped.

    The returned Bundle uses ``type = "transaction"`` so it can be sent in a
    single ``POST`` to ``/fhir/$transaction`` (batch endpoint).

    Args:
        patient_id:    FHIR Patient resource ID (UUID, e.g.
                       ``"a1312c03-cd3f-44b5-9d5f-1ef5751a7550"``).
        facts:         List of fact dicts.  Each dict must have at minimum
                       ``"marker_name"`` and ``"marker_value"`` keys.
                       Compatible with ``evidence_staging`` row dicts returned
                       by ``database.get_pending_markers()``.
        effective_dt:  ISO-8601 UTC datetime for Observation.effectiveDateTime /
                       AllergyIntolerance.recordedDate.  Defaults to *now*.

    Returns:
        dict: A FHIR R4 Transaction Bundle.  Pass to ``json.dumps()`` or
              ``openemr_client.post_bundle()`` to send to OpenEMR.

    Raises:
        ValueError: if ``patient_id`` is empty.

    Example::

        from database import get_pending_markers
        from fhir_mapper import map_to_bundle

        facts  = get_pending_markers(session_id="my-session")
        bundle = map_to_bundle("a1312c03-...", facts)
        # bundle is ready for json.dumps() or POST to /fhir/$transaction
    """
    if not patient_id:
        raise ValueError("patient_id must not be empty.")

    dt = effective_dt or datetime.now(timezone.utc).isoformat()
    entries: List[Dict[str, Any]] = []
    skipped = 0

    for fact in facts:
        marker_name = str(fact.get("marker_name", "")).strip()
        if not marker_name:
            logger.warning("fhir_mapper: fact missing 'marker_name' — skipped: %s", fact)
            skipped += 1
            continue

        entry = _resolve_marker(marker_name)
        if entry is None:
            logger.warning(
                "fhir_mapper: unknown marker '%s' — not in LOINC_REGISTRY or aliases.",
                marker_name,
            )
            skipped += 1
            continue

        resource_type: str = entry["resource_type"]

        if resource_type == "AllergyIntolerance":
            resource = _build_allergy_intolerance(patient_id, entry, fact, dt)
        else:
            resource = _build_observation(patient_id, entry, fact, dt)

        entries.append({
            "fullUrl":  f"urn:uuid:{uuid.uuid4()}",
            "resource": resource,
            "request": {
                "method": "POST",
                "url":    resource_type,
            },
        })

        logger.debug(
            "fhir_mapper: mapped '%s' → %s (LOINC %s).",
            marker_name,
            resource_type,
            entry["code"],
        )

    bundle: Dict[str, Any] = {
        "resourceType": "Bundle",
        "id":           str(uuid.uuid4()),
        "type":         "transaction",
        "timestamp":    dt,
        "total":        len(entries),
        "entry":        entries,
    }

    logger.info(
        "fhir_mapper: built transaction bundle — %d entries, %d skipped (patient=%s).",
        len(entries),
        skipped,
        patient_id,
    )
    return bundle


# ---------------------------------------------------------------------------
# Module-level helpers exposed for downstream use
# ---------------------------------------------------------------------------

def get_loinc_code(marker_name: str) -> Optional[str]:
    """
    Return the LOINC code string for *marker_name*, or ``None`` if unknown.

    Convenience wrapper around ``_resolve_marker`` for callers that only
    need the code (e.g. the sync worker building individual Observations).

    Example::

        code = get_loinc_code("HER2")   # → "85337-4"
        code = get_loinc_code("Temp")   # → "8310-5"
        code = get_loinc_code("bogus")  # → None
    """
    entry = _resolve_marker(marker_name)
    return entry["code"] if entry else None


# ---------------------------------------------------------------------------
# Verification test — run directly to inspect and validate bundle structure
#
#   python fhir_mapper.py
#
# Prints one representative Observation and one AllergyIntolerance as full
# JSON, then checks all four structural requirements against every entry in
# the bundle and reports pass / fail.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

    _SEP  = "─" * 72
    _TEST_PATIENT = "a1312c03-cd3f-44b5-9d5f-1ef5751a7550"   # John Smith (seeded)

    _SAMPLE_FACTS: List[Dict[str, Any]] = [
        # Full registry-key names (lab panel)
        {"marker_name": "HER2 Status",  "marker_value": "positive",      "raw_text": "HER2 IHC 3+ strongly positive", "session_id": "verify-test", "confidence": 1.0},
        {"marker_name": "ER Status",    "marker_value": "negative",      "raw_text": "ER receptor negative"},
        {"marker_name": "PR Status",    "marker_value": "equivocal",     "raw_text": "PR equivocal Allred score 3"},
        # Extractor short-name alias
        {"marker_name": "HER2",         "marker_value": "3+",            "raw_text": "HER2/neu amplified by FISH", "confidence": 0.92},
        {"marker_name": "ER",           "marker_value": "80%",           "raw_text": "ER positive 80%"},
        # Vital signs → valueQuantity
        {"marker_name": "Temp",         "marker_value": "38.5",          "raw_text": "Temperature 38.5 °C"},
        {"marker_name": "Heart Rate",   "marker_value": "88 bpm",        "raw_text": "HR 88"},
        {"marker_name": "BP Sys",       "marker_value": "120",           "raw_text": "BP 120/80 mmHg"},
        {"marker_name": "BP Dia",       "marker_value": "80",            "raw_text": "BP 120/80 mmHg"},
        {"marker_name": "O2 Sat",       "marker_value": "97%",           "raw_text": "SpO2 97% on room air"},
        # Drug allergy → AllergyIntolerance
        {"marker_name": "Drug Allergy", "marker_value": "penicillin",    "raw_text": "NKDA except penicillin allergy",  "source_file": "/data/admit.pdf"},
        # Social history
        {"marker_name": "Tobacco",      "marker_value": "former smoker", "raw_text": "Quit smoking 2018"},
        # Clinical note
        {"marker_name": "Biopsy Report","marker_value": "adenocarcinoma","raw_text": "Core biopsy: invasive ductal carcinoma"},
        # Unknown marker — must be skipped (not appear in bundle)
        {"marker_name": "UNKNOWN_XYZ",  "marker_value": "foo",           "raw_text": "some unknown marker"},
    ]

    bundle = map_to_bundle(_TEST_PATIENT, _SAMPLE_FACTS)
    entries = bundle.get("entry", [])

    # ── 1. Print representative entries ──────────────────────────────────────
    print(_SEP)
    print("FHIR TRANSACTION BUNDLE — SAMPLE ENTRIES")
    print(_SEP)

    # Find first Observation and first AllergyIntolerance for display
    obs_entry    = next((e for e in entries if e["resource"]["resourceType"] == "Observation"),           None)
    allergy_entry= next((e for e in entries if e["resource"]["resourceType"] == "AllergyIntolerance"),   None)

    if obs_entry:
        print("\n[Representative Observation entry]\n")
        print(json.dumps(obs_entry, indent=2))
    if allergy_entry:
        print("\n[Representative AllergyIntolerance entry]\n")
        print(json.dumps(allergy_entry, indent=2))

    # ── 2. Bundle-level metadata ──────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("BUNDLE METADATA")
    print(_SEP)
    print(f"  resourceType : {bundle['resourceType']}")
    print(f"  type         : {bundle['type']}")
    print(f"  id           : {bundle['id']}")
    print(f"  timestamp    : {bundle['timestamp']}")
    print(f"  total        : {bundle['total']}  (entries actually in bundle)")
    print(f"  entries built: {len(entries)}")
    print(f"  skipped      : {len(_SAMPLE_FACTS) - len(entries)}  (unknown markers dropped)")

    # ── 3. Requirement verification (all 4 checks across every entry) ─────────
    print(f"\n{_SEP}")
    print("REQUIREMENT VERIFICATION (per entry)")
    print(_SEP)

    req1_fails: List[str] = []   # request object present with method+url
    req2_fails: List[str] = []   # subject / patient reference present
    req3_fails: List[str] = []   # fullUrl is urn:uuid:…
    req4_fails: List[str] = []   # subject reference points to correct patient

    for i, entry in enumerate(entries):
        rt      = entry["resource"]["resourceType"]
        label   = f"entry[{i}] {rt}"
        request = entry.get("request", {})
        resource= entry["resource"]

        # Req 1 — request object with method=POST and url=<resourceType>
        if not (
            isinstance(request, dict)
            and request.get("method") == "POST"
            and request.get("url") == rt
        ):
            req1_fails.append(f"{label}: request={request!r}")

        # Req 2 — subject (Observation) or patient+subject (AllergyIntolerance)
        if rt == "Observation":
            subj = resource.get("subject", {}).get("reference", "")
            if not subj:
                req2_fails.append(f"{label}: missing subject.reference")
            elif not subj.startswith("Patient/"):
                req2_fails.append(f"{label}: subject.reference={subj!r}")
        else:  # AllergyIntolerance
            pat_ref  = resource.get("patient", {}).get("reference", "")
            subj_ref = resource.get("subject", {}).get("reference", "")
            if not pat_ref:
                req2_fails.append(f"{label}: missing patient.reference")
            if not subj_ref:
                req2_fails.append(f"{label}: missing subject.reference (mirror)")

        # Req 3 — fullUrl is a valid urn:uuid:…
        full_url = entry.get("fullUrl", "")
        if not full_url.startswith("urn:uuid:"):
            req3_fails.append(f"{label}: fullUrl={full_url!r}")

        # Req 4 — reference points to the correct patient ID
        expected = f"Patient/{_TEST_PATIENT}"
        refs = []
        if resource.get("subject"):
            refs.append(resource["subject"].get("reference", ""))
        if resource.get("patient"):
            refs.append(resource["patient"].get("reference", ""))
        for ref in refs:
            if ref and ref != expected:
                req4_fails.append(f"{label}: reference={ref!r} expected={expected!r}")

    def _result(fails: List[str]) -> str:
        return "PASS ✓" if not fails else f"FAIL ✗  ({len(fails)} violation(s))"

    print(f"\n  Req 1 — request {{ method:POST, url:<resourceType> }} : {_result(req1_fails)}")
    for msg in req1_fails:
        print(f"           {msg}")

    print(f"  Req 2 — subject/patient reference present            : {_result(req2_fails)}")
    for msg in req2_fails:
        print(f"           {msg}")

    print(f"  Req 3 — fullUrl is urn:uuid:<random-uuid>            : {_result(req3_fails)}")
    for msg in req3_fails:
        print(f"           {msg}")

    print(f"  Req 4 — reference → Patient/{_TEST_PATIENT[:8]}…     : {_result(req4_fails)}")
    for msg in req4_fails:
        print(f"           {msg}")

    # ── 4. Per-entry summary table ────────────────────────────────────────────
    print(f"\n{_SEP}")
    print(f"{'#':<4} {'resourceType':<22} {'LOINC':<10} {'fullUrl (truncated)':<42} {'request.url'}")
    print(_SEP)
    for i, entry in enumerate(entries):
        rt      = entry["resource"]["resourceType"]
        loinc   = (entry["resource"].get("code", {})
                   .get("coding", [{}])[0].get("code", "?"))
        full_url= entry.get("fullUrl", "")[-36:]   # last 36 chars = UUID part
        req_url = entry.get("request", {}).get("url", "?")
        print(f"{i:<4} {rt:<22} {loinc:<10} urn:uuid:{full_url:<33} {req_url}")

    total_fails = len(req1_fails) + len(req2_fails) + len(req3_fails) + len(req4_fails)
    print(f"\n{'All requirements met.' if total_fails == 0 else f'{total_fails} violation(s) found — see details above.'}")
    sys.exit(0 if total_fails == 0 else 1)
