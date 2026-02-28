"""
test_denial_analyzer.py
-----------------------
AgentForge — Healthcare RCM AI Agent — Tests for denial_analyzer.py
--------------------------------------------------------------------
TDD test suite for the denial_analyzer tool. Tests are written before
the implementation to confirm each test fails first, then passes after
implementation. Covers load_denial_patterns and analyze_denial_risk
across happy path, edge cases, and adversarial inputs.

Run:
    pytest tests/test_denial_analyzer.py -v --tb=short

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denial_analyzer import analyze_denial_risk, load_denial_patterns


# ── load_denial_patterns ──────────────────────────────────────────────────────

def test_load_denial_patterns_returns_list():
    """load_denial_patterns must return a non-empty list."""
    patterns = load_denial_patterns()
    assert isinstance(patterns, list)
    assert len(patterns) > 0


def test_load_denial_patterns_structure():
    """Every pattern must have id, code, description, keywords, risk_level, recommendation."""
    patterns = load_denial_patterns()
    for pattern in patterns:
        assert "id" in pattern
        assert "code" in pattern
        assert "description" in pattern
        assert "keywords" in pattern
        assert "risk_level" in pattern
        assert "recommendation" in pattern


def test_load_denial_patterns_keywords_are_list():
    """Every pattern's keywords field must be a list of strings."""
    patterns = load_denial_patterns()
    for pattern in patterns:
        assert isinstance(pattern["keywords"], list)
        for kw in pattern["keywords"]:
            assert isinstance(kw, str)


def test_load_denial_patterns_valid_risk_levels():
    """Every pattern's risk_level must be one of the accepted values."""
    valid_levels = {"NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
    patterns = load_denial_patterns()
    for pattern in patterns:
        assert pattern["risk_level"] in valid_levels


# ── analyze_denial_risk — return structure ────────────────────────────────────

def test_analyze_returns_required_keys():
    """Result dict must always contain all required keys."""
    extractions = [{"claim": "John Smith takes Metformin.", "citation": "Metformin", "source": "mock_data/medications.json", "verbatim": True}]
    result = analyze_denial_risk(extractions)
    assert "success" in result
    assert "risk_level" in result
    assert "matched_patterns" in result
    assert "recommendations" in result
    assert "denial_risk_score" in result
    assert "source" in result
    assert "error" in result


def test_analyze_source_citation_present():
    """Result must always include source pointing to denial_patterns.json for audit trail."""
    extractions = [{"claim": "John Smith takes Metformin.", "citation": "Metformin", "source": "mock_data/medications.json", "verbatim": True}]
    result = analyze_denial_risk(extractions)
    assert result["source"] == "mock_data/denial_patterns.json"


def test_analyze_score_is_float_in_range():
    """denial_risk_score must always be a float between 0.0 and 1.0."""
    extractions = [{"claim": "John Smith takes Metformin.", "citation": "Metformin", "source": "mock_data/medications.json", "verbatim": True}]
    result = analyze_denial_risk(extractions)
    assert isinstance(result["denial_risk_score"], float)
    assert 0.0 <= result["denial_risk_score"] <= 1.0


# ── analyze_denial_risk — happy path ─────────────────────────────────────────

def test_analyze_no_risk_clean_extraction():
    """Clean extraction with no denial-triggering keywords returns LOW or NONE risk."""
    extractions = [
        {
            "claim": "John Smith is prescribed Metformin 500mg twice daily.",
            "citation": "Metformin 500mg twice daily",
            "source": "mock_data/medications.json",
            "verbatim": True,
        }
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert result["risk_level"] in ("LOW", "NONE")


def test_analyze_allergy_conflict_triggers_critical():
    """Extraction mentioning allergy conflict must trigger CRITICAL risk and ALLERGY_CONTRAINDICATION pattern."""
    extractions = [
        {
            "claim": "Emily Rodriguez has a documented Penicillin allergy. She is prescribed Amoxicillin — allergy conflict detected.",
            "citation": "Penicillin allergy | Amoxicillin prescribed",
            "source": "mock_data/patients.json",
            "verbatim": True,
        }
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert result["risk_level"] == "CRITICAL"
    codes = [p["code"] for p in result["matched_patterns"]]
    assert "ALLERGY_CONTRAINDICATION" in codes


def test_analyze_drug_interaction_high_triggers_high():
    """Extraction with HIGH severity drug interaction triggers HIGH or CRITICAL risk."""
    extractions = [
        {
            "claim": "Drug interaction detected: Warfarin + Aspirin — severity HIGH. Increased bleeding risk.",
            "citation": "Warfarin and Aspirin: HIGH severity interaction",
            "source": "mock_data/interactions.json",
            "verbatim": True,
        }
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert result["risk_level"] in ("HIGH", "CRITICAL")
    codes = [p["code"] for p in result["matched_patterns"]]
    assert "DRUG_INTERACTION_HIGH" in codes


def test_analyze_contraindicated_triggers_high():
    """Extraction with CONTRAINDICATED severity triggers HIGH or CRITICAL risk."""
    extractions = [
        {
            "claim": "Drug interaction detected: Sertraline + Tramadol — severity CONTRAINDICATED. Serotonin syndrome risk.",
            "citation": "Sertraline and Tramadol: CONTRAINDICATED",
            "source": "mock_data/interactions.json",
            "verbatim": True,
        }
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert result["risk_level"] in ("HIGH", "CRITICAL")


def test_analyze_missing_documentation_triggers_medium():
    """Extraction mentioning missing or incomplete documentation triggers at least MEDIUM risk."""
    extractions = [
        {
            "claim": "Required clinical documentation is incomplete — no record of prior authorization.",
            "citation": "incomplete documentation",
            "source": "extractor",
            "verbatim": False,
        }
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert result["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")


def test_analyze_returns_recommendations_when_matched():
    """When patterns match, result must include at least one recommendation string."""
    extractions = [
        {
            "claim": "Allergy conflict: patient is allergic to Codeine.",
            "citation": "allergy",
            "source": "mock_data/patients.json",
            "verbatim": True,
        }
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert len(result["recommendations"]) > 0
    for rec in result["recommendations"]:
        assert isinstance(rec, str)
        assert len(rec) > 0


def test_analyze_matched_patterns_have_required_fields():
    """Every matched pattern in result must have id, code, description, and risk_level."""
    extractions = [
        {
            "claim": "Drug interaction: Warfarin + Aspirin — bleeding risk — HIGH severity.",
            "citation": "Warfarin and Aspirin HIGH",
            "source": "mock_data/interactions.json",
            "verbatim": True,
        }
    ]
    result = analyze_denial_risk(extractions)
    for pattern in result["matched_patterns"]:
        assert "id" in pattern
        assert "code" in pattern
        assert "description" in pattern
        assert "risk_level" in pattern


# ── analyze_denial_risk — edge cases ─────────────────────────────────────────

def test_analyze_empty_extractions():
    """Empty extractions list must return success with NONE risk — must not crash."""
    result = analyze_denial_risk([])
    assert result["success"] is True
    assert result["risk_level"] == "NONE"
    assert result["matched_patterns"] == []
    assert result["denial_risk_score"] == 0.0
    assert result["error"] is None


def test_analyze_multiple_extractions_combined():
    """Multiple extractions are evaluated together as one combined search text."""
    extractions = [
        {
            "claim": "John Smith has known allergies: Penicillin, Sulfa.",
            "citation": "Penicillin, Sulfa",
            "source": "mock_data/patients.json",
            "verbatim": True,
        },
        {
            "claim": "Drug interaction detected: Warfarin + Aspirin — severity HIGH. Bleeding risk.",
            "citation": "Warfarin and Aspirin: HIGH",
            "source": "mock_data/interactions.json",
            "verbatim": True,
        },
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert len(result["matched_patterns"]) >= 1


def test_analyze_malformed_extraction_does_not_crash():
    """Malformed extraction dict (missing standard keys) must not raise an exception."""
    extractions = [{"bad_key": "bad_value"}]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert "risk_level" in result


def test_analyze_risk_score_increases_with_severity():
    """Higher severity matches must produce a higher denial_risk_score than clean extractions."""
    low_extractions = [
        {"claim": "John Smith takes Metformin 500mg.", "citation": "Metformin", "source": "mock_data/medications.json", "verbatim": True}
    ]
    high_extractions = [
        {"claim": "Allergy conflict: patient allergic to Penicillin — Amoxicillin contraindicated — allergy conflict detected.", "citation": "allergy conflict", "source": "mock_data/patients.json", "verbatim": True}
    ]
    low_result = analyze_denial_risk(low_extractions)
    high_result = analyze_denial_risk(high_extractions)
    assert high_result["denial_risk_score"] > low_result["denial_risk_score"]


def test_analyze_recommendations_are_strings():
    """All recommendations must be non-empty strings."""
    extractions = [
        {"claim": "Missing documentation — incomplete fields — no record.", "citation": "incomplete", "source": "extractor", "verbatim": False}
    ]
    result = analyze_denial_risk(extractions)
    for rec in result["recommendations"]:
        assert isinstance(rec, str)
        assert len(rec.strip()) > 0


def test_analyze_no_false_positive_on_common_words():
    """Common clinical words that are NOT denial triggers must not produce HIGH/CRITICAL risk."""
    extractions = [
        {"claim": "Patient is stable. Blood pressure within normal range.", "citation": "stable, normal range", "source": "mock_data/patients.json", "verbatim": True}
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert result["risk_level"] not in ("HIGH", "CRITICAL")


# ── Policy criteria flag filtering ────────────────────────────────────────────

def test_criteria_met_flag_excluded_from_denial_search():
    """CRITERIA_MET entries must not contribute to denial keyword matching.

    Criterion descriptions contain words like 'procedure', 'surgery', 'missing'
    that would produce false-positive HIGH/CRITICAL denial scores if included.
    """
    extractions = [
        {
            "claim": "[Cigna Medical Policy #012] Criteria A NOT MET: Conservative therapy failure. "
                     "Patient must have completed a minimum of 3 months of supervised physical therapy "
                     "or conservative management including NSAIDs, corticosteroid injections, "
                     "or activity modification. Documentation must include therapy dates.",
            "citation": "policy_search:cigna",
            "source": "policy_search",
            "synthetic": True,
            "flag": "CRITERIA_MET",
        }
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert result["risk_level"] in ("NONE", "LOW")


def test_criteria_unmet_flag_excluded_from_denial_search():
    """CRITERIA_UNMET entries must not produce false-positive denial pattern matches.

    'Procedure', 'surgery', 'missing', 'documentation' in unmet criteria text
    must not trigger DP001 (PRIOR_AUTH_MISSING) or DP004 (INCOMPLETE_DOCUMENTATION).
    """
    extractions = [
        {
            "claim": "[Cigna Medical Policy #012] Criteria B NOT MET: Radiographic evidence of "
                     "severe osteoarthritis required. Missing imaging report — surgery not yet "
                     "authorized. Procedure 27447 requires documentation.",
            "citation": "policy_search:cigna",
            "source": "policy_search",
            "synthetic": True,
            "flag": "CRITERIA_UNMET",
        }
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert result["risk_level"] in ("NONE", "LOW")
    codes = [p["code"] for p in result["matched_patterns"]]
    assert "PRIOR_AUTH_MISSING" not in codes
    assert "INCOMPLETE_DOCUMENTATION" not in codes


def test_real_ehr_claims_still_trigger_denial_alongside_criteria():
    """When real EHR claims AND criteria entries are mixed, only EHR claims drive denial score.

    Criteria entries are excluded but real allergy/interaction claims are not.
    """
    extractions = [
        {
            "claim": "Drug interaction detected: Warfarin + Aspirin — severity HIGH. Bleeding risk.",
            "citation": "Warfarin and Aspirin: HIGH severity interaction",
            "source": "mock_data/interactions.json",
            "verbatim": True,
        },
        {
            "claim": "[Cigna Medical Policy #012] Criteria C NOT MET: BMI must be recorded. "
                     "Missing documentation. Procedure authorization incomplete.",
            "citation": "policy_search:cigna",
            "source": "policy_search",
            "synthetic": True,
            "flag": "CRITERIA_UNMET",
        },
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
    assert result["risk_level"] in ("HIGH", "CRITICAL")
    codes = [p["code"] for p in result["matched_patterns"]]
    assert "DRUG_INTERACTION_HIGH" in codes
    assert "INCOMPLETE_DOCUMENTATION" not in codes


def test_no_policy_found_flag_still_feeds_denial_analyzer():
    """NO_POLICY_FOUND entries are NOT filtered — only CRITERIA_MET/UNMET are excluded."""
    extractions = [
        {
            "claim": "No policy criteria found for payer 'humana'. Authorization criteria not available.",
            "citation": "policy_search:humana",
            "source": "policy_search",
            "synthetic": True,
            "flag": "NO_POLICY_FOUND",
        }
    ]
    result = analyze_denial_risk(extractions)
    assert result["success"] is True
