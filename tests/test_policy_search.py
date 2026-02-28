"""
test_policy_search.py
---------------------
AgentForge — Healthcare RCM AI Agent — Tests for policy_search tool
--------------------------------------------------------------------
TDD test suite for tools/policy_search.py covering:
    - Mock mode (keyword-based fallback, always active in test environment)
    - Known payer returns structured criteria results
    - Unknown / empty payer returns no_policy_found cleanly
    - Empty procedure_code does not crash
    - Criteria met/unmet logic based on evidence overlap
    - Result dict always contains required keys (success, policy_id,
      criteria_met, criteria_unmet, source, error)
    - Pinecone mode routes to mock when USE_REAL_PINECONE is unset

Scenarios from UI testing / design discussion (2026-02-28):
    - gs-034 equivalent: cigna + knee replacement (27447) + EHR evidence
    - Policy query with no payer → empty payer_id → no_policy_found (safe)
    - Policy query with unknown payer (aetna not in mock) → no_policy_found
    - Empty procedure_code → still searches by payer, no crash
    - Policy query with procedure description vs CPT code both accepted

Run:
    pytest tests/test_policy_search.py -v --tb=short

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import os
import sys
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force mock mode for all tests — no real Pinecone calls
os.environ.pop("USE_REAL_PINECONE", None)

from tools.policy_search import search_policy, _search_mock, _criterion_supported_by_evidence, _find_supporting_evidence


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _extractions_with_evidence():
    """Clinical extractions that contain keywords matching Cigna criteria."""
    return [
        {
            "claim": "Patient completed 12 weeks of supervised physical therapy with documented functional decline. NSAIDs were trialed and discontinued due to GI intolerance.",
            "source": "patients.json",
        },
        {
            "claim": "X-ray demonstrates severe Kellgren-Lawrence grade 4 osteoarthritis with bone-on-bone contact in medial compartment. Radiologist report signed.",
            "source": "patients.json",
        },
        {
            "claim": "BMI recorded as 28.4 by treating physician. Weight stable for past 6 months.",
            "source": "patients.json",
        },
    ]


def _empty_extractions():
    return []


def _extractions_no_match():
    """Extractions that contain no keywords matching any policy criteria."""
    return [
        {"claim": "Patient reports mild headache. Recommended hydration.", "source": "patients.json"},
    ]


# ── Mock mode: known payer ────────────────────────────────────────────────────

class TestMockSearchKnownPayer:
    def test_known_payer_returns_success_true(self):
        """cigna payer exists in mock data — success must be True."""
        result = search_policy("cigna", "27447", _extractions_with_evidence())
        assert result["success"] is True

    def test_known_payer_returns_policy_id(self):
        """Known payer returns a non-null policy_id."""
        result = search_policy("cigna", "27447", _extractions_with_evidence())
        assert result["policy_id"] is not None
        assert "Cigna" in result["policy_id"]

    def test_known_payer_returns_criteria_lists(self):
        """Result must include criteria_met and criteria_unmet as lists."""
        result = search_policy("cigna", "27447", _extractions_with_evidence())
        assert isinstance(result["criteria_met"], list)
        assert isinstance(result["criteria_unmet"], list)

    def test_known_payer_source_is_mock(self):
        """Mock mode must always return source='mock'."""
        result = search_policy("cigna", "27447", _extractions_with_evidence())
        assert result["source"] == "mock"

    def test_known_payer_no_error(self):
        """Successful mock search returns error=None."""
        result = search_policy("cigna", "27447", _extractions_with_evidence())
        assert result["error"] is None

    def test_known_payer_with_evidence_has_criteria_met(self):
        """With matching clinical evidence, at least one criterion should be met."""
        result = search_policy("cigna", "27447", _extractions_with_evidence())
        assert len(result["criteria_met"]) > 0

    def test_known_payer_no_evidence_criteria_all_unmet(self):
        """With no clinical evidence, no criteria can be met."""
        result = search_policy("cigna", "27447", _empty_extractions())
        assert len(result["criteria_met"]) == 0

    def test_known_payer_no_match_evidence_criteria_all_unmet(self):
        """With unrelated clinical evidence, no policy criteria should be met."""
        result = search_policy("cigna", "27447", _extractions_no_match())
        assert len(result["criteria_met"]) == 0

    def test_criteria_entry_has_required_keys(self):
        """Each criteria entry (met or unmet) must have id, description, evidence."""
        result = search_policy("cigna", "27447", _extractions_with_evidence())
        all_criteria = result["criteria_met"] + result["criteria_unmet"]
        assert len(all_criteria) > 0
        for entry in all_criteria:
            assert "id" in entry
            assert "description" in entry
            assert "evidence" in entry

    def test_payer_case_insensitive(self):
        """Payer ID lookup should be case-insensitive (cigna == CIGNA == Cigna)."""
        result_lower = search_policy("cigna", "27447", _extractions_with_evidence())
        result_upper = search_policy("CIGNA", "27447", _extractions_with_evidence())
        assert result_lower["success"] == result_upper["success"]
        assert result_lower["policy_id"] == result_upper["policy_id"]

    def test_procedure_description_accepted(self):
        """Procedure description ('knee replacement') accepted same as CPT code."""
        result = search_policy("cigna", "knee replacement", _extractions_with_evidence())
        assert result["success"] is True
        assert result["error"] is None

    def test_empty_procedure_code_does_not_crash(self):
        """Empty procedure_code must not raise — returns valid result."""
        result = search_policy("cigna", "", _extractions_with_evidence())
        assert result["success"] is True
        assert isinstance(result["criteria_met"], list)


# ── Mock mode: unknown / empty payer ─────────────────────────────────────────

class TestMockSearchUnknownPayer:
    # "humana" is deliberately absent from mock data (only cigna, aetna, uhc exist)
    _ABSENT_PAYER = "humana"

    def test_unknown_payer_returns_no_policy_found(self):
        """Payer not in mock data sets no_policy_found=True."""
        result = search_policy(self._ABSENT_PAYER, "MRI", _extractions_with_evidence())
        assert result.get("no_policy_found") is True

    def test_unknown_payer_returns_success_true(self):
        """no_policy_found is a valid outcome — success must still be True (not an error)."""
        result = search_policy(self._ABSENT_PAYER, "MRI", _extractions_with_evidence())
        assert result["success"] is True

    def test_unknown_payer_criteria_lists_empty(self):
        """Unknown payer — no criteria to return."""
        result = search_policy(self._ABSENT_PAYER, "MRI", _extractions_with_evidence())
        assert result["criteria_met"] == []
        assert result["criteria_unmet"] == []

    def test_unknown_payer_has_message(self):
        """no_policy_found result must include a human-readable message."""
        result = search_policy(self._ABSENT_PAYER, "MRI", _extractions_with_evidence())
        assert "message" in result
        assert len(result["message"]) > 0

    def test_empty_payer_id_returns_no_policy_found(self):
        """Empty payer_id (not provided by UI or query) must not crash or guess."""
        result = search_policy("", "27447", _extractions_with_evidence())
        assert result["success"] is True
        assert result.get("no_policy_found") is True

    def test_empty_payer_and_empty_procedure_safe(self):
        """Both empty — must return valid result dict, not crash."""
        result = search_policy("", "", [])
        assert "success" in result
        assert "criteria_met" in result
        assert "criteria_unmet" in result

    def test_unknown_payer_no_error_field(self):
        """no_policy_found path must set error=None, not an exception string."""
        result = search_policy(self._ABSENT_PAYER, "99213", _extractions_with_evidence())
        assert result["error"] is None


# ── Result dict structure ─────────────────────────────────────────────────────

class TestResultDictStructure:
    def test_all_required_keys_present_known_payer(self):
        """Known payer result always has all required keys."""
        result = search_policy("cigna", "27447", _extractions_with_evidence())
        for key in ["success", "policy_id", "criteria_met", "criteria_unmet", "source", "error"]:
            assert key in result, f"Missing key: {key}"

    def test_all_required_keys_present_unknown_payer(self):
        """Unknown payer result also has all required keys."""
        result = search_policy("unknown_payer", "99999", [])
        for key in ["success", "policy_id", "criteria_met", "criteria_unmet", "source", "error"]:
            assert key in result, f"Missing key: {key}"

    def test_policy_id_null_for_unknown_payer(self):
        """policy_id is None when payer has no policy in mock data."""
        result = search_policy("unknownpayer", "27447", [])
        assert result["policy_id"] is None


# ── Mock mode routing ─────────────────────────────────────────────────────────

class TestMockModeRouting:
    def test_mock_mode_used_when_use_pinecone_false(self):
        """When _USE_PINECONE is False, search_policy routes to _search_mock."""
        import tools.policy_search as ps
        with patch.object(ps, "_USE_PINECONE", False):
            result = search_policy("cigna", "27447", _extractions_with_evidence())
        assert result["source"] == "mock"

    def test_pinecone_mode_routes_to_pinecone_when_flag_true(self):
        """When _USE_PINECONE is True, search_policy calls _search_pinecone (mocked)."""
        import tools.policy_search as ps
        mock_result = {
            "success": True,
            "policy_id": "Cigna Policy #012",
            "criteria_met": [],
            "criteria_unmet": [],
            "source": "pinecone",
            "error": None,
        }
        with patch.object(ps, "_USE_PINECONE", True), \
             patch.object(ps, "_search_pinecone", return_value=mock_result) as mock_fn:
            result = search_policy("cigna", "27447", _extractions_with_evidence())
        mock_fn.assert_called_once()
        assert result["source"] == "pinecone"


# ── Evidence matching helpers ─────────────────────────────────────────────────

class TestEvidenceHelpers:
    def test_criterion_supported_by_evidence_match(self):
        """Criterion with ≥20% keyword overlap in evidence returns True."""
        criterion = "Patient must have completed supervised physical therapy with documented functional decline"
        evidence = "completed supervised physical therapy functional decline documented over 12 weeks"
        assert _criterion_supported_by_evidence(criterion, evidence) is True

    def test_criterion_supported_by_evidence_no_match(self):
        """Criterion with <20% keyword overlap in evidence returns False."""
        criterion = "Radiographic evidence severe osteoarthritis Kellgren Lawrence sclerosis osteophyte"
        evidence = "patient reports mild headache and fatigue"
        assert _criterion_supported_by_evidence(criterion, evidence) is False

    def test_criterion_supported_empty_evidence(self):
        """Empty evidence string always returns False."""
        criterion = "Patient must have completed physical therapy"
        assert _criterion_supported_by_evidence(criterion, "") is False

    def test_criterion_supported_empty_criterion(self):
        """Empty criterion text (no long words) returns False safely."""
        assert _criterion_supported_by_evidence("", "some evidence text here") is False

    def test_find_supporting_evidence_returns_matching_claim(self):
        """Returns the extraction claim with ≥2 keyword matches to the criterion."""
        criterion = "physical therapy completed functional decline"
        extractions = [
            {"claim": "Patient completed 12 weeks of physical therapy with functional decline documented"},
            {"claim": "Blood pressure 120/80, stable"},
        ]
        result = _find_supporting_evidence(criterion, extractions)
        assert "physical therapy" in result.lower() or "functional" in result.lower()

    def test_find_supporting_evidence_fallback_when_no_match(self):
        """Returns fallback string when no extraction matches criterion keywords."""
        criterion = "Kellgren Lawrence grade 4 radiographic evidence"
        extractions = [{"claim": "Patient reports mild headache"}]
        result = _find_supporting_evidence(criterion, extractions)
        assert result == "See clinical documentation"

    def test_find_supporting_evidence_empty_extractions(self):
        """Empty extractions list returns fallback string."""
        result = _find_supporting_evidence("some criterion text here", [])
        assert result == "See clinical documentation"
