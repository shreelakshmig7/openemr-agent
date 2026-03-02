"""
Unit tests for identity resolution: composite key (name + DOB).

Validates that:
- _normalize_dob produces ISO YYYY-MM-DD.
- get_patient_info with dob resolves to the correct same-name patient in local cache.
- When name matches but DOB differs, get_patient_info returns success=False (new patient).
"""
import os
import sys

# Run from openemr-agent root so tools and langgraph_agent are importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from tools import _normalize_dob, get_patient_info


class TestNormalizeDob:
    """Test DOB normalization to ISO YYYY-MM-DD."""

    def test_iso_unchanged(self):
        assert _normalize_dob("1965-03-15") == "1965-03-15"

    def test_mm_dd_yyyy_slash(self):
        assert _normalize_dob("3/15/1965") == "1965-03-15"
        assert _normalize_dob("03/15/1965") == "1965-03-15"

    def test_mm_dd_yyyy_dash(self):
        assert _normalize_dob("3-15-1965") == "1965-03-15"

    def test_empty_none(self):
        assert _normalize_dob("") is None
        assert _normalize_dob(None) is None

    def test_invalid_returns_none(self):
        assert _normalize_dob("not-a-date") is None
        assert _normalize_dob("32/13/2000") is None


class TestGetPatientInfoCompositeKey:
    """
    Test get_patient_info with optional dob (local cache only).
    Assumes mock_data/patients.json has P001 John Smith 1965-03-15 and P012 John Smith 1990-05-20.
    """

    def test_john_smith_dob_1965_resolves_to_p001(self):
        out = get_patient_info("John Smith", dob="1965-03-15")
        assert out.get("success") is True
        assert out.get("patient") is not None
        assert out["patient"].get("id") == "P001"
        assert out["patient"].get("dob") == "1965-03-15"
        assert "Penicillin" in (out["patient"].get("allergies") or [])
        assert "Sulfa" in (out["patient"].get("allergies") or [])

    def test_john_smith_dob_1990_resolves_to_p012(self):
        out = get_patient_info("John Smith", dob="1990-05-20")
        assert out.get("success") is True
        assert out.get("patient") is not None
        assert out["patient"].get("id") == "P012"
        assert out["patient"].get("dob") == "1990-05-20"
        assert "Latex" in (out["patient"].get("allergies") or [])

    def test_john_smith_name_match_dob_different_returns_not_found(self):
        # DOB that matches no John Smith in local cache → not found (new patient).
        out = get_patient_info("John Smith", dob="1980-01-01")
        assert out.get("success") is False
        assert out.get("patient") is None

    def test_john_smith_no_dob_returns_first_match(self):
        # Legacy: no DOB → first name match (order in patients.json).
        out = get_patient_info("John Smith")
        assert out.get("success") is True
        assert out.get("patient") is not None
        # First John Smith in JSON is P001.
        assert out["patient"].get("id") == "P001"

    def test_p_id_ignores_dob(self):
        out = get_patient_info("P012", dob="1965-03-15")
        assert out.get("success") is True
        assert out["patient"].get("id") == "P012"
        assert out["patient"].get("dob") == "1990-05-20"
