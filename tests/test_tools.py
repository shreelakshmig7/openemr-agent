"""
test_tools.py
-------------
Healthcare AI Agent — Test Suite for tools.py
----------------------------------------------
Test-Driven Development (TDD) test suite for all agent tools.
Tests are organized by tool and cover:
    - Happy path scenarios
    - Edge cases (empty input, invalid IDs)
    - Data structure validation

Run:
    pytest tests/test_tools.py -v --tb=short

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import get_patient_info, get_medications, check_drug_interactions


# ── Tool 1: get_patient_info ──────────────────────────────────────────────────

def test_get_patient_info_found():
    """Happy path: patient exists"""
    result = get_patient_info("John Smith")
    assert result["success"] is True
    assert result["patient"]["id"] == "P001"
    assert result["patient"]["name"] == "John Smith"

def test_get_patient_info_partial_name():
    """Should find patient with partial name match"""
    result = get_patient_info("John")
    assert result["success"] is True
    assert "John" in result["patient"]["name"]

def test_get_patient_info_case_insensitive():
    """Should be case insensitive"""
    result = get_patient_info("john smith")
    assert result["success"] is True
    assert result["patient"]["id"] == "P001"

def test_get_patient_info_not_found():
    """Edge case: patient does not exist"""
    result = get_patient_info("Unknown Patient")
    assert result["success"] is False
    assert result["patient"] is None
    assert "error" in result

def test_get_patient_info_empty_string():
    """Edge case: empty string input must not return any patient."""
    result = get_patient_info("")
    assert result["success"] is False
    assert result["patient"] is None
    assert "error" in result


# ── Tool 2: get_medications ───────────────────────────────────────────────────

def test_get_medications_found():
    """Happy path: patient has medications"""
    result = get_medications("P001")
    assert result["success"] is True
    assert len(result["medications"]) == 3
    assert result["medications"][0]["name"] == "Metformin"

def test_get_medications_all_patients():
    """Happy path: all 3 patients have medications"""
    for pid in ["P001", "P002", "P003"]:
        result = get_medications(pid)
        assert result["success"] is True
        assert len(result["medications"]) > 0

def test_get_medications_structure():
    """Each medication should have name, dose, frequency"""
    result = get_medications("P002")
    for med in result["medications"]:
        assert "name" in med
        assert "dose" in med
        assert "frequency" in med

def test_get_medications_invalid_patient():
    """Edge case: patient ID does not exist"""
    result = get_medications("P999")
    assert result["success"] is False
    assert result["medications"] == []
    assert "error" in result

def test_get_medications_empty_id():
    """Edge case: empty patient ID"""
    result = get_medications("")
    assert result["success"] is False


# ── Tool 3: check_drug_interactions ──────────────────────────────────────────

def test_check_interactions_high_severity():
    """Happy path: known HIGH severity interaction"""
    meds = [{"name": "Warfarin"}, {"name": "Aspirin"}]
    result = check_drug_interactions(meds)
    assert result["success"] is True
    assert result["interactions_found"] is True
    assert result["interactions"][0]["severity"] == "HIGH"

def test_check_interactions_no_interactions():
    """Happy path: no known interactions"""
    meds = [{"name": "Metformin"}, {"name": "Atorvastatin"}]
    result = check_drug_interactions(meds)
    assert result["success"] is True
    assert result["interactions_found"] is False
    assert result["interactions"] == []

def test_check_interactions_multiple_found():
    """Should find multiple interactions"""
    meds = [{"name": "Warfarin"}, {"name": "Aspirin"}, {"name": "Ibuprofen"}]
    result = check_drug_interactions(meds)
    assert result["interactions_found"] is True
    assert result["count"] >= 2

def test_check_interactions_string_input():
    """Should handle plain string list, not just dict list"""
    meds = ["Warfarin", "Aspirin"]
    result = check_drug_interactions(meds)
    assert result["interactions_found"] is True

def test_check_interactions_empty_list():
    """Edge case: empty medication list"""
    result = check_drug_interactions([])
    assert result["success"] is True
    assert result["interactions_found"] is False

def test_check_interactions_single_drug():
    """Edge case: only one drug — no interaction possible"""
    meds = [{"name": "Warfarin"}]
    result = check_drug_interactions(meds)
    assert result["interactions_found"] is False