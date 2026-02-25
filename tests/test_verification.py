"""
test_verification.py
---------------------
AgentForge — Healthcare RCM AI Agent — Test Suite for verification.py
----------------------------------------------------------------------
TDD test suite for the safety verification layer. Tests are written
BEFORE verification.py is implemented — they will fail first, then
pass once verification.py is built correctly.

Tests cover:
    - check_allergy_conflict: match, no match, case-insensitive, empty allergies
    - calculate_confidence: base score, deductions, clamping
    - should_escalate_to_human: threshold 0.90
    - apply_fda_rules: HIGH, CONTRAINDICATED, LOW, MEDIUM

Run:
    pytest tests/test_verification.py -v --tb=short

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Imports ────────────────────────────────────────────────────────────────

def test_verification_imports():
    """Verification and healthcare_guidelines modules should import without error."""
    from verification import (
        check_allergy_conflict,
        calculate_confidence,
        should_escalate_to_human,
        apply_fda_rules,
    )
    from healthcare_guidelines import (
        SEVERITY_LEVELS,
        FDA_RULES,
        CLINICAL_SAFETY_RULES,
    )
    assert check_allergy_conflict is not None
    assert calculate_confidence is not None
    assert should_escalate_to_human is not None
    assert apply_fda_rules is not None


# ── check_allergy_conflict ─────────────────────────────────────────────────

def test_allergy_conflict_match():
    """check_allergy_conflict should detect when drug matches allergy."""
    from verification import check_allergy_conflict
    result = check_allergy_conflict("Penicillin", ["Penicillin", "Sulfa"])
    assert result.get("conflict") is True
    assert result.get("severity") == "HIGH"
    assert "citation" in str(result).lower() or "source" in str(result).lower()


def test_allergy_conflict_case_insensitive():
    """check_allergy_conflict should be case-insensitive."""
    from verification import check_allergy_conflict
    result = check_allergy_conflict("penicillin", ["Penicillin"])
    assert result.get("conflict") is True
    assert result.get("severity") == "HIGH"


def test_allergy_conflict_no_match():
    """check_allergy_conflict should return no conflict when drug not in allergies."""
    from verification import check_allergy_conflict
    result = check_allergy_conflict("Metformin", ["Penicillin", "Sulfa"])
    assert result.get("conflict") is False


def test_allergy_conflict_empty_allergies():
    """check_allergy_conflict with empty allergies should return no conflict."""
    from verification import check_allergy_conflict
    result = check_allergy_conflict("Penicillin", [])
    assert result.get("conflict") is False


# ── calculate_confidence ───────────────────────────────────────────────────

def test_confidence_base_score():
    """calculate_confidence: 3/3 tools, no deductions = 1.0."""
    from verification import calculate_confidence
    score = calculate_confidence(3, 3, False, False)
    assert score == 1.0


def test_confidence_allergy_deduction():
    """calculate_confidence: allergy conflict deducts 0.20."""
    from verification import calculate_confidence
    score = calculate_confidence(3, 3, False, True)
    assert abs(score - 0.80) < 0.001


def test_confidence_interaction_deduction():
    """calculate_confidence: interactions found deducts 0.10."""
    from verification import calculate_confidence
    score = calculate_confidence(3, 3, True, False)
    assert abs(score - 0.90) < 0.001


def test_confidence_both_deductions():
    """calculate_confidence: allergy + interaction deduct 0.30."""
    from verification import calculate_confidence
    score = calculate_confidence(3, 3, True, True)
    assert abs(score - 0.70) < 0.001


def test_confidence_clamped_zero():
    """calculate_confidence: score clamped to min 0.0."""
    from verification import calculate_confidence
    score = calculate_confidence(0, 3, True, True)
    assert score >= 0.0
    assert score <= 1.0


# ── should_escalate_to_human ───────────────────────────────────────────────

def test_escalate_above_threshold():
    """should_escalate_to_human: score >= 0.90 returns escalate False."""
    from verification import should_escalate_to_human
    result = should_escalate_to_human(0.90)
    assert result.get("escalate") is False
    assert "disclaimer" in result or "consult" in str(result).lower()


def test_escalate_below_threshold():
    """should_escalate_to_human: score < 0.90 returns escalate True."""
    from verification import should_escalate_to_human
    result = should_escalate_to_human(0.89)
    assert result.get("escalate") is True
    assert "reason" in result or "escalat" in str(result).lower()


# ── apply_fda_rules ────────────────────────────────────────────────────────

def test_fda_rules_high():
    """apply_fda_rules: HIGH severity requires physician review."""
    from verification import apply_fda_rules
    result = apply_fda_rules("HIGH")
    assert result.get("requires_physician_review") is True
    assert "FDA" in str(result) or "citation" in str(result).lower() or "source" in str(result).lower()


def test_fda_rules_contraindicated():
    """apply_fda_rules: CONTRAINDICATED requires physician review."""
    from verification import apply_fda_rules
    result = apply_fda_rules("CONTRAINDICATED")
    assert result.get("requires_physician_review") is True


def test_fda_rules_low():
    """apply_fda_rules: LOW can auto-approve with monitoring."""
    from verification import apply_fda_rules
    result = apply_fda_rules("LOW")
    assert result.get("requires_physician_review") is False
    assert "monitor" in str(result).lower() or "approve" in str(result).lower()


def test_fda_rules_medium():
    """apply_fda_rules: MEDIUM requires physician review (per spec)."""
    from verification import apply_fda_rules
    result = apply_fda_rules("MEDIUM")
    assert result.get("requires_physician_review") is True
