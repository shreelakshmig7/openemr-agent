"""
verification.py
---------------
AgentForge — Healthcare RCM AI Agent — Safety verification layer
-----------------------------------------------------------------
Four functions enforcing healthcare safety rules: allergy conflict
check, confidence scoring, human escalation, and FDA rule application.
Uses constants from healthcare_guidelines.py. Never raises exceptions
to caller — returns structured dicts.

Key functions:
    - check_allergy_conflict: drug vs allergy list, case-insensitive
    - calculate_confidence: tools success + deductions for allergy/interaction
    - should_escalate_to_human: threshold 0.90
    - apply_fda_rules: severity → physician review required or not

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from typing import List
from langsmith import traceable
from healthcare_guidelines import (
    FDA_RULES,
    CLINICAL_SAFETY_RULES,
    DRUG_CLASS_ALLERGY_MAP,
)

@traceable
def check_allergy_conflict(drug: str, allergies: List[str]) -> dict:
    """
    Check if a drug conflicts with any known allergy (case-insensitive).

    Args:
        drug: Medication name to check.
        allergies: List of known allergy strings.

    Returns:
        dict: {"conflict": bool, "drug": str, "allergy": str|None, "conflict_type": str|None,
               "severity": str|None, "source_citation": str}.
              conflict=True and severity="HIGH" when exact or class-level match found.
    """
    try:
        if not drug or not isinstance(drug, str):
            return {
                "conflict": False,
                "drug": str(drug) if drug else "",
                "allergy": None,
                "conflict_type": None,
                "severity": None,
                "source_citation": "Allergy check: no drug provided.",
            }
        drug_lower = drug.strip().lower()
        if not allergies or not isinstance(allergies, list):
            return {
                "conflict": False,
                "drug": drug,
                "allergy": None,
                "conflict_type": None,
                "severity": None,
                "source_citation": "Allergy check: no allergies on record.",
            }

        # Pass 1 — exact name match (case-insensitive)
        for allergy in allergies:
            if allergy and isinstance(allergy, str) and allergy.strip().lower() == drug_lower:
                return {
                    "conflict": True,
                    "drug": drug,
                    "allergy": allergy,
                    "conflict_type": "exact_name",
                    "severity": "HIGH",
                    "source_citation": (
                        f"Allergy conflict: {drug} matches known allergy '{allergy}'. "
                        "Source: Patient allergy record."
                    ),
                }

        # Pass 2 — drug class match (e.g. Amoxicillin → penicillin class)
        for allergy in allergies:
            if not allergy or not isinstance(allergy, str):
                continue
            allergy_lower = allergy.strip().lower()
            class_drugs = DRUG_CLASS_ALLERGY_MAP.get(allergy_lower, [])
            if drug_lower in class_drugs:
                return {
                    "conflict": True,
                    "drug": drug,
                    "allergy": allergy,
                    "conflict_type": "drug_class",
                    "severity": "HIGH",
                    "source_citation": (
                        f"Allergy conflict: {drug} belongs to the {allergy} drug class. "
                        f"Patient has documented {allergy} allergy. "
                        "Source: Patient allergy record + drug class guidelines."
                    ),
                }

        return {
            "conflict": False,
            "drug": drug,
            "allergy": None,
            "conflict_type": None,
            "severity": None,
            "source_citation": "Allergy check: no conflict found. Source: Patient allergy record.",
        }
    except Exception as e:
        return {
            "conflict": False,
            "drug": str(drug) if drug else "",
            "severity": None,
            "source_citation": f"Allergy check error: {str(e)}",
        }

@traceable
def calculate_confidence(
    tools_succeeded: int,
    tools_total: int,
    interactions_found: bool,
    allergy_conflict: bool,
) -> float:
    """
    Calculate confidence score from tool success rate minus deductions.

    Base = tools_succeeded / tools_total. Deduct 0.20 for allergy conflict,
    0.10 for interactions found. Clamp result between 0.0 and 1.0.

    Args:
        tools_succeeded: Number of tools that succeeded.
        tools_total: Total number of tools called.
        interactions_found: True if drug interactions were detected.
        allergy_conflict: True if allergy conflict was found.

    Returns:
        float: Confidence score in [0.0, 1.0].
    """
    try:
        base = tools_succeeded / tools_total if tools_total > 0 else 0.0
        deduction = 0.0
        if allergy_conflict:
            deduction += CLINICAL_SAFETY_RULES["allergy_conflict_deduction"]
        if interactions_found:
            deduction += CLINICAL_SAFETY_RULES["interaction_found_deduction"]
        score = base - deduction
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0

@traceable
def should_escalate_to_human(confidence_score: float) -> dict:
    """
    Determine if response should be escalated to physician based on confidence.

    Threshold is 0.90 from CLINICAL_SAFETY_RULES. Below threshold returns
    escalate=True with reason; at or above returns escalate=False with disclaimer.

    Args:
        confidence_score: Float in [0.0, 1.0].

    Returns:
        dict: {"escalate": bool, "reason": str|None, "disclaimer": str|None}.
    """
    try:
        threshold = CLINICAL_SAFETY_RULES["confidence_threshold"]
        disclaimer = CLINICAL_SAFETY_RULES["disclaimer"]
        if confidence_score >= threshold:
            return {
                "escalate": False,
                "reason": None,
                "disclaimer": disclaimer,
            }
        return {
            "escalate": True,
            "reason": f"Confidence score {confidence_score:.2f} is below {threshold} threshold. Physician review recommended.",
            "disclaimer": disclaimer,
        }
    except Exception as e:
        return {
            "escalate": True,
            "reason": f"Unable to compute confidence: {str(e)}. Physician review recommended.",
            "disclaimer": CLINICAL_SAFETY_RULES.get("disclaimer", ""),
        }

@traceable
def apply_fda_rules(interaction_severity: str) -> dict:
    """
    Map interaction severity to FDA action requirements.

    HIGH and CONTRAINDICATED always require physician review. LOW can
    auto-approve with monitoring. MEDIUM requires physician review.
    Always includes FDA source citation.

    Args:
        interaction_severity: One of HIGH, CONTRAINDICATED, MEDIUM, LOW.

    Returns:
        dict: {"requires_physician_review": bool, "action": str, "source_citation": str}.
    """
    try:
        severity = (interaction_severity or "").strip().upper()
        source = FDA_RULES.get("source", "FDA Drug Safety Guidelines")
        if severity in FDA_RULES["physician_review_required"]:
            return {
                "requires_physician_review": True,
                "action": "Physician review required before proceeding.",
                "source_citation": source,
            }
        if severity == FDA_RULES["auto_approve_with_monitoring"]:
            return {
                "requires_physician_review": False,
                "action": "Auto-approve with monitoring note. Monitor patient as clinically indicated.",
                "source_citation": source,
            }
        # MEDIUM and any other severity: require physician review (conservative)
        return {
            "requires_physician_review": True,
            "action": "Physician review recommended. Source: FDA Drug Safety Guidelines.",
            "source_citation": source,
        }
    except Exception as e:
        return {
            "requires_physician_review": True,
            "action": "Unable to determine FDA action. Physician review required.",
            "source_citation": "FDA Drug Safety Guidelines",
        }
