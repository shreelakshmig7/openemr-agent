"""
denial_analyzer.py
------------------
AgentForge — Healthcare RCM AI Agent — Denial Risk Analyzer Tool
-----------------------------------------------------------------
Analyzes clinical extractions against historical insurance denial patterns
to predict claim rejection risk before submission. Compares extraction text
against a structured denial pattern database and returns a risk level,
matched patterns, and actionable recommendations for the care team.

This tool is called by the LangGraph extractor node after all clinical
extractions are collected. It does not make LLM calls — all logic is
deterministic keyword matching against mock_data/denial_patterns.json.

Key functions:
    load_denial_patterns: Loads denial patterns from mock_data/denial_patterns.json.
    analyze_denial_risk: Scores a list of clinical extractions against denial patterns.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import json
import logging
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DENIAL_PATTERNS_PATH = os.path.join(BASE_DIR, "mock_data", "denial_patterns.json")

# Risk level severity ordering — higher index = higher severity
_RISK_ORDER: dict = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# Numeric denial risk score per risk level (0.0–1.0)
_RISK_SCORES: dict = {
    "NONE": 0.0,
    "LOW": 0.1,
    "MEDIUM": 0.35,
    "HIGH": 0.65,
    "CRITICAL": 0.90,
}


# ── Data loader ───────────────────────────────────────────────────────────────

def load_denial_patterns() -> List[dict]:
    """
    Load denial patterns from mock_data/denial_patterns.json.

    Args:
        None

    Returns:
        List[dict]: List of denial pattern dicts, each with id, code, description,
                    keywords, risk_level, and recommendation fields.
                    Returns empty list if the file is missing or malformed.

    Raises:
        Never — returns empty list on any failure.
    """
    try:
        with open(DENIAL_PATTERNS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        patterns = data.get("denial_patterns", [])
        if not isinstance(patterns, list):
            logger.warning("denial_patterns.json has unexpected structure — returning empty list.")
            return []
        return patterns
    except FileNotFoundError:
        logger.error("denial_patterns.json not found at %s", DENIAL_PATTERNS_PATH)
        return []
    except json.JSONDecodeError as e:
        logger.error("Failed to parse denial_patterns.json: %s", e)
        return []
    except Exception as e:
        logger.exception("Unexpected error loading denial patterns: %s", e)
        return []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_search_text(extractions: List[dict]) -> str:
    """
    Combine all extraction claim and citation text into one lowercase search string.

    Args:
        extractions: List of extraction dicts from the extractor node.

    Returns:
        str: Lowercased combined text suitable for keyword matching.

    Raises:
        Never.
    """
    parts = []
    for ext in extractions:
        if isinstance(ext, dict):
            parts.append(ext.get("claim", ""))
            parts.append(ext.get("citation", ""))
    return " ".join(parts).lower()


def _score_patterns(
    search_text: str,
    patterns: List[dict],
) -> Tuple[List[dict], str, float, List[str]]:
    """
    Match denial pattern keywords against combined extraction text.

    Iterates every pattern and checks if any of its keywords appear in the
    search text. Returns all matched patterns, the highest risk level found,
    a numeric denial_risk_score, and the list of recommendations.

    Args:
        search_text: Lowercased combined extraction text from _build_search_text.
        patterns: List of denial pattern dicts from load_denial_patterns.

    Returns:
        Tuple of:
            matched (List[dict]): Patterns that had at least one keyword match.
            risk_level (str): Highest risk level across all matches.
            score (float): Numeric score 0.0–1.0 corresponding to risk_level.
            recommendations (List[str]): One recommendation string per matched pattern.

    Raises:
        Never.
    """
    matched: List[dict] = []
    highest_risk = "NONE"
    recommendations: List[str] = []

    for pattern in patterns:
        keywords = [kw.lower() for kw in pattern.get("keywords", [])]
        if any(kw in search_text for kw in keywords):
            matched.append({
                "id": pattern.get("id"),
                "code": pattern.get("code"),
                "description": pattern.get("description"),
                "risk_level": pattern.get("risk_level"),
            })
            recommendation = pattern.get("recommendation", "")
            if recommendation:
                recommendations.append(recommendation)
            pattern_risk = pattern.get("risk_level", "NONE")
            if _RISK_ORDER.get(pattern_risk, 0) > _RISK_ORDER.get(highest_risk, 0):
                highest_risk = pattern_risk

    score = _RISK_SCORES.get(highest_risk, 0.0)
    return matched, highest_risk, score, recommendations


# ── Public tool function ──────────────────────────────────────────────────────

def analyze_denial_risk(extractions: List[dict]) -> dict:
    """
    Analyze clinical extractions against historical denial patterns and return a risk assessment.

    Loads patterns from mock_data/denial_patterns.json, builds a unified search
    string from all extraction claims and citations, and scores against each
    pattern's keywords. Returns the highest matched risk level plus all matched
    patterns and actionable recommendations for the care team.

    Args:
        extractions: List of extraction dicts from the LangGraph extractor node.
                     Each dict should contain claim, citation, source, and verbatim keys.
                     Empty list is valid — returns NONE risk.

    Returns:
        dict: {
            "success": bool,
            "risk_level": str,           # NONE | LOW | MEDIUM | HIGH | CRITICAL
            "matched_patterns": List[dict],
            "recommendations": List[str],
            "denial_risk_score": float,  # 0.0–1.0
            "source": str,               # always "mock_data/denial_patterns.json"
            "error": str | None
        }

    Raises:
        Never — all failures return a structured error dict with success: False.
    """
    try:
        if not extractions:
            return {
                "success": True,
                "risk_level": "NONE",
                "matched_patterns": [],
                "recommendations": [],
                "denial_risk_score": 0.0,
                "source": "mock_data/denial_patterns.json",
                "error": None,
            }

        patterns = load_denial_patterns()
        search_text = _build_search_text(extractions)
        matched, risk_level, score, recommendations = _score_patterns(search_text, patterns)

        return {
            "success": True,
            "risk_level": risk_level,
            "matched_patterns": matched,
            "recommendations": recommendations,
            "denial_risk_score": score,
            "source": "mock_data/denial_patterns.json",
            "error": None,
        }

    except Exception as e:
        logger.exception("analyze_denial_risk failed: %s", e)
        return {
            "success": False,
            "risk_level": "UNKNOWN",
            "matched_patterns": [],
            "recommendations": [],
            "denial_risk_score": 0.0,
            "source": "mock_data/denial_patterns.json",
            "error": f"Unexpected error in denial analyzer: {str(e)}",
        }
