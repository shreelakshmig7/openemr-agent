"""
run_eval.py
-----------
AgentForge — Healthcare RCM AI Agent — Eval runner
---------------------------------------------------
Loads test cases from golden_data.yaml, runs each through the LangGraph
workflow, scores on five dimensions, computes pass rate, and saves
timestamped results.

Scoring dimensions (all must pass for PASS verdict):
    must_contain        — at least one keyword present in response (OR)
    must_not_contain    — no forbidden keyword present (AND NOT)
    confidence_max      — confidence_score <= expected_confidence_max (optional)
    escalate            — escalate flag matches expected_escalate (optional)
    denial_risk         — denial_risk.risk_level matches expected_denial_risk (optional)

Key functions:
    load_test_cases         — parse YAML test cases
    check_must_contain      — OR match, case-insensitive
    check_must_not_contain  — any forbidden word = fail
    check_confidence_max    — confidence_score <= threshold
    check_escalate          — escalate bool matches expected
    check_denial_risk       — risk_level string matches expected
    run_eval                — full runner returning scored result dict

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import os
import sys
import json
import time
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional

# Load .env before importing any module that reads API keys.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables being set

sys.path.insert(0, _REPO_ROOT)

from langgraph_agent.workflow import run_workflow
from verification import should_escalate_to_human


DEFAULT_GOLDEN_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "golden_data.yaml",
)

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "results",
)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_test_cases(path: str) -> List[Dict]:
    """
    Load and return test cases from a Gauntlet-format YAML file.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        List[Dict]: List of test case dicts. Returns empty list on any error.
    """
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        cases = data.get("test_cases", [])
        if not isinstance(cases, list):
            return []
        return cases
    except FileNotFoundError:
        return []
    except Exception:
        return []


# ── Scoring functions ──────────────────────────────────────────────────────────

def check_must_contain(response: str, keywords: List[str]) -> bool:
    """
    Return True if any keyword appears in response (OR match, case-insensitive).

    Args:
        response: Agent response text.
        keywords: List of required keyword strings.

    Returns:
        bool: True if at least one keyword found, or keywords list is empty.
    """
    if not keywords:
        return True
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in keywords)


def check_must_not_contain(response: str, forbidden: List[str]) -> bool:
    """
    Return True (safe) if no forbidden word appears in response (case-insensitive).

    Args:
        response: Agent response text.
        forbidden: List of forbidden keyword strings.

    Returns:
        bool: True if response is clean, False if any forbidden word found.
    """
    if not forbidden:
        return True
    response_lower = response.lower()
    return not any(kw.lower() in response_lower for kw in forbidden)


def check_confidence_max(confidence: float, expected_max: Optional[float]) -> bool:
    """
    Return True if confidence_score is at or below expected_confidence_max.
    Skipped (True) if expected_confidence_max is not specified in the test case.

    Args:
        confidence: Actual confidence_score from run_workflow result.
        expected_max: Maximum allowed confidence (e.g. 0.55 for Scenario A).

    Returns:
        bool: True if within threshold or not specified.
    """
    if expected_max is None:
        return True
    return confidence <= expected_max


def check_escalate(actual_escalate: bool, expected_escalate: Optional[bool]) -> bool:
    """
    Return True if the escalate flag matches expected_escalate.
    Skipped (True) if expected_escalate is not specified in the test case.

    Args:
        actual_escalate: Computed escalation flag from verification layer.
        expected_escalate: Expected value (True or False).

    Returns:
        bool: True if matches or not specified.
    """
    if expected_escalate is None:
        return True
    return actual_escalate == expected_escalate


def check_denial_risk(actual_risk: Optional[str], expected_risk: Optional[str]) -> bool:
    """
    Return True if denial_risk.risk_level matches expected_denial_risk.
    Skipped (True) if expected_denial_risk is not specified in the test case.

    Args:
        actual_risk: risk_level string from denial_risk result (e.g. "HIGH").
        expected_risk: Expected risk level string.

    Returns:
        bool: True if matches or not specified.
    """
    if expected_risk is None:
        return True
    return (actual_risk or "").upper() == expected_risk.upper()


# ── Runner ────────────────────────────────────────────────────────────────────

def run_eval(
    test_cases_path: str = DEFAULT_GOLDEN_DATA_PATH,
    save_results: bool = True,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict:
    """
    Run the full eval suite via the LangGraph workflow and return scored results.

    Loads test cases from YAML, runs each through run_workflow(), scores on
    five dimensions, computes pass rate, and optionally saves timestamped results.

    Args:
        test_cases_path: Path to golden_data.yaml.
        save_results: Whether to write results to results_dir.
        results_dir: Directory to save timestamped result files.

    Returns:
        Dict: {total, passed, failed, pass_rate, results, timestamp}

    Raises:
        Never — individual test failures are caught and recorded.
    """
    test_cases = load_test_cases(test_cases_path)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    per_case_results = []
    passed = 0
    failed = 0

    for case in test_cases:
        case_id = case.get("id", "unknown")
        query = case.get("query", "")
        must_contain = case.get("must_contain", [])
        must_not_contain = case.get("must_not_contain", [])
        pdf_source_file = case.get("pdf_source_file")
        expected_confidence_max = case.get("expected_confidence_max")
        expected_escalate = case.get("expected_escalate")
        expected_denial_risk = case.get("expected_denial_risk")

        start = time.time()
        try:
            result = run_workflow(
                query=query,
                pdf_source_file=pdf_source_file,
            )
            # Use clarification_needed as fallback when final_response is empty.
            # This covers Scenario B (unknown patient hard-stop) and
            # any clarification-node paths where final_response is not set.
            response_text = result.get("final_response") or result.get("clarification_needed", "")
            confidence = float(result.get("confidence_score", 0.0))
            escalate = should_escalate_to_human(confidence_score=confidence).get("escalate", True)
            denial_risk_level = (
                result.get("denial_risk", {}).get("risk_level")
                if result.get("denial_risk")
                else None
            )
        except Exception as e:
            response_text = f"Agent error: {str(e)}"
            confidence = 0.0
            escalate = True   # conservative default on error
            denial_risk_level = None

        latency = round(time.time() - start, 3)

        contain_ok = check_must_contain(response_text, must_contain)
        not_contain_ok = check_must_not_contain(response_text, must_not_contain)
        confidence_ok = check_confidence_max(confidence, expected_confidence_max)
        escalate_ok = check_escalate(escalate, expected_escalate)
        denial_ok = check_denial_risk(denial_risk_level, expected_denial_risk)

        case_passed = (
            contain_ok
            and not_contain_ok
            and confidence_ok
            and escalate_ok
            and denial_ok
        )

        status = "PASS" if case_passed else "FAIL"
        if case_passed:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {case_id} ({latency}s)")
        if not contain_ok:
            print(f"       must_contain FAILED — none of {must_contain} in response")
        if not not_contain_ok:
            print(f"       must_not_contain FAILED — forbidden word found in response")
        if not confidence_ok:
            print(f"       confidence FAILED — got {confidence:.2f}, expected <= {expected_confidence_max}")
        if not escalate_ok:
            print(f"       escalate FAILED — got {escalate}, expected {expected_escalate}")
        if not denial_ok:
            print(f"       denial_risk FAILED — got {denial_risk_level!r}, expected {expected_denial_risk!r}")

        per_case_results.append({
            "id": case_id,
            "category": case.get("category", ""),
            "description": case.get("description", ""),
            "passed": case_passed,
            "scores": {
                "must_contain": contain_ok,
                "must_not_contain": not_contain_ok,
                "confidence_max": confidence_ok,
                "escalate": escalate_ok,
                "denial_risk": denial_ok,
            },
            "actual": {
                "confidence": confidence,
                "escalate": escalate,
                "denial_risk_level": denial_risk_level,
            },
            "latency_seconds": latency,
            "response_preview": response_text[:300],
        })

    total = passed + failed
    pass_rate = round(passed / total, 4) if total > 0 else 0.0

    result_summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "results": per_case_results,
        "timestamp": timestamp,
    }

    print(f"\n===== Eval complete: {passed}/{total} passed ({pass_rate * 100:.1f}%) =====")

    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        filename = f"eval_results_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, "w") as f:
                json.dump(result_summary, f, indent=2)
            print(f"Results saved to {filepath}")
        except Exception as e:
            print(f"Warning: could not save results: {e}")

    return result_summary


if __name__ == "__main__":
    run_eval()
