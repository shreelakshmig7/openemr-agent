"""
run_eval.py
-----------
AgentForge — Healthcare RCM AI Agent — Eval runner
---------------------------------------------------
Loads test cases from golden_data.yaml, runs each through the agent,
scores must_contain and must_not_contain keywords, computes pass rate,
and saves timestamped results. Supports injecting a mock agent for
unit testing.

Key functions:
    - load_test_cases: parse YAML test cases
    - check_must_contain: OR match, case-insensitive
    - check_must_not_contain: any forbidden word = fail
    - run_eval: full runner returning scored result dict

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_GOLDEN_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "golden_data.yaml",
)

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "results",
)


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


def check_must_contain(response: str, keywords: List[str]) -> bool:
    """
    Return True if any keyword appears in response (OR match, case-insensitive).

    Args:
        response: Agent response text.
        keywords: List of required keyword strings.

    Returns:
        bool: True if at least one keyword found.
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


def _normalize_output(raw: Any) -> str:
    """
    Normalize agent output (str or list of blocks) to a plain string.

    Args:
        raw: Raw output from agent.invoke()["output"].

    Returns:
        str: Plain text.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(raw)


def run_eval(
    test_cases_path: str = DEFAULT_GOLDEN_DATA_PATH,
    agent: Optional[Any] = None,
    save_results: bool = True,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict:
    """
    Run the full eval suite and return scored results.

    Loads test cases from YAML, runs each through the agent, scores
    must_contain and must_not_contain, computes pass rate, and
    optionally saves timestamped results to disk.

    Args:
        test_cases_path: Path to golden_data.yaml.
        agent: AgentExecutor to use. If None, creates a fresh one per test.
        save_results: Whether to write results to results_dir.
        results_dir: Directory to save timestamped result files.

    Returns:
        Dict: {total, passed, failed, pass_rate, results, timestamp}
    """
    from agent import create_agent

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

        start = time.time()
        try:
            if agent is not None:
                run_agent = agent
            else:
                run_agent = create_agent()

            raw_response = run_agent.invoke({"input": query})
            response_text = _normalize_output(raw_response.get("output"))
        except Exception as e:
            response_text = f"Agent error: {str(e)}"

        latency = round(time.time() - start, 3)

        contain_ok = check_must_contain(response_text, must_contain)
        not_contain_ok = check_must_not_contain(response_text, must_not_contain)
        case_passed = contain_ok and not_contain_ok

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

        per_case_results.append({
            "id": case_id,
            "category": case.get("category", ""),
            "passed": case_passed,
            "contain_ok": contain_ok,
            "not_contain_ok": not_contain_ok,
            "latency_seconds": latency,
            "response_preview": response_text[:200],
        })

    total = passed + failed
    pass_rate = round(passed / total, 4) if total > 0 else 0.0

    result = {
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
                json.dump(result, f, indent=2)
            print(f"Results saved to {filepath}")
        except Exception as e:
            print(f"Warning: could not save results: {e}")

    return result


if __name__ == "__main__":
    run_eval()
