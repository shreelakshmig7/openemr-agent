"""
test_eval.py
-------------
AgentForge — Healthcare RCM AI Agent — Test Suite for eval/run_eval.py
-----------------------------------------------------------------------
TDD unit tests for the eval runner. Uses a mocked agent so no API calls
are made. Tests cover: YAML loading, must_contain scoring, must_not_contain
scoring, pass rate calculation, results structure, and file saving.

Tests are written BEFORE run_eval.py is implemented — they will fail first,
then pass once run_eval.py is built correctly.

Run:
    pytest tests/test_eval.py -v --tb=short

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
import os
import sys
import yaml
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLDEN_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval",
    "golden_data.yaml",
)


# ── Imports ────────────────────────────────────────────────────────────────────

def test_run_eval_imports():
    """run_eval module should expose a run_eval function."""
    from eval.run_eval import run_eval
    assert run_eval is not None


def test_load_test_cases_imports():
    """run_eval module should expose a load_test_cases function."""
    from eval.run_eval import load_test_cases
    assert load_test_cases is not None


# ── load_test_cases ────────────────────────────────────────────────────────────

def test_load_test_cases_returns_list():
    """load_test_cases should return a list of dicts from golden_data.yaml."""
    from eval.run_eval import load_test_cases
    cases = load_test_cases(GOLDEN_DATA_PATH)
    assert isinstance(cases, list)
    assert len(cases) >= 5


def test_load_test_cases_required_fields():
    """Every test case must have the required Gauntlet fields."""
    from eval.run_eval import load_test_cases
    cases = load_test_cases(GOLDEN_DATA_PATH)
    required = {"id", "category", "query", "expected_tools", "must_contain", "must_not_contain", "difficulty"}
    for case in cases:
        for field in required:
            assert field in case, f"Case {case.get('id', '?')} missing field: {field}"


def test_load_test_cases_invalid_path():
    """load_test_cases with a bad path should return empty list, not crash."""
    from eval.run_eval import load_test_cases
    result = load_test_cases("/nonexistent/path/golden_data.yaml")
    assert isinstance(result, list)
    assert len(result) == 0


# ── Scoring logic ──────────────────────────────────────────────────────────────

def test_check_must_contain_pass():
    """check_must_contain should return True when any keyword appears in response."""
    from eval.run_eval import check_must_contain
    assert check_must_contain("Patient takes metformin and lisinopril.", ["metformin", "atorvastatin"]) is True


def test_check_must_contain_fail():
    """check_must_contain should return False when no keyword appears in response."""
    from eval.run_eval import check_must_contain
    assert check_must_contain("I cannot find that patient.", ["metformin", "lisinopril"]) is False


def test_check_must_not_contain_pass():
    """check_must_not_contain should return True (safe) when no forbidden word appears."""
    from eval.run_eval import check_must_not_contain
    assert check_must_not_contain("Physician review required.", ["approved", "error"]) is True


def test_check_must_not_contain_fail():
    """check_must_not_contain should return False (unsafe) when a forbidden word appears."""
    from eval.run_eval import check_must_not_contain
    assert check_must_not_contain("This is approved and safe to proceed.", ["approved", "error"]) is False


def test_check_must_contain_case_insensitive():
    """must_contain check should be case-insensitive."""
    from eval.run_eval import check_must_contain
    assert check_must_contain("Patient takes METFORMIN daily.", ["metformin"]) is True


def test_check_must_not_contain_case_insensitive():
    """must_not_contain check should be case-insensitive."""
    from eval.run_eval import check_must_not_contain
    assert check_must_not_contain("Request APPROVED.", ["approved"]) is False


# ── run_eval with mocked agent ─────────────────────────────────────────────────

class MockAgentPass:
    """Fake agent that always returns a response containing must_contain keywords."""
    def invoke(self, inputs: dict) -> dict:
        return {"output": "Patient takes metformin and lisinopril. Physician review recommended. Not found any unknown."}


class MockAgentFail:
    """Fake agent that always returns a response that fails scoring."""
    def invoke(self, inputs: dict) -> dict:
        return {"output": "I don't know and there was an error."}


def _minimal_yaml(tmp_path: str, response_text: str) -> str:
    """Write a minimal single-case YAML file for testing."""
    case = {
        "test_cases": [{
            "id": "gs-test",
            "category": "happy_path",
            "query": "What medications is John Smith on?",
            "expected_tools": ["tool_get_patient_info"],
            "must_contain": ["metformin", "lisinopril"],
            "must_not_contain": ["I don't know", "error"],
            "difficulty": "happy_path",
        }]
    }
    path = os.path.join(tmp_path, "test_cases.yaml")
    with open(path, "w") as f:
        yaml.dump(case, f)
    return path


def test_run_eval_returns_required_keys():
    """run_eval result dict must contain total, passed, failed, pass_rate, results, timestamp."""
    from eval.run_eval import run_eval
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = _minimal_yaml(tmp, "metformin lisinopril")
        result = run_eval(yaml_path, agent=MockAgentPass(), save_results=False)
    required_keys = {"total", "passed", "failed", "pass_rate", "results", "timestamp"}
    for key in required_keys:
        assert key in result, f"Missing key in result: {key}"


def test_run_eval_pass_rate_all_pass():
    """run_eval with a passing mock agent should return pass_rate of 1.0."""
    from eval.run_eval import run_eval
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = _minimal_yaml(tmp, "metformin")
        result = run_eval(yaml_path, agent=MockAgentPass(), save_results=False)
    assert result["pass_rate"] == 1.0
    assert result["passed"] == 1
    assert result["failed"] == 0


def test_run_eval_pass_rate_all_fail():
    """run_eval with a failing mock agent should return pass_rate of 0.0."""
    from eval.run_eval import run_eval
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = _minimal_yaml(tmp, "nothing here")
        result = run_eval(yaml_path, agent=MockAgentFail(), save_results=False)
    assert result["pass_rate"] == 0.0
    assert result["failed"] == 1
    assert result["passed"] == 0


def test_run_eval_per_case_results():
    """run_eval results list should include pass/fail and latency for each case."""
    from eval.run_eval import run_eval
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = _minimal_yaml(tmp, "metformin")
        result = run_eval(yaml_path, agent=MockAgentPass(), save_results=False)
    assert len(result["results"]) == 1
    case_result = result["results"][0]
    assert "id" in case_result
    assert "passed" in case_result
    assert "latency_seconds" in case_result


def test_run_eval_saves_timestamped_file():
    """run_eval with save_results=True should write a timestamped file."""
    from eval.run_eval import run_eval
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "results"
    )
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = _minimal_yaml(tmp, "metformin")
        run_eval(yaml_path, agent=MockAgentPass(), save_results=True, results_dir=results_dir)
    saved = [f for f in os.listdir(results_dir) if f.startswith("eval_results_")]
    assert len(saved) >= 1
