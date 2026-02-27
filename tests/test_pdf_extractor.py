"""
test_pdf_extractor.py
---------------------
AgentForge — Healthcare RCM AI Agent — Tests for pdf_extractor.py
------------------------------------------------------------------
TDD test suite for the pdf_extractor tool. Tests are written before the
implementation to confirm each test fails first, then passes after
implementation.

Tests run in two modes:
  - With UNSTRUCTURED_API_KEY set: validates real API call structure.
  - Without UNSTRUCTURED_API_KEY: validates graceful fallback behaviour.

No test is skipped regardless of API key presence — the tool must handle
both states correctly.

Run:
    pytest tests/test_pdf_extractor.py -v --tb=short

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdf_extractor import extract_pdf, _build_extraction_result, _is_api_available

API_KEY_SET = bool(os.getenv("UNSTRUCTURED_API_KEY"))


# ── _is_api_available ─────────────────────────────────────────────────────────

def test_is_api_available_with_key(monkeypatch):
    """Returns True when UNSTRUCTURED_API_KEY is set in environment."""
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "test-key-123")
    assert _is_api_available() is True


def test_is_api_available_without_key(monkeypatch):
    """Returns False when UNSTRUCTURED_API_KEY is not set."""
    monkeypatch.delenv("UNSTRUCTURED_API_KEY", raising=False)
    assert _is_api_available() is False


def test_is_api_available_empty_key(monkeypatch):
    """Returns False when UNSTRUCTURED_API_KEY is set but empty."""
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "")
    assert _is_api_available() is False


# ── _build_extraction_result ──────────────────────────────────────────────────

def test_build_extraction_result_structure():
    """_build_extraction_result must return dict with all required keys."""
    raw_elements = [
        {"type": "NarrativeText", "text": "Patient has hypertension.", "metadata": {"page_number": 1}},
    ]
    result = _build_extraction_result(raw_elements, source_file="test.pdf")
    assert "success" in result
    assert "extractions" in result
    assert "source_file" in result
    assert "element_count" in result
    assert "error" in result


def test_build_extraction_result_success_true():
    """_build_extraction_result with valid elements returns success: True."""
    raw_elements = [
        {"type": "NarrativeText", "text": "Patient diagnosed with Type 2 Diabetes.", "metadata": {"page_number": 1}},
    ]
    result = _build_extraction_result(raw_elements, source_file="clinical_note.pdf")
    assert result["success"] is True
    assert result["error"] is None


def test_build_extraction_result_extractions_are_list():
    """extractions field must always be a list."""
    raw_elements = [
        {"type": "NarrativeText", "text": "Patient prescribed Metformin 500mg.", "metadata": {"page_number": 2}},
    ]
    result = _build_extraction_result(raw_elements, source_file="note.pdf")
    assert isinstance(result["extractions"], list)


def test_build_extraction_result_each_extraction_has_required_fields():
    """Each extraction must have verbatim_quote, page_number, element_type, and source_file."""
    raw_elements = [
        {"type": "NarrativeText", "text": "Diagnosis: hypertension confirmed.", "metadata": {"page_number": 3}},
    ]
    result = _build_extraction_result(raw_elements, source_file="report.pdf")
    for ext in result["extractions"]:
        assert "verbatim_quote" in ext
        assert "page_number" in ext
        assert "element_type" in ext
        assert "source_file" in ext


def test_build_extraction_result_verbatim_quote_matches_input():
    """verbatim_quote must exactly match the text from the raw element."""
    text = "Patient has a Penicillin allergy."
    raw_elements = [
        {"type": "NarrativeText", "text": text, "metadata": {"page_number": 1}},
    ]
    result = _build_extraction_result(raw_elements, source_file="note.pdf")
    assert result["extractions"][0]["verbatim_quote"] == text


def test_build_extraction_result_source_file_preserved():
    """source_file in each extraction must match the source_file argument."""
    raw_elements = [
        {"type": "NarrativeText", "text": "Some text.", "metadata": {"page_number": 1}},
    ]
    result = _build_extraction_result(raw_elements, source_file="discharge_summary.pdf")
    assert result["source_file"] == "discharge_summary.pdf"
    for ext in result["extractions"]:
        assert ext["source_file"] == "discharge_summary.pdf"


def test_build_extraction_result_empty_elements():
    """Empty elements list must return success: True with empty extractions list."""
    result = _build_extraction_result([], source_file="empty.pdf")
    assert result["success"] is True
    assert result["extractions"] == []
    assert result["element_count"] == 0


def test_build_extraction_result_filters_empty_text():
    """Elements with empty or whitespace-only text must be filtered out."""
    raw_elements = [
        {"type": "NarrativeText", "text": "Real clinical content.", "metadata": {"page_number": 1}},
        {"type": "NarrativeText", "text": "   ", "metadata": {"page_number": 1}},
        {"type": "NarrativeText", "text": "", "metadata": {"page_number": 2}},
    ]
    result = _build_extraction_result(raw_elements, source_file="note.pdf")
    assert result["element_count"] == 1
    assert len(result["extractions"]) == 1


def test_build_extraction_result_page_number_defaults_to_none():
    """Element without page_number in metadata must default page_number to None (not crash)."""
    raw_elements = [
        {"type": "NarrativeText", "text": "Some content.", "metadata": {}},
    ]
    result = _build_extraction_result(raw_elements, source_file="note.pdf")
    assert result["success"] is True
    assert result["extractions"][0]["page_number"] is None


def test_build_extraction_result_element_count_matches():
    """element_count must equal the number of non-empty extractions."""
    raw_elements = [
        {"type": "NarrativeText", "text": "First sentence.", "metadata": {"page_number": 1}},
        {"type": "NarrativeText", "text": "Second sentence.", "metadata": {"page_number": 1}},
        {"type": "NarrativeText", "text": "", "metadata": {"page_number": 1}},
    ]
    result = _build_extraction_result(raw_elements, source_file="note.pdf")
    assert result["element_count"] == 2
    assert len(result["extractions"]) == 2


# ── extract_pdf — no API key (graceful fallback) ──────────────────────────────

def test_extract_pdf_no_api_key_returns_structured_error(monkeypatch):
    """When UNSTRUCTURED_API_KEY is not set, must return success: False with clear error — not crash."""
    monkeypatch.delenv("UNSTRUCTURED_API_KEY", raising=False)
    result = extract_pdf("clinical_note.pdf")
    assert result["success"] is False
    assert "error" in result
    assert result["error"] is not None
    assert len(result["error"]) > 0


def test_extract_pdf_no_api_key_has_required_keys(monkeypatch):
    """Fallback result must still contain all standard keys even without API key."""
    monkeypatch.delenv("UNSTRUCTURED_API_KEY", raising=False)
    result = extract_pdf("note.pdf")
    assert "success" in result
    assert "extractions" in result
    assert "source_file" in result
    assert "element_count" in result
    assert "error" in result


def test_extract_pdf_no_api_key_extractions_is_list(monkeypatch):
    """extractions must be an empty list when API key is missing — not None."""
    monkeypatch.delenv("UNSTRUCTURED_API_KEY", raising=False)
    result = extract_pdf("note.pdf")
    assert isinstance(result["extractions"], list)
    assert result["extractions"] == []


def test_extract_pdf_empty_source_returns_error(monkeypatch):
    """Empty source_file string must return success: False with a clear error."""
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "test-key")
    result = extract_pdf("")
    assert result["success"] is False
    assert result["error"] is not None


# ── extract_pdf — with mocked API response ────────────────────────────────────

def test_extract_pdf_mocked_api_success(monkeypatch):
    """With API key set and mocked API response, must return success: True with extractions."""
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "mock-key-for-test")

    mock_elements = [
        MagicMock(
            category="NarrativeText",
            text="Patient John Smith presents with hypertension.",
            metadata=MagicMock(page_number=1),
        ),
        MagicMock(
            category="NarrativeText",
            text="Current medications: Lisinopril 10mg once daily.",
            metadata=MagicMock(page_number=1),
        ),
    ]

    with patch("pdf_extractor._call_unstructured_api", return_value=mock_elements):
        result = extract_pdf("clinical_note.pdf")

    assert result["success"] is True
    assert result["element_count"] == 2
    assert result["source_file"] == "clinical_note.pdf"
    assert len(result["extractions"]) == 2


def test_extract_pdf_mocked_verbatim_quotes(monkeypatch):
    """verbatim_quote in each extraction must exactly match the mocked element text."""
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "mock-key-for-test")

    text1 = "Diagnosis: Type 2 Diabetes confirmed by HbA1c 8.2%."
    text2 = "Allergy: Penicillin — documented anaphylaxis reaction."

    mock_elements = [
        MagicMock(category="NarrativeText", text=text1, metadata=MagicMock(page_number=1)),
        MagicMock(category="NarrativeText", text=text2, metadata=MagicMock(page_number=2)),
    ]

    with patch("pdf_extractor._call_unstructured_api", return_value=mock_elements):
        result = extract_pdf("note.pdf")

    quotes = [e["verbatim_quote"] for e in result["extractions"]]
    assert text1 in quotes
    assert text2 in quotes


def test_extract_pdf_mocked_api_failure_returns_structured_error(monkeypatch):
    """If the unstructured API call raises an exception, result must be success: False — not a crash."""
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "mock-key-for-test")

    with patch("pdf_extractor._call_unstructured_api", side_effect=Exception("API timeout")):
        result = extract_pdf("clinical_note.pdf")

    assert result["success"] is False
    assert "error" in result
    assert result["extractions"] == []


def test_extract_pdf_mocked_empty_api_response(monkeypatch):
    """If the API returns an empty list, result must be success: True with empty extractions."""
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "mock-key-for-test")

    with patch("pdf_extractor._call_unstructured_api", return_value=[]):
        result = extract_pdf("empty_doc.pdf")

    assert result["success"] is True
    assert result["extractions"] == []
    assert result["element_count"] == 0
