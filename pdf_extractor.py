"""
pdf_extractor.py
----------------
AgentForge — Healthcare RCM AI Agent — PDF Extractor Tool
----------------------------------------------------------
Extracts structured clinical text from scanned or digital PDF documents
using the unstructured.io API. Returns verbatim quotes with page numbers
and source file attribution for every element, ensuring the auditor node
can verify every claim against its exact source location.

Design decisions (v2):
  1. All element types are captured — Table and UncategorizedText are no
     longer silently dropped. Table text is surfaced from the `text` field
     with `metadata.text_as_html` as a fallback so drug-interaction tables
     and prior-auth criteria grids are never omitted.
  2. Every extraction chunk carries a required `page_number` metadata field.
     This powers the Logic Roll-up UI and lets the auditor node cite specific
     pages when validating claims.
  3. OCR sensitivity: hi_res strategy is always used so scanned / image-based
     prior-auth documents (common in RCM) are fully OCR'd. Falling back to
     auto for retry if hi_res fails keeps the system resilient.

When UNSTRUCTURED_API_KEY is not set, returns a structured error with an
empty extractions list — the tool never raises an exception to the caller.

Key functions:
    extract_pdf: Main entry point — validates inputs, calls API, returns extractions.
    _call_unstructured_api: Wraps the unstructured-client API call (mockable in tests).
    _build_extraction_result: Converts raw API elements to the standard extraction format.
    _is_api_available: Returns True if UNSTRUCTURED_API_KEY is present and non-empty.
    _get_table_text: Extracts text from Table elements including HTML fallback.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import logging
import os
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

UNSTRUCTURED_SERVER_URL = "https://api.unstructuredapp.io"

# Element types that the Unstructured API may return. We capture ALL of them —
# no type is excluded, so tables, uncategorized text, and list items are all
# surfaced to the agent.
_TABLE_TYPES = {"Table", "FigureCaption"}


# ── API availability check ────────────────────────────────────────────────────

def _is_api_available() -> bool:
    """
    Return True if the UNSTRUCTURED_API_KEY environment variable is set and non-empty.

    Args:
        None

    Returns:
        bool: True if the API key is available, False otherwise.

    Raises:
        Never.
    """
    key = os.getenv("UNSTRUCTURED_API_KEY", "")
    return bool(key and key.strip())


# ── Element attribute helpers (handle dict and object style) ──────────────────

def _get_element_text(el: Any) -> str:
    """
    Extract the text content from a raw element, handling dict and object formats.

    For Table elements, falls back to metadata.text_as_html (stripped of tags)
    when the plain text field is absent or empty, ensuring table rows are not lost.

    Args:
        el: A raw element returned by the unstructured API — either a dict or an object.

    Returns:
        str: The text content, or empty string if not present.

    Raises:
        Never.
    """
    element_type = _get_element_type(el)

    if isinstance(el, dict):
        text = el.get("text", "") or ""
        if not text.strip() and element_type in _TABLE_TYPES:
            text = _get_table_text(el)
        return text
    text = getattr(el, "text", "") or ""
    if not text.strip() and element_type in _TABLE_TYPES:
        text = _get_table_text(el)
    return text


def _get_table_text(el: Any) -> str:
    """
    Extract text from a Table element, using metadata.text_as_html as a fallback.

    The Unstructured API stores a structured HTML table in metadata.text_as_html
    even when the plain-text field is empty. This function strips HTML tags and
    collapses whitespace so the resulting string is readable by the LLM and can
    be used as a verbatim citation.

    Args:
        el: A Table-type element returned by the unstructured API.

    Returns:
        str: Plain-text representation of the table, or empty string.

    Raises:
        Never.
    """
    try:
        if isinstance(el, dict):
            meta = el.get("metadata", {})
            html = (meta.get("text_as_html", "") if isinstance(meta, dict) else "") or ""
        else:
            meta = getattr(el, "metadata", None) or {}
            if isinstance(meta, dict):
                html = meta.get("text_as_html", "") or ""
            else:
                html = getattr(meta, "text_as_html", "") or ""

        if not html.strip():
            return ""

        # Strip HTML tags and collapse whitespace runs to single spaces/newlines.
        text = re.sub(r"<tr[^>]*>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<td[^>]*>|<th[^>]*>", " | ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception as exc:
        logger.warning("_get_table_text failed: %s", exc)
        return ""


def _get_element_type(el: Any) -> str:
    """
    Extract the element type/category from a raw element.

    Args:
        el: A raw element returned by the unstructured API — either a dict or an object.

    Returns:
        str: The element type (e.g. NarrativeText, Title, Table), or empty string.

    Raises:
        Never.
    """
    if isinstance(el, dict):
        return el.get("type", "") or el.get("category", "") or ""
    return getattr(el, "type", "") or getattr(el, "category", "") or ""


def _get_page_number(el: Any) -> Optional[int]:
    """
    Extract the page number from a raw element's metadata.

    Every extraction chunk must carry a page_number so the Logic Roll-up UI
    can anchor citations to exact pages and the auditor can validate them.

    Args:
        el: A raw element returned by the unstructured API — either a dict or an object.

    Returns:
        Optional[int]: The page number, or None if not present.

    Raises:
        Never.
    """
    try:
        if isinstance(el, dict):
            meta = el.get("metadata", {})
            if isinstance(meta, dict):
                return meta.get("page_number")
            return getattr(meta, "page_number", None)
        meta = getattr(el, "metadata", None)
        if meta is None:
            return None
        if isinstance(meta, dict):
            return meta.get("page_number")
        return getattr(meta, "page_number", None)
    except Exception:
        return None


# ── API caller (isolated for mocking in tests) ────────────────────────────────

def _call_unstructured_api(source_file: str, strategy: str = "hi_res") -> List[Any]:
    """
    Call the unstructured.io API to partition a PDF file into structured elements.

    Uses hi_res strategy by default to apply full OCR on scanned / image-based
    prior-auth documents, ensuring no clinical text or table data is omitted.
    Falls back to auto strategy on retry when hi_res raises an error.

    Args:
        source_file: Absolute or relative path to the PDF file on disk.
        strategy: Unstructured partition strategy — "hi_res" or "auto".

    Returns:
        List[Any]: Raw elements returned by the API (dicts or objects).

    Raises:
        FileNotFoundError: If source_file does not exist on disk.
        Exception: Any SDK or network error is re-raised so extract_pdf can catch it.
    """
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import operations, shared

    api_key = os.getenv("UNSTRUCTURED_API_KEY", "").strip()
    client = UnstructuredClient(
        api_key_auth=api_key,
        server_url=UNSTRUCTURED_SERVER_URL,
    )

    with open(source_file, "rb") as f:
        file_content = f.read()

    file_name = os.path.basename(source_file)

    # Map strategy string to SDK enum. hi_res triggers layout-aware OCR so
    # tables and image-embedded text are fully extracted.
    strategy_enum = shared.Strategy.HI_RES if strategy == "hi_res" else shared.Strategy.AUTO

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=file_content,
                file_name=file_name,
            ),
            strategy=strategy_enum,
        ),
    )

    res = client.general.partition(request=req)
    return res.elements or []


# ── Result builder ────────────────────────────────────────────────────────────

def _build_extraction_result(raw_elements: List[Any], source_file: str) -> dict:
    """
    Convert raw unstructured API elements into the standard extraction format.

    All element types are included — Table, UncategorizedText, NarrativeText,
    Title, ListItem, etc. Each extraction carries a mandatory page_number field
    for Logic Roll-up UI anchoring and auditor citation validation.

    Args:
        raw_elements: List of raw elements from _call_unstructured_api or a mock.
        source_file: The PDF file path or name — propagated to each extraction.

    Returns:
        dict: {
            "success": bool,
            "extractions": List[dict],   # one per non-empty element
            "source_file": str,
            "element_count": int,
            "error": None
        }
        Each extraction dict contains:
            "verbatim_quote": str    — full text of the element
            "page_number":    int|None — page where element appears
            "element_type":   str    — Unstructured element category
            "source_file":    str    — originating PDF path

    Raises:
        Never — returns success: True with empty extractions on any element-level failure.
    """
    extractions: List[dict] = []

    for el in raw_elements:
        try:
            text = _get_element_text(el)
            if not text or not text.strip():
                continue
            page_number = _get_page_number(el)
            element_type = _get_element_type(el)
            extractions.append({
                "verbatim_quote": text.strip(),
                "page_number": page_number,
                "element_type": element_type,
                "source_file": source_file,
            })
        except Exception as e:
            logger.warning("Skipping malformed element during extraction: %s", e)
            continue

    logger.info(
        "PDF extraction complete: %d elements from '%s' (%d raw)",
        len(extractions),
        os.path.basename(source_file),
        len(raw_elements),
    )
    return {
        "success": True,
        "extractions": extractions,
        "source_file": source_file,
        "element_count": len(extractions),
        "error": None,
    }


# ── Public tool function ──────────────────────────────────────────────────────

def extract_pdf(source_file: str) -> dict:
    """
    Extract structured clinical text from a PDF using the unstructured.io API.

    Uses hi_res OCR strategy to fully extract image-based prior-auth documents.
    Captures ALL element types (NarrativeText, Table, UncategorizedText, etc.)
    so that drug-interaction tables and criteria grids are not silently dropped.
    Falls back from hi_res to auto strategy on retry when the remote API rejects
    hi_res (e.g. unsupported file format).

    Args:
        source_file: Path to the PDF file to extract. Can be an absolute path
                     or a path relative to the working directory.

    Returns:
        dict: {
            "success": bool,
            "extractions": List[dict],   # empty list on failure
            "source_file": str,
            "element_count": int,
            "error": str | None          # human-readable error message on failure
        }

    Raises:
        Never — all failures return a structured error dict with success: False.
    """
    try:
        if not source_file or not source_file.strip():
            return {
                "success": False,
                "extractions": [],
                "source_file": source_file,
                "element_count": 0,
                "error": "source_file cannot be empty.",
            }

        if not _is_api_available():
            return {
                "success": False,
                "extractions": [],
                "source_file": source_file,
                "element_count": 0,
                "error": (
                    "UNSTRUCTURED_API_KEY is not set. "
                    "Add it to your .env file to enable PDF extraction. "
                    "Sign up at https://unstructured.io to get an API key."
                ),
            }

        # Attempt hi_res first; fall back to auto if hi_res is rejected.
        try:
            logger.info("PDF extraction starting with hi_res strategy: '%s'", source_file)
            raw_elements = _call_unstructured_api(source_file, strategy="hi_res")
        except Exception as hi_res_exc:
            logger.warning(
                "hi_res strategy failed for '%s' (%s) — retrying with auto strategy.",
                source_file,
                hi_res_exc,
            )
            raw_elements = _call_unstructured_api(source_file, strategy="auto")

        return _build_extraction_result(raw_elements, source_file)

    except FileNotFoundError:
        return {
            "success": False,
            "extractions": [],
            "source_file": source_file,
            "element_count": 0,
            "error": f"File not found: '{source_file}'. Verify the path and try again.",
        }
    except Exception as e:
        logger.exception("extract_pdf failed for '%s': %s", source_file, e)
        return {
            "success": False,
            "extractions": [],
            "source_file": source_file,
            "element_count": 0,
            "error": f"PDF extraction failed: {str(e)}",
        }
