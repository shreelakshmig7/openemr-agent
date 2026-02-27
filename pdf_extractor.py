"""
pdf_extractor.py
----------------
AgentForge — Healthcare RCM AI Agent — PDF Extractor Tool
----------------------------------------------------------
Extracts structured clinical text from scanned or digital PDF documents
using the unstructured.io API. Returns verbatim quotes with page numbers
and source file attribution for every element, ensuring the auditor node
can verify every claim against its exact source location.

When UNSTRUCTURED_API_KEY is not set, returns a structured error with an
empty extractions list — the tool never raises an exception to the caller.

Key functions:
    extract_pdf: Main entry point — validates inputs, calls API, returns extractions.
    _call_unstructured_api: Wraps the unstructured-client API call (mockable in tests).
    _build_extraction_result: Converts raw API elements to the standard extraction format.
    _is_api_available: Returns True if UNSTRUCTURED_API_KEY is present and non-empty.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

UNSTRUCTURED_SERVER_URL = "https://api.unstructuredapp.io"


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

    Args:
        el: A raw element returned by the unstructured API — either a dict or an object.

    Returns:
        str: The text content, or empty string if not present.

    Raises:
        Never.
    """
    if isinstance(el, dict):
        return el.get("text", "") or ""
    return getattr(el, "text", "") or ""


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

def _call_unstructured_api(source_file: str) -> List[Any]:
    """
    Call the unstructured.io API to partition a PDF file into structured elements.

    Reads the file from disk and sends it to the unstructured API using the
    official unstructured-client SDK. Returns the raw list of elements.

    Args:
        source_file: Absolute or relative path to the PDF file on disk.

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

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=file_content,
                file_name=file_name,
            ),
            strategy=shared.Strategy.AUTO,
        ),
    )

    res = client.general.partition(request=req)
    return res.elements or []


# ── Result builder ────────────────────────────────────────────────────────────

def _build_extraction_result(raw_elements: List[Any], source_file: str) -> dict:
    """
    Convert raw unstructured API elements into the standard extraction format.

    Filters out elements with empty or whitespace-only text. For each valid
    element, produces a dict with verbatim_quote, page_number, element_type,
    and source_file. Handles both dict-style and object-style raw elements.

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

    Raises:
        Never — returns success: True with empty extractions on any element-level failure.
    """
    extractions: List[dict] = []

    for el in raw_elements:
        try:
            text = _get_element_text(el)
            if not text or not text.strip():
                continue
            extractions.append({
                "verbatim_quote": text,
                "page_number": _get_page_number(el),
                "element_type": _get_element_type(el),
                "source_file": source_file,
            })
        except Exception as e:
            logger.warning("Skipping malformed element during extraction: %s", e)
            continue

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

    Validates that the API key is set and the source_file is non-empty before
    making any API call. On success, returns all extracted elements as verbatim
    quotes with page numbers and source attribution. On failure, returns a
    structured error dict — never raises an exception to the caller.

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

        raw_elements = _call_unstructured_api(source_file)
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
