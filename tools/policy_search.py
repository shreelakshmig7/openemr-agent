"""
policy_search.py
----------------
AgentForge — Healthcare RCM AI Agent — Policy Search Tool
----------------------------------------------------------
RAG search over payer policy criteria using Pinecone + Voyage AI embeddings.
Embeds the query context with Voyage AI, retrieves top matching policy
criteria chunks from Pinecone, and returns structured criteria match results.

Falls back to keyword-based mock search when USE_REAL_PINECONE is not set,
so the tool always returns a valid result dict regardless of environment.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_USE_PINECONE = (
    bool(os.getenv("PINECONE_API_KEY"))
    and os.getenv("USE_REAL_PINECONE", "false").lower() == "true"
)


# ── Lazy client init ──────────────────────────────────────────────────────────

_pinecone_index = None
_voyage_client = None


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _pinecone_index = pc.Index(
            os.getenv("PINECONE_INDEX", "agentforge-rcm-policies")
        )
    return _pinecone_index


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai
        _voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    return _voyage_client


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_query(text: str) -> list:
    """
    Embed query text using Voyage AI voyage-2.
    Uses input_type='query' for retrieval — different from 'document' used at upsert.
    """
    try:
        client = _get_voyage_client()
        result = client.embed(
            [text],
            model="voyage-2",
            input_type="query"
        )
        return result.embeddings[0]
    except Exception as e:
        logger.error("Voyage embedding failed: %s", e)
        return []


# ── Query builder ─────────────────────────────────────────────────────────────

def _build_query_text(
    payer_id: str,
    procedure_code: str,
    extractions: list
) -> str:
    """Build a rich semantic query string from payer, procedure, and clinical extractions."""
    claims = " ".join(
        e.get("claim", "")
        for e in extractions
        if e.get("claim") and not e.get("synthetic")
    )[:500]

    return (
        f"Payer: {payer_id}. "
        f"Procedure code: {procedure_code}. "
        f"Clinical evidence: {claims}"
    )


# ── Evidence helpers ──────────────────────────────────────────────────────────

def _criterion_supported_by_evidence(
    criterion_text: str,
    all_claims_text: str
) -> bool:
    """Check if clinical evidence supports a criterion using keyword overlap (20% threshold)."""
    keywords = [
        w.lower().strip(".,;:()")
        for w in criterion_text.split()
        if len(w) > 5
    ]
    if not keywords:
        return False
    matches = sum(1 for kw in keywords if kw in all_claims_text)
    return (matches / len(keywords)) >= 0.20


def _find_supporting_evidence(
    criterion_text: str,
    extractions: list
) -> str:
    """Find the most relevant extraction claim to cite as evidence for a criterion."""
    criterion_words = set(
        w.lower().strip(".,;:()")
        for w in criterion_text.split()
        if len(w) > 5
    )
    for e in extractions:
        claim = e.get("claim", "")
        claim_words = set(claim.lower().split())
        if len(criterion_words & claim_words) >= 2:
            return claim[:200]
    return "See clinical documentation"


def _error_result(error_msg: str) -> dict:
    return {
        "success": False,
        "policy_id": None,
        "criteria_met": [],
        "criteria_unmet": [],
        "source": "pinecone",
        "error": error_msg,
    }


# ── Real Pinecone search ──────────────────────────────────────────────────────

def _search_pinecone(
    payer_id: str,
    procedure_code: str,
    extractions: list
) -> dict:
    """
    Query Pinecone for matching policy criteria using Voyage AI embeddings.
    Filters by payer in metadata. Scores > 0.75 + keyword evidence = criteria met.
    """
    try:
        query_text = _build_query_text(payer_id, procedure_code, extractions)
        query_vector = _embed_query(query_text)

        if not query_vector:
            return _error_result("Voyage embedding failed — check VOYAGE_API_KEY")

        index = _get_pinecone_index()

        results = index.query(
            vector=query_vector,
            top_k=10,
            filter={"payer": {"$eq": payer_id.lower()}},
            include_metadata=True
        )

        if not results.matches:
            return {
                "success": True,
                "policy_id": None,
                "criteria_met": [],
                "criteria_unmet": [],
                "payer": payer_id,
                "procedure_code": procedure_code,
                "source": "pinecone",
                "message": f"No policy found for payer '{payer_id}'",
                "error": None,
            }

        all_claims_text = " ".join(
            e.get("claim", "").lower()
            for e in extractions
            if e.get("claim")
        )

        policy_id = results.matches[0].metadata.get("policy_id", "Unknown Policy")
        criteria_met = []
        criteria_unmet = []
        seen_criteria = set()

        for match in results.matches:
            meta = match.metadata
            criteria_id = meta.get("criteria_id", "")
            chunk_text = meta.get("text", "")

            if criteria_id in seen_criteria:
                continue
            seen_criteria.add(criteria_id)

            is_met = (
                match.score > 0.75
                and _criterion_supported_by_evidence(chunk_text, all_claims_text)
            )

            entry = {
                "id": criteria_id,
                "description": chunk_text[:200],
                "score": round(match.score, 3),
                "evidence": _find_supporting_evidence(chunk_text, extractions),
            }

            if is_met:
                criteria_met.append(entry)
            else:
                criteria_unmet.append(entry)

        return {
            "success": True,
            "policy_id": policy_id,
            "criteria_met": criteria_met,
            "criteria_unmet": criteria_unmet,
            "payer": payer_id,
            "procedure_code": procedure_code,
            "source": "pinecone",
            "error": None,
        }

    except Exception as e:
        logger.error("Pinecone search failed: %s", e)
        return _error_result(str(e))


# ── Mock fallback ─────────────────────────────────────────────────────────────

def _search_mock(
    payer_id: str,
    procedure_code: str,
    extractions: list
) -> dict:
    """
    Keyword-based policy search against payer_policies_raw.
    Used when USE_REAL_PINECONE is not set or Pinecone is unavailable.
    """
    try:
        from mock_data.payer_policies_raw import PAYER_POLICIES

        payer_data = PAYER_POLICIES.get(payer_id.lower(), [])
        if not payer_data:
            return {
                "success": True,
                "policy_id": None,
                "criteria_met": [],
                "criteria_unmet": [],
                "payer": payer_id,
                "procedure_code": procedure_code,
                "source": "mock",
                "message": f"No mock policy for payer '{payer_id}'",
                "error": None,
            }

        all_claims = " ".join(
            e.get("claim", "").lower()
            for e in extractions
            if e.get("claim")
        )

        policy_id = payer_data[0]["policy_id"]
        criteria_met = []
        criteria_unmet = []

        for chunk in payer_data:
            keywords = [
                w.lower()
                for w in chunk["text"].split()
                if len(w) > 5
            ]
            match_count = sum(1 for kw in keywords if kw in all_claims)
            is_met = bool(keywords) and (match_count / len(keywords)) >= 0.20

            entry = {
                "id": chunk["criteria_id"],
                "description": chunk["text"][:200],
                "evidence": _find_supporting_evidence(chunk["text"], extractions),
            }

            if is_met:
                criteria_met.append(entry)
            else:
                criteria_unmet.append(entry)

        return {
            "success": True,
            "policy_id": policy_id,
            "criteria_met": criteria_met,
            "criteria_unmet": criteria_unmet,
            "payer": payer_id,
            "procedure_code": procedure_code,
            "source": "mock",
            "error": None,
        }

    except Exception as e:
        return _error_result(f"Mock search failed: {e}")


# ── Public interface ──────────────────────────────────────────────────────────

def search_policy(
    payer_id: str,
    procedure_code: str,
    extractions: list
) -> dict:
    """
    Search payer policy criteria for a given procedure and clinical evidence.
    Routes to real Pinecone when USE_REAL_PINECONE=true, mock otherwise.
    """
    if _USE_PINECONE:
        return _search_pinecone(payer_id, procedure_code, extractions)
    return _search_mock(payer_id, procedure_code, extractions)
