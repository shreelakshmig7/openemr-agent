"""
upsert_policies.py
------------------
AgentForge — Healthcare RCM AI Agent
Generates Voyage AI embeddings for all payer policy chunks
and upserts them to Pinecone. Safe to re-run — skips existing vectors.

Usage:
    python scripts/upsert_policies.py
"""

import os
import sys
import json

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)

from pinecone import Pinecone
import voyageai
from mock_data.payer_policies_raw import PAYER_POLICIES

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "agentforge-rcm-policies"))
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

EMBEDDING_MODEL = "voyage-2"  # 1024 dims — matches index


def embed_batch(texts: list) -> list:
    """
    Embed a batch of texts using Voyage AI.
    Uses input_type='document' for upsert (vs 'query' for search).
    Voyage-2 max batch size = 8.
    """
    result = voyage_client.embed(
        texts,
        model=EMBEDDING_MODEL,
        input_type="document"
    )
    return result.embeddings


def upsert_policies():
    # Flatten all chunks
    all_chunks = [
        chunk
        for chunks in PAYER_POLICIES.values()
        for chunk in chunks
    ]

    # Check which vectors already exist — skip them
    all_ids = [f"{c['payer']}_{c['section']}" for c in all_chunks]
    existing = index.fetch(ids=all_ids)
    existing_ids = set(existing.vectors.keys())

    to_embed = [
        c for c in all_chunks
        if f"{c['payer']}_{c['section']}" not in existing_ids
    ]

    if not to_embed:
        print("All vectors already exist — nothing to upsert")
        return

    print(f"Embedding {len(to_embed)} chunks (skipping {len(existing_ids)} existing)...")

    vectors = []
    batch_size = 8  # Voyage-2 max batch

    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_batch(texts)

        for chunk, embedding in zip(batch, embeddings):
            vector_id = f"{chunk['payer']}_{chunk['section']}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "payer": chunk["payer"],
                    "policy_id": chunk["policy_id"],
                    "procedure_codes": json.dumps(chunk["procedure_codes"]),
                    "criteria_id": chunk["criteria_id"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                }
            })
        print(f"  Embedded batch {i // batch_size + 1} ({len(batch)} chunks)")

    # Upsert in batches of 100 (Pinecone limit)
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i + 100]
        index.upsert(vectors=batch)
        print(f"  Upserted {len(batch)} vectors")

    print(f"\nDone — {len(vectors)} vectors upserted total")


if __name__ == "__main__":
    upsert_policies()
