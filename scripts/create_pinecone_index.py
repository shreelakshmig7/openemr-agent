"""
create_pinecone_index.py
------------------------
AgentForge — Healthcare RCM AI Agent
Run once before upsert_policies.py.
Safe to re-run — skips if index already exists.

Usage:
    python scripts/create_pinecone_index.py
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)

from pinecone import Pinecone, ServerlessSpec


def create_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set in .env")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    index_name = os.getenv("PINECONE_INDEX", "agentforge-rcm-policies")
    region = os.getenv("PINECONE_ENV", "us-east-1")

    # voyage-2 outputs 1024 dims — NOT 1536 (OpenAI)
    DIMENSION = 1024

    existing = pc.list_indexes().names()
    if index_name in existing:
        print(f"Index '{index_name}' already exists — skipping creation")
    else:
        print(f"Creating index '{index_name}' (dim={DIMENSION}, region={region})...")
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region)
        )
        print("Index created — waiting for ready state...")

    # Poll until ready
    max_wait = 120
    waited = 0
    while waited < max_wait:
        status = pc.describe_index(index_name).status
        if status.get("ready"):
            print(f"Index '{index_name}' is ready.")
            return
        print(f"  Waiting... ({waited}s)")
        time.sleep(5)
        waited += 5

    print(f"WARNING: Index not ready after {max_wait}s — check Pinecone console")


if __name__ == "__main__":
    create_index()
