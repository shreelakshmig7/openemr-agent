# AgentForge — Setup Guide

**Project:** AgentForge Healthcare RCM AI Agent  
**Author:** Shreelakshmi Gopinatha Rao  
**Repository:** [openemr-agent](https://github.com/shreelakshmigopinatharao/openemr)  
**Deployed:** [Railway](https://openemr-agent-production.up.railway.app)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Environment Variables](#4-environment-variables)
5. [OpenEMR Docker Setup](#5-openemr-docker-setup)
6. [Running the Agent Server](#6-running-the-agent-server)
7. [Verifying the Setup](#7-verifying-the-setup)
8. [Optional Services](#8-optional-services)
9. [Running Tests and Evals](#9-running-tests-and-evals)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Python** | 3.11+ | 3.12 recommended |
| **pip** | 23+ | Comes with Python |
| **Docker** | 24+ | Required for OpenEMR |
| **Docker Compose** | v2.20+ | Bundled with Docker Desktop |
| **Git** | Any recent version | For cloning the repo |

**API Keys required:**

| Key | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** | Powers Claude Haiku (routing) and Claude Sonnet (synthesis) |
| `UNSTRUCTURED_API_KEY` | No | PDF extraction via unstructured.io; mock fallback available |
| `LANGSMITH_API_KEY` | No | Observability tracing; agent runs without it |
| `PINECONE_API_KEY` + `VOYAGE_API_KEY` | No | Payer policy vector search; keyword mock fallback available |

---

## 2. Project Structure

```
openemr-agent/
├── main.py                        # FastAPI server — entry point
├── openemr_client.py              # OpenEMR FHIR R4 async client
├── pdf_extractor.py               # PDF extraction via unstructured.io
├── database.py                    # SQLite session + evidence staging
├── healthcare_guidelines.py       # Clinical safety rules + disclaimer
├── langgraph_agent/
│   ├── workflow.py                # LangGraph graph definition + run_workflow()
│   ├── state.py                   # AgentState TypedDict
│   ├── router_node.py             # Intent classifier (Haiku)
│   ├── orchestrator_node.py       # Tool planner + patient name extractor (Haiku)
│   ├── extractor_node.py          # Tool executor
│   ├── auditor_node.py            # Citation verifier + response synthesizer (Sonnet)
│   ├── clarification_node.py      # Pause/resume for ambiguous queries
│   ├── comparison_node.py         # PDF vs EHR diff; sets HITL sync flag
│   └── sync_execution_node.py     # Writes approved markers to OpenEMR FHIR
├── tools/
│   ├── __init__.py                # Tool registry (8 tools)
│   └── pii_scrubber.py            # Microsoft Presidio PII redaction
├── mock_data/
│   ├── patients.json              # 12 demo patients (FHIR fallback)
│   ├── medications.json           # Demo medication list
│   └── interactions.json          # Drug interaction pairs
├── eval/
│   ├── golden_data.yaml           # 63 labeled test cases
│   └── run_eval.py                # Eval runner
├── static/
│   └── index.html                 # Chat UI (served by FastAPI)
├── scripts/
│   └── seed_production.sql        # OpenEMR seed data (12 patients)
├── tests/
│   └── test_identity_resolution.py
├── agent_checkpoints.sqlite       # LangGraph state persistence (auto-created)
└── uploads/                       # PDF upload directory (auto-created)
```

---

## 3. Installation

### 3.1 Clone the Repository

```bash
git clone https://github.com/shreelakshmigopinatharao/openemr.git
cd openemr/openemr-agent
```

### 3.2 Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3.3 Install Python Dependencies

```bash
pip install --upgrade pip
pip install \
  fastapi \
  uvicorn[standard] \
  langgraph \
  langchain \
  langchain-anthropic \
  langsmith \
  anthropic \
  httpx \
  pydantic \
  python-dotenv \
  pyyaml \
  unstructured-client \
  presidio-analyzer \
  presidio-anonymizer \
  spacy
```

Install the spaCy English language model (required for Presidio NLP-based PII detection):

```bash
python -m spacy download en_core_web_lg
```

**Optional — Pinecone policy search:**

```bash
pip install pinecone-client voyageai
```

### 3.4 Create the `.env` File

Copy the template below and save it as `.env` in the `openemr-agent/` directory:

```bash
# openemr-agent/.env
# Required
ANTHROPIC_API_KEY=sk-ant-...

# OpenEMR connection (defaults shown — change if using a remote instance)
OPENEMR_BASE_URL=https://localhost:9300
OPENEMR_USERNAME=admin
OPENEMR_PASSWORD=pass
# Leave blank to use auto-registration (recommended for local dev)
OPENEMR_CLIENT_ID=
OPENEMR_CLIENT_SECRET=

# Optional — LangSmith observability
LANGSMITH_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentforge-rcm

# Optional — PDF extraction via unstructured.io
UNSTRUCTURED_API_KEY=...

# Optional — Pinecone payer policy vector search
PINECONE_API_KEY=...
VOYAGE_API_KEY=...
USE_REAL_PINECONE=false     # set to "true" to use real Pinecone; "false" uses keyword mock

# Optional — Audit trail endpoint auth
AUDIT_TOKEN=your-secret-token
```

> **Security note:** Never commit `.env` to version control. Add it to `.gitignore`.

---

## 4. Environment Variables

Full reference of all supported environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** | — | Anthropic API key; powers Claude Haiku and Sonnet |
| `OPENEMR_BASE_URL` | No | `https://localhost:9300` | Base URL of the OpenEMR instance |
| `OPENEMR_USERNAME` | No | `admin` | OpenEMR login username |
| `OPENEMR_PASSWORD` | No | `pass` | OpenEMR login password |
| `OPENEMR_CLIENT_ID` | No | auto-registered | OAuth2 client ID; auto-registered if blank |
| `OPENEMR_CLIENT_SECRET` | No | `""` | OAuth2 client secret |
| `LANGSMITH_API_KEY` | No | — | LangSmith observability; agent runs without it |
| `LANGCHAIN_API_KEY` | No | — | Alias for `LANGSMITH_API_KEY` |
| `LANGCHAIN_TRACING_V2` | No | — | Set to `"true"` to enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | No | — | LangSmith project name (e.g. `agentforge-rcm`) |
| `UNSTRUCTURED_API_KEY` | No | — | unstructured.io API key; mock fallback used if absent |
| `PINECONE_API_KEY` | No | — | Pinecone API key for payer policy vector search |
| `VOYAGE_API_KEY` | No | — | Voyage AI API key for policy embeddings |
| `USE_REAL_PINECONE` | No | `false` | Set `"true"` to use live Pinecone; `"false"` uses keyword mock |
| `AUDIT_TOKEN` | No | — | Bearer token for `GET /api/audit/{thread_id}`; endpoint disabled if unset |

---

## 5. OpenEMR Docker Setup

The agent reads live patient data from an OpenEMR FHIR R4 instance running in Docker. This step is optional — all tools fall back to `mock_data/` when OpenEMR is unavailable.

### 5.1 Start OpenEMR

From the root of the repository (not `openemr-agent/`):

```bash
cd docker/development-easy
docker compose up --detach --wait
```

Wait 60–90 seconds for OpenEMR to fully initialize (database migrations run on first boot).

**Default credentials:**

| Field | Value |
|---|---|
| App URL | http://localhost:8300 / https://localhost:9300 |
| Username | `admin` |
| Password | `pass` |
| phpMyAdmin | http://localhost:8310 |

### 5.2 Enable FHIR API and OAuth Password Grant

The OpenEMR Docker image used by this project ships with FHIR and OAuth already configured via environment variables. If you are connecting to a different OpenEMR instance, verify these settings in **Administration → Globals → Connectors**:

```
oauth_password_grant = 3      # password grant enabled
rest_fhir_api = 1             # FHIR REST API enabled
site_addr_oath = https://localhost:9300
```

### 5.3 Seed Demo Patients (optional)

To load the 12 demo patients used in the eval suite:

```bash
docker compose exec openemr mysql -u openemr -popenemr openemr \
  < ../../openemr-agent/scripts/seed_production.sql
```

### 5.4 Test the FHIR Connection

```bash
curl -k https://localhost:9300/fhir/Patient \
  -H "Authorization: Bearer $(python3 -c "
import asyncio, os, sys
sys.path.insert(0, 'openemr-agent')
from openemr_client import OpenEMRClient
async def t():
    async with OpenEMRClient() as c:
        print(c._token)
asyncio.run(t())
")"
```

If the connection fails, the agent automatically falls back to `mock_data/patients.json`.

---

## 6. Running the Agent Server

From the `openemr-agent/` directory (virtual environment activated):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts on **http://localhost:8000**.

**Production (no reload):**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use `--workers 1` only — the LangGraph SQLite checkpointer is not safe for multi-process sharing.

---

## 7. Verifying the Setup

### 7.1 Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "service": "AgentForge Healthcare RCM AI Agent",
  "version": "2.0.0",
  "status": "ok",
  "timestamp": "2026-03-02T00:00:00+00:00"
}
```

### 7.2 Chat UI

Open **http://localhost:8000** in a browser. You should see the AgentForge chat interface.

### 7.3 Send a Test Query

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What medications is John Smith taking?"}' \
  | python3 -m json.tool
```

A successful response includes `answer`, `confidence`, `tool_trace`, and `citation_anchors` fields.

### 7.4 Streaming (SSE)

```bash
curl -s -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What medications is John Smith taking?"}' \
  --no-buffer
```

Each line is an SSE event (`event: node`, `event: done`, or `event: error`).

---

## 8. Optional Services

### 8.1 LangSmith Observability

1. Create an account at [smith.langchain.com](https://smith.langchain.com).
2. Generate an API key under **Settings → API Keys**.
3. Add to `.env`:

```bash
LANGSMITH_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentforge-rcm
```

All graph node executions, LLM calls, token usage, and latency appear in the LangSmith dashboard automatically. The agent runs identically without this key — tracing is simply skipped.

### 8.2 PDF Extraction (unstructured.io)

Without this key, `pdf_extractor` returns a structured error and PDF queries are not processed.

1. Sign up at [unstructured.io](https://unstructured.io) and obtain an API key.
2. Add to `.env`:

```bash
UNSTRUCTURED_API_KEY=...
```

### 8.3 Pinecone Payer Policy Search

Without Pinecone, `policy_search` falls back to a keyword-matching mock.

1. Create a Pinecone index and a Voyage AI API key.
2. Add to `.env`:

```bash
PINECONE_API_KEY=...
VOYAGE_API_KEY=...
USE_REAL_PINECONE=true
```

### 8.4 PII Scrubbing (Presidio)

Presidio is used automatically if installed (see Section 3.3). When unavailable, a 5-regex fallback runs instead. The regex fallback catches `PERSON`, `DATE`, `PHONE_NUMBER`, `EMAIL`, and `US_SSN` patterns — sufficient for demo use. Install Presidio for full HIPAA-grade coverage.

---

## 9. Running Tests and Evals

### 9.1 Unit Tests

```bash
cd openemr-agent
python -m pytest tests/ -v
```

Expected: **16/16 tests passing** (tool registry unit tests).

### 9.2 PII Scrubber Verification

```bash
python scripts/verify_pii_scrubber.py
```

This script tests the Presidio pipeline against sample PHI strings and reports which entities are detected.

### 9.3 Eval Suite (63 golden test cases)

Run via the API (server must be running):

```bash
curl -s -X POST http://localhost:8000/eval | python3 -m json.tool
```

Or run directly from the command line:

```bash
python -c "
from eval.run_eval import run_eval, DEFAULT_GOLDEN_DATA_PATH
result = run_eval(DEFAULT_GOLDEN_DATA_PATH, save_results=True)
print(f\"Passed: {result['passed']}/{result['total']} ({result['pass_rate']:.1%})\")
"
```

> Requires `ANTHROPIC_API_KEY`. Each case makes 1–3 LLM calls. Full suite takes ~5–15 minutes and costs approximately \$0.50–\$2.00 depending on model routing.

### 9.4 Retrieve Latest Eval Results

```bash
curl http://localhost:8000/eval/results | python3 -m json.tool
```

Results are also saved to `tests/results/eval_results_<timestamp>.json`.

---

## 10. Troubleshooting

### `ANTHROPIC_API_KEY` not found

```
anthropic.AuthenticationError: 401 Unauthorized
```

Ensure your `.env` file is in the `openemr-agent/` directory and the server is started from that directory. The server calls `load_dotenv()` on startup.

### OpenEMR FHIR returns 401 / connection refused

The agent falls back to `mock_data/` automatically. If you want live FHIR data:

- Confirm Docker is running: `docker ps | grep openemr`
- Wait 90 seconds after `docker compose up` for the DB to initialize
- Verify FHIR is enabled: `curl -k https://localhost:9300/fhir/metadata`

### `ModuleNotFoundError: No module named 'presidio_analyzer'`

Presidio is not installed. Either install it:

```bash
pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_lg
```

Or ignore — the pipeline uses a regex fallback automatically.

### SQLite `database is locked` error

Only one server process can access `agent_checkpoints.sqlite` at a time. Ensure you are not running multiple `uvicorn` worker processes (`--workers 1`).

### PDF upload fails

- Confirm the `uploads/` directory exists (auto-created on startup).
- Only `.pdf` files are accepted.
- File size is limited by uvicorn's default (~16 MB); increase with `--limit-concurrency` if needed.

### Streaming endpoint hangs

SSE requires the client to support chunked transfer encoding. Use `curl --no-buffer` or a proper EventSource client. Some reverse proxies buffer SSE — set `X-Accel-Buffering: no` on your proxy.

### LangSmith traces not appearing

- Confirm `LANGSMITH_API_KEY` is set and `LANGCHAIN_TRACING_V2=true`.
- Check the LangSmith project name matches `LANGCHAIN_PROJECT`.
- Traces are sent asynchronously — allow 30–60 seconds after a query.

---

## API Reference (Quick)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check |
| `/` | GET | Chat UI (HTML) |
| `/ask` | POST | Natural language query; returns full JSON response |
| `/ask/stream` | POST | Same as `/ask` but streams SSE progress per node |
| `/upload` | POST | Upload clinical PDF; returns `{"path": "uploads/..."}` |
| `/history` | GET | Recent session list (for audit sidebar) |
| `/history/{session_id}/messages` | GET | Full message transcript for a session |
| `/api/audit/{thread_id}` | GET | Audit trail (requires `Authorization: Bearer <AUDIT_TOKEN>`) |
| `/eval` | POST | Run 63-case eval suite; returns pass/fail results |
| `/eval/results` | GET | Latest saved eval results |
| `/pdf` | GET | Serve a PDF file inline (uploads/ or mock_data/ only) |

---

## Architecture Summary

```
User Query (natural language)
    │
    ▼
FastAPI (main.py) ──► LangGraph State Machine
                            │
                    ┌───────┴────────┐
                    ▼                ▼
              Router Node      (OUT_OF_SCOPE → refuse)
            (Claude Haiku)
                    │
                    ▼
           Orchestrator Node   ─── "yes/sync" ──► Sync Execution Node
            (Claude Haiku)                         (writes to OpenEMR FHIR)
                    │
                    ▼
            Extractor Node     ◄─── retry loop (max 3x)
          (runs 8 tools)
                    │
                    ▼
            Auditor Node       ─── citation verification (Claude Sonnet)
                    │
                    ▼
            Output Node        ─── synthesized + cited response
                    │
               SQLite checkpoint saved after every node
```

For full architecture details, see [`agent-architecture-doc.md`](./agent-architecture-doc.md).  
For design decisions and UI blueprint, see [`AgentForge_Design_Prototype.md`](./AgentForge_Design_Prototype.md).
