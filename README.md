# AgentForge — Healthcare AI Agent

**Author:** Shreelakshmi Gopinatha Rao  
**Program:** Gauntlet AI — G4 Week 2  
**Domain:** Healthcare Medication Safety  
**Live Demo:** [Railway Deployment](https://openemr-agent-production.up.railway.app)

---

## What It Does

A healthcare RCM (Revenue Cycle Management) AI agent that helps clinical staff review patient medication histories, check payer policy criteria, and flag denial risk. It accepts natural language queries, runs a LangGraph pipeline (router → orchestrator → extractor → auditor), and calls tools against mock (or real) data. Responses are cited and never from memory.

**Features:** Patient/medication lookup · Drug interaction & allergy checks · PDF extraction (clinical notes) · Payer policy search (Pinecone or mock) · Denial risk analysis · Multi-turn with clarification

**Tech stack:** FastAPI · LangGraph · LangChain · Claude (Anthropic) · LangSmith observability

---

## Quick Start (Local)

```bash
# 1. Clone and enter the project
cd openemr-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY, UNSTRUCTURED_API_KEY (for PDFs), and optionally LANGSMITH_API_KEY

# 5. Run the server
uvicorn main:app --reload --port 8000

# 6. Open the chat UI
open http://localhost:8000
```

---

## Deploy on Railway

The app is already set up for Railway: **Procfile** runs `uvicorn main:app --host 0.0.0.0 --port $PORT`.

### 1. Set environment variables

In the Railway project → **Variables**, add (no `.env` file is deployed):

| Variable | Required | Notes |
|----------|----------|--------|
| `ANTHROPIC_API_KEY` | Yes | From [console.anthropic.com](https://console.anthropic.com/) |
| `UNSTRUCTURED_API_KEY` | Yes* | For PDF extraction; [unstructured.io](https://unstructured.io) |
| `LANGSMITH_API_KEY` | No | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable tracing |
| `LANGCHAIN_PROJECT` | No | e.g. `openemr-agent` |
| `PINECONE_API_KEY` | No | Only if using real Pinecone policy search |
| `PINECONE_ENV` | No | e.g. `us-east-1` |
| `PINECONE_INDEX` | No | e.g. `agentforge-rcm-policies` |
| `VOYAGE_API_KEY` | No | Only if using real Pinecone (embeddings) |
| `USE_REAL_PINECONE` | No | Set to `true` for Pinecone; omit or `false` for mock policy search |

\* Without `UNSTRUCTURED_API_KEY`, PDF extraction will fail; the rest of the agent still works.

### 2. Optional: Python version

To pin the runtime, add a **runtime.txt** in the project root:
```
python-3.11.6
```
Otherwise Railway infers from `requirements.txt`.

### 3. Limitations on Railway

- **Ephemeral filesystem:** SQLite checkpoints (`agent_checkpoints.sqlite`) and in-memory sessions are lost on redeploy or restart. Multi-turn context works within a single run but does not persist across restarts.
- **PDF uploads:** Uploaded PDFs are written to the app’s `uploads/` directory. If you run **multiple replicas**, the instance that serves `POST /upload` may differ from the one that serves `POST /ask`; the second instance won’t have the file, so the agent may respond “I did not have access to that document.” Use a **single replica** for PDF queries, or add shared storage (e.g. S3) and change the app to read/write PDFs there.
- **Policy search:** With `USE_REAL_PINECONE=false` (or unset), policy search uses the mock backend; no Pinecone/Voyage keys needed.

---

## Mock Patient Database

All patient data lives in `mock_data/`. There are **11 patients** available for testing.

### Patients & Their Medications

| ID | Name | Key Medications | Allergies | What to Test |
|----|------|-----------------|-----------|--------------|
| P001 | John Smith | Metformin, Lisinopril, Atorvastatin | Penicillin, Sulfa | Standard multi-med lookup |
| P002 | Mary Johnson | Methotrexate, Folic Acid, Prednisone | Aspirin, Ibuprofen | Allergy conflict with condition |
| P003 | Robert Davis | Warfarin, Digoxin, Furosemide | None | No-allergy patient, HIGH interactions |
| P004 | Sarah Chen | Sertraline, Tramadol, Lisinopril | Codeine, Sulfa | **CONTRAINDICATED** interaction |
| P005 | James Wilson | Metformin, Atorvastatin, Allopurinol, Amlodipine | Penicillin | LOW interaction (Atorvastatin + Amlodipine) |
| P006 | Emily Rodriguez | Amoxicillin, Lisinopril | Penicillin | **Allergy conflict** — prescribed a Penicillin-class drug |
| P007 | David Kim | Lithium, Ibuprofen, Levothyroxine | None | **HIGH** interaction (Lithium + Ibuprofen) |
| P008 | Patricia Moore | Alendronate, Lisinopril, Calcium Carbonate | Aspirin, Codeine | LOW interaction (Calcium Carbonate + Alendronate) |
| P009 | Alex Turner | *(no medications on record)* | None | Edge case: no meds found |
| P010 | Maria Santos | *(empty medication list)* | None | Edge case: empty med record |
| P011 | Thomas Lee | *(no medications on record)* | None | Edge case: no meds found |

### Interaction Rules (mock_data/interactions.json)

| Drug Pair | Severity |
|-----------|----------|
| Warfarin + Aspirin | HIGH |
| Methotrexate + Aspirin | HIGH |
| Warfarin + Ibuprofen | HIGH |
| Lithium + Ibuprofen | HIGH |
| Sertraline + Tramadol | CONTRAINDICATED |
| Atorvastatin + Amlodipine | LOW |
| Calcium Carbonate + Alendronate | LOW |
| Digoxin + Furosemide | MODERATE |
| Warfarin + Digoxin | MODERATE |

---

## Example Queries to Test

Try these in the live chat UI or via the API:

```
# Basic lookup
"What medications is John Smith taking?"
"Look up patient P001"

# Interaction check
"Check drug interactions for Robert Davis"
"Does Sarah Chen have any dangerous drug interactions?"

# Allergy check
"Does Emily Rodriguez have any allergies?"
"What are Mary Johnson's allergies?"

# Follow-up (multi-turn context)
"Tell me about David Kim"
→ "Does he have any drug interactions?"   ← agent must re-call tools, not answer from memory

# Policy search (use request body with payer_id + procedure_code for policy_search)
"Does John Smith meet Cigna criteria for knee replacement?"
"Does John Smith meet BlueCross criteria for knee replacement?"  ← unknown payer → no-policy message

# Edge cases
"What medications is Alex Turner on?"     ← no record found
"What meds does Maria Santos take?"       ← empty record
"Look up patient P999"                    ← patient not found
```

---

## API Reference

### `GET /`
Serves the web chat UI.

### `GET /health`
Returns server status.
```json
{"service": "AgentForge Healthcare RCM AI Agent", "version": "2.0.0", "status": "ok", "timestamp": "2025-02-27T12:00:00.000000Z"}
```

### `POST /ask`
Send a question to the agent.
```json
{
  "question": "What medications is John Smith taking?",
  "session_id": "optional-uuid-for-multi-turn",
  "thread_id": "optional-same-as-session_id",
  "pdf_source_file": "optional-path-e.g-uploads/note.pdf",
  "payer_id": "optional-e.g-cigna",
  "procedure_code": "optional-e.g-27447"
}
```
Response:
```json
{
  "answer": "John Smith is currently taking...",
  "session_id": "uuid",
  "thread_id": "uuid",
  "timestamp": "2025-02-27T12:00:00.000000Z",
  "escalate": false,
  "confidence": 0.9,
  "disclaimer": "...",
  "tool_trace": [...],
  "denial_risk": {"risk_level": "NONE", "matched_patterns": [], ...},
  "citation_anchors": []
}
```

### `POST /upload`
Upload a PDF (multipart/form-data). Returns `{ "success": true, "path": "uploads/filename.pdf", "filename": "..." }` for use as `pdf_source_file` in `POST /ask`.

### `GET /pdf`
Serve an uploaded PDF by path (query param or path). Used by the UI to display documents.

### `GET /api/audit/{thread_id}`
Return audit trail for a thread (messages, extractions, tool_call_history). Requires header `Authorization: Bearer <AUDIT_TOKEN>`. Set `AUDIT_TOKEN` in env to enable.

### `POST /eval`
Run the automated evaluation suite against all golden test cases. Requires `ANTHROPIC_API_KEY`.

### `GET /eval/results`
Retrieve the most recent evaluation results (from `tests/results/`).

---

## Running Tests

### Unit tests (no API key needed)
```bash
source venv/bin/activate
python -m pytest tests/test_tools.py tests/test_verification.py tests/test_denial_analyzer.py tests/test_pdf_extractor.py tests/test_langgraph_state.py tests/test_eval.py tests/test_main.py -v
```

### LangGraph / orchestrator tests (no API key for most)
```bash
python -m pytest tests/test_langgraph_orchestrator.py tests/test_langgraph_extractor.py tests/test_langgraph_workflow.py tests/test_langgraph_clarification.py tests/test_langgraph_auditor.py -v
```

### Conversation / agent tests (requires ANTHROPIC_API_KEY)
```bash
python -m pytest tests/test_conversation.py tests/test_agent.py -v
```

### Full evaluation suite (requires ANTHROPIC_API_KEY)
```bash
python eval/run_eval.py
```
Runs all cases in `eval/golden_data.yaml` (35 test cases). Results are saved to `tests/results/`.

---

## Project Structure

```
openemr-agent/
├── main.py                 # FastAPI server + API endpoints
├── Procfile                 # Railway: uvicorn main:app --host 0.0.0.0 --port $PORT
├── verification.py         # Allergy, confidence, FDA checks
├── healthcare_guidelines.py # Clinical safety rules
├── denial_analyzer.py       # Denial risk pattern matching
├── pdf_extractor.py         # PDF extraction (unstructured.io)
├── tools/                   # Tool implementations (package)
│   ├── __init__.py         # get_patient_info, get_medications, check_drug_interactions
│   └── policy_search.py    # Payer policy search (Pinecone or mock)
├── langgraph_agent/         # LangGraph state machine
│   ├── state.py            # AgentState, create_initial_state
│   ├── workflow.py         # run_workflow, graph build
│   ├── router_node.py      # Intent classification
│   ├── orchestrator_node.py# Tool plan, patient name extraction
│   ├── extractor_node.py   # Tool execution (patient, meds, PDF, policy, denial)
│   ├── auditor_node.py     # Verification, response synthesis
│   └── clarification_node.py
├── mock_data/
│   ├── patients.json       # Mock patients
│   ├── medications.json   # Medication records
│   ├── interactions.json  # Drug interaction rules
│   ├── denial_patterns.json
│   └── payer_policies_raw.py  # Raw policy chunks (Pinecone upsert / mock)
├── scripts/
│   ├── create_pinecone_index.py  # Create Pinecone index (voyage-2, 1024 dims)
│   └── upsert_policies.py        # Embed + upsert policy chunks to Pinecone
├── eval/
│   ├── golden_data.yaml    # Evaluation test cases (35 cases)
│   └── run_eval.py         # Evaluation runner
├── static/
│   └── index.html          # Web chat UI
├── legacy/                  # Legacy LangChain agent (optional)
│   ├── agent.py
│   └── conversation.py
└── tests/
    ├── test_tools.py
    ├── test_langgraph_*.py
    ├── test_verification.py
    ├── test_eval.py
    └── test_main.py
```

---

## Observability

All agent runs are traced in [LangSmith](https://smith.langchain.com). Each API response includes a `tool_trace` field showing exactly which tools were called, with what inputs, and what they returned — so evaluators can verify the agent is querying the database and not answering from memory.
