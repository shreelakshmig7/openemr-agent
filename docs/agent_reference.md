# AgentForge: Healthcare RCM Agent — Implementation Reference

**Author:** Shreelakshmi Gopinatha Rao
**Domain:** Healthcare Revenue Cycle Management (RCM)
**Framework:** LangGraph Multi-Agent State Machine
**LLM Strategy:** Claude Haiku (extraction) + Claude 3.5 Sonnet (reasoning/audit)
**Version:** 2.2 (Final Submission)
**Status:** Implementation reference — describes the current codebase state

---

## Purpose of This Document

This reference consolidates all five project documents (Pre-Search, PRD v2.2, Design Prototype v2.2, Coding Standards, and G4 Week 2 spec) and accurately describes the **current codebase implementation**. Each section reflects what is actually built, not aspirational plans.

### Current Implementation Status

| Component | Status | Notes |
| :--- | :--- | :--- |
| LangGraph Supervisory Router Node | ✅ Implemented | `langgraph_agent/router_node.py` — entry point of the graph |
| LangGraph Extractor Node | ✅ Implemented | `langgraph_agent/extractor_node.py` |
| LangGraph Auditor Node | ✅ Implemented | `langgraph_agent/auditor_node.py` |
| LangGraph Clarification Node | ✅ Implemented | `langgraph_agent/clarification_node.py` |
| LangGraph Output Node | ✅ Implemented | `langgraph_agent/workflow.py` — includes query-intent filtering |
| `get_patient_info` tool | ✅ Implemented | `tools.py` |
| `get_medications` tool | ✅ Implemented | `tools.py` |
| `check_drug_interactions` tool | ✅ Implemented | `tools.py` |
| FastAPI backend | ✅ Implemented | `main.py` |
| Chat UI | ✅ Implemented | `static/index.html` |
| Eval framework (18 cases) | ✅ Implemented | `eval/golden_data.yaml` + `eval/run_eval.py` |
| Verification layer | ✅ Implemented | `verification.py` |
| Healthcare constants | ✅ Implemented | `healthcare_guidelines.py` |
| PII scrubber | ⚠️ Stub | Regex-based stub in `extractor_node.py`; Microsoft Presidio not yet integrated |
| Ledger Write Node | ❌ Not yet built | No OpenEMR ledger integration |
| `evidence_ledger_write` tool | ❌ Not yet built | — |
| `pdf_extractor` tool | ❌ Not yet built | — |
| `policy_search` tool | ❌ Not yet built | — |
| `denial_analyzer` tool | ❌ Not yet built | — |
| `POST /api/rcm/verify` endpoint | ❌ Not yet built | — |
| `rcm_evidence_ledger` SQL table | ❌ Not yet built | — |
| SQLite state checkpointing | ❌ Not yet built | Graph rebuilt fresh per invocation |
| Weighted confidence scoring | ❌ Not yet built | Current: deduction-based formula |
| Multi-step eval cases | ❌ Not yet built | 0 of required 10+ |

---

## Part 1 — Core Architecture (Understand Before Writing Code)

### 1.1 What the Agent Does

AgentForge extracts clinical evidence from provider notes and verifies that evidence against insurance policy criteria to reduce claim denials. A wrong extraction costs $25–$200 per rework plus clinical risk from delayed care authorizations. Accuracy is the top constraint — not speed.

### 1.2 System Topology (Current)

```
[static/index.html Chat UI]  ──HTTP──►  [FastAPI — main.py]
                                               │
                                       [LangGraph State Machine]
                                       [langgraph_agent/workflow.py]
                                               │
                                      Entry point: extractor_node

LangGraph Node Flow (actual):

User Query (POST /ask)
    │
    ▼
┌───────────────────────┐
│  Supervisory Router   │  ── OUT_OF_SCOPE ──► Output Node ──► [END]
│  router_node.py       │     (pirate, weather, non-healthcare)
│  Classifies intent:   │     final_response = standard refusal
│  MEDICATIONS          │
│  ALLERGIES            │
│  INTERACTIONS         │
│  SAFETY_CHECK         │
│  GENERAL_CLINICAL     │
│  OUT_OF_SCOPE         │
└───────────┬───────────┘
            │ CLINICAL_* intent → sets state["query_intent"]
            ▼
┌───────────────────────┐
│    Extractor Node     │  ◄──────────────────────────────┐
│  extractor_node.py    │                                 │
│  Step 0: Haiku        │                          Review Loop
│  extracts patient ID  │                          (Max 3x, from Auditor)
│  Calls 3 tools:       │
│  - get_patient_info   │
│  - get_medications    │
│  - check_drug_        │
│    interactions       │
└───────────┬───────────┘
            │ ambiguous/no patient ID
            ▼
┌───────────────────────┐
│  Clarification Node   │  ──► pending_user_input: True ──► [END]
│  clarification_node.py│      (user's next /ask resumes)
└───────────────────────┘

            │ patient found, extractions ready
            ▼
┌───────────────────────┐
│    Auditor Node       │  ── "missing" (count < 3) ──► Extractor (review loop)
│  auditor_node.py      │  ── "ambiguous"           ──► Clarification Node
│  Validates citations  │  ── "partial" (count >= 3)──► Output Node
│  Sets routing_decision│  ── "pass"                ──► Output Node
└───────────┬───────────┘
            │ "pass" or "partial"
            ▼
┌───────────────────────┐
│    Output Node        │
│  _output_node()       │
│  Filters extractions  │  ← uses state["query_intent"] to return only
│  by query_intent      │    what was asked (allergies/meds/interactions)
│  Appends disclaimer   │
│  Sets confidence      │
└───────────┬───────────┘
            ▼
         [END] → AskResponse returned to UI

External integrations (current):
  - Anthropic Claude Haiku (Step 0 patient extraction)
  - Mock data JSON files (patients, medications, interactions)
  - LangSmith (basic tracing via LANGCHAIN_TRACING_V2=true)
  - PII scrubbing: regex stub in extractor_node.py (not Presidio)
```

### 1.3 LangGraph State Schema (Current — `langgraph_agent/state.py`)

```python
class AgentState(TypedDict):
    input_query: str                          # original user query
    query_intent: str                         # set by Router: MEDICATIONS | ALLERGIES | INTERACTIONS | SAFETY_CHECK | GENERAL_CLINICAL | OUT_OF_SCOPE
    documents_processed: List[str]            # source files read during extraction
    extractions: List[dict]                   # [{claim, citation, source, verbatim}]
    audit_results: List[dict]                 # validation results per extraction
    pending_user_input: bool                  # True = graph paused, awaiting user response
    clarification_needed: str                 # question to surface to user when paused
    clarification_response: str              # user's answer injected on resume
    iteration_count: int                      # review loop counter — hard cap at 3
    confidence_score: float                   # 0.0–1.0; < 0.90 → escalate flag in response
    final_response: str                       # formatted text returned to user
    error: Optional[str]                      # structured error message or None
    routing_decision: str                     # "pass" | "missing" | "ambiguous" | "partial" | "out_of_scope"
    is_partial: bool                          # True when iteration ceiling hit
    insufficient_documentation_flags: List[str]  # gaps listed when is_partial=True
    tool_trace: List[dict]                    # [{tool, input, output}] for every tool call
    extracted_patient_identifier: dict        # {type, value, ambiguous, reason} from Step 0
```

> **Key insight:** The `pending_user_input` flag pauses the workflow without losing work. The session is stored in-memory in `main.py` (`_sessions` dict). The next `POST /ask` to the same `session_id` passes the user's reply as `clarification_response`, which is injected into a fresh `run_workflow()` call.

---

## Part 2 — Build Order (Follow Exactly)

Implement in this sequence. Do not skip ahead. Each step must be working before the next begins.

### 2.2 Memory System & State Persistence

To satisfy the requirement for multi-turn reasoning and clinical continuity, the agent relies on three concepts: state persistence, conversation history, and context management.

#### 1. State Persistence (The "Hard Drive")

* **What it is:** The ability to save the current progress of a "thread" to a database so it survives a crash, a timeout, or a pause.
* **How it works:** In LangGraph, you use a Checkpointer (e.g. `SqliteSaver`). After every node (Router → Extractor → Auditor), the graph's entire state—including variables like `patient_id` or `verified_claims`—is written to a local SQLite file (e.g. `agent_checkpoints.sqlite`).
* **Why you need it:** If the agent reaches the Clarification Node because it's confused about a patient's left vs. right knee, the state is saved. When the human finally replies 10 minutes later, the agent resumes from that exact spot rather than re-running the expensive PDF extraction.

#### 2. Conversation History (The "Recent Memory")

* **What it is:** The list of back-and-forth messages between the user and the agent within a single `thread_id`.
* **How it works:** You store an array of `BaseMessage` objects (`HumanMessage`, `AIMessage`). When the user sends a new query, the agent looks at the previous messages to understand context. In the graph state this is tracked via the `messages` key.
* **The "Pronoun" test:**
  * User: "Look up John Smith."
  * User: "What are his meds?"
  If the history is working, the agent knows "his" refers to the `patient_id` retrieved in the first turn.

#### 3. Context Management (The "Work Desk")

* **What it is:** Deciding what information is currently "active" and relevant to the LLM's reasoning.
* **How it works:** You don't dump hundreds of messages into the LLM (that's expensive and confusing). You manage the context by:
  * **Summarizing:** Condensing old parts of the chat.
  * **Trimming:** Only keeping the last 5–10 messages.
  * **Injecting data:** Automatically pulling in the patient's "Allergy List" into every turn once the patient is identified, so the agent is always aware of safety risks.

The agent maintains a `clinical_context` object in the state (Patient ID, Policy IDs) so it doesn't re-run expensive tools for every follow-up message.

### Step 1 — Project Scaffold

**Goal:** Repo structure, dependency file, environment config, and `.gitignore` in place.

```
openemr-agent/                          (current actual structure)
├── main.py                             # FastAPI server — all endpoints
├── tools.py                            # 3 healthcare tools
├── verification.py                     # Allergy, confidence, FDA, escalation checks
├── healthcare_guidelines.py            # Constants (FDA rules, HIPAA, thresholds)
├── agent.py                            # Legacy LangChain agent (deprecated)
├── conversation.py                     # Legacy conversation manager (deprecated)
├── langgraph_agent/
│   ├── __init__.py
│   ├── state.py                        # AgentState TypedDict + create_initial_state()
│   ├── workflow.py                     # Graph assembly + run_workflow() entry point
│   ├── extractor_node.py               # Extractor Node (Step 0 Haiku + 3 tools)
│   ├── auditor_node.py                 # Auditor Node (citation validator + router)
│   └── clarification_node.py          # Clarification Node (pause/resume)
├── static/
│   └── index.html                      # Chat UI (418 lines)
├── eval/
│   ├── __init__.py
│   ├── run_eval.py                     # Eval runner
│   └── golden_data.yaml               # 18 test cases — Gauntlet YAML format
├── mock_data/
│   ├── patients.json                   # 11 patients (P001–P011)
│   ├── medications.json                # Medications keyed by patient ID
│   └── interactions.json              # 10 drug interaction rules
├── tests/
│   ├── test_main.py
│   ├── test_tools.py
│   ├── test_verification.py
│   ├── test_eval.py
│   ├── test_agent.py
│   ├── test_conversation.py
│   ├── test_langgraph_state.py
│   ├── test_langgraph_workflow.py
│   ├── test_langgraph_extractor.py
│   ├── test_langgraph_auditor.py
│   ├── test_langgraph_clarification.py
│   └── results/                        # Timestamped eval_results_*.json files
├── Procfile                            # Railway: uvicorn main:app
├── requirements.txt
├── .env                                # API keys (gitignored)
└── .gitignore
```

**Environment variables required (never hardcode):**

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=          # GPT-4o fallback only
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

---

### Step 1b — OpenEMR Fork Contribution ❌ NOT YET IMPLEMENTED

This is the open source contribution for the project. Fork OpenEMR and add two things: a new SQL table and a new REST endpoint. This creates a permanent, auditable link between clinical evidence and insurance policy decisions inside the EHR itself.

#### New SQL Table: `rcm_evidence_ledger`

```sql
CREATE TABLE rcm_evidence_ledger (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id       VARCHAR(64)   NOT NULL,
    session_id       VARCHAR(128)  NOT NULL,
    verbatim_quote   TEXT          NOT NULL,
    source_document  VARCHAR(255)  NOT NULL,
    page_number      INTEGER,
    policy_id        VARCHAR(128),
    confidence_score DECIMAL(5,2),
    auditor_result   VARCHAR(16),   -- PASS / FAIL / ESCALATED
    created_at       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
```

#### New REST Endpoint: `POST /api/rcm/verify`

This endpoint is called by the `evidence_ledger_write` tool after the Auditor Node passes. It writes every verified citation into the ledger, creating an immutable audit trail.

```
POST /api/rcm/verify

Request Body:
{
  "patient_id":       "P-00123",
  "verbatim_quote":   "Patient reports bilateral knee pain for 6 months...",
  "source_document":  "DischargeNote.pdf",
  "page_number":      4,
  "policy_id":        "Cigna-MedPolicy-012",
  "confidence_score": 94.0,
  "auditor_result":   "PASS"
}

Response:
{
  "ledger_id":   1042,
  "status":      "WRITTEN",
  "audit_trail": "rcm_evidence_ledger#1042"
}
```

The response `ledger_id` must be included in every agent output response so reviewers can trace any recommendation back to the source record.

---

### Step 2 — PII Scrubber Pre-Processing ⚠️ STUB (Regex, not Presidio)

**Current state:** `_stub_pii_scrubber()` is implemented as a regex function inside `langgraph_agent/extractor_node.py` and `langgraph_agent/clarification_node.py`. It runs before any tool call or LLM invocation.

**What the stub does (current):**

```python
# langgraph_agent/extractor_node.py — _stub_pii_scrubber()
# Strips HIPAA PII patterns via regex before any tool or LLM call
_SSN_PATTERN   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN_PATTERN   = re.compile(r"\bMRN[:\s]*\w+\b", re.IGNORECASE)
_DOB_PATTERN   = re.compile(r"\b(DOB|Date of Birth)[:\s]*[\d/\-]+\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"\b\d{3}[.\-]\d{3}[.\-]\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")
# Replaces matches with [REDACTED-SSN], [REDACTED-MRN], etc.
```

**TODO (not yet built):** Replace with Microsoft Presidio (`presidio-analyzer`, `presidio-anonymizer`) and move to a standalone `langgraph_agent/preprocessing/pii_scrubber.py`.

---

### Step 3 — Tool Registry (3 Tools — Current Implementation in `tools.py`)

All tools load mock JSON data at startup. All return `{"success": bool, ...}` — never raise exceptions to the caller.

**Tool data flow (current):**

```
User Query
    │ (Step 0: Haiku extracts patient identifier)
    ▼
get_patient_info(name_or_id)  ──► patient dict  ──► allergies, conditions
    │
    ▼
get_medications(patient_id)   ──► medications list
    │
    ▼
check_drug_interactions(medications[])  ──► interactions list
    │
    ▼
_format_extractions() → extractions[] with citation + source → Auditor Node
```

| Tool | Signature | Data Source | What It Returns |
| :--- | :--- | :--- | :--- |
| `get_patient_info` | `(name_or_id: str) -> dict` | `mock_data/patients.json` | `{success, patient: {id, name, dob, age, gender, allergies[], conditions[]}}` |
| `get_medications` | `(patient_id: str) -> dict` | `mock_data/medications.json` | `{success, medications: [{name, dose, frequency, prescribed}]}` |
| `check_drug_interactions` | `(medications: list) -> dict` | `mock_data/interactions.json` | `{success, interactions: [{drug1, drug2, severity, description, recommendation}]}` |

**Tool error contract (all tools):**

```python
# On failure — never raises to caller
{"success": False, "error": "human-readable description", ...}
```

**Not yet implemented (planned):**

| Tool | Library | Status |
| :--- | :--- | :--- |
| `pdf_extractor` | unstructured.io | ❌ Not built |
| `policy_search` | Pinecone | ❌ Not built |
| `denial_analyzer` | Custom logic | ❌ Not built |
| `evidence_ledger_write` | OpenEMR fork | ❌ Not built |

---

### Step 4 — LangGraph State Machine (5 Nodes — Current Implementation)

The graph has 5 nodes. The Ledger Write node is not yet built.

#### Node 0: Supervisory Router Node ✅

**Purpose:** Classify every incoming query before anything else runs. This is the first node in the graph. It decides whether a query is clinical (should proceed to the Extractor) or out-of-scope (should be refused immediately without touching any patient data or tools).

**Why this exists (not in PRD verbatim — see [Hotfixes section](#part-14--hotfixes--deviations-from-prd)):**

Before this node was added, every query — including "talk like a pirate" and "what's the weather?" — entered the clinical pipeline and returned John Smith's full medical record because the session context prepend (`Regarding John Smith: ...`) caused the Extractor to always run. This was observed in live UI testing on 2026-02-26.

**Intent classifications (`state["query_intent"]`):**

| Value | Trigger Example | Next Node |
| :--- | :--- | :--- |
| `MEDICATIONS` | "What medications is John on?" | Extractor |
| `ALLERGIES` | "Does he have any allergies?" | Extractor |
| `INTERACTIONS` | "Check drug interactions for Mary" | Extractor |
| `SAFETY_CHECK` | "Can I give penicillin to him?" | Extractor |
| `GENERAL_CLINICAL` | Any other healthcare question | Extractor |
| `OUT_OF_SCOPE` | "Talk like a pirate", "What's the weather?" | Output (refusal) |

**Out-of-scope response (verbatim from PRD):**
> "I am a specialized Healthcare RCM agent. I can only assist with clinical documentation and insurance verification."

**Fail-safe:** If the LLM call errors, the router falls back to `GENERAL_CLINICAL` — it will never falsely refuse a clinical query due to an API failure.

**`query_intent` is also used by the Output Node** to filter extractions to only what was asked. See Node 4 (Output Node).

#### Node 1: Extractor Node ✅

**Purpose:** Extract patient data by calling all 3 tools in sequence. Step 0 uses Claude Haiku to identify the patient from natural language before any tool runs.

**Actual flow (`langgraph_agent/extractor_node.py`):**

1. Run `_stub_pii_scrubber()` on the raw query
2. **Step 0** — Haiku extracts patient identifier as JSON `{type, value, ambiguous, reason}`
3. If ambiguous or no identifier → set `pending_user_input: True`, route to Clarification Node
4. Call `get_patient_info(patient_identifier)`
5. Call `get_medications(patient_id)`
6. Call `check_drug_interactions(medications)`
7. Call `_format_extractions()` → `extractions[]` each with `{claim, citation, source, verbatim: True}`

**Extraction item shape (current):**

```json
{
    "claim": "John Smith is prescribed Metformin 500mg twice daily.",
    "citation": "Metformin 500mg twice daily",
    "source": "mock_data/medications.json",
    "verbatim": true
}
```

#### Node 2: Auditor Node ✅

**Purpose:** Validate every extraction from the Extractor Node. Makes ALL routing decisions.

**Actual routing logic (`langgraph_agent/auditor_node.py`):**

```
if iteration_count >= 3       → routing_decision = "partial" → Output Node
if any extraction.ambiguous   → routing_decision = "ambiguous" → Clarification Node
if any citation fails verify  → routing_decision = "missing" → Extractor (review loop)
if all citations pass         → routing_decision = "pass" → Output Node
```

**Citation verification** (`_verify_citation_exists_in_source`): splits the citation into tokens (3+ chars), checks every token appears in the source JSON file as a substring. No LLM call — pure string matching against `mock_data/*.json`.

**Confidence on pass:** `max(0.0, 0.95 - (iteration_count * 0.05))` — starts at 0.95, drops 0.05 per review loop.

**On partial (ceiling hit):** returns `is_partial: True`, `insufficient_documentation_flags[]`, `confidence_score: 0.5`.

#### Node 3: Clarification Node ✅

**Purpose:** Pause the workflow when the patient identifier is ambiguous or missing.

**Trigger conditions (current):**
- Step 0 (Haiku) returns `ambiguous: true` — e.g., first name only, or no name found
- Auditor sets `routing_decision: "ambiguous"` — e.g., extraction contains an ambiguous claim

**Actual logic (`langgraph_agent/clarification_node.py`):**

```python
def clarification_node(state: AgentState) -> AgentState:
    state["pending_user_input"] = True
    # clarification_needed already set by Extractor or Auditor before routing here
    # No SQLite checkpoint — state is held in main.py _sessions dict
    return state  # Graph ends here (→ END); user's next /ask resumes workflow
```

**Resume mechanism:** `main.py` detects `pending_user_input: True` in `_sessions[session_id]`. The user's next `POST /ask` to the same `session_id` is passed as `clarification_response` to a new `run_workflow()` call, which injects it via `resume_from_clarification()`.

#### Node 4: Output Node ✅

**Purpose:** Filter extractions by `query_intent`, append disclaimer to `final_response`, and set `confidence_score` default if not set.

**Query-intent filtering (added 2026-02-26 — not in original PRD):**

Before this was added, every query returned the full data dump regardless of what was asked. "Does he have any allergies?" returned medications + allergies + interactions. The output node now uses `state["query_intent"]` set by the Router Node to return only the relevant subset:

| `query_intent` | Extractions included in response |
| :--- | :--- |
| `MEDICATIONS` | Only medication extractions |
| `ALLERGIES` | Only the allergy extraction |
| `INTERACTIONS` | Only interaction extractions |
| `SAFETY_CHECK` | Allergy + medication extractions (answers "can I give X?") |
| `GENERAL_CLINICAL` | All extractions (original behavior) |
| `OUT_OF_SCOPE` | None — `final_response` is already set by Router to the refusal string |

**Actual response shape returned to `POST /ask`:**

```json
{
    "answer": "John Smith is prescribed Metformin 500mg... [Source: mock_data/medications.json]",
    "session_id": "abc-123",
    "timestamp": "2026-02-26T...",
    "escalate": false,
    "confidence": 0.95,
    "disclaimer": "This information is generated by an AI assistant...",
    "tool_trace": [
        {"tool": "tool_get_patient_info", "input": "John Smith", "output": {...}},
        {"tool": "tool_get_medications", "input": "P001", "output": {...}},
        {"tool": "tool_check_drug_interactions", "input": ["Metformin", ...], "output": {...}}
    ]
}
```

#### Graph Assembly (Current — `langgraph_agent/workflow.py`)

```python
graph = StateGraph(AgentState)

graph.add_node("router", router_node)        # Entry point — classifies intent
graph.add_node("extractor", extractor_node)
graph.add_node("auditor", auditor_node)
graph.add_node("clarification", clarification_node)
graph.add_node("output", _output_node)

graph.set_entry_point("router")              # Router is now the entry point

graph.add_conditional_edges("router", _route_from_router, {
    "extractor": "extractor",               # CLINICAL_* intent
    "output": "output",                     # OUT_OF_SCOPE — goes straight to output with refusal
})
graph.add_conditional_edges("extractor", _route_from_extractor, {
    "clarification": "clarification",
    "auditor": "auditor",
})
graph.add_conditional_edges("auditor", _route_from_auditor, {
    "output": "output",                     # "pass" or "partial"
    "extractor": "extractor",               # "missing" (review loop)
    "clarification": "clarification",       # "ambiguous"
})
graph.add_edge("clarification", END)        # Paused; session holds state
graph.add_edge("output", END)

# No SQLite checkpointing — graph compiled fresh on each run_workflow() call
return graph.compile()
```

---

### Step 5 — FastAPI Backend (Current — `main.py`)

| Endpoint | Method | What It Does |
| :--- | :--- | :--- |
| `/` | GET | Serves `static/index.html` chat UI |
| `/health` | GET | Returns `{service, version, status, timestamp}` |
| `/ask` | POST | Main query endpoint — routes through LangGraph, handles session pause/resume |
| `/eval` | POST | Runs `eval/golden_data.yaml` test suite, saves results to `tests/results/` |
| `/eval/results` | GET | Returns latest saved eval results JSON |

**Session management:** In-memory `_sessions` dict keyed by `session_id`. Detects `pending_user_input: True` on the prior result and treats the next `/ask` as a clarification response.

**Follow-up context:** If the new question has no patient identifier and the session has a prior patient, prepends `"Regarding {last_patient}: {question}"` before invoking the workflow.

**`POST /ask` request/response:**

```python
# Request
{"question": "What medications is John Smith on?", "session_id": "abc-123"}  # session_id optional

# Response (AskResponse)
{"answer": str, "session_id": str, "timestamp": str,
 "escalate": bool, "confidence": float, "disclaimer": str, "tool_trace": list}
```

**Not yet implemented:**
- `POST /agent/query` (planned rename of `/ask`)
- `POST /agent/resume` (planned dedicated resume endpoint)
- `POST /eval/run` (planned rename of `/eval`)
- `POST /api/rcm/verify` (planned OpenEMR fork endpoint)

---

### Step 6 — Observability (LangSmith — Basic Tracing Active)

**Current state:** Basic LangSmith tracing is active. `LANGCHAIN_TRACING_V2=true` and `LANGSMITH_API_KEY` are set in `.env`. The `verification.py` functions (`check_allergy_conflict`, `calculate_confidence`, `should_escalate_to_human`, `apply_fda_rules`) are decorated with `@traceable`.

**What is currently traced:**
- LangSmith traces for `@traceable` functions in `verification.py`
- LangChain/LangGraph native traces (node-level input/output, LLM calls)
- Each request's tool calls appear in the `tool_trace` field of the response

**What is NOT yet implemented (planned):**
- Custom domain metrics: Faithfulness, Citation Accuracy, Review Loop Rate with >20% alert
- LangSmith dataset/eval integration
- Per-request `ledger_id` in traces
- LangSmith dashboard with threshold alerts

---

### Step 7 — Domain Verification Layer

These 4 verification checks are non-negotiable. Implement all before calling any output production-ready.

#### Verification 1: Evidence Attribution (Core Rule)
- Every claim **must** have a `verbatim_quote`, `page_number`, `section`, and `document_name`
- No quote = claim is invalid and must not appear in output
- Enforced by: `citation_verifier` tool + Auditor Node

#### Verification 2: Allergy Conflict Check
- Compare every prescribed/mentioned drug against the patient's known allergy list via `med_retrieval`
- If a match is found → flag immediately, do not proceed to output

#### Verification 3: FDA Severity Escalation
- If a drug interaction severity is `HIGH` or `CONTRAINDICATED` → trigger "Physician Review Required" escalation
- This is a hard stop — never surface high-severity interactions directly to end users

#### Verification 4: Confidence Scoring (Current — Deduction-Based)

Confidence is calculated in the Auditor Node on a pass, then checked in `main.py` to set the `escalate` flag.

**Auditor confidence formula (current — `auditor_node.py`):**

```python
# On "pass" routing decision:
confidence = max(0.0, 0.95 - (iteration_count * 0.05))
# 0 review loops → 0.95
# 1 review loop  → 0.90
# 2 review loops → 0.85
```

**Escalation check (`main.py`):**

```python
escalate = confidence < CLINICAL_SAFETY_RULES["confidence_threshold"] or result.get("is_partial", False)
# confidence_threshold = 0.90 (from healthcare_guidelines.py)
# is_partial = True when iteration ceiling (3) was hit
```

**Separate `calculate_confidence()` in `verification.py` (tool-level, not used by LangGraph flow):**

```python
base = tools_succeeded / tools_total
deduction += 0.20 if allergy_conflict else 0
deduction += 0.10 if interactions_found else 0
score = max(0.0, min(1.0, base - deduction))
```

**Not yet implemented:** Weighted hybrid formula (Auditor Pass Rate ×50% + Citation Density ×30% + normalized Self-Assessment ×20%).

---

### Step 8 — Evaluation Framework (Current — 18 Test Cases)

**File structure:**

```
eval/
├── run_eval.py           # eval runner — loads YAML, calls /ask, scores must_contain/must_not_contain
└── golden_data.yaml      # 18 test cases — Gauntlet YAML format
```

**Current test case breakdown:**

| Category | Count | IDs |
| :--- | :--- | :--- |
| happy_path | 7 | gs-001, gs-002, gs-003, gs-004, gs-012, gs-014, gs-015 |
| edge_case | 7 | gs-005, gs-006, gs-007, gs-013, gs-016, gs-017, gs-018 |
| adversarial | 4 | gs-008, gs-009, gs-010, gs-011 |
| **multi_step** | **0** | **None — required 10+ not yet written** |

**Current tool names in `expected_tools` (actual names used):**
- `tool_get_patient_info`
- `tool_get_medications`
- `tool_check_drug_interactions`

**Sample case from current `eval/golden_data.yaml`:**

```yaml
- id: "gs-001"
  category: "happy_path"
  query: "What medications is John Smith on?"
  expected_tools:
    - tool_get_patient_info
    - tool_get_medications
  expected_sources:
    - mock_data/patients.json
    - mock_data/medications.json
  must_contain:
    - "metformin"
    - "lisinopril"
  must_not_contain:
    - "I don't know"
    - "not found"
    - "error"
  difficulty: "happy_path"
```

**Eval runner (`eval/run_eval.py`):**
- Loads `golden_data.yaml`
- Calls `run_workflow()` directly (not via HTTP)
- Scores: `must_contain` (OR — any one match = pass), `must_not_contain` (AND — all must be absent)
- Saves timestamped results to `tests/results/eval_results_YYYYMMDD_HHMMSS.json`
- Triggered via `POST /eval` endpoint

**Gaps vs target:**
- Need 32+ more cases to reach 50+ total
- Need 10+ multi-step cases (none exist)
- `expected_tools` names will need updating when tools are renamed

---

### Step 9 — Chat UI (Current — `static/index.html`, 418 lines)

The UI is a single-page HTML/JS app served by `GET /`. It communicates with the backend via `POST /ask` and `POST /eval`.

**What is currently implemented:**
- Message thread (user + agent messages)
- Displays `answer` field from `AskResponse`
- Shows `confidence` score and `escalate` flag
- Shows `tool_trace` (expandable)
- Shows `disclaimer` text
- Handles `pending_user_input` — when the agent asks for clarification, the same input box accepts the clarification response (sent to the same `session_id`)
- "Run Eval" button → `POST /eval` → displays pass/fail results

**What is NOT yet in the UI (planned):**
- Inline verification badges (`PII Scrubbed`, `Verbatim ✓`, `FDA Severity`)
- Ledger reference display (`rcm_evidence_ledger#id`)
- Confidence breakdown (weighted formula components)
- "Export LangSmith Trace" button
- Clarification modal with structured answer buttons (Left/Right/Both)

---

## Part 3 — Failure Modes & Handling

Never guess. Every unresolvable ambiguity is escalated or flagged explicitly.

| Failure Mode | Example | Agent Response |
| :--- | :--- | :--- |
| Ambiguous clinical text | "Left leg pain" but MRI for "Right leg" | Clarification Node → `pending_user_input: true` → pause without losing work |
| Contradictory notes | "Stable" Monday, "Worsening" Wednesday | Flag contradiction → cite both quotes → escalate to human reviewer |
| Missing evidence | Payer requires "failed conservative therapy" — not documented | Auditor triggers Review Loop → re-extract → if still missing, flag as "Insufficient Documentation" |
| Unreadable PDF section | Scanned handwriting or low-resolution scan | Mark section "Extraction Failed" → flag for manual review → continue with readable sections |
| Policy not in vector DB | New payer policy not yet indexed | Return "Policy Not Found" → suggest manual lookup → never guess |
| Max review loops reached | Auditor sends back 3 times with no resolution | Return partial results with gaps explicitly flagged → full chart escalated to human |
| Out-of-scope query | "Talk like a pirate" | Supervisory Router → "I am a specialized Healthcare RCM agent. I can only assist with clinical documentation and insurance verification." |

### 3.3 Internal Messaging Integration

The agent is integrated into the OpenEMR Internal Mail/Portal system to automate administrative tasks:

* **Trigger:** The agent pre-scans incoming patient messages for medication or procedure requests.
* **Pre-Processing:** Before a provider opens a message, the agent runs the `Extractor` and `Auditor` nodes against the patient's chart.
* **Drafting:** The agent generates a "Verification Summary" that is prepended to the message thread, providing the doctor with a pre-audited recommendation.

---

## Part 4 — Disaster Recovery

### RTO / RPO Targets

| Metric | Definition | Target |
| :--- | :--- | :--- |
| RTO (Recovery Time Objective) | Maximum acceptable time to restore service | < 15 minutes for critical failures |
| RPO (Recovery Point Objective) | Maximum acceptable data loss window | < 1 hour of processing work lost |

### Component Failovers

| Component | Failure | Recovery | RTO |
| :--- | :--- | :--- | :--- |
| Claude API | Anthropic unreachable or rate limited | Automatic fallback to GPT-4o via LangChain model swap | < 30 seconds |
| Pinecone Vector DB | Service outage | Fallback to local FAISS index (snapshot updated nightly) | < 2 minutes |
| OpenEMR FHIR API | Docker instance down | Return cached patient data; flag as "Live Data Unavailable" | < 1 minute |
| LangGraph workflow crash | Mid-document processing failure | Resume from last SQLite checkpoint — not from page 1 | < 5 minutes |
| FastAPI server crash | Application process dies | Railway/Render auto-restarts container; health check every 30s | < 2 minutes |
| Full infrastructure outage | Hosting provider down | Manual failover to backup Railway project in different region | < 15 minutes |

---

## Part 5 — Performance Targets

| Metric | Target | Notes |
| :--- | :--- | :--- |
| Latency per document | < 30 seconds | Accuracy over speed in RCM domain |
| Single-tool query latency | < 5 seconds | Per project requirements |
| Multi-step chain latency | < 15 seconds | 3+ tool chains |
| Batch throughput | 20–50 charts/hour | Realistic for authorization teams |
| Tool success rate | > 95% | Per project requirements |
| Eval pass rate | > 80% | Blocks deployment if dropped below |
| Hallucination rate | < 5% | Enforced by Evidence Attribution rule |
| Verification accuracy | > 90% | Per project requirements |

### 5.1 Real-Time Monitoring

During the final demo, the following LangSmith dashboards will be utilized to prove system integrity:

* **Trace View:** Visualizing the "Review Loop" between the Auditor and Extractor.
* **Latency Heatmap:** Confirming the < 30s processing time for complex multi-agent chains.
* **Score Distribution:** Real-time view of the Faithfulness and Citation Accuracy scores for the 50-case eval suite.

### Performance Regression Triggers

| Condition | Threshold | Action |
| :--- | :--- | :--- |
| End-to-end latency | > 30s on P90 | Alert + investigate bottleneck layer |
| Tool success rate | < 95% | Alert + switch affected tool to fallback |
| Eval pass rate | < 80% | **Block deployment + rollback to previous version** |
| LLM error rate | > 2% | Alert + activate Claude → GPT-4o fallback |
| Latency regression | > 20% increase vs baseline | Flag PR, require manual approval before merge |
| Hallucination rate | > 5% | **Immediate rollback + root cause analysis required** |

---

## Part 6 — Deployment & CI/CD

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| Backend API | FastAPI + Uvicorn | Async, fast, auto-generates API docs |
| Agent Framework | LangGraph | Multi-agent state machine with review loops and state persistence |
| PII Protection | Microsoft Presidio | Local scrubbing before cloud LLM — HIPAA 2026 compliant |
| Vector DB | Pinecone | Scalable semantic search over payer policy PDFs |
| Observability | LangSmith | Native LangGraph integration, domain-specific metrics |
| Deployment | Railway or Render | Simple Docker deployment, free tier, public URL for demo |
| CI/CD | GitHub Actions | Auto-deploy on push to main; eval suite runs on every PR |
| Checkpointing | SQLite | State persisted at every node; retained 30 days |
| Audit Logs | Append-only storage | 7-year retention per healthcare compliance requirements |

---

## Part 7 — AI Cost Analysis

### LLM Strategy (Hybrid)

| Task | Model | Reason |
| :--- | :--- | :--- |
| PDF text extraction | Claude Haiku | Cheap, fast, sufficient for structured extraction |
| Policy matching and reasoning | Claude 3.5 Sonnet | Complex medical reasoning requires full capability |
| Final verification and audit | Claude 3.5 Sonnet | High-stakes decision — worth the cost |
| Embedding generation | text-embedding-3-small | Cost-effective vector indexing of policy docs |

### Monthly Cost Projections

| Scale | Users | Charts/Month | Est. Monthly Cost (USD) |
| :--- | :--- | :--- | :--- |
| Pilot | 100 | ~5,000 | $80 – $150 |
| Small clinic | 1,000 | ~50,000 | $600 – $1,200 |
| Mid-size hospital | 10,000 | ~500,000 | $5,000 – $9,000 |
| Enterprise | 100,000 | ~5,000,000 | $40,000 – $70,000 |

**Assumptions:** ~50 charts per user per month; average 10K tokens per chart (input + output); Haiku for extraction (~80% of calls), Sonnet for verification (~20%); Pinecone at $0.096/1M vectors.

---

## Part 8 — Open Source Contribution

### Primary: Healthcare RCM Eval Dataset (Required by Sunday)

- Release 50+ test cases as a public dataset on HuggingFace or GitHub
- Each test case includes: input document, expected extractions, expected citations, pass/fail criteria
- Covers: evidence extraction, citation verification, contradiction detection, adversarial inputs
- License: MIT

### Stretch Goal: OpenEMR RCM Agent Package (PyPI)

- `pip install openemr-rcm-agent`
- Only attempted if all primary deliverables are completed before Sunday

---

## Part 9 — Final Submission Checklist

| Deliverable | Requirement | Status |
| :--- | :--- | :--- |
| GitHub Repository | Setup guide, architecture overview, deployed link | ☐ |
| Demo Video (3–5 min) | Show the "Pirate Fix" refusal + automated eval runner | ☐ |
| Pre-Search Document | Completed (AgentForge_PreSearch_Final_v2.md) | ✓ |
| Agent Architecture Doc | 1–2 page breakdown using PRD template | ☐ |
| AI Cost Analysis | Dev spend + projections for 100/1K/10K/100K users | ☐ |
| Eval Dataset | 50+ test cases published on GitHub/HuggingFace | ☐ |
| Open Source Link | Published dataset, package, or PR | ☐ |
| Deployed Application | Publicly accessible agent interface | ☐ |
| Social Post | Tag `@GauntletAI` on X or LinkedIn with demo screenshot | ☐ |

**Deadline: Sunday 10:59 PM CT**

---

## Part 10 — Coding Standards (Enforced on Every File)

These rules apply to every Python file written for this project. Follow them in the exact order listed before committing any code.

---

### 10.1 Test-Driven Development — Mandatory Sequence

Every single file must follow this exact sequence. No exceptions.

1. Write the test file first
2. Run tests — confirm they **FAIL**
3. Write the implementation file
4. Run tests — confirm they **PASS**
5. Save results to `tests/results/` with a descriptive filename

Never write implementation before tests exist. Never skip saving results — they are proof of TDD.

---

### 10.2 Module Docstring — Every File

The very first thing in every Python file must be a docstring in this exact format:

```python
"""
filename.py
-----------
AgentForge — Healthcare RCM AI Agent — [One line description]
--------------------------------------------------------------
[2-3 sentences describing what this module does and why it exists]

[List of key functions or classes if applicable]

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""
```

---

### 10.3 Function Docstring — Every Function

Every function must have a docstring in this exact format:

```python
def function_name(param1: str, param2: int) -> dict:
    """
    One sentence describing what this function does.

    Args:
        param1: What this parameter is and what values are valid
        param2: What this parameter is and what values are valid

    Returns:
        dict: What the return value contains

    Raises:
        ValueError: When and why this is raised (if applicable)
    """
```

---

### 10.4 Type Annotations — Every Function

Every function parameter and return value must have a type annotation.

Required on:
- All function parameters
- All return values
- Module-level constants

Accepted types: `str`, `int`, `float`, `bool`, `dict`, `list`, `List`, `Tuple`, `Optional`, `Any`, `Union`

---

### 10.5 Error Handling — Every Function

Every function must handle failures gracefully. Raw exceptions must never reach the user.

```python
def function_name(param: str) -> dict:
    try:
        # main logic
        return {"success": True, "data": result}
    except SpecificException as e:
        return {"success": False, "error": str(e), "data": None}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}", "data": None}
```

Rules:
- Always catch the most specific exception first
- Always catch generic `Exception` as final fallback
- Always return a structured dict — never raise to the caller
- Error message must be human-readable — not a raw Python traceback

---

### 10.6 Hardcoded Healthcare Constants

These values must be defined in `agent/constants.py` and imported everywhere. Never use magic numbers or inline strings for these.

```python
# agent/constants.py

# Drug interaction severity levels
SEVERITY_REQUIRES_REVIEW = ["HIGH", "CONTRAINDICATED"]
SEVERITY_AUTO_APPROVE     = ["LOW"]  # with monitoring note appended

# Confidence thresholds
CONFIDENCE_ESCALATION_THRESHOLD = 0.90   # anything below triggers human escalation
CONFIDENCE_DEDUCTION_ALLERGY    = 0.20   # deducted when allergy conflict found
CONFIDENCE_DEDUCTION_INTERACTION = 0.10  # deducted when drug interaction found

# LangGraph limits
MAX_REVIEW_LOOP_ITERATIONS = 3   # never exceed

# HIPAA — fields that must never appear in logs or LangSmith traces
HIPAA_FORBIDDEN_FIELDS = ["name", "ssn", "dob", "address", "phone", "email", "mrn"]

# Disclaimer — append to every agent response
AGENT_DISCLAIMER = (
    "This information is generated by an AI assistant and is intended to support — "
    "not replace — clinical judgment. Always consult a qualified healthcare "
    "professional before making medical decisions."
)
```

Source citations are required for:
- Every drug interaction claim
- Every allergy conflict finding
- Every verification result

---

### 10.7 Eval Test Case Format

All test cases must follow Gauntlet's YAML format exactly.

**File location:** `eval/golden_data.yaml`

```yaml
test_cases:
  - id: "gs-001"
    category: "happy_path"
    query: "What medications is John Smith on?"
    expected_tools:
      - tool_get_patient_info
      - tool_get_medications
    expected_sources:
      - mock_data/medications.json
    must_contain:
      - "metformin"
      - "lisinopril"
    must_not_contain:
      - "I don't know"
      - "I cannot help"
    difficulty: "happy_path"
```

**Required fields:** `id`, `category`, `query`, `expected_tools`, `must_contain`, `must_not_contain`, `difficulty`

`must_not_contain` must always include at minimum:
- `"I don't know"`
- `"error"`
- Any harmful medical information relevant to the test case

---

### 10.8 Regression Testing Rule

After every new feature is added, run the full eval suite before moving on.

```bash
python eval/run_eval.py
```

If any test case that previously passed now fails — **stop**. Fix the regression before proceeding. Do not move forward with a regression present.

Results must be saved to `tests/results/` with a descriptive filename after every run.

---

### 10.9 File Naming Conventions

| File Type | Convention | Example |
| :--- | :--- | :--- |
| Python modules | snake_case | `healthcare_guidelines.py` |
| Test files | `test_` prefix | `test_verification.py` |
| Test results | descriptive name | `test_verification_results.txt` |
| Eval data | descriptive name | `golden_data.yaml` |
| Config files | standard names | `.env`, `.gitignore` |

---

### 10.10 What Never Goes in Code

- API keys — always in `.env`, never hardcoded
- Magic numbers for healthcare thresholds — always import from `agent/constants.py`
- Patient names or identifiers in log statements
- Raw exception tracebacks returned to users
- `TODO` comments in committed code — finish it or remove it

---

## Part 11 — Correction PR (Fix Before Any New Feature)

These are bugs, standards violations, and inconsistencies in the **current codebase** that must be fixed in a single clean PR before any new feature work begins. No behavior changes — only corrections.

| # | File | Issue | What to Fix |
| :--- | :--- | :--- | :--- |
| 1 | `eval/golden_data.yaml` | Header comment says "10 test cases: 4 happy_path, 3 edge_case, 3 adversarial" — file has **18 cases (7/7/4)** | Update comment to match reality |
| 2 | `eval/__init__.py` | Empty file — violates coding standards (every Python file must have a module docstring) | Add module docstring |
| 3 | `langgraph_agent/__init__.py` | Has `""" """` empty docstring — violates coding standards | Add proper module docstring |
| 4 | `.env.example` | Missing entirely — `.env` with real API keys exists but no template for contributors | Create `.env.example` with placeholder values |
| 5 | `agent.py` + `conversation.py` | Deprecated legacy LangChain code in root. Tests still import from them. Creates confusion about what's canonical | Move to `legacy/` folder or delete; update any tests that import from them |
| 6 | `golden_data.yaml` `expected_tools` values | Uses LangChain wrapper names from deprecated `agent.py` (`tool_get_patient_info`, etc.) — not actual Python function names from `tools.py` (`get_patient_info`, etc.) | Update to match actual function names in `tools.py` |
| 7 | `10.10 What Never Goes in Code` | References `agent/constants.py` — actual file is `healthcare_guidelines.py` at root | Fix path reference throughout |

**Rule:** This PR is corrections only. If a fix requires behavior change, it belongs in a feature PR, not here.

---

## Part 12 — Feature PR Roadmap (Build in Order After Correction PR)

Each PR builds on the previous. Do not start a PR until the prior one is merged and tests pass.

| PR | Feature | Key Files | Depends On |
| :--- | :--- | :--- | :--- |
| ✅ **PR 1** | Supervisory Router Node — classifies `CLINICAL_RCM` vs `OUT_OF_SCOPE` before Extractor runs; "pirate fix" + query-intent filtering in Output Node | `langgraph_agent/router_node.py`, `langgraph_agent/state.py`, `langgraph_agent/workflow.py` | Correction PR |
| **PR 2** | OpenEMR fork — `rcm_evidence_ledger` SQL table + `POST /api/rcm/verify` endpoint | `main.py`, SQL migration file | Correction PR |
| **PR 3** | `evidence_ledger_write` tool + Ledger Write Node — wires PR 2 into graph after Auditor PASS | `tools.py`, `langgraph_agent/ledger_write_node.py`, `langgraph_agent/workflow.py` | PR 2 |
| **PR 4** | Real PII Scrubber — replace regex stub with Microsoft Presidio; move to own module | `langgraph_agent/extractor_node.py`, `langgraph_agent/clarification_node.py`, new `langgraph_agent/preprocessing/pii_scrubber.py` | Correction PR |
| **PR 5** | `pdf_extractor` tool — unstructured.io for clinical PDFs | `tools.py` | Correction PR |
| **PR 6** | `policy_search` tool — Pinecone RAG over payer policy PDFs | `tools.py` | Correction PR |
| **PR 7** | `denial_analyzer` tool — historical denial pattern matching | `tools.py` | Correction PR |
| **PR 8** | Weighted confidence scoring — replace `0.95 - (iteration_count * 0.05)` with Auditor Pass Rate ×50% + Citation Density ×30% + normalized Self-Assessment ×20% | `langgraph_agent/auditor_node.py`, `verification.py` | Correction PR |
| **PR 9** | Eval expansion — grow from 18 → 50+ cases; add 10+ multi-step cases using `evidence_ledger_write` and `policy_search` in `expected_tools` | `eval/golden_data.yaml` | PR 3, PR 6 |
| **PR 10** | LangSmith domain metrics — Faithfulness, Citation Accuracy, Review Loop Rate >20% alert | `langgraph_agent/auditor_node.py`, `verification.py` | PR 8 |

---

## Part 13 — Interview Preparation

Be ready to answer these questions with specifics from this project:

- **Why LangGraph over LangChain?** — LangGraph's native cyclic graph support enables the Auditor → Review Loop → Extractor pattern. A single-agent LangChain architecture cannot route work back for re-extraction cleanly.
- **Why the Evidence Attribution rule?** — Hallucination in RCM is a liability issue, not just a quality issue. A fabricated "failed conservative therapy" claim causes denial, financial loss, and compliance exposure.
- **Why the `pending_user_input` flag?** — Ambiguity on page 23 of a 50-page chart must not discard 22 pages of completed work. The flag pauses the graph at that exact node and resumes from that node.
- **Why PII scrubbing runs first?** — Patient names, SSNs, DOBs, and MRNs must never reach a cloud LLM. Presidio runs locally. Even if the Anthropic API were compromised, no identifying data would be exposed.
- **Why Claude 3.5 Sonnet for verification?** — 200K token context window handles full 50-page clinical charts without chunking errors. Diagnoses on page 3 stay connected to procedures on page 47.
- **What happens at max review loops?** — After 3 failed Auditor verifications, partial results with explicitly flagged gaps are returned and the full chart is escalated to human review. The agent never guesses.
- **What is your open source contribution to OpenEMR?** — A new SQL table `rcm_evidence_ledger` and a REST endpoint `POST /api/rcm/verify` that writes every audited verbatim citation directly into the EHR. This creates an immutable, traceable audit trail linking insurance approval decisions to the exact clinical note lines that supported them — a capability OpenEMR did not have before.
- **Why does the `evidence_ledger_write` tool run after the Auditor, not after the Extractor?** — Only verified citations should be written to the ledger. Writing a citation that the Auditor later fails would create a false audit trail. The ledger records the result of the full verification chain, not the raw extraction.
- **How is confidence scoring different from a simple self-assessment?** — Self-assessment alone is unreliable because LLMs are overconfident. The weighted hybrid formula (Auditor Pass Rate 50% + Citation Density 30% + normalized Self-Assessment 20%) grounds confidence in measurable, programmatic signals. The normalization step specifically prevents model overconfidence from inflating the final score.

---

## Part 14 — Hotfixes & Deviations from PRD

This section records every implementation decision that differs from or extends the PRD v2.2 and Design Prototype v2.2. These fixes were made based on live testing and must not be reverted. The reason each change was made is documented here so future contributors understand the intent.

---

### Hotfix 1 — LLM Model Name (2026-02-26)

**File:** `langgraph_agent/extractor_node.py`

**Change:** `claude-3-5-haiku-20241022` → `claude-haiku-4-5`

**Why:** The PRD and design docs specified "Claude Haiku" without pinning an exact model string. The string `claude-3-5-haiku-20241022` was used during development but returns a `404 Not Found` from the Anthropic API on the account used for this project. The Anthropic Console confirmed the available models on this account are `claude-haiku-4-5` and `claude-sonnet-4`. This is a configuration-only fix — no logic changed.

**Do not revert** unless the Anthropic account plan changes. The PRD intent ("use Haiku for cheap, fast extraction") is fully preserved.

---

### Hotfix 2 — Supervisory Router Node: Out-of-Scope Queries (2026-02-26)

**Files:** `langgraph_agent/router_node.py` (new), `langgraph_agent/state.py`, `langgraph_agent/workflow.py`

**Change:** Added a new Router Node as the graph entry point. It classifies every query as `OUT_OF_SCOPE` or one of 5 clinical intents before the Extractor runs.

**Why:** The PRD mentions the Router Node but does not specify what happens at the implementation level when session context is active. Live testing showed that queries like "Talk like a pirate" and "What's the weather today?" were returning full patient medical records because `main.py` prepends `"Regarding John Smith: ..."` to all session follow-ups before the workflow runs. The Extractor received `"Regarding John Smith: Talk like a pirate"`, extracted "John Smith" as the patient identifier, called all 3 tools, and returned his medications, allergies, and drug interactions. This is both a safety issue (clinical data returned in response to irrelevant queries) and a professional standard violation. The Router Node intercepts before any patient data is touched.

**Do not revert.** Removing the Router Node causes patient medical data to be returned in response to non-healthcare queries. This is a clinical data handling issue, not a cosmetic one.

---

### Hotfix 3 — Query-Intent Filtering in Output Node (2026-02-26)

**File:** `langgraph_agent/workflow.py` (`_output_node` function)

**Change:** The Output Node now filters `extractions[]` using `state["query_intent"]` before building `final_response`. Previously it always returned all extractions regardless of what was asked.

**Why:** Live testing showed that "Does he have any allergies?" returned the full dump: allergies + all medications + all drug interactions. The intent was clear but the response was not. "Can I give penicillin to him?" similarly returned unrelated medication data instead of answering the allergy question directly. This is a response relevance problem — the data was correct but the signal-to-noise ratio was unacceptable for clinical use. The Extractor still fetches all data (cheap from mock JSON), but the Output Node now returns only what matches the classified intent.

**Intent → extraction mapping:**

| `query_intent` | What is returned |
| :--- | :--- |
| `MEDICATIONS` | Medication extractions only |
| `ALLERGIES` | Allergy extraction only |
| `INTERACTIONS` | Interaction extractions only |
| `SAFETY_CHECK` | Allergy + medication extractions |
| `GENERAL_CLINICAL` | All extractions (original behavior) |
| `OUT_OF_SCOPE` | Standard refusal string — no extractions |

**Do not revert.** Returning all extractions regardless of intent makes the agent clinically unusable — a doctor asking about allergies should not have to parse through a full medication and interaction list to find the answer.

---
