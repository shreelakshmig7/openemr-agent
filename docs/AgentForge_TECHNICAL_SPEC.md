# AgentForge — Healthcare RCM AI Agent
## Technical Specification

**Author:** Shreelakshmi Gopinatha Rao
**Project:** AgentForge — Gauntlet AI Program
**Version:** 1.0
**Date:** February 2026

---

## 1. System Overview

The agent is a Python application built on LangChain (MVP) and LangGraph (full project). It exposes a FastAPI REST API that accepts natural language queries, routes them through an AI reasoning layer, calls healthcare data tools, applies safety verification, and returns a cited response.

---

## 2. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.9 |
| AI Framework (MVP) | LangChain | Latest |
| AI Framework (Full) | LangGraph | Latest |
| LLM — Reasoning | Claude Sonnet | claude-sonnet-4-20250514 |
| LLM — Extraction | Claude Haiku | claude-haiku-4-5-20251001 |
| API Server | FastAPI + Uvicorn | Latest |
| Observability | LangSmith | Latest |
| Data (MVP) | Mock JSON files | — |
| Data (Full) | OpenEMR FHIR R4 | — |

---

## 3. File Structure

### MVP

```
openemr-agent/
├── .env                          API keys and config
├── .gitignore                    Protects secrets and venv
├── .cursorignore                 Protects docs folder
├── mock_data/
│   ├── patients.json             3 mock patients
│   ├── medications.json          Medications per patient
│   └── interactions.json         Drug interaction rules
├── tools.py                      ✅ BUILT — 3 core tools
├── agent.py                      ✅ BUILT — LangChain agent
├── conversation.py               Multi-turn conversation manager
├── verification.py               Safety verification layer
├── healthcare_guidelines.py      Hardcoded compliance constants
├── eval/
│   ├── golden_data.yaml          Test cases in Gauntlet format
│   └── run_eval.py               Eval runner
├── main.py                       FastAPI server
├── tests/
│   ├── __init__.py               ✅ BUILT
│   ├── test_tools.py             ✅ BUILT — 16 tests
│   ├── test_agent.py             ✅ BUILT — 5 tests
│   ├── test_conversation.py      To be built
│   ├── test_verification.py      To be built
│   ├── test_eval.py              To be built
│   └── results/                  Test result logs
└── docs/                         Reference documents — DO NOT EDIT
```

### Full Project (additions)

```
openemr-agent/
├── langgraph_agent/
│   ├── __init__.py
│   ├── state.py                  LangGraph state schema
│   ├── extractor_node.py         Node A — extracts clinical data
│   ├── auditor_node.py           Node B — validates extractions
│   ├── clarification_node.py     Handles ambiguous inputs
│   └── workflow.py               Assembles full graph
├── observability/
│   ├── __init__.py
│   ├── tracer.py                 LangSmith tracing setup
│   └── metrics.py                Custom RCM metrics
└── eval_dataset/
    ├── README.md
    ├── healthcare_rcm_eval_v1.json
    └── LICENSE
```

---

## 4. Component Descriptions

### 4.1 tools.py ✅ BUILT

Three functions that retrieve data from mock JSON files.

**get_patient_info(name: str) -> dict**
- Accepts full or partial patient name
- Case-insensitive search
- Returns patient ID, age, gender, allergies, conditions
- Returns error dict if name is empty or patient not found

**get_medications(patient_id: str) -> dict**
- Accepts patient ID string
- Returns list of medications with name, dose, frequency, prescribed date
- Returns error dict if patient ID not found

**check_drug_interactions(medications: list) -> dict**
- Accepts list of medication names (strings or dicts with name key)
- Checks every pair against interaction database
- Returns all interactions found with severity and recommendation
- Returns no-interaction message if none found

**Data loading:**
- All 3 JSON files loaded once at module startup
- Each load wrapped in try/except — module never crashes on missing files
- Falls back to empty list/dict if file is missing or malformed

---

### 4.2 agent.py ✅ BUILT

LangChain agent connecting Claude to the 3 tools.

**create_agent() -> AgentExecutor**
- Initializes Claude Sonnet as the LLM (temperature=0)
- Wraps all 3 tools as LangChain @tool decorated functions
- System prompt enforces: tool call order, citation requirement, safety focus
- Returns configured AgentExecutor with verbose=True, max_iterations=5
- LangSmith tracing enabled via environment variables

**Tool call order enforced by system prompt:**
1. tool_get_patient_info → get patient ID
2. tool_get_medications → get medication list
3. tool_check_drug_interactions → check interactions
4. Synthesize into cited response

---

### 4.3 conversation.py — ✅ BUILT

Manages multi-turn conversation history.

**create_conversation_agent() -> Tuple[AgentExecutor, List]**
- Creates a new agent instance
- Returns agent and empty history list

**chat(agent, history: List, message: str) -> Tuple[str, List]**
- Sends message to agent with history context prepended
- Handles empty input without crashing
- Updates and returns history after each turn
- Returns error message string on failure — never raises exception

---

### 4.4 healthcare_guidelines.py — ✅ BUILT

Module-level constants only. No functions. Contains:

- FDA_RULES: severity levels requiring physician review, auto-approve threshold
- HIPAA_RULES: PII fields to scrub, retention requirements
- ICD10_RULES: code format validation pattern
- CLINICAL_SAFETY_RULES: confidence threshold (0.90), escalation rules, disclaimer text
- SEVERITY_LEVELS: definitions for CONTRAINDICATED, HIGH, MEDIUM, LOW

---

### 4.5 verification.py — ✅ BUILT

Four functions enforcing healthcare safety rules.

**check_allergy_conflict(drug: str, allergies: List[str]) -> dict**
- Case-insensitive comparison of drug against allergy list
- Returns conflict details with HIGH severity if match found
- Always includes source citation

**calculate_confidence(tools_succeeded: int, tools_total: int, interactions_found: bool, allergy_conflict: bool) -> float**
- Base score = tools_succeeded / tools_total
- Deducts 0.20 for allergy conflict
- Deducts 0.10 for interactions found
- Returns float clamped between 0.0 and 1.0

**should_escalate_to_human(confidence_score: float) -> dict**
- Compares score against CLINICAL_SAFETY_RULES threshold (0.90)
- Returns escalate: True with reason if below threshold
- Returns escalate: False with disclaimer if above threshold

**apply_fda_rules(interaction_severity: str) -> dict**
- Maps severity string to FDA action requirements
- HIGH and CONTRAINDICATED always require physician review
- LOW can auto-approve with monitoring note
- Always includes FDA source citation

---

### 4.6 eval/golden_data.yaml — ✅ BUILT

Test cases in Gauntlet YAML format. Minimum 5 for MVP, 50+ for full project.

**Required fields per test case:**
- id: unique identifier (gs-001 format)
- category: happy_path, edge_case, or adversarial
- query: natural language input
- expected_tools: list of tool names expected to be called
- must_contain: list of keywords response must include
- must_not_contain: list of keywords response must never include
- difficulty: happy_path, edge_case, or adversarial

---

### 4.7 eval/run_eval.py — ✅ BUILT

**run_eval(test_cases_path: str) -> dict**
- Loads test cases from golden_data.yaml
- Creates fresh agent for each test case
- Checks must_contain keywords (any match = pass)
- Checks must_not_contain keywords (any match = fail)
- Returns total, passed, failed, pass_rate, per-case results, timestamp
- Saves results to tests/results/ with timestamp
- Prints pass/fail per case with latency

**Regression rule:**
- run_eval must be called after every new feature is added
- A previously passing case that now fails = regression = do not proceed

---

### 4.8 main.py — TO BUILD

FastAPI server exposing the agent as a REST API.

**GET /health**
- Returns service name, version, status, timestamp

**POST /ask**
- Request body: question (string), session_id (optional string)
- Creates or retrieves session by session_id
- Calls chat() with session history
- Returns answer, session_id, timestamp

**POST /eval**
- Runs run_eval() against golden_data.yaml
- Saves results to tests/results/
- Returns full results dict

**GET /eval/results**
- Returns latest saved eval results
- Returns helpful message if no results exist yet

---

## 5. LangGraph Multi-Agent Architecture (Full Project)

### State Schema

```
AgentState:
  input_query: str
  documents_processed: List[str]
  extractions: List[dict]
  audit_results: List[dict]
  pending_user_input: bool
  clarification_needed: str
  iteration_count: int
  confidence_score: float
  final_response: str
  error: Optional[str]
```

**pending_user_input is critical:** When True, the entire workflow pauses without discarding any completed work. On user response, workflow resumes from exactly where it stopped.

### Node Routing

```
START
  → Extractor Node
      → Auditor Node
          → [audit passes] Output Node → END
          → [citation missing, iteration_count < 3] Review Loop → Extractor Node
          → [ambiguity detected] Clarification Node → [user responds] Extractor Node
          → [iteration_count >= 3] Output Node (partial) → END
```

### Model Split

| Node | Model | Reason |
|------|-------|--------|
| Extractor | claude-haiku-4-5-20251001 | Cost-efficient for extraction |
| Auditor | claude-sonnet-4-20250514 | Accuracy required for verification |
| Clarification | claude-sonnet-4-20250514 | Nuanced understanding required |

---

## 6. Data Flow (MVP)

```
User query (natural language)
    ↓
FastAPI /ask endpoint
    ↓
conversation.py — prepend history context
    ↓
agent.py — Claude reasons about which tools to call
    ↓
tools.py — fetch patient data from mock JSON
    ↓
verification.py — allergy check, confidence score, FDA rules
    ↓
Escalation check — if confidence < 0.90, add physician review recommendation
    ↓
Synthesized cited response
    ↓
FastAPI response → user
```

---

## 7. Environment Variables

```
ANTHROPIC_API_KEY         Required — Claude API access
LANGSMITH_API_KEY         Required — observability tracing
LANGCHAIN_TRACING_V2      Set to "true" to enable tracing
LANGCHAIN_PROJECT         Project name in LangSmith (default: openemr-agent)
OPENEMR_BASE_URL          OpenEMR instance URL (full project)
```

---

## 8. Dependencies

```
langchain
langchain-anthropic
langchain-core
langgraph
anthropic
fastapi
uvicorn
python-dotenv
pytest
pyyaml
```

---

## 9. Deployment

**Platform:** Railway or Render
**Process:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
**Required env vars:** Set in Railway dashboard — never in code
**Health check:** GET /health must return 200 OK

---

## 10. Git Remote Setup

```
origin    → private GitHub repo    daily work
fork      → public OpenEMR fork    open source contribution only
upstream  → openemr/openemr        upstream updates
```

At submission: cherry-pick eval dataset commits to fork, open PR to upstream.
