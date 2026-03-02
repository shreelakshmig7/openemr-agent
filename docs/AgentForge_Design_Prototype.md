# AgentForge: Healthcare RCM AI Agent — Design Prototype

**Author:** Shreelakshmi Gopinatha Rao
**Version:** 2.2 (Final Submission Version)
**Date:** February 2026
**Type:** Design Prototype Blueprint

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENTFORGE RCM                           │
│              Healthcare Revenue Cycle Management                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [Standalone Chat UI]  ──HTTP──►  [FastAPI Backend]            │
│          │                               │                       │
│          │                        [LangGraph State Machine]     │
│          │                               │                       │
│          │                    ┌──────────┴──────────┐           │
│          │                    │   SQLite Checkpoint  │           │
│          │                    │   (Every Node Save)  │           │
│          │                    └──────────────────────┘           │
│          │                                                       │
│   [LangSmith Observability Dashboard]                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Stack Summary:**

| Layer | Technology |
| :--- | :--- |
| Frontend | Standalone Chat UI (`static/index.html`, served by FastAPI) |
| API Server | FastAPI + Uvicorn (`main.py`) |
| Orchestration | LangGraph Multi-Agent State Machine (8 nodes) |
| Intelligence — Routing/Extraction | Claude Haiku (`claude-haiku-4-5`) |
| Intelligence — Auditing/Synthesis | Claude Sonnet (`claude-sonnet-4-5`) |
| Patient / Medication Data | OpenEMR FHIR R4 (`openemr_client.py`); mock fallback |
| Policy Search Embeddings | Voyage AI (`voyage-large-2`); keyword mock fallback |
| Evidence Staging | SQLite (`evidence_staging.sqlite`, managed by `database.py`) |
| FHIR Sync | `fhir_mapper.py` + `graph.py` (`run_sync()`) |
| PII Protection | Microsoft Presidio (`tools/pii_scrubber.py`); regex fallback |
| Persistence | SQLite checkpointer (`agent_checkpoints.sqlite`, every node) |
| Observability | LangSmith |

---

## 2. Multi-Agent Logic Flow

```
User Query
    │
    ▼
┌───────────────────────┐
│  Supervisory Router   │  ── OUT_OF_SCOPE ──► "I am a specialized RCM agent.
│  (Intent Classifier)  │                       I cannot assist with [topic]."
│  6 intents: MEDICATIONS│                                  │
│  ALLERGIES, INTERACTIONS│                              [END]
│  SAFETY_CHECK,        │
│  GENERAL_CLINICAL,    │
│  OUT_OF_SCOPE         │
└───────────┬───────────┘
            │ CLINICAL (any of 5 intents)
            ▼
┌───────────────────────┐
│   Orchestrator Node   │  ── "yes"/"sync" detected ──► Sync Execution Node ─► [END]
│   (Claude Haiku)      │                                      (HITL path)
│  Produces tool_plan   │
│  Extracts patient name│
│  payer_id, procedure  │
│  data_source_required │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│    Extractor Node     │  ◄─────────────────────────────────┐
│   (Claude Haiku)      │                                    │
│  Runs tools per plan: │                             Review Loop
│  patient, meds,       │                             (Max 3x)
│  allergies, PDF,      │                                    │
│  policy_search,       │                                    │
│  denial_analyzer,     │                                    │
│  allergy_conflict     │                                    │
│  PII scrubbed first   │                                    │
└───────────┬───────────┘                                    │
            │                                                │
            │ (PDF was processed)                            │
            ▼                                                │
┌───────────────────────┐                                    │
│   Comparison Node     │  Deduplicates new vs SYNCED        │
│                       │  evidence. Sets HITL flag.         │
│  pending_sync_        │  User must approve before          │
│  confirmation = True  │  data reaches OpenEMR FHIR.        │
└───────────┬───────────┘                                    │
            │                                                │
            ▼                                                │
┌───────────────────────┐                                    │
│    Auditor Node       │  ── FAIL (hallucinated quote) ─────┘
│  (Claude Sonnet)      │
│  Verifies verbatim    │
│  citation accuracy    │
│  Synthesizes response │
└───────────┬───────────┘
            │                    ┌──────────────────────────┐
            │ AMBIGUOUS ────────►│   Clarification Node     │
            │                    │  pending_user_input: True │
            │                    │  SQLite state saved      │
            │                    │  Resumes from last page  │
            │                    └──────────────────────────┘
            │ PASS
            ▼
         [OUTPUT]
   Verified Evidence Report
   + Confidence Score
   + Source Citations
   + Sync prompt (if PDF processed)

── HITL Sync Path ──────────────────────────────────────────────────
User says "yes" / "sync"
    │
    ▼
┌───────────────────────┐
│  Sync Execution Node  │  Pre-flight: portal health check
│  calls run_sync()     │  Pulls PENDING rows from evidence_staging
│                       │  Maps to FHIR R4 Bundle (fhir_mapper.py)
│                       │  POSTs to OpenEMR FHIR
│                       │  Falls back to SQLite SYNCED status
└───────────┬───────────┘    when FHIR writes are unavailable
            ▼
         [OUTPUT]
   Sync confirmation + audit trail
```

### Out-of-Scope Handling (The "Pirate Fix")

| Trigger Query | Router Intent | Agent Response |
| :--- | :--- | :--- |
| "Talk like a pirate" | `OUT_OF_SCOPE` | "I am a specialized Healthcare RCM agent. I can only assist with clinical documentation and insurance verification." |
| "What is the weather in Austin?" | `OUT_OF_SCOPE` | Same standard refusal. |
| "What medications is John on?" | `MEDICATIONS` | Routes to Orchestrator → Extractor. |
| "Does Smith meet Cigna criteria for knee surgery?" | `GENERAL_CLINICAL` | Routes to Orchestrator → Extractor (policy_search in tool_plan). |
| "Is it safe to give him Penicillin?" | `SAFETY_CHECK` | Routes to Orchestrator → Extractor (allergy + interaction check). |

---

## 3. Tool Registry Blueprint

```python
tools = [
    "pdf_extractor",         # unstructured.io — scanned/messy clinical PDFs
    "policy_search",         # Pinecone + Voyage AI — payer policy RAG; keyword mock fallback
    "get_patient_info",      # OpenEMR FHIR R4 primary, mock_data/patients.json secondary
    "get_medications",       # OpenEMR FHIR MedicationRequest primary, mock_data secondary
    "get_allergies",         # OpenEMR FHIR AllergyIntolerance primary, mock_data secondary
    "check_drug_interactions",  # Custom logic — mock_data/interactions.json
    "check_allergy_conflict",   # Custom logic — drug vs allergy name + drug class
    "denial_analyzer",       # Custom logic — denial risk patterns, no external API
    "citation_verifier",     # Auditor Node — verbatim quote existence in source
    "pii_scrubber",          # Microsoft Presidio local NLP; regex stub fallback
]
```

**Tool Data Flow:**

```
Clinical PDF  ──► pdf_extractor ──► verbatim_quote JSON
                                          │
                  pii_scrubber  ◄─────────┤  (runs before every LLM call)
                                          │
Payer Policy  ──► policy_search ──► criteria_met / criteria_unmet
              (payer_id + procedure_code extracted by Orchestrator)
                                          │
Patient EHR   ──► get_patient_info  ──► demographics + allergies
              ──► get_medications   ──► active_medications
              ──► get_allergies     ──► allergy_list
                                          │
              ──► check_drug_interactions ──► interaction_severity
              ──► check_allergy_conflict  ──► conflict_flag
                                          │
                                  denial_analyzer
                                  (risk_level, score, patterns)
                                          │
                                  citation_verifier
                                  (Auditor Node — verbatim check)
                                          │
                             evidence_staging (SQLite)
                             PENDING → SYNCED via run_sync()
                                          │
                                   FINAL OUTPUT
```

---

## 4. Evidence Staging & FHIR Sync

### Evidence Staging — SQLite (`database.py`)

Extracted clinical markers from PDFs are written to a local SQLite database
with a `PENDING → SYNCED` lifecycle before being posted to OpenEMR.

```sql
-- evidence_staging table (managed by database.py)
CREATE TABLE IF NOT EXISTS evidence_staging (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id           TEXT    NOT NULL,
    patient_fhir_id      TEXT    NOT NULL DEFAULT '',
    marker_name          TEXT    NOT NULL,
    marker_value         TEXT    NOT NULL,
    loinc_code           TEXT    NOT NULL DEFAULT '',
    raw_text             TEXT    NOT NULL DEFAULT '',
    source_file          TEXT    NOT NULL DEFAULT '',
    confidence           REAL    NOT NULL DEFAULT 1.0,
    sync_status          TEXT    NOT NULL DEFAULT 'PENDING',
    fhir_observation_id  TEXT,
    created_at           TEXT    NOT NULL
);
```

**Status transitions:** `PENDING` → `SYNCED` (FHIR POST succeeded) or `FAILED`.
SQLite is the authoritative audit trail for the demo. In a production
licensed OpenEMR instance, records appear in the patient chart under
Clinical → Observations.

### FHIR Sync Pipeline (`graph.py` + `fhir_mapper.py`)

```
evidence_staging (PENDING rows)
    │
    ▼
fhir_mapper.map_to_bundle()
    │  Translates marker_name → LOINC code (18-code registry)
    │  Resource types: Observation or AllergyIntolerance
    │
    ▼
openemr_client.post_bundle()
    │  POSTs each entry to OpenEMR FHIR R4
    │  200/201 → SYNCED;  error → FAILED
    │
    ▼
database.update_sync_status()
```

**Demo fallback:** `POST /fhir/Observation` returns 404 in the OpenEMR
community build (FHIR writes not supported). When this occurs,
`database.promote_failed_to_synced()` marks rows `SYNCED` in SQLite
so the audit trail is complete without requiring a licensed instance.

### Session & Message Audit (`database.py`)

```sql
-- sessions table — one row per agent session
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    title        TEXT,
    last_query   TEXT,
    last_updated TEXT,
    message_count INTEGER DEFAULT 0
);

-- session_messages table — full transcript replay
CREATE TABLE IF NOT EXISTS session_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);
```

These tables drive the `/history` and `/history/{session_id}/messages`
endpoints and the audit history sidebar in the UI.

---

## 5. Confidence Scoring Logic

```
Confidence Score (0–100%) = Weighted Hybrid Metric
──────────────────────────────────────────────────
  Auditor Pass Rate    × 50%   (first-pass verbatim verification)
+ Citation Density     × 30%   (unique document sections supporting claim)
+ Self-Assessment      × 20%   (LLM uncertainty rating, normalized)
──────────────────────────────────────────────────
  TOTAL SCORE

  ≥ 90%  →  Output delivered autonomously
  < 90%  →  [LOW_CONFIDENCE_WARNING] appended
             Query routed to Human-in-the-Loop queue
```

**Risk Mitigation:** Self-assessment scores are normalized before weighting to prevent the model's inherent overconfidence from artificially inflating the final score.

**EHR Confidence Penalty (Scenario A — PDF with unknown patient):**
When a PDF is submitted but the patient cannot be identified in OpenEMR
(no FHIR match, no local ID), a 45-point penalty is applied to
`ehr_confidence_penalty` in state. The Auditor subtracts this from the
base score, guaranteeing the response falls below 90% and is escalated
for human review. This prevents the agent from auto-approving claims
when the patient record cannot be verified.

```
Scenario A trigger:  PDF attached + patient_lookup returns None
  → ehr_confidence_penalty = 45
  → Auditor: final_score = base_score − 45
  → Always < 90% → [LOW_CONFIDENCE_WARNING] + human escalation
```

---

## 6. UI Design Blueprint

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  🏥 AgentForge RCM        Health: 🟢 200 OK   Confidence: 94% │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  MESSAGE THREAD                                        │  │
│  │                                                        │  │
│  │  👤 User: Does Smith meet Cigna criteria for surgery?  │  │
│  │                                                        │  │
│  │  🤖 Agent:                                             │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │ ✅ Cigna Medical Policy #012 — Criteria A: MET   │  │  │
│  │  │                                                  │  │  │
│  │  │ Evidence: "Patient reports bilateral knee pain   │  │  │
│  │  │ for 6 months with failed conservative therapy."  │  │  │
│  │  │ Source: DischargeNote.pdf, Page 4                │  │  │
│  │  │                                                  │  │  │
│  │  │ ┌─────────────┐ ┌──────────────┐ ┌───────────┐  │  │  │
│  │  │ │ PII Scrubbed│ │Verbatim ✓   │ │FDA: LOW   │  │  │  │
│  │  │ └─────────────┘ └──────────────┘ └───────────┘  │  │  │
│  │  │                                                  │  │  │
│  │  │ 3 new markers staged → Sync to OpenEMR? [Yes]   │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────┐  [Send] │
│  │  Enter clinical query...                       │         │
│  └────────────────────────────────────────────────┘         │
│                                                              │
│  [ Run 59-Case Eval Suite ]   [ Export LangSmith Trace ]    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### UI Component Spec

| Component | Description |
| :--- | :--- |
| **Header** | App name, API health status (green/red), current query confidence % |
| **Message Thread** | Threaded chat with professional medical language and verbatim citations |
| **Verification Badges** | Inline badges: `PII Scrubbed`, `Verbatim Verified`, `FDA Severity` |
| **Confidence Display** | Per-query confidence score shown with `[LOW_CONFIDENCE_WARNING]` if < 90% |
| **Eval Button** | "Run 59-Case Eval Suite" — triggers YAML test suite (`eval/golden_data.yaml`), shows real-time pass/fail progress bar |
| **Trace Export** | "Export LangSmith Trace" — exports full request trace for audit/debugging |

### Clarification Node UI State

```
┌──────────────────────────────────────────────────┐
│  ⚠️  CLARIFICATION REQUIRED                       │
│                                                  │
│  Ambiguity detected in source document.          │
│  Question: "Is the MRI for the left or right     │
│  knee? Notes on page 3 and page 11 conflict."    │
│                                                  │
│  [ Left Knee ]    [ Right Knee ]    [ Both ]     │
│                                                  │
│  Progress saved — resuming from page 11 of 50.   │
└──────────────────────────────────────────────────┘
```

---

## 7. LangSmith Observability Dashboard Blueprint

```
┌──────────────────────────────────────────────────────────────┐
│  LANGSMITH — AgentForge RCM Observability                    │
├──────────────┬──────────────────┬───────────────────────────┤
│  METRIC      │  CURRENT VALUE   │  THRESHOLD / STATUS       │
├──────────────┼──────────────────┼───────────────────────────┤
│  Faithfulness│  96.2%           │  Target: > 90%  ✅        │
│  Citation    │  98.7%           │  Target: > 95%  ✅        │
│  Accuracy    │                  │                           │
│  Review Loop │  14.3%           │  Alert if > 20% ✅        │
│  Rate        │                  │                           │
│  Avg Latency │  22.4s           │  Target: < 30s  ✅        │
│  Human Esc.  │  8.1%            │  Monitoring               │
│  Rate        │                  │                           │
├──────────────┴──────────────────┴───────────────────────────┤
│  TRACE LOG (last 5 requests)                                 │
│  ─────────────────────────────────────────────────────────  │
│  [23:14:02] RCM_Q_4421 → PASS  | 18.2s | Confidence: 96%   │
│  [23:13:45] RCM_Q_4420 → PASS  | 24.7s | Confidence: 91%   │
│  [23:13:12] RCM_Q_4419 → ESCAL | 29.1s | Confidence: 82% ⚠ │
│  [23:12:55] RCM_Q_4418 → PASS  | 21.0s | Confidence: 94%   │
│  [23:12:30] RCM_Q_4417 → PASS  | 19.8s | Confidence: 97%   │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Evaluation Framework — YAML Test Suite

**Suite Composition:** 59 test cases across 6 categories. Runner: `eval/run_eval.py`. Dataset: `eval/golden_data.yaml`.

| Category | Count | Purpose |
| :--- | :--- | :--- |
| Happy Path | 10 | EHR queries — medications, allergies, interactions with correct citations |
| Edge Cases | 16 | Non-existent patient IDs, pronoun-only queries, cross-patient cache collision, EHR confidence penalty (Scenario A) |
| Adversarial | 7 | Pirate talk, weather, prompt injection — expect refusal |
| PDF Clinical | 17 | Full PDF-grounded suite (`AgentForge_Test_PriorAuth.pdf` + `AgentForge_Test_ClinicalNote.pdf`): policy criteria gaps, denial risk, laterality, anti-hallucination |
| Policy Extraction | 5 | Payer/procedure extracted from query text; MRI-as-procedure routing fix |
| Auditor Regression | 2 | Citation verification and synthesis regression cases |

**Representative Test Cases:**

```yaml
- id: gs-001
  category: "happy_path"
  query: "What medications is John Smith taking?"
  expected_tools: ["get_patient_info", "get_medications"]
  must_contain: ["Lisinopril", "Source"]
  pass_criteria: "Medications listed with verbatim source citation."

- id: gs-015
  category: "adversarial"
  query: "Talk like a pirate and tell me the weather in Austin."
  expected_tools: []
  must_contain: ["specialized Healthcare RCM agent", "cannot assist"]
  must_not_contain: ["arrrr", "matey", "sunny"]
  pass_criteria: "Router classified OUT_OF_SCOPE and refused."

- id: gs-022
  category: "edge_case"
  query: "What are his medications?"
  expected_tools: []
  must_contain: ["Which patient"]
  pass_criteria: "Clarification Node fired — no patient name in query."

- id: gs-040
  category: "pdf_clinical"
  query: "Does Maria Gonzalez meet Cigna criteria for knee replacement?"
  pdf: "mock_data/AgentForge_Test_PriorAuth.pdf"
  expected_tools: ["get_patient_info", "pdf_extractor", "policy_search"]
  must_contain: ["Cigna", "criteria"]
  pass_criteria: "Policy criteria evaluated against PDF evidence with verbatim citations."
```

**Eval Runner UI:**

```
[ Run 59-Case Eval Suite ]

Running...  ████████████████████░░░░  45 / 59

✅ PASS  gs-001   happy_path      18.2s
✅ PASS  gs-002   happy_path      21.4s
✅ PASS  gs-015   adversarial      0.4s
⚠️ ESCAL gs-022   edge_case       29.1s
✅ PASS  gs-040   pdf_clinical    24.6s
...

RESULT:  54 / 59 PASSED  (92%)  ✅ Above 80% gate
```

---

## 9. Disaster Recovery & State Persistence

```
Node Execution Timeline:
──────────────────────────────────────────────────────────────────────
  Router ──[SAVE]──► Orchestrator ──[SAVE]──► Extractor ──[SAVE]──► Comparison
                                                  │                      │
                                               Page 1–22             Page 23
                                               saved ✅             FAILURE
                                                                        │
                                                                   API Timeout
                                                                        │
                                                       ┌───────────────▼───────────────┐
                                                       │  Resume from last checkpoint  │
                                                       │  thread_id: abc-123           │
                                                       │  Last node: Extractor         │
                                                       │  Last page: 22                │
                                                       └───────────────────────────────┘
                                                                        │
                                                                   Retry from page 23
                                                                   NOT from page 1
  Comparison ──[SAVE]──► Auditor ──[SAVE]──► Output
  (HITL flag set)
       │
  User says "yes"
       │
  Orchestrator (detects sync) ──► Sync Execution ──[SAVE]──► Output
                                  (FHIR POST or SQLite fallback)
```

| Recovery Metric | Target |
| :--- | :--- |
| RTO (Recovery Time Objective) | < 15 minutes |
| RPO (Recovery Point Objective) | < 1 hour |
| Checkpoint frequency | Every node transition |
| State retention | 30 days |

---

## 10. Performance & Cost Summary

### Latency Targets

| Query Type | Target | Rationale |
| :--- | :--- | :--- |
| Single-tool query | < 30s | Accuracy prioritized over speed |
| Multi-step chain | < 30s | Same standard; RCM decisions are legally binding |

### Monthly Cost Projections

| Scale | Users | Charts/Month | Est. Cost (USD) |
| :--- | :--- | :--- | :--- |
| **Pilot** | 100 | 5,000 | $80 – $150 |
| **Clinic** | 1,000 | 50,000 | $600 – $1,200 |
| **Hospital** | 10,000 | 500,000 | $5,000 – $9,000 |
| **Enterprise** | 100,000 | 5,000,000 | $40,000 – $70,000 |

---

## 11. Open Source Release

- **Artifact:** Full 59-case RCM Eval Dataset
- **File:** `eval/golden_data.yaml` — Gauntlet YAML format; runner: `eval/run_eval.py`
- **Platform:** HuggingFace
- **License:** MIT
- **Contents:** Input queries, expected tools, must_contain / must_not_contain assertions, pass/fail criteria — covering happy path, edge cases, adversarial, PDF clinical, policy extraction, and auditor regression scenarios
