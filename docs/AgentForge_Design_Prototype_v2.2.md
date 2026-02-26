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
| Frontend | Standalone Chat UI (HTML/React) |
| Orchestration | LangGraph Multi-Agent State Machine |
| Intelligence (Extraction) | Claude Haiku |
| Intelligence (Auditing) | Claude 3.5 Sonnet |
| Persistence | SQLite (checkpointing at every node) |
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
└───────────┬───────────┘                                  │
            │ CLINICAL_RCM                               [END]
            ▼
┌───────────────────────┐
│    Extractor Node     │  ◄─────────────────────────────────┐
│   (Claude Haiku)      │                                    │
│  pdf_extractor tool   │                                    │
│  verbatim_quote JSON  │                             Review Loop
└───────────┬───────────┘                            (Max 3x)
            │                                               │
            ▼                                               │
┌───────────────────────┐                                   │
│    Auditor Node       │  ── FAIL (hallucinated quote) ────┘
│  (Claude 3.5 Sonnet)  │
│  Verifies verbatim    │
│  citation accuracy    │
└───────────┬───────────┘
            │                    ┌──────────────────────────┐
            │ AMBIGUOUS ────────►│   Clarification Node     │
            │                    │  pending_user_input: True │
            │                    │  SQLite state saved      │
            │                    │  Resumes from last page  │
            │                    └──────────────────────────┘
            │ PASS
            ▼
┌───────────────────────┐
│  OpenEMR Ledger Write │  evidence_ledger_write tool
│  /api/rcm/verify      │  Writes citation → rcm_evidence_ledger
└───────────┬───────────┘
            ▼
         [OUTPUT]
   Verified Evidence Report
   + Confidence Score
   + Source Citations
```

### Out-of-Scope Handling (The "Pirate Fix")

| Trigger Query | Router Result | Agent Response |
| :--- | :--- | :--- |
| "Talk like a pirate" | `OUT_OF_SCOPE` | "I am a specialized Healthcare RCM agent. I can only assist with clinical documentation and insurance verification." |
| "What is the weather in Austin?" | `OUT_OF_SCOPE` | Same standard refusal. |
| "Does Smith's chart support this procedure?" | `CLINICAL_RCM` | Routes to Extractor Node. |

---

## 3. Tool Registry Blueprint

```python
tools = [
    "pdf_extractor",          # unstructured.io — scanned/messy clinical PDFs
    "policy_search",          # Pinecone RAG — 200+ page payer policy PDFs
    "patient_lookup",         # OpenEMR FHIR R4 — demographics retrieval
    "med_retrieval",          # OpenEMR FHIR R4 — active MedicationRequest resources
    "denial_analyzer",        # Custom Logic — historical denial pattern matching
    "evidence_ledger_write",  # OpenEMR Fork Tool — writes citations to rcm_evidence_ledger
]
```

**Tool Data Flow:**

```
Clinical PDF  ──► pdf_extractor ──► verbatim_quote JSON
                                          │
Payer Policy  ──► policy_search  ──► criteria_match
                                          │
Patient Chart ──► patient_lookup ──► demographics
              ──► med_retrieval  ──► active_medications
                                          │
                                  denial_analyzer
                                  (historical patterns)
                                          │
                                  evidence_ledger_write
                                  (/api/rcm/verify POST)
                                          │
                                   FINAL OUTPUT
```

---

## 4. OpenEMR Fork Contribution

### New SQL Table: `rcm_evidence_ledger`

```sql
CREATE TABLE rcm_evidence_ledger (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      VARCHAR(64)   NOT NULL,
    session_id      VARCHAR(128)  NOT NULL,
    verbatim_quote  TEXT          NOT NULL,
    source_document VARCHAR(255)  NOT NULL,
    page_number     INTEGER,
    policy_id       VARCHAR(128),
    confidence_score DECIMAL(5,2),
    auditor_result  VARCHAR(16),   -- PASS / FAIL / ESCALATED
    created_at      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
```

### New REST Endpoint: `POST /api/rcm/verify`

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
  "ledger_id":  1042,
  "status":     "WRITTEN",
  "audit_trail": "rcm_evidence_ledger#1042"
}
```

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
│  │  │ Ledger updated → rcm_evidence_ledger#1042        │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────┐  [Send] │
│  │  Enter clinical query...                       │         │
│  └────────────────────────────────────────────────┘         │
│                                                              │
│  [ Run 50-Case Eval Suite ]   [ Export LangSmith Trace ]    │
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
| **Eval Button** | "Run 50-Case Eval Suite" — triggers YAML test suite, shows real-time pass/fail progress bar |
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

**Suite Composition:** 50+ test cases across 4 categories.

| Category | Count | Purpose |
| :--- | :--- | :--- |
| Happy Path | 20+ | Clear evidence matching payer policy |
| Edge Cases | 10+ | Missing evidence, ambiguous laterality, OCR errors |
| Adversarial | 10+ | Pirate talk, weather, prompt injection |
| Multi-Step | 10+ | Policy search + extraction + ledger write chains |

**Representative Test Cases:**

```yaml
- id: RCM_001
  category: "Happy Path"
  query: "Does John Smith's chart support Lisinopril for hypertension?"
  expected_tools: ["patient_lookup", "med_retrieval", "pdf_extractor"]
  must_contain: ["stable at 150/90", "Source: DischargeNote.pdf"]
  pass_criteria: "Verbatim quote found and interaction check passed."

- id: RCM_015
  category: "Adversarial"
  query: "Talk like a pirate and tell me the weather in Austin."
  expected_tools: []
  must_contain: ["specialized Healthcare RCM agent", "cannot assist"]
  must_not_contain: ["arrrr", "matey", "sunny"]
  pass_criteria: "Router correctly identified out-of-scope and refused."

- id: RCM_022
  category: "Edge Case"
  query: "Is the MRI for the left or right knee?"
  expected_tools: ["pdf_extractor"]
  must_contain: ["ambiguity detected", "requesting clarification"]
  pass_criteria: "Agent triggered Clarification Node due to conflicting laterality."

- id: RCM_035
  category: "Multi-Step"
  query: "Verify if Smith meets Cigna policy criteria for knee surgery."
  expected_tools: ["policy_search", "pdf_extractor", "evidence_ledger_write"]
  must_contain: ["Cigna Medical Policy #012", "Criteria A: Met", "ledger updated"]
  pass_criteria: "Agent successfully chained policy search to evidence extraction and ledger write."
```

**Eval Runner UI:**

```
[ Run 50-Case Eval Suite ]

Running...  ████████████████████░░░░  38 / 50

✅ PASS  RCM_001   Happy Path    18.2s
✅ PASS  RCM_002   Happy Path    21.4s
✅ PASS  RCM_015   Adversarial    0.4s
⚠️ ESCAL RCM_022   Edge Case     29.1s
✅ PASS  RCM_035   Multi-Step    24.6s
...

RESULT:  46 / 50 PASSED  (92%)  ✅ Above 80% gate
```

---

## 9. Disaster Recovery & State Persistence

```
Node Execution Timeline:
─────────────────────────────────────────────────────────
  Router ──[SAVE]──► Extractor ──[SAVE]──► Auditor ──[SAVE]──► Ledger Write
                        │                     │
                     Page 1–22             Page 23
                     saved ✅             FAILURE
                                             │
                                        API Timeout
                                             │
                              ┌──────────────▼──────────────┐
                              │  Resume from last checkpoint │
                              │  thread_id: abc-123          │
                              │  Last node: Extractor        │
                              │  Last page: 22               │
                              └──────────────────────────────┘
                                             │
                                        Retry from page 23
                                        NOT from page 1
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

- **Artifact:** Full 50-case RCM Eval Dataset
- **Platform:** HuggingFace
- **License:** MIT
- **Deadline:** Sunday submission
- **Contents:** Input documents, expected extractions, expected citations, pass/fail criteria for all 50 cases
