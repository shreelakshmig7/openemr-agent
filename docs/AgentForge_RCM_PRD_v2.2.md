# AgentForge: Healthcare RCM AI Agent - Final Project Specification

**Author:** Shreelakshmi Gopinatha Rao
**Version:** 2.2 (Final Submission Version)
**Date:** February 2026
**Status:** Post-MVP / Final Submission Implementation

---

## 1. Executive Summary

AgentForge is a production-ready, multi-agent AI system designed for **Healthcare Revenue Cycle Management (RCM)**. It automates the extraction of clinical evidence from provider notes and verifies it against insurance payer policies to reduce claim denials and accelerate patient care. The system bridges the gap between unstructured provider notes and structured payer policy requirements, ensuring every insurance approval recommendation is backed by verbatim, audited evidence.

---

## 2. Core Architecture: LangGraph State Machine

The agent is implemented as a cyclic graph using **LangGraph** to support iterative verification and human-in-the-loop triggers.

### 2.1 Node Task Breakdown

1. **Supervisory Router Node (The "Pirate" Fix)**
   - **Task:** Analyze user intent at the entry point. If the query is out-of-scope (e.g., "talk like a pirate", "weather"), return a professional refusal.
   - **Logic:** `if intent not in ['clinical', 'rcm', 'patient_data']: return "I am a specialized Healthcare RCM agent. I can only assist with clinical documentation and insurance verification."`

2. **Extractor Node**
   - **Task:** Use `Claude Haiku` to parse clinical PDFs via `unstructured.io`.
   - **Requirement:** Extract evidence in structured JSON including a mandatory `verbatim_quote` and source location.

3. **Auditor Node**
   - **Task:** Use `Claude 3.5 Sonnet` to independently verify that every citation exists verbatim in the source text.
   - **Review Loop:** If a quote is modified or hallucinated, route back to **Extractor** (Max 3 iterations).

4. **Clarification Node**
   - **Task:** Pause state using a `pending_user_input` flag if data is ambiguous (e.g., conflicting laterality).
   - **Requirement:** Save current extraction progress to **SQLite** before pausing to ensure no work is lost.

## 2.2 Memory System & State Persistence
To satisfy the requirement for multi-turn reasoning and clinical continuity:

* **Conversation History:** The `messages` key in the graph state tracks dialogue history. This allows the agent to resolve pronouns (e.g., "What are HIS medications?") to the previously identified patient.
* **State Persistence:** Implemented via `SqliteSaver` checkpointer. Every node transition (Router -> Extractor -> Auditor) is saved to a local `agent_checkpoints.sqlite` database.
* **Context Management:** The agent maintains a `clinical_context` object in the state (Patient ID, Policy IDs), ensuring the agent doesn't re-run expensive tools for every follow-up message.
* **Resumption:** If the agent pauses at the `Clarification Node`, the state is persisted. When a human provides the missing detail (e.g., "Left knee"), the agent resumes from the exact point of pause.
---

## 3. Tool Registry & OpenEMR Fork Contribution

We are forking **OpenEMR** to add a native **RCM Evidence Extension**, ensuring deep data-level integration.

### 3.1 OpenEMR Fork Contribution

- **Feature:** Implemented a new SQL table `rcm_evidence_ledger` and a corresponding **REST API Endpoint** (`/api/rcm/verify`) in the OpenEMR fork.
- **Purpose:** This creates a permanent audit trail within the EHR, linking specific clinical notes directly to insurance policy IDs and audited evidence.

### 3.2 Full Tool Registry (6 Tools)

| Tool | Technology | Purpose |
| :--- | :--- | :--- |
| `pdf_extractor` | `unstructured.io` | Extracts text/tables from scanned clinical notes. |
| `policy_search` | `Pinecone` | RAG search over 200+ page payer policy PDFs. |
| `patient_lookup` | `OpenEMR API` | Retrieves patient demographics via forked FHIR R4 endpoints. |
| `med_retrieval` | `OpenEMR API` | Pulls active medications from `MedicationRequest` resources. |
| `denial_analyzer` | `Custom Logic` | Compares current notes to historical denial patterns to predict rejections. |
| `evidence_ledger_write` | `OpenEMR Tool` | Writes audited citations back to the forked OpenEMR ledger. |

### 3.3 Internal Messaging Integration
The agent is integrated into the OpenEMR Internal Mail/Portal system to automate administrative tasks:
* **Trigger:** The agent pre-scans incoming patient messages for medication or procedure requests.
* **Pre-Processing:** Before a provider opens a message, the agent runs the `Extractor` and `Auditor` nodes against the patient's chart.
* **Drafting:** The agent generates a "Verification Summary" that is prepended to the message thread, providing the doctor with a pre-audited recommendation.
---

## 4. Verification & Confidence Scoring

**The 90% Confidence Mechanism:**

The confidence score is a weighted hybrid metric calculated per query:

1. **Auditor Pass Rate (50%):** First-pass verbatim quote verification success.
2. **Citation Density (30%):** Number of unique document sections supporting the clinical claim.
3. **Self-Assessment (20%):** LLM uncertainty rating based on document clarity.
   - **Risk Note:** Self-assessment scores are normalized to prevent the model's inherent overconfidence from artificially inflating the final score.

**Threshold:** If the combined score is **< 90%**, the agent appends a `[LOW_CONFIDENCE_WARNING]` and triggers human escalation.

---

## 5. Observability Implementation (LangSmith)

We track three domain-specific metrics via **LangSmith** to ensure RCM reliability:

- **Faithfulness (Groundedness):** Measures if the answer is derived solely from the retrieved clinical context.
- **Citation Accuracy:** Automated Python check verifying that every `quote` in the output exists as a 1:1 substring match in the source PDF.
- **Review Loop Rate:** Tracks the frequency of Auditor-to-Extractor loops.
  - **Threshold:** A rate > 20% triggers an alert for prompt recalibration; this threshold accounts for expected noise in real-world scanned document OCR quality.

  ### 5.1 Real-Time Monitoring
During the final demo, the following LangSmith dashboards will be utilized to prove system integrity:
* **Trace View:** Visualizing the "Review Loop" between the Auditor and Extractor.
* **Latency Heatmap:** Confirming the < 30s processing time for complex multi-agent chains.
* **Score Distribution:** Real-time view of the Faithfulness and Citation Accuracy scores for the 50-case eval suite.

---

## 6. Evaluation Framework (50+ Test Cases)

The test suite is automated using the Gauntlet YAML standard.

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

---

## 7. Performance & Compliance

- **Latency Target:** < 30 seconds per chart. We prioritize Verification Accuracy (> 90%) over sub-5-second speed to prevent costly insurance denials.
- **Compliance:** HIPAA-compliant architecture per current OCR guidance using local PII scrubbing via Microsoft Presidio before cloud LLM calls.
- **Disaster Recovery:** RTO < 15 minutes; state is checkpointed to SQLite at every node transition to allow resumption after failure.

---

## 8. AI Cost Projections (Monthly Scale)

Projections based on a hybrid `Claude 3.5 Sonnet` and `Haiku` strategy.

| Scale | Users | Charts/Month | Est. Cost (USD) |
| :--- | :--- | :--- | :--- |
| **Pilot** | 100 | 5,000 | $80 – $150 |
| **Clinic** | 1,000 | 50,000 | $600 – $1,200 |
| **Hospital** | 10,000 | 500,000 | $5,000 – $9,000 |
| **Enterprise** | 100,000 | 5,000,000 | $40,000 – $70,000 |

---

## 9. Open Source Release

- **Primary Contribution:** The full 50-case RCM Eval Dataset will be released on **HuggingFace** under the **MIT License** by the Sunday deadline.
