# AgentForge — Healthcare RCM AI Agent
## Product Requirements Document (PRD)

**Author:** Shreelakshmi Gopinatha Rao
**Project:** AgentForge — Gauntlet AI Program
**Domain:** Healthcare Revenue Cycle Management
**Version:** 1.0
**Date:** February 2026

---

## 1. Problem Statement

Healthcare providers lose billions of dollars annually to insurance claim denials. A significant portion of these denials are caused by insufficient or incorrect clinical documentation — doctors prescribing medications that conflict with patient allergies, undetected dangerous drug interactions, and clinical notes that fail to meet payer coverage criteria.

Currently, reviewing a patient's medication history, checking for drug interactions, and verifying allergy conflicts is a manual, time-consuming process prone to human error. There is no intelligent system that connects patient records, drug interaction databases, and insurance coverage criteria in one place.

---

## 2. Solution

An AI agent that connects to a healthcare data system, retrieves patient records on demand, and automatically checks for dangerous drug interactions and allergy conflicts. The agent responds to natural language queries from doctors and clinical staff, synthesizes data from multiple sources, and flags safety concerns with evidence-based citations.

---

## 3. Domain

**Primary Domain:** Healthcare Revenue Cycle Management (RCM)

**Specific Focus:** Medication safety review — verifying that prescribed medications are safe for a patient given their known allergies, existing medications, and conditions.

**Data Sources:**
- Patient records (name, age, allergies, conditions)
- Current medication lists (drug name, dosage, frequency)
- Drug interaction database (severity levels, recommendations)
- Insurance payer policy criteria

---

## 4. Users

**Primary User:** Doctors and clinical staff reviewing patient medication safety before prescribing or authorizing treatment.

**Secondary User:** Insurance reviewers verifying clinical documentation supports coverage criteria.

---

## 5. User Stories

- As a doctor, I want to ask in plain English about a patient's medications so I don't have to navigate multiple systems.
- As a doctor, I want to be warned immediately if two medications interact dangerously so I can intervene before harm occurs.
- As a clinical staff member, I want to check if a new prescription conflicts with a patient's known allergies so I can flag it before dispensing.
- As a doctor, I want the agent to remember context within a conversation so I don't have to repeat patient information.
- As a reviewer, I want every agent claim to cite its source so I can verify accuracy independently.

---

## 6. MVP Requirements

The following are hard requirements. All must be met for MVP to be considered complete.

### 6.1 Natural Language Understanding
The agent must respond to plain English questions about patients and medications without requiring structured input or specific syntax.

**Acceptance Criteria:**
- Agent correctly interprets queries about named patients
- Agent correctly interprets queries about medications and interactions
- Agent responds in clear, professional medical language
- Agent never returns a raw error or crash to the user

### 6.2 Tool Execution
The agent must have at least 3 functional tools it can invoke to retrieve real data.

**Tools Required:**
- Patient lookup by name
- Medication retrieval by patient ID
- Drug interaction check across a list of medications

**Acceptance Criteria:**
- Each tool returns structured data
- Each tool handles invalid input gracefully
- Each tool returns an error message — never crashes
- Tools are invoked automatically based on the user's query

### 6.3 Multi-Tool Reasoning
The agent must be able to chain multiple tool calls in the correct order to answer a single query.

**Acceptance Criteria:**
- Agent calls patient lookup first, then medication retrieval, then interaction check — in that order
- Agent synthesizes results from all 3 tools into a single coherent response
- Agent cites the source of every claim in its response

### 6.4 Conversation History
The agent must maintain context across multiple turns in a single conversation.

**Acceptance Criteria:**
- Agent remembers patient context from earlier in the conversation
- A follow-up question referencing "he" or "she" is correctly resolved to the previously discussed patient
- History is maintained for the duration of the session

### 6.5 Error Handling
The agent must handle all failure scenarios gracefully.

**Acceptance Criteria:**
- Unknown patient returns a clear "not found" message — not a crash
- Empty input returns a helpful prompt asking for more information
- API failures return a safe error message directing the user to consult medical staff
- No raw Python exceptions are ever returned to the user

### 6.6 Domain Verification
The agent must perform at least one domain-specific safety check before returning a response.

**Verification Required:**
- Allergy conflict check: flag if a prescribed drug matches any of the patient's known allergies
- Confidence scoring: calculate certainty of response based on tool success rate
- Human escalation: trigger physician review when confidence falls below 90%
- FDA rule enforcement: HIGH severity interactions must always require physician review

**Acceptance Criteria:**
- Allergy conflict is always detected and flagged with HIGH severity
- HIGH severity drug interactions are never auto-approved
- Confidence below 90% always triggers an escalation recommendation
- Every verification result includes a source citation

### 6.7 Evaluation Framework
The agent must have an automated evaluation suite with at least 5 test cases covering different scenarios.

**Test Case Categories Required:**
- Happy path: valid queries with known expected outputs
- Edge cases: missing data, unknown patients, empty inputs
- Adversarial: attempts to extract harmful information or bypass safety rules

**Eval Format:** Gauntlet YAML standard
```
Each test case must contain:
  - id
  - category
  - query
  - expected_tools
  - must_contain
  - must_not_contain
  - difficulty
```

**Acceptance Criteria:**
- Minimum 5 test cases for MVP
- Pass rate must exceed 80%
- Eval suite runs automatically after every new feature (regression testing)
- Results saved with timestamp as proof

### 6.8 Deployment
The agent must be publicly accessible via a REST API.

**Endpoints Required:**
- Health check endpoint
- Query endpoint accepting natural language input
- Eval endpoint that runs the test suite on demand
- Results endpoint returning latest eval results

**Acceptance Criteria:**
- Public URL accessible without authentication
- Health check returns 200 OK
- Query endpoint returns response within 30 seconds
- Eval endpoint returns pass rate and per-case results

---

## 7. Full Project Requirements

Builds on MVP. All MVP requirements must be met before starting full project.

### 7.1 Multi-Agent Architecture
Replace the single LangChain agent with a LangGraph multi-agent state machine.

**Architecture:**
- Extractor Node: retrieves and extracts clinical evidence
- Auditor Node: independently validates every claim from Extractor
- Clarification Node: pauses workflow when input is ambiguous, requests human input without losing work
- Review Loop: Auditor can send work back to Extractor up to 3 times if evidence is missing

**Acceptance Criteria:**
- Agent correctly routes between nodes based on state
- Auditor catches at least one extraction error in adversarial test cases
- Clarification Node preserves full state when pausing — no work is lost on resume
- Review loop never exceeds 3 iterations

### 7.2 Observability
Full tracing and domain-specific metrics via LangSmith.

**Metrics Required:**
- Faithfulness: percentage of claims with valid source citations
- Answer Relevancy: did the response address the actual question
- Citation Accuracy: do cited quotes exist in source documents
- Review Loop Rate: how often Auditor sends back to Extractor
- Human Escalation Rate: how often confidence falls below 90%
- Tool Success Rate: percentage of tool calls that succeed

**Acceptance Criteria:**
- Every agent run produces a LangSmith trace
- All 6 metrics are calculated and logged per run
- LangSmith dashboard is shown in demo video

### 7.3 Full Evaluation Dataset
Minimum 50 test cases across all categories.

**Required Distribution:**
- 20+ happy path cases
- 10+ edge cases
- 10+ adversarial cases
- 10+ multi-step reasoning cases (3 or more tool calls)

**Acceptance Criteria:**
- Pass rate exceeds 80% on full dataset
- Results saved and timestamped
- Dataset published publicly as open source contribution

### 7.4 Open Source Contribution
One public contribution to the open source community.

**Primary:** Healthcare RCM evaluation dataset published on HuggingFace or GitHub with MIT license.

**Stretch Goal:** Installable Python package (only if all other requirements are met first).

---

## 8. Healthcare Compliance Requirements

These rules are non-negotiable and must be enforced programmatically — not left to LLM judgment.

| Rule | Standard | Requirement |
|------|----------|-------------|
| Drug interaction severity | FDA Drug Safety Guidelines | HIGH and CONTRAINDICATED interactions must always require physician review |
| Patient data privacy | HIPAA Privacy Rule | Patient names and identifiers must never appear in logs |
| Diagnosis codes | ICD-10-CM | All diagnosis codes must be validated before use |
| Clinical safety | Clinical best practices | Agent must never suggest dosage changes |
| Source citation | Evidence Attribution | Every claim must cite its source |
| Uncertainty | Clinical safety | Agent must escalate when confidence is below 90% |

---

## 9. Performance Targets

| Metric | Target |
|--------|--------|
| Single tool latency | < 5 seconds |
| Multi-step latency | < 30 seconds |
| Tool success rate | > 95% |
| Eval pass rate | > 80% |
| Hallucination rate | < 5% |
| Verification accuracy | > 90% |

---

## 10. Out of Scope for MVP

- Real OpenEMR patient data (mock data used for MVP)
- Pinecone vector database (mock interaction data used for MVP)
- Microsoft Presidio PII scrubbing (added in full project)
- LangGraph multi-agent architecture (added in full project)
- LangSmith observability (wired in full project)

---

## 11. Development Status

### Completed

| Component | Description | Test Coverage |
|-----------|-------------|---------------|
| Mock Data | 3 patients, medications, 6 drug interaction rules | — |
| Patient Lookup Tool | Look up patient by name, case-insensitive, partial match | 5/5 tests |
| Medication Tool | Retrieve medications by patient ID | 5/5 tests |
| Drug Interaction Tool | Check medication list against interaction database | 6/6 tests |
| LangChain Agent | Claude connected to all 3 tools with system prompt | 5/5 tests |

**Total tests passing: 21/21**

### Remaining — MVP

| Component | Description |
|-----------|-------------|
| Conversation Manager | Multi-turn history across sessions |
| Verification Layer | Allergy check, confidence scoring, FDA rules, escalation |
| Eval Framework | 5+ test cases in Gauntlet YAML format with regression runner |
| API Server | FastAPI with health, query, eval, and results endpoints |
| Deployment | Publicly accessible via Railway or Render |

### Remaining — Full Project

| Component | Description |
|-----------|-------------|
| LangGraph Multi-Agent | Extractor, Auditor, Clarification nodes with review loop |
| LangSmith Observability | 6 domain-specific metrics per run |
| Full Eval Dataset | 50+ test cases published as open source |

---

## 12. Submission Requirements

Due Sunday 10:59 PM CT:

| Item | Description |
|------|-------------|
| GitHub Repository | Setup guide, architecture overview, deployed link |
| Demo Video | 3-5 minutes showing agent, eval results, observability |
| Pre-Search Document | Architecture decisions and rationale |
| Agent Architecture Doc | 1-2 page technical breakdown |
| AI Cost Analysis | Dev spend and projections at 100/1K/10K/100K users |
| Eval Dataset | 50+ test cases with results |
| Open Source Link | Published dataset or package URL |
| Deployed Application | Publicly accessible agent |
| Social Post | X or LinkedIn post tagging @GauntletAI |
