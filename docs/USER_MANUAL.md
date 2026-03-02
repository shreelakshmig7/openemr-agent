# AgentForge AI — User Manual

**Product:** AgentForge Healthcare RCM AI Agent
**Audience:** Clinical staff, RCM coordinators, and system administrators
**Version:** 1.0 — March 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Getting Started](#2-getting-started)
3. [The Chat Interface](#3-the-chat-interface)
4. [Asking Questions](#4-asking-questions)
5. [PDF Upload and Inline Citations](#5-pdf-upload-and-inline-citations)
6. [Streaming Responses](#6-streaming-responses)
7. [Session and Chat History](#7-session-and-chat-history)
8. [New Case Workflow](#8-new-case-workflow)
9. [Drug Interaction Checking](#9-drug-interaction-checking)
10. [Allergy Checking](#10-allergy-checking)
11. [Medication Lookup](#11-medication-lookup)
12. [Denial Risk Analysis](#12-denial-risk-analysis)
13. [FHIR Evidence Sync (HITL)](#13-fhir-evidence-sync-hitl)
14. [PII Scrubbing and Privacy](#14-pii-scrubbing-and-privacy)
15. [Escalation and Confidence](#15-escalation-and-confidence)
16. [Tool Trace Inspector](#16-tool-trace-inspector)
17. [Audit Trail](#17-audit-trail)
18. [Eval System](#18-eval-system)
19. [System Health](#19-system-health)
20. [API Reference](#20-api-reference)
21. [Troubleshooting](#21-troubleshooting)

---

## 1. Overview

AgentForge is a multi-agent AI system embedded directly inside OpenEMR. It answers clinical Revenue Cycle Management (RCM) questions — medication lookups, allergy checks, drug interactions, safety checks, and prior authorisation support — by querying the live OpenEMR EHR via FHIR R4, processing uploaded clinical PDFs, and synthesising answers grounded in real patient data.

### What AgentForge does

| Capability | Source |
|---|---|
| Medication lookup | OpenEMR FHIR `MedicationRequest` |
| Allergy lookup | OpenEMR REST API + FHIR `AllergyIntolerance` |
| Drug interaction checking | Built-in interaction rules engine |
| Allergy conflict / safety check | Cross-referenced against live patient allergy list |
| Clinical PDF analysis | Uploaded PDF → page-level extraction → inline citation |
| Denial risk scoring | RCM pattern-matching engine |
| Evidence FHIR sync | Extracted biomarkers → OpenEMR Observations (with your approval) |
| Session history | All conversations persisted to SQLite, replayed on demand |

### What AgentForge does NOT do

- It does not modify patient records without your explicit confirmation (HITL sync).
- It does not answer non-healthcare questions — it will politely decline.
- It does not access external internet sources beyond OpenEMR and uploaded documents.

---

## 2. Getting Started

### Accessing AgentForge

1. Log in to OpenEMR at `http://<your-server>:8300/`
2. Click **AgentForge AI** in the top navigation bar
3. The AgentForge tab opens inside the OpenEMR tabbed interface

> **Important:** Do not use the browser's refresh button on the AgentForge page. The OpenEMR tab interface uses one-time session tokens — a browser refresh will log you out. To "re-open" AgentForge, simply click the **AgentForge AI** menu item again.

### Interface Layout

```
┌────────────────────────────────────────────────────────────┐
│  🏥 AgentForge AI   Healthcare RCM · Medication Safety     │
│                              [ Agent online ] [ Eval ] [ New Case ] │
├──────────────┬─────────────────────────────────────────────┤
│ SYSTEM AUDIT │                                             │
│ HISTORY      │            Chat area                        │
│              │                                             │
│  Session 1   │                                             │
│  Session 2   │                                             │
│  Session 3   │                                             │
│              ├─────────────────────────────────────────────┤
│              │  [ 📎 ] [  Type your question...      ] [Send] │
└──────────────┴─────────────────────────────────────────────┘
```

---

## 3. The Chat Interface

### Header Bar

| Element | Description |
|---|---|
| **Agent online** (green dot) | Confirms the AI backend is reachable and healthy |
| **Eval** button | Opens the evaluation results panel (see [Section 18](#18-eval-system)) |
| **New Case** button | Starts a fresh session while preserving history (see [Section 8](#8-new-case-workflow)) |

### Welcome Screen

When no conversation is active, AgentForge shows a welcome card with five clickable **prompt chips**:

- 💊 What medications is John Smith on?
- ⚠️ Check drug interactions for Mary Johnson
- 🚨 Is it safe to give Robert Davis Aspirin?
- 🩺 Does John Smith have any known allergies?
- 🚨 Is it safe to give Emily Rodriguez Amoxicillin?

Click any chip to instantly send that query. The chips are intended as starting examples — replace patient names and drug names as needed in the text box.

### Input Bar

| Element | Behaviour |
|---|---|
| **📎** (paperclip icon) | Opens the file picker to attach a PDF |
| **PDF badge** | Appears after upload. Shows filename. Click **✕** to remove the attachment |
| **Text area** | Type your question. Auto-expands as you type |
| **Enter** | Sends the message |
| **Shift + Enter** | Inserts a line break without sending |
| **Send button** | Greyed out while a response is in-flight |

---

## 4. Asking Questions

AgentForge understands natural language questions about patients registered in OpenEMR. The agent automatically identifies the patient from your query — no need for IDs or codes.

### Supported Question Types

| Intent | Example |
|---|---|
| **Medications** | "What medications is Sarah Connor on?" |
| **Allergies** | "Does John Smith have any known allergies?" |
| **Drug interactions** | "Check drug interactions for Mary Johnson" |
| **Safety check** | "Is it safe to give Robert Davis Aspirin?" |
| **General clinical** | "Summarise the clinical findings in this PDF" |

### Patient Identification

The agent identifies patients by **name and date of birth**. If two patients share the same name, the agent will ask for a date of birth to disambiguate — this prevents wrong-patient data from being shown.

### Multi-Turn Conversations

You can ask follow-up questions within the same session using pronouns — the agent remembers context:

```
You:   What medications is John Smith on?
Agent: John Smith is on Metformin 500mg and Lisinopril 10mg.

You:   Does he have any allergies?
Agent: John Smith has a documented allergy to Penicillin.
```

### Out-of-Scope Queries

If you ask a non-clinical question (e.g., "What's the weather?"), the agent will politely decline and redirect you to healthcare questions.

---

## 5. PDF Upload and Inline Citations

AgentForge can analyse clinical documents — prior authorisation letters, lab reports, pathology notes, imaging reports — and answer questions directly grounded in the document text.

### Uploading a PDF

1. Click the **📎** icon in the input bar
2. Select a PDF file from your computer (max size depends on your server configuration)
3. A **PDF badge** appears: `📄 filename.pdf [✕]`
4. Type your question and click **Send**

The agent will extract text from every page and search for clinically relevant information.

### Inline Citation Anchors

When the agent cites a specific page in the document, a **citation button** appears below the answer:

```
📄 AgentForge_Test_PriorAuth.pdf  p.3
```

Click this button to open the **PDF Viewer Pane**, which slides in from the right side of the screen and automatically jumps to the cited page with a **gold pulsing highlight** so you can see exactly what the agent read.

### PDF Viewer Pane

| Element | Description |
|---|---|
| **Citation pills** | Row of `📄 filename p.N` buttons at the top; click any to jump to that page |
| **Page canvas** | Full PDF page rendered via pdf.js |
| **Gold pulse highlight** | Animated border on the cited page — confirms exactly what the agent referenced |
| **✕ Close** | Slides the pane away; the citation button remains in the chat bubble |

The PDF viewer supports multi-page navigation. If the agent found relevant content on multiple pages, each page gets its own citation pill. The most query-relevant page is highlighted automatically.

---

## 6. Streaming Responses

AgentForge streams its responses in real time — you see progress as the agent works through its multi-step pipeline, rather than waiting for the full answer.

### Typing Indicator

While the agent processes your question, a **typing indicator** shows the current step:

```
● ● ●  Identifying patient...
● ● ●  Gathering data from EHR...
● ● ●  Checking drug interactions...
● ● ●  Reviewing findings...
● ● ●  Building answer...
```

These steps correspond to the agent's internal workflow nodes (Router → Orchestrator → Extractor → Auditor → Output). You always know where the agent is in its reasoning process.

### How Streaming Works

AgentForge uses **Server-Sent Events (SSE)** over the `/ask/stream` endpoint. Each workflow node emits a `node` event updating the status text. When the full answer is ready, a `done` event delivers the complete response and removes the typing indicator.

### Tab-Close Safety Net

If you close the tab while a question is being processed, the agent uses `navigator.sendBeacon` to save your in-flight question to the database before the page unloads. Your question is never lost even if the connection drops mid-flight.

---

## 7. Session and Chat History

### How Sessions Work

Every conversation is automatically saved to a persistent database. A **session** is one continuous conversation thread identified by a unique `session_id`. Sessions capture:

- Patient name and PID
- All questions and answers
- Tool traces (which tools were called and with what data)
- Confidence scores, denial risk, and escalation flags
- PDF paths for transcript replay

### System Audit History Sidebar

The left sidebar shows your **30 most recent sessions**, ordered by most recently updated. Each entry displays:

- **Date** of the last message
- **Intent badge** (colour-coded: Medications, Allergies, Interactions, Safety Check, etc.)
- **Patient name** and PID
- **Query summary** (first 80 characters of the last question)

### Resuming a Past Session

Click any entry in the sidebar to **replay the full transcript** — all messages, tool traces, denial badges, citation anchors, and PDF viewer state are restored exactly as they were. You can then continue the conversation from where you left off.

### Persistence Across Logout

Sessions are stored in SQLite on a persistent volume. They survive:

- Logging out and logging back in
- Closing the browser
- Server restarts and Railway redeployments

---

## 8. New Case Workflow

Click **New Case** in the top-right corner to start a fresh conversation while keeping your history intact.

### What New Case does

- Generates a new `session_id` — follow-up messages go to a new thread
- Clears the chat area and shows the welcome card
- Clears the PDF attachment and closes the PDF viewer pane
- Refreshes the sidebar so the previous session is immediately visible in history
- Inserts a timestamped **"New Case"** divider in the sidebar (if applicable)

### What New Case does NOT do

- It does **not** delete previous conversations
- It does **not** clear the sidebar history
- It does **not** log you out

Use New Case at the start of each new patient encounter to ensure clean session boundaries for audit purposes.

---

## 9. Drug Interaction Checking

AgentForge checks for drug-drug interactions by fetching the patient's current medication list from OpenEMR and running it through the interaction rules engine.

### Example Query

```
Check drug interactions for Mary Johnson
```

### What the Agent Does

1. Looks up Mary Johnson in OpenEMR
2. Fetches her active `MedicationRequest` list via FHIR
3. Cross-references every medication pair against the interaction database
4. Reports interactions by severity: **HIGH**, **MODERATE**, **LOW**, or **NONE**

### Interpreting Results

| Severity | Meaning | Action |
|---|---|---|
| **HIGH** | Potentially dangerous combination | Physician review required (auto-escalated) |
| **MODERATE** | Monitor closely | Review dosing and frequency |
| **LOW** | Minor — generally safe | Note for awareness |
| **NONE** | No known interactions found | No action required |

---

## 10. Allergy Checking

AgentForge fetches allergy data from two sources in priority order:

1. **OpenEMR REST API** (`GET /api/patient/{uuid}/allergy`) — exact allergy titles as recorded by clinical staff
2. **OpenEMR FHIR API** (`GET /fhir/AllergyIntolerance`) — structured FHIR allergy resources as fallback

### Safety Check (Allergy Conflict)

Ask whether it is safe to give a specific drug:

```
Is it safe to give Robert Davis Aspirin?
```

The agent will:
1. Fetch Robert Davis's allergy list from OpenEMR
2. Check the proposed drug against each allergy, including **drug-class matching** (e.g., Aspirin is an NSAID — if the patient is allergic to NSAIDs, this is flagged even if "Aspirin" is not listed by name)
3. Return a clear SAFE / CONFLICT result with clinical context

---

## 11. Medication Lookup

```
What medications is John Smith on?
```

AgentForge returns the patient's **active medication list** from OpenEMR, including drug name, dose, and frequency where available.

If the patient is not found in OpenEMR (Scenario A), the agent applies a **45-point confidence penalty** and falls back to local mock data, clearly noting the data source limitation. If the patient is completely unknown (Scenario B), the agent asks a clarifying question before proceeding.

---

## 12. Denial Risk Analysis

Every agent response includes a **Denial Risk badge** scored by the RCM denial pattern engine.

### Risk Levels

| Badge | Colour | Meaning |
|---|---|---|
| **NONE** | Green | No denial risk indicators detected |
| **LOW** | Green | Minor risk — standard documentation recommended |
| **MEDIUM** | Yellow | Review documentation completeness |
| **HIGH** | Red | High likelihood of denial — physician review recommended |
| **CRITICAL** | Dark red | Immediate review required — auto-escalation triggered |

The badge shows the `risk_level`, a `denial_risk_score` percentage (0–100%), and matched **denial pattern codes** that explain why the risk was flagged.

---

## 13. FHIR Evidence Sync (HITL)

When you upload a clinical PDF containing biomarker data (e.g., a pathology report with HER2, ER, PR, KI67 results), AgentForge can sync the extracted evidence to OpenEMR as structured FHIR Observations.

### How It Works

1. You upload a PDF and ask a clinical question
2. The agent extracts clinical markers (e.g., "HER2 positive", "ER 90%") and stages them to the local SQLite database
3. After extraction, the agent presents a **sync summary**:

```
I found the following new clinical markers not yet in OpenEMR:
  • HER2: Positive
  • ER: 90%
  • KI67: 25%

Would you like me to sync these to OpenEMR? Reply "yes" to proceed.
```

4. Type **yes** (or "sync" / "proceed" / "confirm") to approve
5. The agent POSTs each marker to OpenEMR as a FHIR R4 `Observation` resource, mapped to the correct **LOINC code**
6. Results are reported: how many markers were synced, superseded (duplicates), or failed

### Evidence Status Lifecycle

| Status | Meaning |
|---|---|
| `PENDING` | Extracted from PDF, waiting for sync |
| `SYNCED` | Successfully posted to OpenEMR as a FHIR Observation |
| `FAILED` | POST to OpenEMR failed (network error or API rejection) |
| `SUPERSEDED` | Duplicate of a champion row — absorbed without re-posting |

> **Nothing is synced without your explicit approval.** The HITL (Human-In-The-Loop) gate ensures clinical staff always review before any data is written to the EHR.

---

## 14. PII Scrubbing and Privacy

Every question you type is automatically scrubbed of Personally Identifiable Information (PII) before being processed by the AI models.

### What Gets Scrubbed

| Entity Type | Example | Replaced With |
|---|---|---|
| Person names | "John Smith" | `<PERSON>` |
| Dates and times | "born 01/15/1980" | `<DATE_TIME>` |
| Phone numbers | "555-1234" | `<PHONE_NUMBER>` |
| Email addresses | "patient@email.com" | `<EMAIL_ADDRESS>` |
| Social Security Numbers | "123-45-6789" | `<US_SSN>` |
| Medical licence numbers | "ML-12345" | `<MEDICAL_LICENSE>` |
| Account numbers | "ACC-2024-00123" | `<ACCOUNT_NUMBER>` |

The scrubbing uses **Microsoft Presidio** with the spaCy `en_core_web_lg` model. A graceful 5-regex fallback activates if Presidio is unavailable.

### Privacy Accordion

Below every agent response, a **🔒 Privacy** accordion shows the PII-redacted version of your question — so you can confirm exactly what was sent to the AI model. Click to expand it.

---

## 15. Escalation and Confidence

### Confidence Score

Every agent response carries a **confidence score** (0–100%) indicating how certain the agent is in its answer, based on:

- Whether source data was found in OpenEMR
- Whether citations are verbatim and unambiguous (Auditor Node validation)
- Whether the patient was identified with certainty
- Whether the PDF contained the requested information

### Auto-Escalation

A **🔴 Physician Review Recommended** banner appears on the agent bubble when any of these conditions are met:

- Confidence is below 90%
- Denial risk is HIGH or CRITICAL
- The Auditor Node flagged the answer as partial or ambiguous
- A drug safety conflict was detected

Escalation is a signal that a qualified clinician should review the case before any action is taken. It does not prevent you from reading the answer.

---

## 16. Tool Trace Inspector

Every agent response includes a collapsible **🛠️ Tool Trace** section showing exactly which tools were called during the workflow.

### How to Read It

Click **🛠️ Tool Trace** to expand. Each step shows:

- **Tool name** (e.g., `get_patient_info`, `get_medications`, `check_drug_interactions`)
- **Input** — the JSON parameters sent to the tool
- **Output** — the JSON data returned (formatted in green monospace)

The tool trace is your audit record proving exactly where every piece of information came from.

### Tool Inventory

| Tool | What It Does |
|---|---|
| `get_patient_info` | Looks up patient in OpenEMR FHIR, falls back to mock data |
| `get_medications` | Fetches active `MedicationRequest` resources |
| `get_allergies` | Fetches allergy list via REST API (primary) and FHIR (fallback) |
| `check_drug_interactions` | Runs medication list through interaction rules engine |
| `check_allergy_conflict` | Checks a proposed drug against the patient's allergy list |
| `extract_pdf` | Extracts page-level text from an uploaded PDF |
| `clinical_marker_scan` | Scans PDF text for biomarkers and stages them to SQLite |
| `analyze_denial_risk` | Scores findings against RCM denial patterns |
| `policy_search` | Searches payer policy documents using semantic embedding search |

---

## 17. Audit Trail

### Session Audit API

Every session has a protected audit endpoint:

```
GET /api/audit/{thread_id}
Authorization: Bearer <token>
```

Returns a full audit record including:

- All message turns (user and agent)
- All tool calls and their inputs/outputs
- Confidence scores per turn
- EHR confidence penalty (applied when patient is not found in OpenEMR)
- Denial risk levels

### SQLite Evidence Staging

All clinical markers extracted from PDFs are recorded in the `evidence_staging` table with full provenance:

- Which session extracted the marker
- Which patient it belongs to
- Which PDF page it came from
- The verbatim raw text excerpt
- The FHIR Observation ID after sync

This table is the authoritative local audit record even if the FHIR sync to OpenEMR fails.

---

## 18. Eval System

AgentForge has a built-in evaluation framework that tests the agent against a curated golden dataset of 62 clinical test cases.

### Running an Eval

1. Click the **Eval** button in the top-right header
2. The eval overlay opens showing the last saved results
3. Click **▶ Run Eval** to start a fresh evaluation run (takes several minutes)
4. Results update automatically when complete

### Eval Categories

| Category | Cases | Description |
|---|---|---|
| **Happy Path** | 10 | Standard medication, allergy, and interaction queries |
| **Edge Case** | 16 | Unknown patients, invalid IDs, drug-class allergies, EHR unavailable |
| **Adversarial** | 7 | Jailbreak attempts, SSN extraction, manipulation attempts |
| **PDF Clinical** | 17 | PDF-grounded queries: policy criteria, ICD/CPT alignment, denial risk |
| **Policy Extraction** | 5 | Payer/CPT extraction, imaging classification, no-payer fallback |
| **Auditor Fix** | 2 | Auditor regression tests |
| **Identity Resolution** | 3 | DOB-based disambiguation, same-name collision prevention |

### Scoring Dimensions

Each test case is scored on 5 dimensions:

| Dimension | Check |
|---|---|
| **MC** — Must Contain | At least one expected keyword is present in the response |
| **MNC** — Must Not Contain | No forbidden keyword appears in the response |
| **CF** — Confidence Floor | Confidence score is within the expected range |
| **ES** — Escalate | Escalation flag matches the expected value |
| **DR** — Denial Risk | Denial risk level matches the expected level |

### Filtering Results

Use the filter bar to view subsets:

- **All** — all 62 cases
- **✅ Pass** / **❌ Fail** — filter by outcome
- **📄 PDF Clinical** / **✅ Happy Path** / **⚠️ Edge Case** / **🛡 Adversarial** — filter by category

Use the **search box** to find a specific case by ID (e.g., `gs-027`) or by response content.

---

## 19. System Health

### Agent Status Indicator

The **Agent online** green dot in the header confirms the Railway backend is reachable. If the dot is absent or the header shows an error, the backend may be restarting.

### Health Endpoint

```
GET https://<railway-url>/health
```

Returns:

```json
{
  "service": "AgentForge Healthcare RCM AI Agent",
  "version": "1.0.0",
  "status": "healthy",
  "timestamp": "2026-03-02T05:00:00Z"
}
```

---

## 20. API Reference

All endpoints are available at `https://web-production-5d03.up.railway.app`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/ask` | Synchronous NL query → full JSON response |
| `POST` | `/ask/stream` | Streaming SSE response with node progress events |
| `POST` | `/upload` | Upload a PDF file |
| `GET` | `/pdf?path=<path>` | Serve an uploaded or mock PDF |
| `GET` | `/history` | List recent sessions (`?limit=30`) |
| `GET` | `/history/{session_id}/messages` | Full transcript for a session |
| `GET` | `/api/audit/{thread_id}` | Protected audit trail (Bearer token required) |
| `POST` | `/save-message` | Persist an in-flight message (used by `sendBeacon` on tab close) |
| `POST` | `/eval` | Run the full golden dataset eval suite |
| `GET` | `/eval/results` | Retrieve the most recently saved eval results |

---

## 21. Troubleshooting

### Agent shows "offline" or fails to respond

- Check the Railway deployment is running at the production URL
- Visit `/health` directly in a browser to confirm the service is up
- Check Railway Logs for startup errors

### History sidebar shows "No sessions yet"

- This is expected on first use — have a conversation to create the first session
- If sessions existed before but are gone, a Railway redeployment without a persistent volume wiped the database. Ensure the Railway volume is mounted at `/data` and `DB_PATH=/data/evidence_staging.sqlite` is set in Railway Variables

### Refreshing the page logs me out

- This is OpenEMR's tab-based session behaviour — the `token_main` URL parameter is a one-time token
- **Do not refresh** the AgentForge page. Instead close the tab and click **AgentForge AI** from the top navigation menu to reopen it

### Allergies not found for a patient

- Confirm the patient has allergies recorded in OpenEMR under `Patient → Allergies`
- The agent uses the REST API endpoint `GET /api/patient/{uuid}/allergy` as the primary source
- If the allergy list is empty in OpenEMR, the agent will correctly report no allergies found

### PDF citations are not clickable or viewer does not open

- Ensure the PDF was uploaded via the **📎** button in the current session
- The PDF viewer requires the file to be accessible at `GET /pdf?path=uploads/<filename>`
- Try uploading the PDF again if the badge is missing

### Eval results are empty

- Click **▶ Run Eval** inside the Eval overlay to generate results
- The first run requires the full 62-case suite to complete — allow 5–10 minutes
- Check Railway Logs if the eval endpoint returns an error

---

*AgentForge is a clinical decision support tool. All outputs must be reviewed by a qualified healthcare professional before clinical action is taken.*
