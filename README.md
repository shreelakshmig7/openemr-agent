# AgentForge — Healthcare AI Agent

**Author:** Shreelakshmi Gopinatha Rao  
**Program:** Gauntlet AI — G4 Week 2  
**Domain:** Healthcare Medication Safety  
**Live Demo:** [Railway Deployment](https://openemr-agent-production.up.railway.app)

---

## What It Does

A healthcare AI agent that helps clinical staff review patient medication histories and flag dangerous drug interactions. Accepts natural language queries, calls tools to fetch real data from a mock patient database, and always cites its source — never answers from memory.

**Tech stack:** FastAPI · LangChain · Claude Sonnet (Anthropic) · LangSmith observability

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
# Edit .env — add ANTHROPIC_API_KEY and LANGSMITH_API_KEY

# 5. Run the server
uvicorn main:app --reload --port 8000

# 6. Open the chat UI
open http://localhost:8000
```

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

# Edge cases
"What medications is Alex Turner on?"     ← no record found
"What meds does Maria Santos take?"       ← empty record
"Look up patient P999"                    ← patient not found
```

---

## API Reference

### `GET /`
Opens the web chat UI.

### `GET /health`
Returns server status.
```json
{"status": "ok", "version": "1.0.0"}
```

### `POST /ask`
Send a question to the agent.
```json
{
  "question": "What medications is John Smith taking?",
  "session_id": "optional-uuid-for-multi-turn"
}
```
Response:
```json
{
  "answer": "John Smith is currently taking...",
  "session_id": "uuid",
  "confidence": 0.9,
  "requires_human_review": false,
  "disclaimer": "...",
  "tool_trace": [
    {"tool": "tool_get_patient_info", "input": {"name_or_id": "John Smith"}, "output": {...}},
    {"tool": "tool_get_medications", "input": {"patient_id": "P001"}, "output": {...}}
  ]
}
```

### `POST /eval`
Run the automated evaluation suite against all golden test cases.

### `GET /eval/results`
Retrieve the most recent evaluation results.

---

## Running Tests

### Unit tests (no API key needed)
```bash
source venv/bin/activate
python -m pytest tests/test_tools.py tests/test_verification.py tests/test_eval.py tests/test_main.py -v
```

### Conversation tests (requires ANTHROPIC_API_KEY)
```bash
python -m pytest tests/test_conversation.py -v
```

### Full evaluation suite (requires ANTHROPIC_API_KEY + running server)
```bash
python eval/run_eval.py
```
Results are saved to `tests/results/`.

---

## Project Structure

```
openemr-agent/
├── main.py              # FastAPI server + API endpoints
├── agent.py             # LangChain agent + system prompt
├── conversation.py      # Multi-turn conversation management
├── tools.py             # Tool implementations (patient, meds, interactions)
├── verification.py      # Domain safety checks (allergy, confidence, FDA)
├── mock_data/
│   ├── patients.json    # 11 mock patients
│   ├── medications.json # Medication records per patient
│   └── interactions.json# Drug interaction rules
├── eval/
│   ├── golden_data.yaml # 18 evaluation test cases
│   └── run_eval.py      # Evaluation runner
├── static/
│   └── index.html       # Web chat UI
└── tests/
    ├── test_tools.py
    ├── test_conversation.py
    ├── test_verification.py
    ├── test_eval.py
    └── test_main.py
```

---

## Observability

All agent runs are traced in [LangSmith](https://smith.langchain.com). Each API response includes a `tool_trace` field showing exactly which tools were called, with what inputs, and what they returned — so evaluators can verify the agent is querying the database and not answering from memory.
