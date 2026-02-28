# OpenEMR Agent (AgentForge RCM)

Healthcare RCM (Revenue Cycle Management) AI agent: natural language → LangGraph pipeline → tools (patient/meds, PDF extraction, policy search, denial risk) → cited, auditable responses.

See **memory-bank/agent-reference.md** for architecture, tool status, and decisions.

## Mock data and test PDFs

Mock inputs live in **`mock_data/`**. See **[mock_data/README.md](mock_data/README.md)** for the full list.

- **JSON:** `patients.json`, `medications.json`, `interactions.json` (used by mock tools).
- **Test PDFs (required for evals gs-036–gs-052):**
  - **`AgentForge_Test_PriorAuth.pdf`** — Prior-authorization test document.
  - **`AgentForge_Test_ClinicalNote.pdf`** — Clinical note test document.

Eval cases gs-036 through gs-052 depend on these two PDFs being present in `mock_data/`. Confirm they are committed before running the full eval suite.

## Runbook

- **Live:** [Railway](https://openemr-agent-production.up.railway.app)
- **Eval:** 52 cases in `eval/golden_data.yaml`; runner `eval/run_eval.py`; results under `tests/results/`.
- **Coding standards:** `docs/CODING_STANDARDS.md`
