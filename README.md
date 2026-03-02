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

## Open Source Contributions

As part of this project, we have released a public RCM evaluation dataset containing 59 validated clinical test cases. This dataset is now hosted within our OpenEMR fork at:

**https://github.com/shreelakshmig7/openemr/tree/master/contrib/benchmarks/rcm_agent_evals**

The benchmark covers four clinical categories: Clinical Extraction, Denial Logic (Aetna CPB #0876), Adversarial/Safety, and Edge Cases. All patient names and identifiers are entirely fictional — safe for public use under HIPAA guidelines. See [`contrib/benchmarks/rcm_agent_evals/README.md`](contrib/benchmarks/rcm_agent_evals/README.md) for usage instructions.

## Runbook

- **Live:** [Railway](https://openemr-agent-production.up.railway.app)
- **Eval:** 62 cases in `eval/golden_data.yaml`; runner `eval/run_eval.py`; results under `tests/results/`.
- **Benchmark:** `contrib/benchmarks/rcm_agent_evals/golden_data.yaml`
- **Coding standards:** `docs/CODING_STANDARDS.md`
