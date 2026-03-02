# OpenEMR Agent (AgentForge RCM)

Healthcare RCM (Revenue Cycle Management) AI agent: natural language → LangGraph pipeline → tools (patient/meds, PDF extraction, policy search, denial risk) → cited, auditable responses.

## Live Application

| Environment | URL |
|---|---|
| **OpenEMR (Production EHR)** | [http://64.225.50.120:8300](http://64.225.50.120:8300) |
| **AgentForge API (Railway)** | [https://web-production-5d03.up.railway.app](https://web-production-5d03.up.railway.app) |

Login to OpenEMR and navigate to the **AgentForge** tab to use the AI agent. The agent is backed by 12 live patient records with medications, allergies, insurance, and encounter data.

## Documentation

| Document | Description |
|---|---|
| [`docs/AgentForge_PreSearch_Doc.md`](docs/AgentForge_PreSearch_Doc.md) | Full system design, architecture, and pre-search research |
| [`docs/agent-architecture-doc.md`](docs/agent-architecture-doc.md) | 8-node LangGraph pipeline — node roles, state schema, tool inventory |
| [`docs/BOUNTY.md`](docs/BOUNTY.md) | Standout feature submission — safety gate and staging-to-sync workflow |
| [`docs/AI_COST_ANALYSIS.md`](docs/AI_COST_ANALYSIS.md) | Token usage and cost breakdown per agent run |
| [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) | Local and production setup instructions |
| [`docs/CODING_STANDARDS.md`](docs/CODING_STANDARDS.md) | Code style and contribution standards |
| [`eval/DATASET_README.md`](eval/DATASET_README.md) | RCM benchmark dataset documentation |

## Open Source Contributions

As part of this project, we have released a public RCM evaluation dataset
containing 59 validated clinical test cases. This dataset is now hosted
within our OpenEMR fork at:

https://github.com/shreelakshmig7/openemr/tree/master/contrib/benchmarks/rcm_agent_evals

## Runbook

- **OpenEMR UI:** [http://64.225.50.120:8300](http://64.225.50.120:8300) — login `admin`, navigate to AgentForge tab
- **Agent API:** [https://web-production-5d03.up.railway.app](https://web-production-5d03.up.railway.app)
- **Eval:** 62 cases in `eval/golden_data.yaml`; runner `eval/run_eval.py`; results under `tests/results/`
- **Benchmark:** `contrib/benchmarks/rcm_agent_evals/golden_data.yaml`
- **Setup:** [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md)
