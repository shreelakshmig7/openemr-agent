# Mock data and test assets

This directory holds mock JSON used by the agent tools and **test PDFs** used by the eval suite.

## Required files

### JSON (mock tools)

- **`patients.json`** — Used by `get_patient_info` (patient_lookup stand-in).
- **`medications.json`** — Used by `get_medications` (med_retrieval stand-in).
- **`interactions.json`** — Used by `check_drug_interactions`.

### Test PDFs (eval suite)

The following PDFs **must** be committed here for eval cases **gs-036** through **gs-052** to run:

| File | Purpose |
|------|--------|
| **`AgentForge_Test_PriorAuth.pdf`** | Prior-auth test document for PDF extraction / denial-analysis evals. |
| **`AgentForge_Test_ClinicalNote.pdf`** | Clinical note test document for PDF extraction / denial-analysis evals. |

If these files are missing, the eval runner will skip or fail the cases that reference them. Ensure both are present in `mock_data/` before running the full eval suite.
