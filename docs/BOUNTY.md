# BOUNTY.md: AgentForge Standout Feature Submission

## 1. The Target Customer
**Profile:** Clinical Revenue Cycle Management (RCM) Coordinators and Pharmacy Auditors in mid-sized healthcare facilities.
**The Pain Point:** Coordinators must manually cross-reference unstructured clinical notes (PDFs) against structured EHR data. This process is prone to "data bleeding" between patients and AI hallucinations where an LLM might suggest a medication that contradicts a patient's documented allergy profile.

## 2. The Bounty Features
I have implemented two high-impact engineering solutions to solve these clinical risks:

* **Bounty Feature 1: Deterministic Safety Gate**
    A hard-coded clinical logic layer that overrides probabilistic LLM outputs. Before the agent approves any medication found in a PDF, it is programmatically forced to query the OpenEMR Allergy list. If a conflict is detected, the system triggers a **High-Severity Red Alert**, preventing the LLM from "guessing" or hallucinating a safe outcome.
    
* **Bounty Feature 2: Staging-to-Sync Workflow**
    A stateful CRUD operation designed for high-integrity environments. Extracted data (ICD-10 codes, CPT codes, and medications) is staged in an intermediate SQLite audit store. This creates a "Human-in-the-Loop" gate where a clinician must click **Approve & Sync** to move the "Verified Champions" into the live OpenEMR database.



## 3. The Data Source: OpenEMR
The agent is natively integrated into **OpenEMR**, an industry-standard open-source EHR.
* **Integration Method:** Direct integration via **FHIR R4 API** (Observations/Meds) and **MariaDB SQL** (Allergy/Patient Demographics).
* **Environment Scale:** Validated against a live production environment of **12 patient records**.
* **Key Fields Consulted:**
    * `Patient`: PID, Name, DOB (used for strict Composite Key Identity Resolution).
    * `AllergyIntolerance`: Substance, Manifestation, Severity.
    * `MedicationRequest`: Active orders for cross-referencing new clinical notes.



## 4. Operational Impact
* **Hallucination Elimination:** Successfully caught a **High-Severity Codeine Allergy** for Robert Kim by using coded Python constraints to override the LLM's general suggestion.
* **Identity Precision:** Despite having 12 patients in the database, the agent uses **Composite Key Matching** (Name + DOB) to ensure Robert Kim's orthopedic data never "bleeds" into Sarah Chen’s oncology thread.
* **Auditability:** The **Staging-to-Sync** workflow provides a 100% immutable audit trail in SQLite, proving exactly which sentence in a PDF justified a specific billing code or clinical update.



---

## Technical Workflow Details

### 1. Identity Resolution
The agent performs a fuzzy search via `tool_get_patient_info` and locks the context using a unique PID. This ensures that even in a multi-patient session, the agent maintains strict context isolation.

### 2. The Logic Override (Bounty Flex)
When a medication is identified in a PDF (e.g., Codeine):
1.  The agent calls `tool_get_allergies` from the MariaDB backend.
2.  A **Python Constraint Node** compares the drug against the allergy list.
3.  If a conflict exists, it injects a **Hard Constraint** into the state, forcing the UI to display a Red Alert regardless of the LLM's internal weights.



### 3. Closing the Loop (Sync)
Through the UI, the user manages the lifecycle of the data:
* **Stage:** Data is automatically saved to `evidence_staging.sqlite`.
* **Review:** User verifies findings in the System Audit History.
* **Sync:** User clicks "Approve & Sync" to push validated data into the EHR.