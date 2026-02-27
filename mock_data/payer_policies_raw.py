"""
payer_policies_raw.py
---------------------
AgentForge — Healthcare RCM AI Agent
Raw payer policy text chunks for Pinecone upsert.
Each chunk = one policy criterion for one payer.
In production: replace with real PDF extraction via unstructured.io.
"""

PAYER_POLICIES = {
    "cigna": [
        {
            "policy_id": "Cigna Medical Policy #012",
            "procedure_codes": ["27447", "27440"],
            "payer": "cigna",
            "section": "criteria_A",
            "criteria_id": "A",
            "text": (
                "Criteria A: Conservative therapy failure. "
                "Patient must have completed a minimum of 3 months of supervised "
                "physical therapy or conservative management including NSAIDs, "
                "corticosteroid injections, or activity modification. "
                "Documentation must include therapy dates, provider name, "
                "and documented functional decline despite treatment."
            ),
        },
        {
            "policy_id": "Cigna Medical Policy #012",
            "procedure_codes": ["27447", "27440"],
            "payer": "cigna",
            "section": "criteria_B",
            "criteria_id": "B",
            "text": (
                "Criteria B: Radiographic evidence of severe osteoarthritis. "
                "Radiographs must demonstrate Kellgren-Lawrence grade 3 or 4 changes, "
                "significant joint space narrowing, subchondral sclerosis, "
                "osteophyte formation, or bone-on-bone contact. "
                "X-ray or MRI report signed by a radiologist must be included "
                "in the prior authorization request."
            ),
        },
        {
            "policy_id": "Cigna Medical Policy #012",
            "procedure_codes": ["27447", "27440"],
            "payer": "cigna",
            "section": "criteria_C",
            "criteria_id": "C",
            "text": (
                "Criteria C: BMI and weight management requirements. "
                "Patient BMI must be below 40 at the time of authorization request. "
                "Patients with BMI 40 or above must have completed a documented "
                "weight management or bariatric program within the past 12 months. "
                "BMI must be recorded by the treating physician."
            ),
        },
        {
            "policy_id": "Cigna Medical Policy #012",
            "procedure_codes": ["27447", "27440"],
            "payer": "cigna",
            "section": "criteria_D",
            "criteria_id": "D",
            "text": (
                "Criteria D: Functional limitation documentation. "
                "Patient must demonstrate significant functional limitation "
                "affecting activities of daily living. Documentation must include "
                "a VAS pain score of 7 or higher, or an objective functional "
                "assessment tool (KOOS, WOMAC, or Oxford Knee Score). "
                "Functional assessment must be performed by the treating physician."
            ),
        },
        {
            "policy_id": "Cigna Medical Policy #012",
            "procedure_codes": ["27447", "27440"],
            "payer": "cigna",
            "section": "criteria_E",
            "criteria_id": "E",
            "text": (
                "Criteria E: Surgical clearance requirements. "
                "Pre-operative cardiac clearance is required for patients over 60 "
                "or with documented cardiovascular risk factors. "
                "Clearance must be documented by a cardiologist or internist "
                "within 90 days of the planned procedure date."
            ),
        },
    ],
    "aetna": [
        {
            "policy_id": "Aetna Clinical Policy Bulletin #0549",
            "procedure_codes": ["27447", "27440"],
            "payer": "aetna",
            "section": "criteria_A",
            "criteria_id": "A",
            "text": (
                "Criteria A: Diagnosis confirmation with ICD-10 coding. "
                "Primary diagnosis of severe osteoarthritis confirmed by imaging. "
                "Accepted codes: M17.11 right knee, M17.12 left knee. "
                "Clinical examination findings must be consistent with imaging."
            ),
        },
        {
            "policy_id": "Aetna Clinical Policy Bulletin #0549",
            "procedure_codes": ["27447", "27440"],
            "payer": "aetna",
            "section": "criteria_B",
            "criteria_id": "B",
            "text": (
                "Criteria B: Failed conservative treatment over 6 months. "
                "Minimum 6 months of conservative treatment including at least two: "
                "supervised physical therapy, oral NSAIDs, corticosteroid injections, "
                "or viscosupplementation. All treatments must be documented with "
                "start/end dates and provider signatures."
            ),
        },
        {
            "policy_id": "Aetna Clinical Policy Bulletin #0549",
            "procedure_codes": ["27447", "27440"],
            "payer": "aetna",
            "section": "criteria_C",
            "criteria_id": "C",
            "text": (
                "Criteria C: Absence of contraindications. "
                "Patient must not have active infection, uncontrolled diabetes "
                "(HbA1c > 9), or severe peripheral vascular disease. "
                "Allergy conflicts with anesthesia agents must be documented "
                "with an alternative anesthesia plan provided."
            ),
        },
    ],
    "uhc": [
        {
            "policy_id": "UHC Medical Policy MP #2023.045",
            "procedure_codes": ["27447"],
            "payer": "uhc",
            "section": "criteria_A",
            "criteria_id": "A",
            "text": (
                "Criteria A: Medical necessity — end-stage joint disease. "
                "Total knee arthroplasty is medically necessary when the member "
                "has end-stage knee joint disease with bone-on-bone contact on imaging, "
                "severe pain rated 8/10 or higher on VAS scale, and failure of "
                "at least 3 months of non-surgical treatment including physical therapy."
            ),
        },
        {
            "policy_id": "UHC Medical Policy MP #2023.045",
            "procedure_codes": ["27447"],
            "payer": "uhc",
            "section": "criteria_B",
            "criteria_id": "B",
            "text": (
                "Criteria B: Laterality and imaging documentation. "
                "Authorization must specify right knee or left knee. "
                "Bilateral replacement requires separate authorization per side. "
                "MRI or X-ray from within 6 months must accompany the request."
            ),
        },
    ],
}
