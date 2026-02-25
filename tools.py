"""
tools.py
--------
Healthcare AI Agent — OpenEMR Drug Interaction Checker
--------------------------------------------------------
This module defines the core tools used by the LangChain agent to interact
with patient data. All data is loaded from mock JSON files for MVP development.

Tools:
    - get_patient_info(name): Look up a patient by name
    - get_medications(patient_id): Get a patient's current medications
    - check_drug_interactions(medications): Check for dangerous drug interactions

Data Sources (MVP):
    - mock_data/patients.json
    - mock_data/medications.json
    - mock_data/interactions.json

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import json
import os
import re
from langsmith import traceable

# Load mock data once at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "mock_data/patients.json")) as f:
        PATIENTS_DB = json.load(f)["patients"]
except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
    PATIENTS_DB = []
    print(f"WARNING: Could not load patients.json — {e}")

try:
    with open(os.path.join(BASE_DIR, "mock_data/medications.json")) as f:
        MEDICATIONS_DB = json.load(f)["medications"]
except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
    MEDICATIONS_DB = {}
    print(f"WARNING: Could not load medications.json — {e}")

try:
    with open(os.path.join(BASE_DIR, "mock_data/interactions.json")) as f:
        INTERACTIONS_DB = json.load(f)["interactions"]
except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
    INTERACTIONS_DB = []
    print(f"WARNING: Could not load interactions.json — {e}")


# ── Tool 1: Get Patient Info ──────────────────────────────────────────────────
@traceable
def get_patient_info(name_or_id: str) -> dict:
    """
    Look up a patient by name or by patient ID (e.g. P001, P002, P003).
    Returns patient ID, name, age, gender, allergies, and conditions.

    Args:
        name_or_id: Full or partial patient name, or a patient ID like P001.

    Returns:
        dict: {"success": bool, "patient": dict | None, "error": str | None}

    Raises:
        Never — all failures return a structured error dict.
    """
    try:
        if not name_or_id or not name_or_id.strip():
            return {
                "success": False,
                "error": "Name or ID cannot be empty",
                "patient": None,
            }
        query = name_or_id.strip()

        # ID lookup: matches P001, P002, p003 etc.
        if re.match(r'^[Pp]\d+$', query):
            query_upper = query.upper()
            for patient in PATIENTS_DB:
                if patient.get("id", "").upper() == query_upper:
                    return {"success": True, "patient": patient}
            return {
                "success": False,
                "error": f"No patient found with ID '{query}'",
                "patient": None,
            }

        # Name lookup (original behaviour — unchanged)
        query_lower = query.lower()
        for patient in PATIENTS_DB:
            if query_lower in patient["name"].lower():
                return {"success": True, "patient": patient}

        return {
            "success": False,
            "error": f"No patient found with name '{name_or_id}'",
            "patient": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error during patient lookup: {str(e)}",
            "patient": None,
        }


# ── Tool 2: Get Medications ───────────────────────────────────────────────────
@traceable
def get_medications(patient_id: str) -> dict:
    """
    Get current medications for a patient by their ID.
    Returns list of medications with dosage and frequency.
    """
    if patient_id not in MEDICATIONS_DB:
        return {
            "success": False,
            "error": f"No medications found for patient ID '{patient_id}'",
            "medications": []
        }
    return {
        "success": True,
        "patient_id": patient_id,
        "medications": MEDICATIONS_DB[patient_id]
    }


# ── Tool 3: Check Drug Interactions ──────────────────────────────────────────
@traceable
def check_drug_interactions(medications: list) -> dict:
    """
    Check a list of medication names for dangerous interactions.
    Returns list of interactions found with severity and recommendations.
    """
    drug_names = [m["name"] if isinstance(m, dict) else m for m in medications]
    found_interactions = []

    for interaction in INTERACTIONS_DB:
        drug1 = interaction["drug1"]
        drug2 = interaction["drug2"]
        if drug1 in drug_names and drug2 in drug_names:
            found_interactions.append(interaction)

    if not found_interactions:
        return {
            "success": True,
            "interactions_found": False,
            "message": "No known dangerous interactions found.",
            "interactions": []
        }

    return {
        "success": True,
        "interactions_found": True,
        "count": len(found_interactions),
        "interactions": found_interactions
    }