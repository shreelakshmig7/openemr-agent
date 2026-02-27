"""
tools — AgentForge Healthcare RCM AI Agent
------------------------------------------
Core tools for patient data, medications, and drug interactions.
Policy search lives in tools.policy_search.
"""

import json
import os
import re
from langsmith import traceable

# Repo root (parent of tools/) so mock_data paths resolve
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


@traceable
def get_patient_info(name_or_id: str) -> dict:
    """
    Look up a patient by name or by patient ID (e.g. P001, P002, P003).
    Returns patient ID, name, age, gender, allergies, and conditions.
    """
    try:
        if not name_or_id or not name_or_id.strip():
            return {
                "success": False,
                "error": "Name or ID cannot be empty",
                "patient": None,
            }
        query = name_or_id.strip()

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


@traceable
def get_medications(patient_id: str) -> dict:
    """Get current medications for a patient by their ID."""
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


@traceable
def check_drug_interactions(medications: list) -> dict:
    """Check a list of medication names for dangerous interactions."""
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
