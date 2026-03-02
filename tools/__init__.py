"""
tools — AgentForge Healthcare RCM AI Agent
------------------------------------------
Core tools for patient data, medications, and drug interactions.
Policy search lives in tools.policy_search.

Patient lookup priority (get_patient_info):
  1. PRIMARY   — OpenEMR FHIR R4 API  (live EHR record, source of truth).
                 GET /fhir/Patient?family=...&given=...
  2. SECONDARY — Local mock_data/patients.json  (offline / demo records).
  3. FINAL     — Both sources empty → "Patient Not Found" + offer to create.

The response always includes a 'data_source' field so the orchestrator and
UI can clearly label whether the record came from the Live EHR or the
Local Cache.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
from datetime import date
from typing import Optional, Tuple

from langsmith import traceable

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# FHIR fallback helpers
# ---------------------------------------------------------------------------

def _parse_name_for_fhir(query: str) -> Tuple[str, str]:
    """
    Parse a free-text name query into (given, family) tokens for FHIR search.

    Handles common formats:
      "Maria Gonzalez"       → ("Maria",  "Gonzalez")
      "Maria J. Gonzalez"    → ("Maria",  "Gonzalez")   ← middle initial stripped
      "Maria J Gonzalez"     → ("Maria",  "Gonzalez")
      "Gonzalez, Maria"      → ("Maria",  "Gonzalez")   ← last-first format
      "Gonzalez"             → ("",       "Gonzalez")   ← single token

    Returns:
        Tuple[str, str]: (given_name, family_name) — either may be empty.
    """
    # Strip middle initials: single uppercase letter optionally followed by a period
    cleaned = re.sub(r'\b[A-Z]\.\s*', '', query)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # "Last, First" format
    if ',' in cleaned:
        parts = [p.strip() for p in cleaned.split(',', 1)]
        return parts[1], parts[0]   # given, family

    parts = cleaned.split()
    if len(parts) >= 2:
        return parts[0], parts[-1]  # first token = given, last token = family
    return "", cleaned              # single token — treat as family name


def _fhir_patient_to_local(fhir_patient: dict) -> dict:
    """
    Map a FHIR R4 Patient resource to the local patients.json record format
    so downstream tools (orchestrator, extractor) receive a consistent shape.

    Fields populated from FHIR:
        id, name, dob, age, gender, fhir_id, source

    Fields defaulted to empty (require additional FHIR queries):
        allergies  → would need GET /fhir/AllergyIntolerance?patient=<id>
        conditions → would need GET /fhir/Condition?patient=<id>

    The caller (extractor / orchestrator) handles empty allergies/conditions
    gracefully — they simply won't trigger interaction or allergy checks.
    """
    # Name — prefer "official" use, fall back to first entry
    names     = fhir_patient.get("name", [])
    name_rec  = next((n for n in names if n.get("use") == "official"), names[0] if names else {})
    family    = name_rec.get("family", "")
    given     = " ".join(name_rec.get("given", []))
    full_name = " ".join(filter(None, [given, family])).strip() or "Unknown"

    # Age from birthDate
    dob = fhir_patient.get("birthDate", "")
    age: Optional[int] = None
    if dob:
        try:
            birth = date.fromisoformat(dob)
            today = date.today()
            age   = today.year - birth.year - (
                (today.month, today.day) < (birth.month, birth.day)
            )
        except ValueError:
            pass

    gender_map = {"male": "Male", "female": "Female", "other": "Other", "unknown": "Unknown"}
    gender     = gender_map.get(fhir_patient.get("gender", "").lower(), "Unknown")
    fhir_id    = fhir_patient.get("id", "")

    # Allergies — prefer the value co-fetched by _fhir_name_lookup_async (same
    # client session, no extra round-trip).  Fall back to a separate async call
    # only when the resource arrived via a different code path (e.g. ID lookup).
    # Filter out generic placeholders ("Unknown", etc.) that OpenEMR may emit
    # for allergies that were seeded via SQL without a SNOMED/RxNorm code.
    _ALLERGY_PLACEHOLDERS = {"unknown", "unspecified", "other", "none", ""}

    def _filter_allergies(raw: list[str]) -> list[str]:
        return [a for a in (raw or []) if a and a.strip().lower() not in _ALLERGY_PLACEHOLDERS]

    if "_prefetched_allergies" in fhir_patient:
        allergies: list[str] = _filter_allergies(fhir_patient["_prefetched_allergies"])
        if allergies:
            logger.info(
                "tools._fhir_patient_to_local: using %d pre-fetched allerg%s for %s",
                len(allergies), "y" if len(allergies) == 1 else "ies", full_name,
            )
    elif fhir_id:
        allergies = _filter_allergies(_run_async_in_thread(_fhir_allergies_async(fhir_id)))
        if allergies:
            logger.info(
                "tools._fhir_patient_to_local: fetched %d allerg%s for %s from FHIR (fallback)",
                len(allergies), "y" if len(allergies) == 1 else "ies", full_name,
            )
    else:
        allergies = []

    return {
        "id":         fhir_id,   # FHIR UUID — used as subject reference in bundles
        "fhir_id":    fhir_id,
        "name":       full_name,
        "dob":        dob,
        "age":        age,
        "gender":     gender,
        "allergies":  allergies,  # live from FHIR AllergyIntolerance
        "conditions": [],         # use FHIR Condition endpoint if needed in future
        "source":     "openemr_fhir",
    }


async def _fhir_name_lookup_async(given: str, family: str) -> Optional[dict]:
    """
    Coroutine: search OpenEMR FHIR for a patient by name.

    Tries (family + given) first; if that returns nothing, retries with
    family only.  Returns the first matching FHIR Patient resource dict
    (with a ``_prefetched_allergies`` key added in-band), or None if no
    match or if OpenEMR is unreachable.

    Allergies are fetched inside the same ``OpenEMRClient`` session so we
    only pay the OAuth2 + registration overhead once per patient lookup.
    The result is stored under ``resource["_prefetched_allergies"]`` so
    ``_fhir_patient_to_local`` can use it without opening a second client.
    """
    # Import here to avoid circular import at module load time
    from openemr_client import OpenEMRClient

    search_attempts = []
    if family and given:
        search_attempts.append({"family": family, "given": given})
    if family:
        search_attempts.append({"family": family})

    try:
        async with OpenEMRClient() as client:
            for params in search_attempts:
                bundle = await client.get_patients(**params)
                entries = bundle.get("entry", [])
                if entries:
                    resource = entries[0]["resource"]
                    fhir_id = resource.get("id", "")
                    if fhir_id:
                        try:
                            resource["_prefetched_allergies"] = await client.get_fhir_allergies(fhir_id)
                        except Exception as allergy_exc:
                            logger.warning("tools: allergy prefetch failed for %s — %s", fhir_id, allergy_exc)
                            resource["_prefetched_allergies"] = []
                    return resource
    except Exception as exc:
        logger.warning("tools: FHIR patient lookup failed — %s", exc)

    return None


def _fhir_patient_lookup(given: str, family: str) -> Optional[dict]:
    """
    Synchronous bridge: runs ``_fhir_name_lookup_async`` in an isolated
    thread so it is safe to call from either a synchronous LangGraph node
    or a FastAPI async endpoint (avoids "cannot run nested event loops").
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            asyncio.run,
            _fhir_name_lookup_async(given, family),
        )
        try:
            return future.result(timeout=15)
        except concurrent.futures.TimeoutError:
            logger.warning("tools: FHIR patient lookup timed out after 15 s.")
            return None
        except Exception as exc:
            logger.warning("tools: FHIR patient lookup thread error — %s", exc)
            return None


@traceable
def get_patient_info(name_or_id: str) -> dict:
    """
    Look up a patient using a three-tier priority hierarchy.

    Priority
    --------
    1. PRIMARY   — OpenEMR FHIR R4 API (live EHR record).
                   Always attempted first for name queries so the agent
                   works with the most current clinical data.
    2. SECONDARY — Local mock_data/patients.json.
                   Used when FHIR is unreachable or returns no results.
    3. FINAL     — Both miss → structured "not found" response with a
                   suggestion to create the patient record.

    Special case: P-prefixed IDs (e.g. "P001") skip FHIR and go directly
    to the local cache, as these are local demo identifiers.

    Returns
    -------
    dict with keys:
        success     bool   — True when a patient was found.
        patient     dict   — Patient record (None on failure).
        data_source str    — "Live EHR (OpenEMR FHIR)" | "Local Cache" | None
        source      str    — internal tag: "openemr_fhir" | "local_cache"
        error       str    — Human-readable error (only when success=False).
        suggestion  str    — Next-step hint (only when success=False).
    """
    try:
        if not name_or_id or not name_or_id.strip():
            return {
                "success":     False,
                "patient":     None,
                "data_source": None,
                "error":       "Name or ID cannot be empty.",
                "suggestion":  "Provide a patient name or ID, or attach a clinical PDF.",
            }

        query = name_or_id.strip()

        # ── Special case: local P-ID lookup (P001 … P999) ────────────────
        if re.match(r'^[Pp]\d+$', query):
            query_upper = query.upper()
            for patient in PATIENTS_DB:
                if patient.get("id", "").upper() == query_upper:
                    patient.setdefault("source", "local_cache")
                    logger.info("tools: ID '%s' resolved from Local Cache.", query)
                    return {
                        "success":     True,
                        "patient":     patient,
                        "data_source": "Local Cache",
                        "source":      "local_cache",
                    }
            return {
                "success":     False,
                "patient":     None,
                "data_source": None,
                "error":       f"No patient found with ID '{query}' in the local cache.",
                "suggestion":  "Verify the ID, or search by full name.",
            }

        # ── Step 1 (PRIMARY): OpenEMR FHIR API ───────────────────────────
        logger.info("tools: querying OpenEMR FHIR API for '%s' (primary).", query)
        given, family = _parse_name_for_fhir(query)
        fhir_resource = _fhir_patient_lookup(given, family)

        if fhir_resource:
            patient = _fhir_patient_to_local(fhir_resource)
            logger.info(
                "tools: '%s' resolved from Live EHR — name='%s', fhir_id='%s'.",
                query, patient["name"], patient["fhir_id"],
            )
            return {
                "success":     True,
                "patient":     patient,
                "data_source": "Live EHR (OpenEMR FHIR)",
                "source":      "openemr_fhir",
            }

        logger.info(
            "tools: '%s' not found in Live EHR — falling back to Local Cache (secondary).",
            query,
        )

        # ── Step 2 (SECONDARY): Local cache ──────────────────────────────
        query_lower = query.lower()
        for patient in PATIENTS_DB:
            if query_lower in patient["name"].lower():
                patient.setdefault("source", "local_cache")
                logger.info(
                    "tools: '%s' resolved from Local Cache (FHIR returned no results).",
                    query,
                )
                return {
                    "success":     True,
                    "patient":     patient,
                    "data_source": "Local Cache",
                    "source":      "local_cache",
                }

        # ── Step 3 (FINAL): Both sources exhausted ────────────────────────
        logger.warning(
            "tools: '%s' not found in Live EHR or Local Cache.", query
        )
        return {
            "success":     False,
            "patient":     None,
            "data_source": None,
            "error": (
                f"Patient '{name_or_id}' was not found in the Live EHR "
                "(OpenEMR FHIR API) or the Local Cache."
            ),
            "suggestion": (
                "Please verify the spelling of the patient's name, "
                "or attach a clinical PDF containing the patient's information. "
                "To create a new record in OpenEMR, use the patient registration portal at "
                "https://localhost:9300."
            ),
        }

    except Exception as exc:
        logger.exception("tools: unexpected error in get_patient_info.")
        return {
            "success":     False,
            "patient":     None,
            "data_source": None,
            "error":       f"Unexpected error during patient lookup: {exc}",
            "suggestion":  "Check server logs for details.",
        }


def _is_fhir_uuid(value: str) -> bool:
    """Return True if value looks like a FHIR UUID (8-4-4-4-12 hex pattern)."""
    return bool(re.match(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        (value or "").lower(),
    ))


def _resolve_local_id_from_uuid(fhir_uuid: str) -> Optional[str]:
    """
    Resolve a FHIR UUID to a local P00X patient ID by matching via patient name.

    Looks up the patient name from PATIENTS_DB using FHIR UUID matching against
    the fhir_id field (if populated), or falls back to a name-based match
    between the FHIR patient record and local patients.json.

    Args:
        fhir_uuid: A FHIR Patient UUID string.

    Returns:
        Local patient ID string (e.g. "P001") or None if not resolvable.
    """
    # Try direct fhir_id match in local DB first (fastest path)
    for p in PATIENTS_DB:
        if p.get("fhir_id", "") == fhir_uuid or p.get("id", "") == fhir_uuid:
            return p["id"]

    # Fall back to FHIR lookup to get patient name, then match by name
    try:
        from openemr_client import OpenEMRClient
        import asyncio, concurrent.futures

        async def _lookup():
            async with OpenEMRClient() as client:
                data = await client._request("GET", f"/Patient/{fhir_uuid}")
                names = data.get("name", [])
                name_rec = next(
                    (n for n in names if n.get("use") == "official"), names[0] if names else {}
                )
                family = name_rec.get("family", "").lower()
                given  = (name_rec.get("given") or [""])[0].lower()
                return family, given

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            family, given = pool.submit(asyncio.run, _lookup()).result(timeout=10)

        for p in PATIENTS_DB:
            p_name  = p.get("name", "").lower()
            p_parts = p_name.split()
            if family and family in p_name and (not given or given in p_name):
                return p["id"]
    except Exception as exc:
        logger.debug("tools._resolve_local_id_from_uuid: %s", exc)

    return None


async def _fhir_medications_async(patient_uuid: str) -> list[dict]:
    """Fetch medications from OpenEMR FHIR MedicationRequest endpoint."""
    from openemr_client import OpenEMRClient
    async with OpenEMRClient() as client:
        return await client.get_fhir_medications(patient_uuid)


async def _fhir_allergies_async(patient_uuid: str) -> list[str]:
    """Fetch allergies from OpenEMR FHIR AllergyIntolerance endpoint."""
    from openemr_client import OpenEMRClient
    async with OpenEMRClient() as client:
        return await client.get_fhir_allergies(patient_uuid)


def _run_async_in_thread(coro) -> any:
    """Run an async coroutine safely from a sync context via a thread pool."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            return pool.submit(asyncio.run, coro).result(timeout=15)
        except Exception as exc:
            logger.warning("tools._run_async_in_thread: %s", exc)
            return None


@traceable
def get_medications(patient_id: str) -> dict:
    """
    Get current medications for a patient.

    Lookup priority:
      1. OpenEMR FHIR MedicationRequest (if patient_id is a FHIR UUID)
      2. Local mock_data/medications.json via local P00X ID
         (resolves UUID → local ID via name matching if needed)
      3. Structured "not found" response

    Args:
        patient_id: Either a FHIR UUID or a local patient ID (e.g. "P001").

    Returns:
        dict with keys: success, patient_id, medications, source, error.
    """
    # ── Step 1: FHIR MedicationRequest (UUID input) ──────────────────────
    _GENERIC_MED_NAMES = {"unknown", "unspecified", "other", "none", ""}

    if _is_fhir_uuid(patient_id):
        logger.info("tools.get_medications: trying FHIR MedicationRequest for %s", patient_id)
        raw_fhir_meds = _run_async_in_thread(_fhir_medications_async(patient_id))
        # Filter out any entries whose name resolved to a generic placeholder.
        fhir_meds = [
            m for m in (raw_fhir_meds or [])
            if m.get("name", "").strip().lower() not in _GENERIC_MED_NAMES
        ]
        if fhir_meds:
            logger.info(
                "tools.get_medications: FHIR returned %d medications for %s",
                len(fhir_meds), patient_id,
            )
            return {
                "success":    True,
                "patient_id": patient_id,
                "medications": fhir_meds,
                "source":     "Live EHR (OpenEMR FHIR)",
            }
        # FHIR empty or all-generic — resolve UUID to local ID for mock_data fallback
        logger.info(
            "tools.get_medications: FHIR returned 0 usable medications for %s (raw=%s) — falling back to mock_data",
            patient_id, raw_fhir_meds,
        )
        local_id = _resolve_local_id_from_uuid(patient_id)
    else:
        local_id = patient_id

    # ── Step 2: mock_data/medications.json via local ID ───────────────────
    if local_id and local_id in MEDICATIONS_DB:
        logger.info(
            "tools.get_medications: local mock_data hit for %s (local_id=%s)",
            patient_id, local_id,
        )
        return {
            "success":    True,
            "patient_id": local_id,
            "medications": MEDICATIONS_DB[local_id],
            "source":     "Local Cache (mock_data)",
        }

    # ── Step 3: not found ─────────────────────────────────────────────────
    return {
        "success":    False,
        "error":      f"No medications found for patient ID '{patient_id}'",
        "medications": [],
        "source":     None,
    }


@traceable
def get_allergies(patient_id: str) -> dict:
    """
    Fetch documented allergies for a patient.

    Lookup priority:
      1. OpenEMR FHIR AllergyIntolerance (if patient_id is a FHIR UUID)
      2. Local mock_data/patients.json via local P00X ID

    Returns dict with keys: success, patient_id, allergies (list[str]), source.
    Always succeeds — returns empty list when no allergies are found.
    """
    local_id: Optional[str] = None

    _GENERIC_PLACEHOLDERS = {"unknown", "unspecified", "other", "none", ""}

    if _is_fhir_uuid(patient_id):
        logger.info("tools.get_allergies: fetching FHIR AllergyIntolerance for %s", patient_id)
        raw_fhir_allergies = _run_async_in_thread(_fhir_allergies_async(patient_id))
        # Filter out generic placeholders that OpenEMR may emit for uncoded allergies.
        fhir_allergies = [
            a for a in (raw_fhir_allergies or [])
            if a and a.strip().lower() not in _GENERIC_PLACEHOLDERS
        ]
        if fhir_allergies:
            # FHIR returned at least one named allergy — use it as authoritative.
            logger.info(
                "tools.get_allergies: FHIR returned %d allerg%s for %s",
                len(fhir_allergies), "y" if len(fhir_allergies) == 1 else "ies", patient_id,
            )
            return {
                "success":    True,
                "patient_id": patient_id,
                "allergies":  fhir_allergies,
                "source":     "Live EHR (OpenEMR FHIR)",
            }
        # FHIR returned empty list, all-generic names, or failed — fall back to mock_data.
        logger.info(
            "tools.get_allergies: FHIR returned 0 usable allergies for %s (raw=%s) — falling back to mock_data",
            patient_id, raw_fhir_allergies,
        )
        local_id = _resolve_local_id_from_uuid(patient_id)
    else:
        local_id = patient_id

    # ── Fallback: mock_data/patients.json ─────────────────────────────────
    # Used when FHIR AllergyIntolerance returns nothing (common in demo builds
    # where the data is seeded via SQL but FHIR layer doesn't expose it).
    if local_id:
        for p in PATIENTS_DB:
            if p.get("id", "").upper() == local_id.upper():
                local_allergies = p.get("allergies", [])
                logger.info(
                    "tools.get_allergies: mock_data fallback for %s → %s",
                    local_id, local_allergies,
                )
                return {
                    "success":    True,
                    "patient_id": local_id,
                    "allergies":  local_allergies,
                    "source":     "Local Cache (mock_data)",
                }

    return {
        "success":    True,
        "patient_id": patient_id,
        "allergies":  [],
        "source":     "not_found",
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
