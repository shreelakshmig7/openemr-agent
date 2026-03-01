"""
test_langgraph_auditor.py
-------------------------
AgentForge — Healthcare RCM AI Agent — Tests for Auditor Node
--------------------------------------------------------------
Tests that the Auditor Node correctly validates extractions, sets
routing_decision, enforces the 3-iteration ceiling, detects hallucinated
facts, and handles ambiguous patient cases. No real API calls.

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import pytest
from unittest.mock import patch
from langgraph_agent.state import create_initial_state
from langgraph_agent.auditor_node import auditor_node
from healthcare_guidelines import HIPAA_RULES


# ── Fixture extractions ────────────────────────────────────────────────────────

VALID_EXTRACTION = {
    "claim": "John Smith is prescribed Metformin 500mg twice daily.",
    "citation": "Metformin 500mg twice daily",
    "source": "mock_data/medications.json",
    "verbatim": True
}

MISSING_CITATION_EXTRACTION = {
    "claim": "John Smith takes Metformin.",
    "citation": "",
    "source": "",
    "verbatim": False
}

NON_VERBATIM_EXTRACTION = {
    "claim": "John Smith takes a diabetes medication twice a day.",
    "citation": "Patient is on diabetes management",
    "source": "mock_data/medications.json",
    "verbatim": False
}

HALLUCINATED_EXTRACTION = {
    "claim": "John Smith has failed conservative therapy.",
    "citation": "patient showed resistance to initial treatment",
    "source": "mock_data/medications.json",
    "verbatim": True
}

AMBIGUOUS_EXTRACTION = {
    "claim": "AMBIGUOUS: Multiple patients found matching 'John'",
    "citation": "",
    "source": "",
    "verbatim": False,
    "ambiguous": True
}


def make_state(extractions, iteration_count=0):
    """Create a test state with given extractions and iteration_count."""
    state = create_initial_state("test query")
    state["extractions"] = extractions
    state["iteration_count"] = iteration_count
    return state


# ── Tests: pass path ──────────────────────────────────────────────────────────

class TestAuditorRoutingPass:
    def test_auditor_passes_valid_extractions(self):
        """Auditor sets routing_decision='pass' when all claims have verbatim citations."""
        state = make_state([VALID_EXTRACTION])
        with patch("langgraph_agent.auditor_node._verify_citation_exists_in_source", return_value=True):
            result = auditor_node(state)
        assert result["routing_decision"] == "pass"


# ── Tests: review loop ────────────────────────────────────────────────────────

class TestAuditorRoutingReviewLoop:
    def test_auditor_sends_back_when_missing_citations(self):
        """Auditor sets routing_decision='missing' when a claim has no citation."""
        state = make_state([MISSING_CITATION_EXTRACTION])
        result = auditor_node(state)
        assert result["routing_decision"] == "missing"

    def test_auditor_sends_back_on_non_verbatim_citation(self):
        """Auditor sets routing_decision='missing' when citation is a paraphrase, not verbatim."""
        state = make_state([NON_VERBATIM_EXTRACTION])
        result = auditor_node(state)
        assert result["routing_decision"] == "missing"

    def test_auditor_detects_hallucinated_fact(self):
        """Auditor flags a claim whose citation does not exist verbatim in the source."""
        state = make_state([HALLUCINATED_EXTRACTION])
        with patch("langgraph_agent.auditor_node._verify_citation_exists_in_source", return_value=False):
            result = auditor_node(state)
        assert result["routing_decision"] == "missing"

    def test_auditor_increments_iteration_count_on_sendback(self):
        """iteration_count increases by 1 each time Auditor sends back to Extractor."""
        state = make_state([MISSING_CITATION_EXTRACTION], iteration_count=0)
        result = auditor_node(state)
        assert result["iteration_count"] == 1

    def test_auditor_recovery_on_second_try(self):
        """After one loop back, Extractor fixes citations, Auditor passes with iteration_count=1."""
        state = make_state([VALID_EXTRACTION], iteration_count=1)
        with patch("langgraph_agent.auditor_node._verify_citation_exists_in_source", return_value=True):
            result = auditor_node(state)
        assert result["routing_decision"] == "pass"
        assert result["iteration_count"] == 1


# ── Tests: iteration ceiling ──────────────────────────────────────────────────

class TestAuditorIterationCeiling:
    def test_auditor_caps_at_three_iterations(self):
        """When iteration_count==3, Auditor routes to 'partial' and does not loop again."""
        state = make_state([MISSING_CITATION_EXTRACTION], iteration_count=3)
        result = auditor_node(state)
        assert result["routing_decision"] == "partial"
        assert result["iteration_count"] == 3

    def test_auditor_ceiling_flags_insufficient_documentation(self):
        """At iteration ceiling, output includes is_partial=True and 'Insufficient Documentation' flags."""
        state = make_state([MISSING_CITATION_EXTRACTION], iteration_count=3)
        result = auditor_node(state)
        assert result["is_partial"] is True
        assert len(result["insufficient_documentation_flags"]) > 0
        assert any(
            "Insufficient Documentation" in flag
            for flag in result["insufficient_documentation_flags"]
        )

    def test_auditor_ceiling_with_forced_count_four(self):
        """If iteration_count is somehow 4 (state corruption), system exits with partial — no crash."""
        state = make_state([MISSING_CITATION_EXTRACTION], iteration_count=4)
        result = auditor_node(state)
        assert isinstance(result, dict)
        assert result["routing_decision"] == "partial"
        assert result["is_partial"] is True


# ── Tests: ambiguity ──────────────────────────────────────────────────────────

class TestAuditorAmbiguity:
    def test_auditor_detects_ambiguous_patient_sets_clarification(self):
        """Auditor sets pending_user_input=True and a non-empty clarification_needed on ambiguity."""
        state = make_state([AMBIGUOUS_EXTRACTION])
        result = auditor_node(state)
        assert result["routing_decision"] == "ambiguous"
        assert result["pending_user_input"] is True
        assert len(result["clarification_needed"]) > 0

    def test_auditor_routes_ambiguity_to_clarification_not_loop(self):
        """Ambiguity does not increment iteration_count — it is a separate path from the review loop."""
        state = make_state([AMBIGUOUS_EXTRACTION], iteration_count=0)
        result = auditor_node(state)
        assert result["routing_decision"] == "ambiguous"
        assert result["iteration_count"] == 0

    def test_auditor_clarification_question_contains_no_pii(self):
        """The clarification question must not expose raw PII (SSN, MRN) to the user."""
        state = make_state([AMBIGUOUS_EXTRACTION])
        result = auditor_node(state)
        question = result.get("clarification_needed", "").lower()
        for field in ["ssn", "mrn"]:
            assert field not in question, f"PII field '{field}' found in clarification question"


# ── Tests: virtual-source short-circuit (os.path.isfile guard) ───────────────
#
# After the auditor fix, any source string that is not a real file on disk
# (e.g. "openemr_fhir", "policy_search", "EHR_UNAVAILABLE") must pass citation
# verification immediately — there is no local file to compare against, so the
# tool result is trusted directly.  This prevents the 3× retry loop that was
# triggered by policy criteria extractions and FHIR EHR extractions.

FHIR_EXTRACTION = {
    "claim": "John Smith has known allergies: Penicillin, Sulfa.",
    "citation": "Penicillin, Sulfa",
    "source": "openemr_fhir",
    "verbatim": True,
}

POLICY_EXTRACTION = {
    "claim": "[Cigna Medical Policy #012] Criteria A MET: Conservative therapy failure.",
    "citation": "Criteria A: Conservative therapy failure",
    "source": "policy_search",
    "verbatim": True,
}

EHR_UNAVAILABLE_EXTRACTION = {
    "claim": "Allergy status could not be verified — no EHR record found for this patient.",
    "citation": "EHR_UNAVAILABLE",
    "source": "EHR_UNAVAILABLE",
    "verbatim": True,
    "synthetic": True,
}

PDF_CONTENT_EXTRACTION = {
    "claim": "Section 3: Patient has documented Penicillin anaphylaxis.",
    "citation": "Section 3: Patient has documented Penicillin anaphylaxis.",
    "source": "/tmp/mock_data/test_chart.pdf",
    "verbatim": True,
    "kind": "pdf_content",
}


class TestAuditorVirtualSourceShortCircuit:
    """
    Validates that virtual source strings (not real file paths) short-circuit
    _verify_citation_exists_in_source to True so the auditor does not attempt
    to open them as files and fail — which was causing the 3× retry loop.
    """

    def test_fhir_source_passes_without_file_verification(self):
        """openemr_fhir is not a real file — auditor must pass it immediately (routing=pass)."""
        state = make_state([FHIR_EXTRACTION])
        result = auditor_node(state)
        assert result["routing_decision"] == "pass", (
            "FHIR extraction should pass auditor without file-verification retry loop"
        )

    def test_policy_search_source_passes_without_retry_loop(self):
        """
        Policy criteria extractions have synthetic=True but no verbatim field.
        Before the fix, the verbatim check (False by default) fired BEFORE the
        synthetic check, adding all criteria to failed_extractions and triggering
        the 3× retry loop (25 tool calls).  After the fix, synthetic is checked
        first so criteria pass immediately.
        """
        state = make_state([POLICY_EXTRACTION])
        result = auditor_node(state)
        assert result["routing_decision"] == "pass", (
            "Policy criteria extraction should pass auditor in one shot — "
            "was previously causing 25-step retry loop (3 extractor reruns)"
        )

    def test_synthetic_check_fires_before_verbatim_check(self):
        """
        Regression test: synthetic=True with verbatim absent/False must PASS,
        not fail.  If the verbatim gate fired first this would be 'missing'.
        """
        synthetic_no_verbatim = {
            "claim": "[Cigna Medical Policy #012] Criteria A MET: Conservative therapy failure.",
            "citation": "policy_search:cigna",
            "source": "policy_search",
            "synthetic": True,
            # verbatim intentionally absent — mirrors real policy criteria extractions
        }
        state = make_state([synthetic_no_verbatim])
        result = auditor_node(state)
        assert result["routing_decision"] == "pass", (
            "synthetic=True must short-circuit before the verbatim check fires"
        )
        assert result["iteration_count"] == 0, "No retry loop should have occurred"

    def test_ehr_unavailable_source_passes(self):
        """EHR_UNAVAILABLE synthetic extractions must pass auditor (synthetic=True skips verify)."""
        state = make_state([EHR_UNAVAILABLE_EXTRACTION])
        result = auditor_node(state)
        # synthetic=True takes the early-continue path in the auditor loop.
        assert result["routing_decision"] == "pass"

    def test_pdf_content_extraction_passes_via_pdf_guard(self):
        """PDF extractions (.pdf suffix) pass via the existing PDF guard — unchanged behaviour."""
        state = make_state([PDF_CONTENT_EXTRACTION])
        result = auditor_node(state)
        assert result["routing_decision"] == "pass", (
            "PDF content extractions must pass auditor via the .pdf suffix guard"
        )

    def test_mixed_fhir_and_policy_extractions_pass_in_one_shot(self):
        """All virtual-source extractions in a single state must pass without retry."""
        state = make_state([FHIR_EXTRACTION, POLICY_EXTRACTION], iteration_count=0)
        result = auditor_node(state)
        assert result["routing_decision"] == "pass"
        assert result["iteration_count"] == 0, (
            "iteration_count must stay 0 — no retry loop should fire"
        )
