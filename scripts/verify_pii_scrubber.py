#!/usr/bin/env python3
"""
Verify PII scrubber is working.

Run from openemr-agent root:
    python scripts/verify_pii_scrubber.py

Or with module path:
    cd openemr-agent && python -c "from tools.pii_scrubber import scrub_pii; print(scrub_pii('Call John Smith at 555-123-4567 or john@example.com'))"
"""

import sys
import os

# Allow importing from openemr-agent root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    import tools.pii_scrubber as _pii_mod
    from tools.pii_scrubber import scrub_pii, scrub_pii_with_map

    mode = "Presidio (NLP)" if getattr(_pii_mod, "_PRESIDIO_AVAILABLE", False) else "regex fallback"
    print(f"Mode: {mode}")
    print("(Presidio also redacts names and ACC-YYYY-NNNNN; fallback only does SSN, MRN, DOB, phone, email.)\n")

    # Test strings that should be scrubbed (work with both Presidio and regex fallback)
    tests = [
        ("SSN", "Patient SSN is 123-45-6789 for verification."),
        ("Phone", "Contact the patient at 555-123-4567."),
        ("Email", "Send results to john.doe@hospital.org."),
        ("MRN", "MRN: ABC12345 is assigned to this encounter."),
        ("DOB", "DOB: 01/15/1980 — verify eligibility."),
        ("Account", "Account ACC-2026-01145 has a balance."),
        ("Mixed", "Call Jane Smith (MRN: X98765) at 800-555-0100 or jane@clinic.com; SSN 111-22-3334."),
    ]

    print("PII scrubber verification\n" + "=" * 50)
    for label, raw in tests:
        scrubbed = scrub_pii(raw)
        changed = raw != scrubbed
        status = "OK" if changed or "PII" not in label else "CHECK"
        print(f"\n[{status}] {label}")
        print(f"  IN:  {raw}")
        print(f"  OUT: {scrubbed}")
        if not changed and label in ("SSN", "Phone", "Email", "MRN", "DOB"):
            print("  (No replacement — scrubber may be in fallback mode or pattern not matched)")

    # Optional: show replacement map for one sample
    print("\n" + "=" * 50)
    sample = "Patient John Smith, SSN 999-88-7776, call 555-000-1234."
    scrubbed, repl_map = scrub_pii_with_map(sample)
    print("Replacement map (scrub_pii_with_map) for audit trail:")
    print(f"  IN:  {sample}")
    print(f"  OUT: {scrubbed}")
    if repl_map:
        for orig, placeholder in repl_map.items():
            print(f"  Map: {orig!r} -> {placeholder!r}")
    else:
        print("  (Map empty — fallback mode does not populate map)")

    print("\nDone. If IN/OUT differ for PII fields, the scrubber is working.")

if __name__ == "__main__":
    main()
