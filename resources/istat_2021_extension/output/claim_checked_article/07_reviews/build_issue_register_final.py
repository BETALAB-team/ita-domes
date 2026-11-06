"""Append independent final closure without altering frozen cycle registers."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
prior_path = here / "issue_register_04.json"
prior_bytes = prior_path.read_bytes()
prior = json.loads(prior_bytes.decode("utf-8"))
closure = here / "final_closure.md"
closure_hash = hashlib.sha256(closure.read_bytes()).hexdigest()
assert closure_hash == "0853158f1092e462a8c753b2cf55a7e87bb3876f2c2565f9caf8a368088618e6"
expected_draft = "8a641ab68bd80fb96aad3016f3deea37ba0dbd4ab50ce9d984d8b59cfe50da7e"
assert prior["revision_sha256"] == expected_draft
assert hashlib.sha256((root / "06_claim_drafts/article_claims_v04.md").read_bytes()).hexdigest() == expected_draft
assert (root / "06_claim_drafts/article_claims_final.md").read_bytes() == (root / "06_claim_drafts/article_claims_v04.md").read_bytes()

final_wording = {
    "R1-01": "Closed for the bounded v5 ideal-sensible single-commit adapter; no latent/system/general dispatcher certification.",
    "R1-02": "Closed for reported hours versus modeled high-setpoint duration and actual positive heating distinction; retain it in prose and figures.",
    "R1-03": "Closed; DAT-004 is a bounded derived-estimate inventory claim, not observed meter evidence.",
    "R2-01": "Closed for declared v5 valid-IR-to-per-surface implementation; no claim of exact hourly radiation physics or measured accuracy.",
    "R2-02": "Closed for historical command annotation and current complete, isolated, byte-identical replay provenance; replay is not another scientific sample.",
    "R3-01": "Closed for exact v5 per-surface correction and conditional fixed-case result admission; 66 pre-v5 annual executions remain quarantined.",
    "R3-02": "Closed for required annual-mean sky, native solar-time and unquantified ranking-sensitivity disclosure.",
    "R3-03": "Closed for explicit 2022 three-band and ±2 h timing precedent; novelty remains provisional and no first-method claim follows.",
    "R3-04": "Closed for current v5 status pointer and requested claim/ledger schema; frozen historical states remain identifiable.",
    "R4-01": "Closed after v04 current-version permission, geography, output locator and EPW field-dictionary repairs in both readable and machine ledgers.",
}
assert {e["issue_id"] for e in prior["entries"]} == set(final_wording)
for entry in prior["entries"]:
    entry["still_open"] = False
    entry["final_disposition"] = final_wording[entry["issue_id"]]
    entry["final_closure_path"] = "07_reviews/final_closure.md"
    entry["final_closure_sha256"] = closure_hash
    if "07_reviews/final_closure.md" not in entry["evidence_paths"]:
        entry["evidence_paths"].append("07_reviews/final_closure.md")

final = {
    **prior,
    "register_type": "final scientific issue closure and current handoff",
    "accepted_claim_snapshot": "06_claim_drafts/article_claims_final.md",
    "accepted_claim_sha256": expected_draft,
    "final_closure_path": "07_reviews/final_closure.md",
    "final_closure_sha256": closure_hash,
    "final_independent_closure_pending": False,
    "unresolved_critical_or_major_scientific_issues": [],
    "administrative_handoff_dependencies": [
        "Finish versioned final source-registry supplement with reviewer04 diagnostic identity and current canonical-summary hashes before prose dispatch.",
        "Refresh mutable current-status/index pointers to cite final closure; preserve frozen historical snapshots.",
    ],
    "scientific_handoff_boundary": "Truthful incomplete first prose manuscript from v04/final byte-identical claims only; no empirical validation, benchmark, general software certification, broad novelty, population or sizing claim.",
}
target = here / "issue_register_final.json"
if target.exists():
    raise SystemExit(f"Refusing overwrite: {target}")
payload = json.dumps(final, ensure_ascii=False, indent=2) + "\n"
target.write_text(payload, encoding="utf-8")
(here / "issue_register.json").write_text(payload, encoding="utf-8")
lines = [
    "# Final scientific issue register",
    "",
    "Four claim-draft/independent-review/author-response/revision cycles are complete. Independent final closure authorizes a bounded, incomplete first prose manuscript from the byte-identical v04 final claim snapshot. The frozen pre-closure issue_register_04.json remains unchanged.",
    "",
    f"Accepted claims SHA-256: `{expected_draft}`. Final independent closure SHA-256: `{closure_hash}`.",
    "",
]
for entry in final["entries"]:
    lines += [f"## {entry['issue_id']} — {entry['severity']} — CLOSED", "", entry["final_disposition"], "", f"Claims: {', '.join(entry['claim_ids'])}.", "", f"Closure evidence: `07_reviews/final_closure.md` (SHA-256 `{closure_hash}`).", ""]
lines += ["## Continuing scientific limits", ""]
for item in final["standing_limitations"]:
    lines += [f"- {item['claim_id']}: {item['description']}"]
lines += ["", "## Administrative handoff work", ""]
for item in final["administrative_handoff_dependencies"]:
    lines += [f"- {item}"]
(here / "issue_register.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
assert prior_path.read_bytes() == prior_bytes
print(json.dumps({"issues": len(final["entries"]), "open": [e["issue_id"] for e in final["entries"] if e["still_open"]], "closure_sha256": closure_hash, "accepted_claim_sha256": expected_draft}))
