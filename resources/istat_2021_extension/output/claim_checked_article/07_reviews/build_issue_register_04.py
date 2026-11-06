"""Freeze cycle-04 register while keeping independent final closure explicit."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
prior = json.loads((here / "issue_register_03.json").read_text(encoding="utf-8"))
entries = prior["entries"]
by = {e["issue_id"]: e for e in entries}

by["R3-01"].update(
    disposition="Accepted; specific v4 per-surface forcing defect corrected locally in v5. Fresh review04 independently reconstructed all 20 v5 actual solver-forcing/state/accounting files and reproduced conditional results, with no new critical/major numerical issue. Earlier 66 annual executions remain quarantined. Final independent closure of critical and central result claims is still required before accepted handoff.",
    revision_locations=["article_claims_v04.md#met-003", "article_claims_v04.md#met-005", "article_claims_v04.md#met-009", "article_claims_v04.md#met-011", "article_claims_v04.md#res-004", "article_claims_v04.md#res-005", "article_claims_v04.md#res-007", "article_claims_v04.md#lim-007"],
    evidence_paths=["07_reviews/reviewer_03/v5_forcing_checks.json", "07_reviews/reviewer_03/independent_audit_v5.json", "07_reviews/reviewer_04/independent_recheck.json", "05_simulations/climate_campaign_v5/Padova/completion_manifest.json", "04_analysis/paired_schedule_contrasts_v5.csv"],
    still_open=True,
    closure_condition="Independent reviewer04 final closure pass confirms v04 metadata/provenance repairs, bounded v5 forcing/state/results and preservation of approximation, novelty and validation limits; no later blocker arises.",
)
by["R3-02"].update(
    disposition="Accepted. Fresh review04 confirms annual-mean sky and unapplied native solar-time offset limits are material and present in v03; v04 removes stale method permission wording and retains those limits across central claims. No robustness experiment or accurate hourly phase claim is asserted.",
    revision_locations=["article_claims_v04.md#met-003", "article_claims_v04.md#met-009", "article_claims_v04.md#res-007", "article_claims_v04.md#lim-008"],
    evidence_paths=["03_methods/weather_correction_protocol_v4.md", "07_reviews/review_04.md", "06_claim_drafts/claim_ledger_v04.json"],
    still_open=True,
    closure_condition="Final independent closure confirms current permitted wording and result interpretation keep annual-mean/timing and unquantified peak-ranking limits.",
)
by["R3-03"].update(
    disposition="Accepted. Fresh review04 independently opened Prataviera 2022 and confirms v03 correctly acknowledges three-band randomization and ±2 h timing perturbation. V04 retains exact fixed-case distinction and unverified broad novelty status.",
    revision_locations=["article_claims_v04.md#lit-002", "article_claims_v04.md#gap-001", "article_claims_v04.md#nov-001"],
    evidence_paths=["02_evidence/source_ledger.md SRC-UA2022", "07_reviews/review_04.md", "06_claim_drafts/claim_ledger_v04.json"],
    still_open=True,
    closure_condition="Final independent closure confirms no first-method, exhaustive gap or causal novelty language enters accepted claims/prose instructions.",
)
by["R3-04"].update(
    disposition="Closed with continuing maintenance obligation. Canonical methods/validation records now lead with v5 candidate disposition and historical quarantine; fourth review found no current-pointer conflict. V04 uses only allowed kind/evidence/status values in readable and JSON claim ledgers.",
    revision_locations=["article_claims_v04.md#met-007", "article_claims_v04.md#met-010", "article_claims_v04.md#lim-007", "claim_ledger_v04.json schema fields"],
    evidence_paths=["03_methods/simulation_protocol.md", "03_methods/validation_protocol.md", "03_methods/validation_matrix.md", "06_claim_drafts/claim_ledger_v04.json"],
    still_open=False,
    closure_condition="Current-version pointers and allowed enums continue to match the final frozen snapshot; historical protocols remain unchanged.",
)

entries.append({
    "source_round": 4,
    "issue_id": "R4-01",
    "severity": "minor",
    "claim_ids": ["DAT-002", "MET-002", "MET-003", "MET-004", "MET-005", "MET-009", "MET-010", "RES-004", "RES-005", "RES-006", "RES-007", "RES-008", "INT-001"],
    "disposition": "Accepted. V04 replaces stale v3 invariant provenance, reviewer02 action, pending-weather permission wording, and Padova-only INT-001 geography with current v5 evidence/scope; repairs dead EPW24.1 link to inspected 24.2 dictionary mirror; harmonizes machine-level residual wording and result permissions. Central numbers and scientific limits are unchanged.",
    "revision_locations": ["article_claims_v04.md#dat-002", "article_claims_v04.md#met-002", "article_claims_v04.md#met-003", "article_claims_v04.md#met-004", "article_claims_v04.md#met-005", "article_claims_v04.md#met-009", "article_claims_v04.md#met-010", "article_claims_v04.md#res-004", "article_claims_v04.md#res-005", "article_claims_v04.md#res-006", "article_claims_v04.md#res-007", "article_claims_v04.md#res-008", "article_claims_v04.md#int-001"],
    "evidence_paths": ["07_reviews/reviewer_04/independent_recheck.json", "05_simulations/climate_campaign_v5/Padova/summary.json", "04_analysis/paired_schedule_contrasts_v5.csv", "02_evidence/local_registry_audit_v03.md", "06_claim_drafts/claim_ledger_v04.json"],
    "still_open": True,
    "closure_condition": "Final independent reviewer checks v04 current-version permitted wording, output/source locators, machine-ledger parity and all-site scope; final source-registry audit finds zero missing aliases and no dead declared primary source locator.",
})

standing = prior["standing_limitations"]
review = here / "review_04.md"
response = here / "response_04.md"
draft = root / "06_claim_drafts/article_claims_v04.md"
register = {
    "through_round": 4,
    "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
    "revision_filename": draft.name,
    "revision_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
    "entries": entries,
    "standing_limitations": standing,
    "final_independent_closure_pending": True,
}
snapshot = here / "issue_register_04.json"
if snapshot.exists():
    raise SystemExit(f"Refusing overwrite: {snapshot}")
payload = json.dumps(register, ensure_ascii=False, indent=2) + "\n"
snapshot.write_text(payload, encoding="utf-8")
(here / "issue_register.json").write_text(payload, encoding="utf-8")
lines = ["# Current scientific issue register", "", "Four complete claim-draft/independent-review/author-response/revision cycles are recorded. V04 is frozen; a separate independent closure of prior critical/major gates and current metadata remains pending before accepted prose handoff.", "", f"V04 SHA-256: `{register['revision_sha256']}`. Review04 SHA-256: `{register['review_sha256']}`. Response04 SHA-256: `{register['response_sha256']}`.", ""]
for e in entries:
    lines += [f"## {e['issue_id']} — {e['severity']} — {'OPEN' if e['still_open'] else 'CLOSED / QUALIFIED'}", "", e["disposition"], "", f"Claim IDs: {', '.join(e['claim_ids'])}.", "", f"Closure condition / continuing obligation: {e['closure_condition']}", "", f"Evidence: {'; '.join(e['evidence_paths'])}.", ""]
lines += ["## Standing limits", ""]
for s in standing:
    lines += [f"- {s['claim_id']}: {s['description']}"]
(here / "issue_register.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(json.dumps({"revision_sha256": register["revision_sha256"], "response_sha256": register["response_sha256"], "register_entries": len(entries), "open": [e["issue_id"] for e in entries if e["still_open"]]}))
