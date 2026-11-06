"""Freeze cycle-02 issue register and current readable/machine views."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
prior = json.loads((here / "issue_register_01.json").read_text(encoding="utf-8"))
entries = prior["entries"]
by_id = {x["issue_id"]: x for x in entries}
closure = "07_reviews/closure_01_after_review02.md"

by_id["R1-01"].update(
    disposition="Conditionally closed for the specific ideal-sensible 1C single-commit state repair after fresh independent transition and 175,200-hour balance checks. This does not admit physical annual results; separate R2-01 remains open for invalid sky forcing and new runs.",
    revision_locations=["article_claims_v02.md#met-003", "article_claims_v02.md#met-005", "article_claims_v02.md#met-008", "article_claims_v02.md#lim-006", "article_claims_v02.md#lim-007"],
    evidence_paths=["07_reviews/reviewer_02/independent_audit.json", "07_reviews/reviewer_02/runtime_checks.json", closure],
    still_open=False,
    closure_condition="Narrow dispatcher correction independently verified within the declared sensible 1C scope; physical-result admission is tracked in R2-01.",
)
by_id["R1-02"].update(
    disposition="Closed for wording and estimand: survey-reported mean operating hours, encoded high-setpoint hours and actual positive heating are separate. Independent source-profile checks confirm mapping; old v3 partition quantity is withheld under R2-01.",
    revision_locations=["article_claims_v02.md#gap-001", "article_claims_v02.md#met-001", "article_claims_v02.md#met-002", "article_claims_v02.md#res-003", "article_claims_v02.md#res-006"],
    evidence_paths=["C:/Works/GRINS/GRINS-Tool/classes/EndUse.py", "07_reviews/reviewer_02/runtime_checks.json", closure],
    still_open=False,
    closure_condition="Continue the distinction in later prose/figures; future corrected-run partitions require separate result review under R2-01.",
)
by_id["R1-03"].update(
    disposition="Closed: DAT-004 remains a bounded derived-estimate inventory claim in readable and machine ledgers; no observed measurement is asserted.",
    revision_locations=["article_claims_v02.md#dat-004", "claim_ledger_v02.json DAT-004.evidence_class"],
    evidence_paths=["06_claim_drafts/claim_ledger_v02.json", closure],
    still_open=False,
    closure_condition="Preserve derived-estimate class and bounded inventory scope.",
)

entries.extend([
    {
        "source_round": 2,
        "issue_id": "R2-01",
        "severity": "critical",
        "claim_ids": ["DAT-002", "MET-003", "MET-005", "MET-006", "MET-008", "MET-009", "RES-004", "RES-005", "RES-006", "RES-007", "INT-001", "LIM-005", "LIM-007"],
        "disposition": "Accepted. All v3 physical annual/peak, setback-partition and summer-temperature values withdrawn; 20 v3 runs retained only as defective-forcing diagnostics. Narrow state correction remains supported. A documented local longwave/IR or justified alternate pathway and complete weather QC must precede new Padova/all-site annual runs and independent review.",
        "revision_locations": ["article_claims_v02.md#dat-002", "article_claims_v02.md#met-005", "article_claims_v02.md#met-006", "article_claims_v02.md#met-009", "article_claims_v02.md#res-004", "article_claims_v02.md#res-005", "article_claims_v02.md#res-006", "article_claims_v02.md#res-007", "article_claims_v02.md#int-001", "article_claims_v02.md#lim-005", "article_claims_v02.md#lim-007"],
        "evidence_paths": ["07_reviews/reviewer_02/independent_audit.json", "07_reviews/reviewer_02/runtime_checks.json", "C:/Works/EUReCA/EUReCA/eureca_building/weather.py", "C:/Works/EUReCA/EUReCA/eureca_building/thermal_zone.py", "05_simulations/climate_inputs_v2/acquisition_manifest.json"],
        "still_open": True,
        "closure_condition": "Predeclared and documented valid sky treatment, guards for all consumed fields, independent night/zero-solar and forcing diagnostics, versioned new Padova/all-site annual outputs, passed state/repeat/warm-up and arithmetic checks, and fresh independent scientific review. No v3 numerical effect is promoted by state closure alone.",
    },
    {
        "source_round": 2,
        "issue_id": "R2-02",
        "severity": "minor",
        "claim_ids": ["MET-003", "MET-005", "MET-006", "MET-010", "RES-004", "RES-005", "RES-007"],
        "disposition": "Accepted. Historic v3 manifest omits required SITE argument. Response records complete Padova command syntax as a labelled provenance correction, without rewriting manifests or executing fixed-output historic script. New separate-destination replay entry point and exact command/environment record are pending.",
        "revision_locations": ["article_claims_v02.md#met-010", "response_02.md#r2-02--minor-incomplete-recorded-command"],
        "evidence_paths": ["07_reviews/reviewer_02/recorded_command_replay.txt", "04_analysis/run_climate_campaign_v3.py", "05_simulations/climate_campaign_v3/Padova/completion_manifest.json"],
        "still_open": True,
        "closure_condition": "A tested replay command for new versioned runs records executable, working directory, complete site argv and isolated output destination, with output/source hashes; historical manifests stay immutable with annotation.",
    },
])

standing = prior["standing_limitations"] + [{"claim_id": "LIM-007", "description": "V3 weather-forcing annual outputs are quarantined despite correct narrow state-equation closure."}]
response = here / "response_02.md"
revision = root / "06_claim_drafts/article_claims_v02.md"
review = here / "review_02.md"
register = {
    "through_round": 2,
    "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
    "revision_filename": revision.name,
    "revision_sha256": hashlib.sha256(revision.read_bytes()).hexdigest(),
    "review_01_closure_sha256": hashlib.sha256((here / "closure_01_after_review02.md").read_bytes()).hexdigest(),
    "entries": entries,
    "standing_limitations": standing,
}
snapshot = here / "issue_register_02.json"
if snapshot.exists():
    raise SystemExit(f"Refusing to overwrite immutable register: {snapshot}")
payload = json.dumps(register, ensure_ascii=False, indent=2) + "\n"
snapshot.write_text(payload, encoding="utf-8")
(here / "issue_register.json").write_text(payload, encoding="utf-8")
lines = ["# Current scientific issue register", "", "Through independent review 02 and author revision v02. Prior cycle registers remain immutable. ‘Closed’ for R1-01 applies only to the bounded single-commit dispatcher repair; physical outputs remain blocked by R2-01.", "", f"Revision SHA-256: `{register['revision_sha256']}`. Review 02 SHA-256: `{register['review_sha256']}`. Response 02 SHA-256: `{register['response_sha256']}`.", ""]
for e in entries:
    lines += [f"## {e['issue_id']} — {e['severity']} — {'OPEN' if e['still_open'] else 'CLOSED / QUALIFIED'}", "", e["disposition"], "", f"Claim IDs: {', '.join(e['claim_ids'])}.", "", f"Closure condition / continuing obligation: {e['closure_condition']}", "", f"Evidence: {'; '.join(e['evidence_paths'])}.", ""]
lines += ["## Standing limits", ""]
for s in standing:
    lines += [f"- {s['claim_id']}: {s['description']}"]
(here / "issue_register.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(json.dumps({"register": str(snapshot), "entries": len(entries), "open": [x["issue_id"] for x in entries if x["still_open"]], "response_sha256": register["response_sha256"], "revision_sha256": register["revision_sha256"]}))
