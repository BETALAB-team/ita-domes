"""Freeze cycle-03 issue register and refresh current views."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
prior = json.loads((here / "issue_register_02.json").read_text(encoding="utf-8"))
entries = prior["entries"]
by = {x["issue_id"]: x for x in entries}

by["R2-01"].update(
    disposition="Conditionally closed for the bounded sky-to-envelope implementation after valid-IR scalar, guarded active fields and independently checked v5 per-surface actual forcing. V3/v4 physical effects remain quarantined. Fresh review04 of v03 central conditional results is tracked by R3-01 and is not empirical validation.",
    revision_locations=["article_claims_v03.md#met-009", "article_claims_v03.md#met-011", "article_claims_v03.md#lim-007", "article_claims_v03.md#lim-008"],
    evidence_paths=["03_methods/weather_correction_protocol_v4.md", "03_methods/surface_correction_protocol_v5.md", "07_reviews/reviewer_03/v5_forcing_checks.json", "07_reviews/closure_02_after_review03.md", "07_reviews/v5_verification_supplement.md"],
    still_open=False,
    closure_condition="Bounded missing-cloud-to-sky and per-surface implementation verified in v5; final fixed-case effect interpretation remains under R3-01/review04 and retains annual-mean/timing limits.",
)
by["R2-02"].update(
    disposition="Closed. Historical v3 SITE omission is explicitly annotated without rewriting manifests; current runner records complete executable/cwd/argv and refuses existing destinations. Separate Padova seeded replay is byte-identical and not an extra scientific sample.",
    revision_locations=["article_claims_v03.md#met-010", "response_03.md#prior-issue-closure-after-review-03"],
    evidence_paths=["05_simulations/recorded_command_addendum.md", "05_simulations/v4_replay_interface_checks.json", "05_simulations/v5_replay_interface_checks.json", "05_simulations/climate_campaign_v5_replay/Padova/completion_manifest.json", "07_reviews/closure_02_after_review03.md"],
    still_open=False,
    closure_condition="Preserve complete command/working-directory/source-hash and isolated replay discipline for later versions.",
)

entries.extend([
    {
        "source_round": 3,
        "issue_id": "R3-01",
        "severity": "critical",
        "claim_ids": ["MET-003", "MET-005", "MET-008", "MET-009", "MET-011", "RES-004", "RES-005", "RES-006", "RES-007", "RES-008", "INT-001", "LIM-007"],
        "disposition": "Accepted. V4 roof shortwave/longwave and order-dependent forcing invalidated all v4 physical values; all 66 pre-v5 annual executions are quarantined. A new study-local per-surface v5 correction, Padova-first then four-site campaign, 20 annual outputs and one separate replay passed analytic, actual-solver forcing, order, state, repeat and arithmetic checks. Independent reviewer03 supplement conditionally verifies exact bounded remedy. V03 effects remain candidate pending fresh review04.",
        "revision_locations": ["article_claims_v03.md#met-009", "article_claims_v03.md#met-011", "article_claims_v03.md#met-005", "article_claims_v03.md#res-004", "article_claims_v03.md#res-005", "article_claims_v03.md#res-006", "article_claims_v03.md#res-007", "article_claims_v03.md#res-008", "article_claims_v03.md#int-001", "article_claims_v03.md#lim-007"],
        "evidence_paths": ["07_reviews/reviewer_03/runtime_forcing_checks.json", "03_methods/surface_correction_protocol_v5.md", "04_analysis/surface_forcing_1c.py", "07_reviews/reviewer_03/v5_forcing_checks.json", "07_reviews/reviewer_03/independent_audit_v5.json", "06_claim_drafts/author_v5_check_for_v03.json", "06_claim_drafts/author_v5_posthoc_peak_check_v03.json", "07_reviews/v5_verification_supplement.md"],
        "still_open": True,
        "closure_condition": "Fresh independent review04 of v03 independently accepts source-to-profile mapping, exact v5 per-surface and IR forcing, all-run state/arithmetical provenance, conditional annual/peak/post-hoc effects and limits. No empirical accuracy or general package validation is implied.",
    },
    {
        "source_round": 3,
        "issue_id": "R3-02",
        "severity": "minor",
        "claim_ids": ["MET-009", "RQ-001", "RQ-002", "RES-004", "RES-005", "RES-007", "INT-001", "LIM-008"],
        "disposition": "Accepted. New LIM-008 and revised central method/result/interpretation claims state annual-mean sky exchange and native fixed-UTC+1 solar position without application of recorded irradiation offset; no timing/longwave sensitivity or robust hourly-peak ranking is claimed.",
        "revision_locations": ["article_claims_v03.md#met-009", "article_claims_v03.md#rq-002", "article_claims_v03.md#res-007", "article_claims_v03.md#lim-008"],
        "evidence_paths": ["03_methods/weather_correction_protocol_v4.md", "04_analysis/ir_sky_forcing.py", "C:/Works/EUReCA/EUReCA/eureca_building/weather.py", "06_claim_drafts/claim_ledger_v03.json"],
        "still_open": True,
        "closure_condition": "Fresh review04 confirms the limitation remains attached to small annual effects and hourly maxima, and no unsupported exact-solar-phase or robust peak-ranking claim appears.",
    },
    {
        "source_round": 3,
        "issue_id": "R3-03",
        "severity": "minor",
        "claim_ids": ["LIT-002", "GAP-001", "NOV-001"],
        "disposition": "Accepted. LIT-002 now explicitly identifies Prataviera 2022 Section3.1.1 ±2 h startup perturbation alongside three-band random profiles. GAP-001 and provisional NOV-001 distinguish the present fixed-respondent within-band placement contrast without a first-method assertion.",
        "revision_locations": ["article_claims_v03.md#lit-002", "article_claims_v03.md#gap-001", "article_claims_v03.md#nov-001"],
        "evidence_paths": ["02_evidence/source_ledger.md SRC-UA2022", "06_claim_drafts/claim_ledger_v03.json"],
        "still_open": True,
        "closure_condition": "Fresh review04 confirms exact prior-work correspondence and bounded novelty language; no exhaustive gap claim is inferred from the targeted search.",
    },
    {
        "source_round": 3,
        "issue_id": "R3-04",
        "severity": "minor",
        "claim_ids": ["MET-007", "MET-010", "LIM-006", "LIM-007"],
        "disposition": "Accepted and corrected. Canonical simulation/validation protocol and matrix now lead with one current v5 candidate status, retain historical chronology and quarantine, and point to exact v5 outputs. Frozen run snapshots remain untouched. Claim-status enum audit also corrected five v02 non-enum `withheld` values in v03.",
        "revision_locations": ["article_claims_v03.md#met-007", "article_claims_v03.md#met-010", "article_claims_v03.md#lim-007", "claim_ledger_v03.json status fields"],
        "evidence_paths": ["03_methods/simulation_protocol.md", "03_methods/validation_protocol.md", "03_methods/validation_matrix.md", "06_claim_drafts/claim_ledger_v03.json"],
        "still_open": False,
        "closure_condition": "Canonical current-status pointer and allowed status enum persist in future versions; frozen historical records retain their chronology.",
    },
])

standing = prior["standing_limitations"] + [
    {"claim_id": "LIM-008", "description": "Annual-mean sky exchange and unapplied native solar irradiance timing offset limit hourly-peak interpretation."},
    {"claim_id": "RES-008", "description": "Bolzano zero annual-peak differences arise at a common modeled season-onset recovery in v5; this is a post-hoc context, not an equipment-sizing inference."},
]
review = here / "review_03.md"
response = here / "response_03.md"
draft = root / "06_claim_drafts/article_claims_v03.md"
register = {
    "through_round": 3,
    "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
    "revision_filename": draft.name,
    "revision_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
    "review_02_closure_sha256": hashlib.sha256((here / "closure_02_after_review03.md").read_bytes()).hexdigest(),
    "v5_verification_supplement_sha256": hashlib.sha256((here / "v5_verification_supplement.md").read_bytes()).hexdigest(),
    "entries": entries,
    "standing_limitations": standing,
}
snapshot = here / "issue_register_03.json"
if snapshot.exists():
    raise SystemExit(f"Refusing overwrite: {snapshot}")
payload = json.dumps(register, ensure_ascii=False, indent=2) + "\n"
snapshot.write_text(payload, encoding="utf-8")
(here / "issue_register.json").write_text(payload, encoding="utf-8")
lines = ["# Current scientific issue register", "", "Through independent review 03 and author revision v03. Earlier cycle registers and run snapshots remain immutable. V5 numerical effects are candidate conditional model results pending fresh review 04.", "", f"Revision SHA-256: `{register['revision_sha256']}`. Review 03 SHA-256: `{register['review_sha256']}`. Response 03 SHA-256: `{register['response_sha256']}`.", ""]
for e in entries:
    lines += [f"## {e['issue_id']} — {e['severity']} — {'OPEN' if e['still_open'] else 'CLOSED / QUALIFIED'}", "", e["disposition"], "", f"Claim IDs: {', '.join(e['claim_ids'])}.", "", f"Closure condition / continuing obligation: {e['closure_condition']}", "", f"Evidence: {'; '.join(e['evidence_paths'])}.", ""]
lines += ["## Standing limits", ""]
for s in standing:
    lines += [f"- {s['claim_id']}: {s['description']}"]
(here / "issue_register.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(json.dumps({"revision_sha256": register["revision_sha256"], "response_sha256": register["response_sha256"], "register_entries": len(entries), "open": [e["issue_id"] for e in entries if e["still_open"]]}))
