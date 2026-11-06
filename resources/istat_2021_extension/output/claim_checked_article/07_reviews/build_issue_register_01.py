"""Write issue register after the first complete author response."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
snapshot = HERE / "issue_register_01.json"
if snapshot.exists():
    raise SystemExit(f"Refusing overwrite: {snapshot}")
review = HERE / "review_01.md"
response = HERE / "response_01.md"
draft = ROOT / "06_claim_drafts/article_claims_v01.md"
entries = [
    {
        "source_round": 1, "issue_id": "R1-01", "severity": "critical",
        "claim_ids": ["MET-003", "MET-005", "MET-008", "RES-004", "RES-005", "RES-007", "INT-001", "RQ-002", "MET-006", "LIM-005", "LIM-006"],
        "disposition": "Accepted. Quarantined all 25 old v1/v2 annual outputs. Local single-commit v3 correction and 20 new runs supply candidate values; author-side integration and declared full-step checks pass. Independent scientific correction review is still required.",
        "revision_locations": ["article_claims_v01.md#met-003", "article_claims_v01.md#met-005", "article_claims_v01.md#met-008", "article_claims_v01.md#res-004", "article_claims_v01.md#res-005", "article_claims_v01.md#res-007", "article_claims_v01.md#lim-006"],
        "evidence_paths": ["07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json", "03_methods/correction_protocol_v3.md", "05_simulations/climate_campaign_v3/Padova/single_commit_1c.py", "05_simulations/climate_campaign_v3/Padova/full_step_balance_checks.json", "04_analysis/paired_schedule_contrasts_v3.csv", "06_claim_drafts/author_result_check_v01.json"],
        "still_open": True, "closure_condition": "Fresh independent review confirms single-commit adapter, transition tests, exported full-step residual arithmetic, paired input equality and candidate numerical results."
    },
    {
        "source_round": 1, "issue_id": "R1-02", "severity": "major",
        "claim_ids": ["GAP-001", "RQ-001", "MET-001", "MET-002", "RES-003", "RES-004", "RES-005", "RES-006", "RES-007", "INT-001"],
        "disposition": "Accepted. Recast intervention as equal encoded 22°C high-setpoint periods with 19°C setback heating allowed. Added post-hoc corrected-run partition, separately labelled from primary endpoints.",
        "revision_locations": ["article_claims_v01.md#gap-001", "article_claims_v01.md#rq-001", "article_claims_v01.md#met-001", "article_claims_v01.md#met-002", "article_claims_v01.md#res-006"],
        "evidence_paths": ["istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf pp41–43", "C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–608", "04_analysis/posthoc_setpoint_partition_v3.csv", "06_claim_drafts/author_posthoc_check_v01.json"],
        "still_open": True, "closure_condition": "Independent review accepts the high-setpoint versus actual-heating distinction and the post-hoc computation."
    },
    {
        "source_round": 1, "issue_id": "R1-03", "severity": "minor",
        "claim_ids": ["DAT-004"],
        "disposition": "Accepted and corrected DAT-004 evidence class from observed measurement to derived estimate (bounded inventory audit) in both Markdown and JSON.",
        "revision_locations": ["article_claims_v01.md#dat-004", "claim_ledger_v01.json DAT-004.evidence_class"],
        "evidence_paths": ["04_analysis/input_audit.json GIS_inventory", "06_claim_drafts/claim_ledger_v01.json"],
        "still_open": False, "closure_condition": "Field remains derived estimate in future readable and machine ledgers."
    },
]
register = {
    "through_round": 1,
    "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
    "revision_filename": draft.name,
    "revision_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
    "entries": entries,
    "standing_limitations": [
        {"claim_id": "LIM-001", "description": "One unweighted respondent and assumed dwelling do not support city/population inference."},
        {"claim_id": "LIM-002", "description": "No compatible independent empirical meter validation has been established."},
        {"claim_id": "LIM-003", "description": "Early/late placements are neither observed schedules nor extrema over all alternatives."},
        {"claim_id": "LIM-004", "description": "Current code/config is not established as the conference executable."},
        {"claim_id": "LIM-005", "description": "No-cooling case does not establish comfort behavior."},
        {"claim_id": "LIM-006", "description": "Original dispatcher-based v1/v2 outputs remain quarantined."},
    ],
}
snapshot.write_text(json.dumps(register, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(HERE / "issue_register.json").write_text(json.dumps(register, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = ["# Scientific issue register through round 01", "",
         f"Review SHA256: `{register['review_sha256']}`. Revision: `../06_claim_drafts/{draft.name}` SHA256 `{register['revision_sha256']}`.", "",
         "| Issue | Severity | Disposition | Open? | Claims |", "|---|---|---|---|---|"]
for e in entries:
    lines.append(f"| {e['issue_id']} | {e['severity']} | {e['disposition']} | {'yes' if e['still_open'] else 'no'} | {', '.join(e['claim_ids'])} |")
lines += ["", "## Standing scientific limits", ""]
for item in register["standing_limitations"]:
    lines.append(f"- {item['claim_id']}: {item['description']}")
lines += ["", "A closed field correction does not certify the pending numerical effects. The corrected adapter and associated results require independent review; no empirical validation has been performed.", ""]
(HERE / "issue_register.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"snapshot": str(snapshot), "open_issues": sum(e["still_open"] for e in entries), "revision_sha256": register["revision_sha256"]}))
