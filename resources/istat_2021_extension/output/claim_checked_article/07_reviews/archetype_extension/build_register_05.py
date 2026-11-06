"""Record review-05 dispositions while preserving frozen issue snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


entries = [
    {
        "issue_id": "R5-01", "severity": "minor",
        "claim_ids": ["MET-013", "MET-014", "MET-015", "MET-016", "LIM-010"],
        "disposition": "Accepted and corrected in the current executed-v2 validation matrix and v06 claim wording. The retained v1 matrix is labelled historical, superseded and unexecuted; the two representations share four components but have three versus twelve free probabilities. Independent closure awaits review06.",
        "revision_locations": ["06_claim_drafts/article_claims_v06.md#met-013", "06_claim_drafts/article_claims_v06.md#met-014", "06_claim_drafts/article_claims_v06.md#lim-010"],
        "evidence_paths": ["03_methods/archetype_extension/validation_matrix.md", "03_methods/archetype_extension/validation_matrix_pre_review05.md", "03_methods/archetype_extension/mixture_protocol_v2.md", "07_reviews/archetype_extension/review_05.md"],
        "still_open": True,
        "closure_condition": "Fresh reviewer confirms that authoritative validation documentation specifies the executed v2 four-shared-component, two-representation, squared-loss design and identifies v1 as superseded."
    },
    {
        "issue_id": "R5-02", "severity": "minor",
        "claim_ids": ["RES-010", "RES-011", "RES-013", "INT-003", "LIM-008"],
        "disposition": "Accepted with new explicitly post-hoc v2 Bolzano peak-timing claim. Direct independent author and methods raw-hourly checks yield 27/32 held-out and 2/4 component restart-day maxima, with 18/32 and 2/4 at restart hour. All primary maxima and losses are unchanged. Independent closure awaits review06.",
        "revision_locations": ["06_claim_drafts/article_claims_v06.md#res-010", "06_claim_drafts/article_claims_v06.md#res-013", "06_claim_drafts/article_claims_v06.md#int-003"],
        "evidence_paths": ["05_simulations/archetype_campaign_v2/Bolzano/summary.json", "04_analysis/archetype_extension/posthoc_peak_context_v1/diagnostic_manifest.json", "04_analysis/archetype_extension/posthoc_peak_context_v1/cohort_counts.csv", "03_methods/archetype_extension/posthoc_peak_context_report.md", "06_claim_drafts/author_peak_context_check_v2.json", "07_reviews/archetype_extension/review_05.md"],
        "still_open": True,
        "closure_condition": "Fresh reviewer verifies raw role-linked maximum hours and exact denominators, retained primary endpoint/losses, and descriptive rather than causal or equipment-sizing wording."
    }
]
review = HERE / "review_05.md"
response = HERE / "response_05.md"
revision = ROOT / "06_claim_drafts/article_claims_v06.md"
assert sha(review) == "1b62bd323c4ca0682d3ac6428891172e45d458cc8ac15b3746e0b4142eede899"
assert sha(revision) == "0075415fec3e985bbd55a6cfc363fd94a7d73b9421ddc9699ff82a4ae79f5c7b"
register = {"through_round": 5, "review_sha256": sha(review), "response_sha256": sha(response),
            "revision_filename": revision.name, "revision_sha256": sha(revision), "entries": entries}
snapshot = HERE / "issue_register_05.json"
assert not snapshot.exists(), snapshot
snapshot.write_text(json.dumps(register, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(HERE / "issue_register.json").write_text(json.dumps(register, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = ["# Archetype-extension issue register after review 05", "",
         f"Reviewed v05 SHA256 `{sha(ROOT / '06_claim_drafts/article_claims_v05.md')}`; review05 SHA256 `{sha(review)}`; response05 SHA256 `{sha(response)}`; revised v06 SHA256 `{sha(revision)}`.", ""]
for entry in entries:
    lines.extend((f"## {entry['issue_id']} — {entry['severity']}", "", entry["disposition"], "",
                  f"Claims: {', '.join(entry['claim_ids'])}.", "",
                  f"Closure condition: {entry['closure_condition']}", ""))
(HERE / "issue_register.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(json.dumps({"review": sha(review), "response": sha(response), "revision": sha(revision), "register": sha(snapshot)}))
