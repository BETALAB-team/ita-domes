"""Prepare a procedural v07 after a no-correction independent review06."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
prior = HERE / "article_claims_v06.md"
review = ROOT / "07_reviews/archetype_extension/review_06.md"
ledger = json.loads((HERE / "claim_ledger_v06.json").read_text(encoding="utf-8"))
by_id = {c["id"]: c for c in ledger["claims"]}

# Review06 independently checked the full bounded study and requested no
# scientific correction. These changes update elapsed procedural dependencies.
updates = {
    "RQ-003": {"dependencies_next_action_owner": "Independent reviews05/06 checked the source-to-input bridge, bounded contribution and post-hoc peak context. Author: retain the one-dwelling, small-holdout, model-reference scope in prose; final review-chain closure remains procedural."},
    "MET-012": {"dependencies_next_action_owner": "Independent review06 reconstructed all frozen source profiles and roles. Author: keep the executed-v2 validation matrix as the current methods pointer; final closure remains procedural."},
    "MET-014": {"dependencies_next_action_owner": "Independent review06 reconstructed squared-loss/RMSE comparisons and the expectation-of-individual-maxima estimand. Author: preserve two-representation wording and three-versus-twelve probability-parameter limit in prose."},
    "MET-015": {"dependencies_next_action_owner": "Independent review06 reproduced training branch probabilities and shared component identities. Author: preserve the input-only fitting and same-four-component distinction."},
    "MET-016": {"dependencies_next_action_owner": "Independent review06 reintegrated all 152 annual files and checked numerical state/forcing consistency. Author: preserve the sensible-only and no-measured-validation boundaries; final closure remains procedural."},
    "LIM-010": {"dependencies_next_action_owner": "Author: keep three-versus-twelve free-probability, single-split and small-holdout limits beside model-reference deviations; independent review06 found no new scientific correction."},
    "LIM-011": {"dependencies_next_action_owner": "Author: retain fixed physical/internal-gain and unused dwelling-allocation boundaries; independent review06 checked the run configurations without requiring a scientific change."},
    "RES-009": {
        "exact_claim": by_id["RES-009"]["exact_claim"].replace("These conditional model-reference deviations await fresh independent review and are not measured prediction errors.", "These conditional model-reference deviations are not measured prediction errors."),
        "dependencies_next_action_owner": "Independent review06 reproduced all-site annual endpoints and weighted losses from primary hourly files. Author: retain model-reference, capacity and subgroup qualifiers in prose; final closure remains procedural."
    },
    "RES-010": {"dependencies_next_action_owner": "Independent review06 reproduced the individual-maximum losses and raw post-hoc seasonal counts. Author: keep expected individual maximum distinct from coincident system peak and retain seasonal context."},
    "RES-011": {"dependencies_next_action_owner": "Independent review06 reproduced both adverse Bolzano subgroup comparisons. Author: place these alongside any overall-conditional-improvement sentence and retain the post-hoc seasonal context."},
    "RES-012": {"dependencies_next_action_owner": "Independent review06 reproduced detailed-input and compressed-mixture spread. Author: do not call the modeled spread measured household variability."},
    "RES-013": {"dependencies_next_action_owner": "Independent review06 directly recovered the 27/32 and 2/4 restart-day counts and found no ties. Author: label the diagnostic post hoc and descriptive, beside RES-010/011; retain every original annual maximum."},
    "INT-003": {"dependencies_next_action_owner": "Independent review06 assessed the conditional contrast and post-hoc peak interpretation without a new scientific correction. Author: retain counterexamples, seasonal context and same-model-reference language in abstract/conclusion prose."}
}
assert "await fresh independent review" not in updates["RES-009"]["exact_claim"]
assert len(updates) == 13
patch = {
    "prior_draft_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
    "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    "status_note": "Procedural v07 snapshot after full independent review06 found no new scientific correction. The 106 claim quantities, statuses, source locators, numerical endpoints and study limits remain as in v06; elapsed review-pending dependencies are updated. Separate prior-issue/final closure and source binding remain required before the revised article prose gate. Original v04 and extension v05/v06 remain immutable.",
    "updates": updates, "new_claims": []
}
target = HERE / "revision_07.json"
assert not target.exists(), target
target.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(target)
