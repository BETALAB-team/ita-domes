"""Prepare the review-05 response patch; leaves v05 untouched."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
prior = HERE / "article_claims_v05.md"
review = ROOT / "07_reviews/archetype_extension/review_05.md"
ledger = json.loads((HERE / "claim_ledger_v05.json").read_text(encoding="utf-8"))
by_id = {c["id"]: c for c in ledger["claims"]}
updates: dict[str, dict] = {}


def set_fields(cid: str, **changes: object) -> None:
    updates.setdefault(cid, {}).update(changes)


def append_locator(cid: str, locator: str) -> None:
    set_fields(cid, locators=[*updates.get(cid, {}).get("locators", by_id[cid]["locators"]), locator])


matrix = "AE-V2-VALIDATION | 03_methods/archetype_extension/validation_matrix.md, authoritative executed-v2 table and explicitly superseded unexecuted v1 table"
peak = "AE-POSTHOC-PEAK | 04_analysis/archetype_extension/posthoc_peak_context_v1/diagnostic_manifest.json and cohort_counts.csv; 03_methods/archetype_extension/posthoc_peak_context_report.md; all 144 scientific raw-file timestamps, post-hoc descriptive count"

for cid in ("MET-013", "MET-014", "MET-015", "MET-016", "LIM-010"):
    append_locator(cid, matrix)
set_fields("MET-013", assumptions_and_uncertainty=by_id["MET-013"]["assumptions_and_uncertainty"] + " The current validation matrix explicitly marks the five-role v1 design as superseded and unexecuted.")
set_fields("MET-014",
           counterevidence_alternative_explanation="The two representations share four components, but the conditional mixture has twelve free branch-probability parameters versus three pooled; lower RMSE would not isolate demographic information from added capacity.",
           dependencies_next_action_owner="Author corrected two-representation wording and current validation matrix; fresh reviewer06 to assess the squared-loss/RMSE and individual-maximum definition.")
set_fields("MET-015", dependencies_next_action_owner="Author and independent review05 checked training weights and shared components; fresh reviewer06 to assess current validated design wording.")
set_fields("MET-016", dependencies_next_action_owner="Independent review05 reintegrated all 152 annual files and reconstructed node/surface checks; fresh reviewer06 to assess revised peak-context interpretation. Numerical consistency still does not establish measured accuracy.")
set_fields("LIM-010", dependencies_next_action_owner="Author: keep three-versus-twelve free-probability and small-holdout limits adjacent to every reported model-reference deviation; fresh reviewer06 to inspect the corrected current matrix.")

append_locator("RES-010", peak)
set_fields("RES-010",
           assumptions_and_uncertainty="Under Bolzano exposure many primary maxima coincide with the imposed 16 October seasonal restart (RES-013); this post-hoc context does not alter the maximum endpoint. Expected individual maxima do not equal the maximum of an averaged hourly curve or equipment sizing need. Annual-mean sky and native solar-time limitations remain.",
           dependencies_next_action_owner="Independent review05 reproduced the peak losses; author independently recomputed post-hoc raw peak-hour counts, corroborated by a separate all-site diagnostic; fresh reviewer06 to assess seasonal wording.",
           permitted_manuscript_wording="The conditional mixture has lower overall within-model RMSE for expected individual annual maxima at all four exposures, with Bolzano subgroup exceptions; many Bolzano maxima occur on the imposed seasonal restart day (post-hoc diagnostic).",
           related_claims_figures="RES-008; RES-009; RES-011; RES-013; LIM-008; LIM-010")
set_fields("RES-011",
           assumptions_and_uncertainty="Eight test respondents per group, one physical case and Bolzano's imposed seasonal recovery; the subgroup differences are descriptive, not inferential or measured.",
           dependencies_next_action_owner="Independent review05 reproduced the subgroup losses; fresh reviewer06 to assess their adjacency to the post-hoc Bolzano seasonal context.",
           related_claims_figures="RES-010; RES-013; LIM-010")
append_locator("INT-003", peak)
set_fields("INT-003",
           exact_claim="The v2 conditional branch mixtures have lower eligible-stratum-weighted overall within-model RMSE than pooled branch weights for annual useful heating and expected individual annual maxima at each of four TMY exposures, but Bolzano subgroup peak counterexamples, a post-hoc predominance of restart-day individual maxima there, large detailed-input spread, extra conditional parameters and absent measurements prevent a blanket algorithm-superiority or empirical-accuracy conclusion.",
           assumptions_and_uncertainty="The conditional mixture differs only in training source-control branch weights, not shared branch components; group membership may proxy controls without causally determining them. The post-hoc Bolzano count is a timing description, not a causal restart effect.",
           dependencies_next_action_owner="Fresh reviewer06: assess conditional contrast and post-hoc peak interpretation; author: retain counterexamples, seasonal context and model-reference language in every abstract/conclusion statement.",
           permitted_manuscript_wording="In this bounded model-reference experiment, conditional branch frequencies modestly reduce overall loss, with subgroup exceptions and many Bolzano maxima on the imposed restart day; there is no measured-accuracy inference.",
           related_claims_figures="RES-009; RES-010; RES-011; RES-012; RES-013; LIM-008; LIM-010; GAP-005")
set_fields("LIM-008", related_claims_figures="MET-009; RES-007; RES-010; RES-013; INT-001; INT-003")

# Stale next-review placeholders are updated without changing the verified claim substance.
replacement = {
    "RQ-003": "Independent review05 checked the source/input/run bridge and bounded contribution; fresh reviewer06 to inspect the revised seasonal-peak context before updated prose.",
    "MET-012": "Independent review05 reconstructed source profiles and frozen roles; fresh reviewer06 to inspect the current validation matrix and claim chain.",
    "LIM-011": "Author: retain fixed physical/internal-gain and unused dwelling-allocation boundaries; fresh reviewer06 may audit run configurations.",
    "RES-009": "Independent review05 reproduced all four weather-site annual losses from raw outputs; fresh reviewer06 to inspect revised integration and scope; author to retain capacity and model-reference qualifiers.",
    "RES-012": "Independent review05 reproduced detailed-input and mixture spread; fresh reviewer06 to check current result interpretation; author to avoid calling it measured household variability.",
}
for cid, value in replacement.items():
    set_fields(cid, dependencies_next_action_owner=value)

new_claim = {
    "id": "RES-013", "label": "Post-hoc Bolzano peak timing in the new campaign",
    "exact_claim": "In a post-hoc diagnostic of the v2 Bolzano-coordinate annual files, 27 of 32 held-out respondents and two of four shared training components attain their individual annual maximum hourly sensible-heating load on the imposed season-restart day, model hours 6912–6935 (16 October); 18 of 32 respondents and those two components peak exactly at hour 6912. The remaining five respondents and two components peak in January. This is a descriptive calendar coincidence, not an alternative or excluded peak endpoint.",
    "kind": "result", "evidence_class": "simulation output", "status": "partially supported",
    "source_ids": ["AE-RUNS", "AE-POSTHOC-PEAK", "AE-V2-PROTOCOL"],
    "locators": [
        "AE-RUNS | 05_simulations/archetype_campaign_v2/Bolzano/summary.json and 36 role-linked primary *_hourly.csv; role_manifest.csv identifies 32 test and four shared component inputs; raw peak hours recomputed without technical repeat/warm-up files",
        peak,
        "AE-V2-PROTOCOL | 03_methods/archetype_extension/mixture_protocol_v2.md, frozen individual-annual-maximum endpoint and season convention"
    ],
    "outputs": ["04_analysis/archetype_extension/posthoc_peak_context_v1/cohort_counts.csv", "04_analysis/archetype_extension/posthoc_peak_context_v1/scientific_peak_context.csv"],
    "population_geography_period_comparator_units": "Bolzano-coordinate TMY; 32 held-out 2013 survey respondents and four separately simulated shared training components, never combined as a 36-person sample; model hours 6912–6935, individual hourly ideal sensible-load maxima in kW.",
    "assumptions_and_uncertainty": "The diagnostic was added after review05 identified the pattern, then independently recomputed by the author and methods owner from raw files. No tied positive maxima, resimulation, peak exclusion or primary-loss change. Imposed 2023 nonleap fixed-UTC+1 season calendar and model assumptions may shape the pattern.",
    "counterevidence_alternative_explanation": "Five held-out respondents and two shared components peak in January; old v5 pilot's equal Bolzano peaks are a separate one-respondent result. The count alone does not isolate a causal seasonal-restart effect.",
    "falsification_or_weakening": "Different raw first-maximum hours, source-role denominators, calendar indexing, run hashes or tied-maxima handling would change the count; a separately designed season sensitivity test would be required for causal attribution.",
    "dependencies_next_action_owner": "Fresh reviewer06: verify raw-role counts and that the post-hoc timing description is kept separate from the predeclared maximum and model-reference losses; author: retain beside RES-010/011.",
    "permitted_manuscript_wording": "Post hoc, 27/32 held-out Bolzano cases and 2/4 shared components peak on the imposed 16 October restart day (18/32 and 2/4 at its first hour); this describes the saved model outputs and does not alter the annual-maximum comparison.",
    "related_claims_figures": "RES-008; RES-010; RES-011; INT-003; LIM-008"
}

patch = {
    "prior_draft_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
    "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    "status_note": "Revised after independent review05. The executed v2 validation matrix now explicitly separates superseded v1, and a separately labelled post-hoc raw-hourly count qualifies Bolzano individual maxima. Primary endpoints and all frozen runs remain unchanged. New archetype claims remain conditional pending fresh review06 and final closure; the original v04 remains immutable for its original scope.",
    "updates": updates, "new_claims": [new_claim],
}
target = HERE / "revision_06.json"
assert not target.exists(), target
target.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(target)
