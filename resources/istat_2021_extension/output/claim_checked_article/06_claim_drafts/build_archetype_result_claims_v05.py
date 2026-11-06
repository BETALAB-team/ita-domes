"""Build candidate run-backed v05 claim supplement from author integration checks."""
from __future__ import annotations

import json
from pathlib import Path

from prepare_archetype_extension_v05 import add

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / "archetype_extension_work"
AUTHOR = HERE / "author_archetype_result_check_v2.json"
source = json.loads(AUTHOR.read_text(encoding="utf-8"))
assert source["cross_check"]["all_method_overall_metrics_match"]
assert source["cross_check"]["subgroup_rmse_rows"] == 32
assert list(source["sites"]) == ["Padova", "Bolzano", "Roma", "Palermo"]

campaign = "AE-CAMPAIGN | 05_simulations/archetype_campaign_v2/campaign_completion_manifest.json, all-site scientific/technical counts and immutable run identities"
gates = "AE-SITE-GATES | 05_simulations/archetype_campaign_v2/{Padova,Bolzano,Roma,Palermo}/site_gate.json, completion_manifest.json, full_step_balance_checks.json, independent_surface_forcing_checks.json and profile_preservation_checks.json"
runs = "AE-RUNS | 05_simulations/archetype_campaign_v2/{Padova,Bolzano,Roma,Palermo}/summary.json and 36 role-linked primary *_hourly.csv per site; per-file hashes in summaries/manifests"
metrics = "AE-METRICS | 04_analysis/archetype_extension/results_v2/weighted_metrics.csv and individual_deviations.csv, predeclared weighted MSE/RMSE and group/site records"
author = "AUTHOR-AE-RESULT | 06_claim_drafts/author_archetype_result_check_v2.json, independent integration of 152 hourly CSVs and recomputation of all eight overall/site plus 32 subgroup RMSE rows"
protocol = "AE-V2-PROTOCOL | 03_methods/archetype_extension/mixture_protocol_v2.md, shared-component mixture endpoint and evaluation definitions"

claims = []
add(claims, "MET-016", "Completed and checked archetype campaign",
    "The v2 archetype campaign completed 144 scientific annual executions (36 distinct input profiles × four coordinate TMYs) and eight repeat/warm-up checks, exporting 1,331,520 hourly rows; all four site gates and independent author hourly-integration, source-hash and mixture-arithmetic checks pass for this bounded sensible-only model path.",
    "method", "simulation output", "partially supported", [campaign, gates, author],
    "32 held-out respondents and four shared real training branch components; one fixed dwelling; Padova, Bolzano, Roma and Palermo coordinate TMYs; hourly sensible-only runs.",
    "The 152 executions are not independent households; 144 scientific profiles include the same 36 inputs at four exposures. Technical checks cannot establish empirical model accuracy.",
    "Earlier 66 defective original executions remain quarantined and are not used for new effects; v1 archetype medoid design had no annual execution.",
    "Any failed source/profile hash, all-hour balance, per-surface forcing, repeat/warm-up gate, or independent CSV integration would withdraw affected outcomes.",
    "Fresh independent review05: inspect campaign source/input lineage, all four gates and author result arithmetic before claiming conditional effects.",
    "All 144 v2 scientific runs and eight technical checks passed their declared bounded implementation gates.",
    "MET-012; MET-013; MET-014; MET-015; RES-009; RES-010",
    ["05_simulations/archetype_campaign_v2/campaign_completion_manifest.json", "06_claim_drafts/author_archetype_result_check_v2.json"])

sites = ["Padova", "Bolzano", "Roma", "Palermo"]
heat = [source["sites"][s]["losses"]["heat_kWh"] for s in sites]
peak = [source["sites"][s]["losses"]["peak_kW"] for s in sites]
def pair(x: dict, places: int = 2) -> str:
    return f"{x['pooled_weighted_rmse']:.{places}f}→{x['conditional_weighted_rmse']:.{places}f}"

heat_exact = (
    "In v2, eligible-stratum-weighted held-out annual useful-heating RMSE against detailed-input outputs from the same model "
    + "(pooled→group-conditional) is "
    + ", ".join(f"{s} {pair(x)} kWh" for s, x in zip(sites, heat))
    + "; paired weighted MSE differences (conditional minus pooled) are "
    + ", ".join(f"{s} {x['paired_weighted_mse_difference_conditional_minus_pooled']:.2f} kWh²" for s, x in zip(sites, heat))
    + ". These conditional model-reference deviations await fresh independent review and are not measured prediction errors."
)
add(claims, "RES-009", "Overall annual-heating compression fidelity",
    heat_exact, "result", "simulation output", "partially supported", [runs, metrics, author, protocol],
    "Unweighted 13,624-record eligible archived cohort reexpressed by 32 stratified test responses; four weather files; annual ideal useful sensible heating kWh and MSE kWh².",
    "Both methods use the same four simulated training branch components; conditional mixture has 12 branch-probability parameters versus pooled three. Detailed-input outputs are model references, not metered truth.",
    "Within-branch variation and the small holdout may dominate errors; observed group-response association cannot establish demographic causation or population-wide superiority.",
    "Any raw hourly integration, role/probability/weight recomputation, or fresh physical-model review failure would withdraw the values.",
    "Reviewer05: independently check source-to-role mapping, all four weather gates, group/pooled predictions and weighted losses; author: retain endpoint and capacity qualifiers.",
    "In this fixed dwelling and eligible held-out cohort, group-conditional branch weights give lower overall within-model annual-heating RMSE than pooled branch weights at each of four TMY exposures.",
    "RQ-003; MET-014; MET-016; LIM-010; RES-011",
    ["04_analysis/archetype_extension/results_v2/weighted_metrics.csv", "06_claim_drafts/author_archetype_result_check_v2.json"])

peak_exact = (
    "In v2, eligible-stratum-weighted held-out expected-individual-annual-maximum sensible-load RMSE against detailed-input outputs from the same model "
    + "(pooled→group-conditional) is "
    + ", ".join(f"{s} {pair(x, 4)} kW" for s, x in zip(sites, peak))
    + "; paired weighted MSE differences (conditional minus pooled) are "
    + ", ".join(f"{s} {x['paired_weighted_mse_difference_conditional_minus_pooled']:.4f} kW²" for s, x in zip(sites, peak))
    + ". The prediction is a weighted expectation of individual annual maxima, not a coincident or diversified system peak."
)
add(claims, "RES-010", "Overall individual-peak compression fidelity",
    peak_exact, "result", "simulation output", "partially supported", [runs, metrics, author, protocol],
    "Same 32 held-out respondents and four weather exposures; each individual annual maximum hourly ideal sensible load kW, MSE kW².",
    "Maxima can occur at common imposed season onset; expected individual maxima do not equal the maximum of an averaged hourly curve or equipment sizing need.",
    "Some group-specific peak RMSE values worsen despite a lower overall cohort RMSE; see RES-011.",
    "Any peak extraction, role/probability/weight, seasonal-calendar or fresh review error would weaken the reported comparison.",
    "Reviewer05: check per-person maxima and weighted expectation, not a coincident system peak; author: include subgroup counterexamples.",
    "The conditional mixture has lower overall within-model RMSE for expected individual annual maxima at all four exposures, with subgroup exceptions.",
    "RES-008; RES-009; RES-011; LIM-008; LIM-010",
    ["04_analysis/archetype_extension/results_v2/weighted_metrics.csv", "06_claim_drafts/author_archetype_result_check_v2.json"])

g2 = peak[1]["group_rmse"]["2"]
g4 = peak[1]["group_rmse"]["4"]
assert g2["conditional"] > g2["pooled"] and g4["conditional"] > g4["pooled"]
add(claims, "RES-011", "Bolzano subgroup counterexamples",
    f"Despite the lower overall Bolzano expected-individual-annual-maximum RMSE, group 2 has pooled→conditional group-weighted RMSE {g2['pooled']:.6f}→{g2['conditional']:.6f} kW and group 4 {g4['pooled']:.6f}→{g4['conditional']:.6f} kW; conditional weighting is worse for these two held-out subgroups.",
    "result", "simulation output", "partially supported", [metrics, author, runs],
    "Bolzano-coordinate TMY; eight held-out responses per group; within-group eligible-branch weights; expected individual annual maximum kW.",
    "Small group-specific holdouts and one case; the differences are descriptive, not inferential or measured.",
    "Group-conditional weighting improves some other site/endpoint/group comparisons; a universal improvement claim is false in these saved outputs.",
    "Group row mismatch with independently reintegrated hourly peaks or corrected source inputs would require amendment.",
    "Reviewer05: inspect subgroup predictions/weights and peak timestamps; author: place these exceptions next to any all-site aggregate statement.",
    "For Bolzano individual annual maxima, group-conditional weights worsen held-out RMSE for archetypes 2 and 4 despite the lower overall cohort RMSE.",
    "RES-010; LIM-010",
    ["04_analysis/archetype_extension/results_v2/weighted_metrics.csv", "06_claim_drafts/author_archetype_result_check_v2.json"])

padova_rows = [r for r in json.loads((HERE / "author_archetype_result_check_v2.json").read_text(encoding="utf-8"))["sites"]["Padova"]["individual"] if r["endpoint"] == "heat_kWh"]
observed = [r["detailed_model"] for r in padova_rows]
cond_pred = sorted({r["conditional_prediction"] for r in padova_rows})
assert len(observed) == 32 and len(cond_pred) == 4
add(claims, "RES-012", "Within-model detailed-input spread persists",
    f"For Padova-coordinate weather the 32 held-out detailed-input model annual useful-heating outputs span {min(observed):.2f}–{max(observed):.2f} kWh, while the four group-conditional mixture predictions span {min(cond_pred):.2f}–{max(cond_pred):.2f} kWh and the pooled prediction is {heat[0]['pooled_prediction']:.2f} kWh; the four branch mixtures compress substantially more respondent-input variation than they retain.",
    "result", "simulation output", "partially supported", [runs, metrics, author],
    "One assumed dwelling; Padova-coordinate TMY; 32 fixed held-out detailed response profiles versus four group endpoint predictions, kWh.",
    "Range is descriptive for this selected small holdout and model; it is not a survey-population quantile or empirical consumption range.",
    "Other weather/sites or a richer within-branch representation may alter the spreads.",
    "A raw-output re-integration or role-linkage discrepancy would invalidate these ranges.",
    "Reviewer05: verify detailed profile extrema and mixture predictions; author: avoid interpreting the spread as measured household variability.",
    "In the Padova case, the held-out detailed-input model outputs span a much wider annual-heating range than the four conditional mixture predictions.",
    "RES-009; LIM-010",
    ["04_analysis/archetype_extension/results_v2/individual_deviations.csv", "06_claim_drafts/author_archetype_result_check_v2.json"])

add(claims, "INT-003", "Bounded interpretation of conditional mixtures",
    "The v2 conditional branch mixtures have lower eligible-stratum-weighted overall within-model RMSE than pooled branch weights for annual useful heating and expected individual annual maxima at each of four TMY exposures, but Bolzano subgroup peak counterexamples, large detailed-input spread, extra conditional parameters and absent measurements prevent a blanket algorithm-superiority or empirical-accuracy conclusion.",
    "interpretation", "simulation output", "partially supported", [metrics, author, protocol, campaign],
    "32 held-out archived-assigned 2013 responses; one illustrative physical dwelling; four coordinate TMY exposures; useful heat kWh and individual maximum kW.",
    "The conditional mixture differs only in training source-control branch weights, not shared branch components; group membership may proxy controls without causally determining them.",
    "The subgroup exceptions and within-branch information loss are direct counterevidence to uniform improvement; no independent measured comparator exists.",
    "A reproduced metric reversal, source-to-profile failure, or matched measured evaluation showing opposite predictive ranking would narrow or overturn this interpretation.",
    "Reviewer05: assess contrast/inference scope; author: retain counterexamples and model-reference language in every abstract/conclusion statement.",
    "In this bounded model-reference experiment, conditional branch frequencies modestly reduce overall loss, with subgroup exceptions and no measured-accuracy inference.",
    "RES-009; RES-010; RES-011; RES-012; LIM-010; GAP-005",
    ["04_analysis/archetype_extension/results_v2/weighted_metrics.csv", "06_claim_drafts/author_archetype_result_check_v2.json"])

assert len(claims) == 6 and len({x["id"] for x in claims}) == 6
WORK.mkdir(exist_ok=True)
target = WORK / "checked_result_claims_v05.json"
target.write_text(json.dumps({"campaign": "archetype_campaign_v2", "claims": claims}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"claim_ids": [x["id"] for x in claims], "path": str(target)}))
