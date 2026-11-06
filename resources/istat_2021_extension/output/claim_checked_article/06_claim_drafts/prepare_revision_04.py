"""Resolve review-04 provenance/permission metadata without changing central results."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
old = json.loads((HERE / "claim_ledger_v03.json").read_text(encoding="utf-8"))
by = {x["id"]: x for x in old["claims"]}
updates = {}


def u(cid, **fields):
    updates[cid] = fields


u("DAT-002",
  counterevidence_alternative_explanation="The archived v3 weather path consumed missing opaque cloud cover as a physical value; its loads remain quarantined. V5 uses the acquired EPW's valid horizontal IR through the documented local correction. Byte-identical acquisition alone never established model compatibility.",
  dependencies_next_action_owner="Preserve acquisition hashes and current v5 IR/per-surface code paths; do not describe provenance alone as validation of building-energy output.")

u("MET-002",
  locators=["SIM-PROTOCOL | 03_methods/simulation_protocol.md current v5 alternatives and profile invariants", "CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625 and 643–656 source-derived 22/19°C mapping", "RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/summary.json daily_on_profile and primary hourly files", "INDEPENDENT-V5 | 07_reviews/reviewer_04/independent_recheck.json source/profile checks"],
  outputs=["05_simulations/climate_campaign_v5/Padova/summary.json", "07_reviews/reviewer_04/independent_recheck.json"],
  dependencies_next_action_owner="Author/prose worker: use v5 input profiles and label preserved *high-setpoint* duration, not positive heating-operation duration.",
  permitted_manuscript_wording="The v5 placements preserve 4/4/1 encoded 22°C high-setpoint hours across the three bands, while actual positive heating duration may differ.")

u("MET-003",
  locators=["CORRECTION-PROTOCOL-V3 | 03_methods/correction_protocol_v3.md original bounded single-commit method and scope", "CORRECTION-CODE-V3 | 04_analysis/single_commit_1c.py unchanged one-old-state/one-commit module, SHA256 e549b4e5645ad8ae159fddf644d225d28622e1495f6fd64768c5242389b82c16", "RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/single_commit_1c.py and completion_manifest.json retained source hash", "INDEPENDENT-V5 | 07_reviews/reviewer_04/independent_recheck.json actual 20-file state/forcing check"],
  outputs=["05_simulations/climate_campaign_v5/Padova/single_commit_1c.py", "05_simulations/climate_campaign_v5/Padova/completion_manifest.json"],
  assumptions_and_uncertainty="The one-state correction is supported only for this ideal sensible 1C path; v5's separate IR/per-surface correction now handles the previously invalid forcing under declared annual-mean sky and native solar-time conventions. No general EUReCA or empirical validation follows.",
  counterevidence_alternative_explanation="The original v1/v2 recursive dispatcher failed state balance; later v3/v4 input/preprocessing defects were distinct and remain quarantined. V5 independently checks both state and actual forcing.",
  dependencies_next_action_owner="Final closure reviewer: confirm the v5 bounded state path and wording against direct checks; prose worker: keep this an implementation-specific verification claim.",
  permitted_manuscript_wording="The study-local single-commit 1C stepper and v5 per-surface forcing passed bounded independent consistency checks under the stated sky and solar-time approximations.")

u("MET-004",
  locators=["SIM-PROTOCOL | 03_methods/simulation_protocol.md current v5 useful-sensible demand and 1 h maximum definitions", "RUN-CLIMATE-V5 | 04_analysis/paired_schedule_contrasts_v5.csv annual_sensible_heating_kWh and peak_sensible_heating_kW", "AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json direct CSV integration/peak arithmetic", "INDEPENDENT-V5 | 07_reviews/reviewer_04/independent_recheck.json independently recomputed all 20 annual sums/maxima"],
  outputs=["04_analysis/paired_schedule_contrasts_v5.csv", "06_claim_drafts/author_v5_check_for_v03.json", "07_reviews/reviewer_04/independent_recheck.json"],
  assumptions_and_uncertainty="P_t is the modeled ideal-zone sensible load in W; positive heating is integrated at 1 h resolution. The maximum is the predeclared annual hourly endpoint, including any imposed season-start recovery, not a system-capacity estimate.",
  dependencies_next_action_owner="No pending reviewer-02 task. Prose worker: use the declared 1 h positive-load integration and same-site seeded denominators; retain the seasonal-maximum interpretation limit.",
  permitted_manuscript_wording="Annual useful sensible heating sums positive hourly zone load; the reported peak is the largest positive hourly zone load in the modeled year.")

u("MET-005",
  exact_claim="The v5 20-case campaign passed direct transition, source-profile, repeat/warm-up and complete air/surface/mass state checks: all 175,200 exported hours have exact old-state continuity and no residual above 1e-6 W; separate independent summation orders find maxima below 3×10⁻¹⁰ W. These tests verify this bounded numerical implementation, not measured accuracy.",
  locators=by["MET-005"]["locators"] + ["INDEPENDENT-V5 | 07_reviews/reviewer_04/independent_recheck.json independent direct-file node residuals below 3e-10 W"],
  assumptions_and_uncertainty="Machine-level maximum residual differs slightly with algebraic summation order; both independent checks are far below the declared 1e-6 W tolerance. Numerical repeatability and state closure do not validate observed heating or native solar-time physics.",
  permitted_manuscript_wording="All v5 hourly node balances closed within the declared 1e-6 W tolerance; this is numerical verification, not measured heating accuracy.")

u("MET-009",
  locators=["EPW-SPEC | https://bigladdersoftware.com/epx/docs/24-2/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html opaque sky cover 99 and horizontal IR 9999 missing codes; inspected EnergyPlus 24.2 field dictionary mirror", *by["MET-009"]["locators"][1:]],
  dependencies_next_action_owner="Final closure reviewer: confirm v5 valid-IR/per-surface actual-forcing evidence and retained annual-mean/time limits. The old EnergyPlus 24.1 PDF locator returned 404 and is replaced by the inspected 24.2 dictionary mirror.")

u("MET-010",
  population_geography_period_comparator_units="Historical v3 command omission plus current v5 Padova and four-site run/replay provenance; executable/cwd/argv/source hashes, no physical units or added scientific sample.")

u("RES-004",
  dependencies_next_action_owner="Final independent closure: confirm review04's numerical and scope findings; prose worker: report only conditional v5 fixed-case model outputs with no observed-accuracy claim.",
  permitted_manuscript_wording="Conditional v5 result: in this fixed Padova-coordinate model case, early/late placement changed annual useful heating by +0.894/+0.337% and maximum hourly load by +2.029/−2.103% relative to seeded.")

u("RES-005",
  dependencies_next_action_owner="Final independent closure: confirm review04's all-site numerical/scope findings; prose worker: retain coordinate-weather and no-causal-variable limits.",
  permitted_manuscript_wording="Conditional v5 result: same-case annual placement effects differ across four coordinate TMY exposures; they are not city housing-stock estimates.")

u("RES-006",
  dependencies_next_action_owner="Final independent closure: confirm the raw v5 partition and post-hoc label; prose worker: keep modeled setpoint duration separate from observed operation.",
  permitted_manuscript_wording="Post-hoc v5 model diagnostic: positive useful heating also occurs at the 19°C setback, so encoded high-setpoint hours do not equal heater runtime.")

u("RES-007",
  dependencies_next_action_owner="Final independent closure: confirm all-site maxima and Bolzano season-onset context; prose worker: retain the annual maximum endpoint and avoid equipment-sizing or robust-ranking claims.",
  permitted_manuscript_wording="Conditional v5 result: maximum hourly loads vary with placement in three exposures; Bolzano's equal maxima occur at the common imposed heating-season restart.")

u("INT-001",
  population_geography_period_comparator_units="One assumed dwelling and selected 2013 Veneto response under Padova, Bolzano, Roma and Palermo coordinate TMYs; within-site seeded/early/late useful heating kWh and annual maximum hourly load kW, not local populations.",
  falsification_or_weakening="A v5 source/profile, IR/per-surface forcing, state, integration or matched-input failure would withdraw the conditional interpretation; an alternative declared sky/time or physical assumption could narrow its transfer or ranking.",
  dependencies_next_action_owner="Final independent closure and prose worker: preserve one-case/four-weather conditional scope, native sky/time limits and no measured/causal inference.",
  permitted_manuscript_wording="Under the stated single-dwelling, respondent, seasonal and weather-time assumptions, v5 paired outputs show conditional schedule-placement sensitivity across four coordinate TMYs; no population or accuracy conclusion follows.")

u("RES-008",
  dependencies_next_action_owner="Final independent closure: confirm hour 6912 and adjacent setpoints; prose worker: label this a post-hoc context for the unchanged annual-maximum endpoint.",
  permitted_manuscript_wording="Post-hoc v5 peak context: Bolzano's equal annual maxima arise at the shared imposed heating-season restart, without implying equal intraday peaks or equipment-sizing needs.")

patch = {
  "prior_draft_sha256": hashlib.sha256((HERE / "article_claims_v03.md").read_bytes()).hexdigest(),
  "review_sha256": hashlib.sha256((ROOT / "07_reviews/review_04.md").read_bytes()).hexdigest(),
  "status_note": "Status: review-04 author response, 20 October 2026. The fresh independent reviewer reproduced the bounded v5 conditional model contrasts and found no new critical or major numerical issue; this v04 corrects current-version claim wording, metadata and EPW locator. Final independent issue-closure and acceptance gate remain pending. All 66 pre-v5 annual executions stay quarantined; no empirical validation, population inference, system sizing or broad novelty is claimed. Current primary-source aliases and hashes are tracked in 02_evidence/local_source_ledger.md and local_sources.json (v04 alias reconciliation pending).",
  "updates": updates,
  "new_claims": [],
}
target = HERE / "revision_04.json"
if target.exists():
    raise SystemExit(f"Refusing overwrite: {target}")
target.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"updates": len(updates), "patch": str(target)}))
