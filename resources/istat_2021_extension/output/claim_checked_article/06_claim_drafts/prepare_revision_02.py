"""Prepare the review-02 author patch; v01 and earlier snapshots stay immutable."""
import copy
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
old = json.loads((here / "claim_ledger_v01.json").read_text(encoding="utf-8"))
by_id = {c["id"]: c for c in old["claims"]}
updates = {}


def update(cid, **fields):
    updates[cid] = fields


weather_locators = [
    "EPW-SPEC | https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v24.1.0/AuxiliaryPrograms.pdf EPW opaque sky cover: 0–10 tenths; 99 missing; horizontal IR W/m² field",
    "CODE-EURECA-WEATHER | C:/Works/EUReCA/EUReCA/eureca_building/weather.py lines 120–127, 349–404 raw opaque-cover dispatch and sky calculation",
    "WEATHER-DIAGNOSTIC-R2 | 07_reviews/reviewer_02/independent_audit.json four raw EPWs, sentinel counts, effective-sky IR diagnostic",
    "WEATHER-RUNTIME-R2 | 07_reviews/reviewer_02/runtime_checks.json actual WeatherFile air-minus-sky outputs",
]

update("RQ-001",
       status="partially supported",
       exact_claim="For one illustrative dwelling and selected Veneto survey response, the predeclared Padova-coordinate TMY comparison concerns annual useful sensible heating and maximum hourly load under three placements of the same 4/4/1 encoded 22°C high-setpoint hours. A 19°C setback permits heating outside those periods; valid physical effects await weather-forcing correction and new annual runs.",
       assumptions_and_uncertainty="The fixed-case schedule question and profile invariants survive the weather defect; v1/v2 state-defective and v3 forcing-defective loads do not answer its physical endpoints.",
       dependencies_next_action_owner="Methods owner: document, guard and independently test longwave input; run new Padova gate and matched alternatives. Author: admit effects only after new hourly checks and fresh independent review.",
       permitted_manuscript_wording="We test three placements of the same encoded high-setpoint hours in a fixed illustrative case; physical effect estimates are withheld pending corrected weather forcing.")

update("RQ-002",
       status="partially supported",
       exact_claim="The predeclared extension compares matched schedule placements under consistently retrieved Padova, Bolzano, Roma and Palermo coordinate TMYs, holding dwelling, response, setpoints and climate-E operating season fixed; all four-site physical load effects await valid longwave preprocessing and new annual runs.",
       dependencies_next_action_owner="Methods owner: complete weather QC and corrected reruns for all four sites after Padova gate; author/reviewer: verify paired inputs, hourly arithmetic and physical forcing before any comparison.",
       permitted_manuscript_wording="Four coordinate TMYs define controlled weather exposures for a pending matched comparison.")

update("DAT-002",
       assumptions_and_uncertainty="Retrieval metadata and byte identity remain supported, but all 8,760 opaque-cover values are missing sentinel 99 in each EPW; provenance does not establish compatibility with the unguarded model sky routine. TMY is not a historical meter-matched year.",
       counterevidence_alternative_explanation="The weather reader consumes missing cloud cover as a physical value; the resulting v3 load simulations are not scientifically admissible. The acquired EPW's horizontal IR field offers a possible documented alternative after checks.",
       dependencies_next_action_owner="Methods owner: retain acquired hashes; document an IR-based or otherwise justified longwave treatment with consumed-field QC and an independent forcing audit.",
       permitted_manuscript_wording="Padova's TMY matches the documented PVGIS acquisition; its opaque sky-cover field is missing and requires explicit handling before load simulation.")

update("MET-003",
       status="supported",
       evidence_class="derived estimate",
       locators=by_id["MET-003"]["locators"] + [
           "STATE-AUDIT-R2 | 07_reviews/reviewer_02/independent_audit.json 175,200 air/surface/mass residual and continuity checks",
           "STATE-RUNTIME-R2 | 07_reviews/reviewer_02/runtime_checks.json direct actual stepper transition checks",
       ],
       exact_claim="For the bounded ideal sensible 1C case, the local single-commit adapter evaluates candidates from one previous state and commits one accepted hourly state. Independent direct-transition and 175,200-hour equation checks support this narrow state correction; they do not validate the weather forcing or physical annual effects.",
       assumptions_and_uncertainty="Only the declared ideal sensible 1C network was tested; no general EUReCA implementation, conference replication, measured accuracy or weather-input correctness follows.",
       dependencies_next_action_owner="Methods owner: reuse the narrow adapter with corrected weather forcing and a complete new gate. Preserve historical command/source hashes and do not edit external checkout.",
       permitted_manuscript_wording="The study-local single-commit 1C stepper passed independent state-equation checks; weather forcing still requires correction.")

update("MET-005",
       status="supported",
       exact_claim="The v3 single-commit outputs passed transition tests, all 175,200 exported hourly air/surface/mass balance checks (maximum absolute residual 2.024×10⁻¹⁰ W), continuity, repeat and warm-up arithmetic. These are narrow numerical consistency findings for a simulation with invalid opaque-cloud forcing, not a physical load-admission gate.",
       locators=by_id["MET-005"]["locators"] + [
           "STATE-AUDIT-R2 | 07_reviews/reviewer_02/independent_audit.json all-site independent residual/repeat/warm-up checks",
           "WEATHER-DIAGNOSTIC-R2 | 07_reviews/reviewer_02/independent_audit.json forcing defect",
       ],
       assumptions_and_uncertainty="Equation residuals close because the incorrect weather-derived forcing is an input to the checked network; a passing residual cannot validate its physical meaning.",
       counterevidence_alternative_explanation="v1/v2 failed transition balance; v3 fixes that defect but still treats cloud sentinel 99 as physical cover, producing nonphysical longwave gain.",
       falsification_or_weakening="A new state discrepancy would narrow the solver finding; no physical effect may be admitted without a separate weather-forcing gate and fresh annual runs.",
       dependencies_next_action_owner="Methods owner: add consumed-field QC, night/zero-solar longwave checks, and rerun all cases with exported intermediate forcing; reviewer: independently audit new gate.",
       permitted_manuscript_wording="The local solver closed its declared hourly equations; the resulting annual loads are withheld because weather preprocessing was invalid.")

update("MET-006",
       status="partially supported",
       exact_claim="The four coordinate TMY files share documented PVGIS source metadata and give contrasting weather descriptors, but their opaque cloud cover is missing for every hour. Their unmodified use in the current EUReCA sky routine fails the weather-admission gate; no existing v3 physical climate effects can be used.",
       locators=by_id["MET-006"]["locators"] + weather_locators,
       assumptions_and_uncertainty="The acquired TMY meteorology and descriptive contrast remain usable as inputs after explicit missing-value handling. The one Palermo negative wind-speed record identified during provisional v4 QC also requires a documented consumed-field disposition before reruns; no v4 audit is claimed passed.",
       counterevidence_alternative_explanation="Homogeneous provider metadata and matched within-site inputs do not cancel a nonlinear forcing error in a controlled heating model.",
       dependencies_next_action_owner="Methods owner: pass a documented pre-run longwave and all-consumed-field audit, then new Padova and four-site runs. Record site-specific executable/arguments and isolated replay destination.",
       permitted_manuscript_wording="The four TMY acquisitions are comparable as source data; their current load simulations are withheld until valid sky forcing is implemented.")

update("MET-008",
       status="supported",
       exact_claim="The narrow single-commit 1C state gate independently passes direct mode-transition and all-hour air/surface/mass balance tests; annual load admission additionally requires a distinct checked weather-forcing gate because correct state equations can propagate incorrect environmental forcing.",
       locators=by_id["MET-008"]["locators"] + [
           "STATE-AUDIT-R2 | 07_reviews/reviewer_02/independent_audit.json independent network checks",
           "STATE-RUNTIME-R2 | 07_reviews/reviewer_02/runtime_checks.json direct transition cases",
           "WEATHER-DIAGNOSTIC-R2 | 07_reviews/reviewer_02/independent_audit.json EPW sentinel/IR diagnostics",
       ],
       assumptions_and_uncertainty="The verified network is ideal sensible-only. State closure and weather admissibility are separate necessary tests; neither alone proves empirical accuracy.",
       dependencies_next_action_owner="Methods owner: retain state gate and add independent weather-forcing gate before new annual outputs are considered.",
       permitted_manuscript_wording="The state solver passed a narrow independent consistency audit, while the weather-forcing gate remains open.")

update("RES-004",
       label="Padova effects withheld pending forcing correction",
       status="withheld",
       exact_claim="No Padova annual-heating or maximum-hourly-load effect estimate is currently admissible for the specified schedule-placement comparison: v1/v2 have defective state transitions, and v3 consumes missing cloud cover as a physical sky input. The predeclared kWh/kW endpoints require new weather-corrected runs and independent audit.",
       locators=by_id["RES-004"]["locators"] + weather_locators,
       assumptions_and_uncertainty="The v3 values remain arithmetically reproducible archival run properties; their effect signs and magnitudes must not be presented as physical findings or inferred to survive correction.",
       counterevidence_alternative_explanation="The valid horizontal IR diagnostic indicates a much cooler effective sky than the engine's erroneous sky calculation; nonlinear load clipping and controls preclude assuming paired cancellation.",
       falsification_or_weakening="Only a documented longwave correction, new paired annual outputs, checks and independent review can replace this hold.",
       dependencies_next_action_owner="Methods owner: pass weather QC and new Padova gate; author: independently integrate new CSVs; next independent reviewer: assess forcing and physical interpretation.",
       permitted_manuscript_wording="[RESULT WITHHELD: corrected-forcing Padova annual and peak contrasts are pending.]")

update("RES-005",
       label="Four-site annual effects withheld",
       status="withheld",
       exact_claim="No four-site annual useful-heating placement effect is presently admissible. The v3 matched table is an archived arithmetic product of annual runs affected at every site by missing-cloud sentinel use; a corrected longwave input and full new matched campaign are required.",
       locators=by_id["RES-005"]["locators"] + weather_locators,
       assumptions_and_uncertainty="All four v3 site contrasts remain file-reproducible but cannot establish even direction of the physical schedule effect. One assumed dwelling and response preclude population inference regardless of correction.",
       counterevidence_alternative_explanation="Erroneous site-specific sky gains can alter heating/nonheating state and paired differences; equal erroneous forcing within site does not prove cancellation.",
       falsification_or_weakening="Weather-corrected annual outputs and independent source-to-forcing/load checks are required to replace the hold.",
       dependencies_next_action_owner="Methods owner: rerun all matched coordinate TMY cases after complete pre-run QC; author/reviewer: recalculate paired table from new hourly files.",
       permitted_manuscript_wording="[RESULT WITHHELD: four-site annual placement effects await valid forcing and new runs.]")

update("RES-006",
       label="Setback diagnostic withheld as quantity",
       status="withheld",
       exact_claim="The fact that the two-level thermostat permits positive heating at 19°C setback is supported by the source mapping and solver control logic. The v3 Padova post-hoc counts and kWh partition, however, depend on invalid weather forcing and cannot be reported as physical results; they must be recomputed from new hourly output.",
       locators=by_id["RES-006"]["locators"] + weather_locators,
       population_geography_period_comparator_units="One assumed Padova TMY case; post-hoc 19°C versus 22°C partition; counts and kWh withheld pending corrected forcing.",
       assumptions_and_uncertainty="The conceptual high-setpoint-versus-runtime distinction is source-supported and remains. The old quantitative partition is an archival numerical diagnostic only, not a predeclared endpoint or observation.",
       dependencies_next_action_owner="Methods owner: regenerate hourly file; author: recompute descriptive setpoint partition separately from primary endpoints; reviewer: check source mapping and arithmetic.",
       permitted_manuscript_wording="The model allows setback heating, so encoded high-setpoint duration does not equal heater runtime; corrected-run counts are pending.")

update("RES-007",
       label="Four-site peak effects withheld",
       status="withheld",
       exact_claim="No four-site maximum-hourly-heating-load placement effect is presently admissible. The v3 peak table is an archived arithmetic product of runs affected by invalid longwave forcing; new hourly runs and a fresh peak extraction are required.",
       locators=by_id["RES-007"]["locators"] + weather_locators,
       assumptions_and_uncertainty="Hourly peaks are especially sensitive to forcing and the selected hour; old signs/magnitudes cannot be carried forward. No measured peak or probabilistic interval exists.",
       dependencies_next_action_owner="Methods owner: rerun all four matched sites after longwave guard passes; author/reviewer: independently extract and compare new hourly maxima.",
       permitted_manuscript_wording="[RESULT WITHHELD: four-site hourly peak placement effects await valid forcing and new runs.]")

update("INT-001",
       label="Interpretation on scientific hold",
       status="withheld",
       exact_claim="The predeclared schedule-placement sensitivity question remains interpretable as a deterministic fixed-case comparison, but no physical direction or magnitude of annual or peak response can be inferred from v1/v2 or v3 annual outputs. New weather-corrected, state-checked runs are required before mechanism interpretation.",
       locators=by_id["INT-001"]["locators"] + weather_locators,
       assumptions_and_uncertainty="Source schedule mapping and controlled design survive; numerical effects and climate interactions do not. Even accepted future results would be conditional on this one assumed case.",
       counterevidence_alternative_explanation="Erroneous sky forcing changes internal and envelope heat balance; matched errors can interact nonlinearly with thermostat clipping.",
       dependencies_next_action_owner="Author: withhold effect interpretation until new validated-forcing outputs and fresh independent scientific review.",
       permitted_manuscript_wording="[PHYSICAL SENSITIVITY INTERPRETATION WITHHELD.] This remains a predeclared fixed-case comparison.")

update("LIM-005",
       status="supported",
       exact_claim="The assumed case has active cooling disabled and therefore cannot support comfort or whole-year conditioned-building conclusions. The numerical v3 summer maximum air temperature is affected by invalid sky forcing and is withheld pending a new run.",
       locators=by_id["LIM-005"]["locators"] + weather_locators,
       population_geography_period_comparator_units="One assumed zone under coordinate TMY; cooling disabled; quantitative air-temperature maximum withheld.",
       assumptions_and_uncertainty="The no-cooling design limit is independent of the weather error; no numerical summer temperature or comfort adequacy is currently admissible.",
       dependencies_next_action_owner="Methods owner: recompute maximum zone air temperature after corrected forcing; author: restrict any future statement to a model state, not observed comfort.",
       permitted_manuscript_wording="Cooling is disabled; the case does not evaluate summer comfort.")

new = []
def add(base_id, cid, **fields):
    c = copy.deepcopy(by_id[base_id])
    c.update(fields)
    c["id"] = cid
    c["source_ids"] = [s.split(" | ")[0] for s in c["locators"]]
    new.append(c)

add("MET-008", "MET-009",
    label="Weather-forcing admission gate",
    exact_claim="Before new annual effects are admitted, a study-local longwave method must explicitly handle EPW opaque-cover sentinel 99, document the selected valid horizontal-IR or justified alternative pathway, units and timestamps, guard all consumed weather fields, and pass independent night/zero-solar and envelope-forcing checks.",
    kind="method", evidence_class="planned test", status="planned",
    locators=weather_locators + ["CODE-EURECA-ZONE | C:/Works/EUReCA/EUReCA/eureca_building/thermal_zone.py lines 837–842 sky-temperature contribution to zone forcing"],
    outputs=["07_reviews/reviewer_02/independent_audit.json", "07_reviews/reviewer_02/runtime_checks.json"],
    population_geography_period_comparator_units="Four 8760-hour PVGIS EPWs; opaque cover in tenths, horizontal IR W/m², sky/air temperatures °C or K, zone forcing W.",
    assumptions_and_uncertainty="The horizontal IR calculation in the independent audit is a diagnostic, not yet an approved method or completed replacement campaign. Provisional v4 pre-run QC has not passed.",
    counterevidence_alternative_explanation="All four v3 EPWs encode missing opaque sky cover as 99; the current engine treats 99 as physical cover and produces mean air-minus-sky near −80 to −96 K.",
    falsification_or_weakening="A missing/invalid consumed field, timestamp mismatch, implausible night forcing, non-reproducing independent sky calculation or failed annual gate prevents admission.",
    dependencies_next_action_owner="Methods owner: specify and implement local forcing method and complete pre-run audit before new Padova/all-site runs; independent reviewer: inspect code, forcing arrays and outputs.",
    permitted_manuscript_wording="A separate weather-forcing gate is required before the predeclared heating contrasts can be estimated.",
    related_claims_figures="MET-003; MET-005; RES-004; LIM-007")

add("MET-006", "MET-010",
    label="Executable command provenance",
    exact_claim="The archived v3 manifests omit the required SITE argument in their recorded run command. The corrected historical invocation is the recorded Python executable, `-B`, the frozen v3 script path, and one explicit site argument; this is a provenance correction, not a claim that the historical manifest was executable as written. Future campaigns require a tested isolated replay entry point and exact executable, working directory and arguments.",
    kind="method", evidence_class="derived estimate", status="partially supported",
    locators=[
        "RUN-CLIMATE-V3 | 05_simulations/climate_campaign_v3/Padova/completion_manifest.json recorded command without site",
        "CODE-RUN-V3 | 04_analysis/run_climate_campaign_v3.py line 8 SITE=sys.argv[1] and fixed output directory",
        "COMMAND-REPLAY-R2 | 07_reviews/reviewer_02/recorded_command_replay.txt exact recorded invocation and IndexError",
    ],
    outputs=["05_simulations/climate_campaign_v3/Padova/completion_manifest.json", "07_reviews/reviewer_02/recorded_command_replay.txt"],
    population_geography_period_comparator_units="Archived v3 run manifests for Padova, Bolzano, Roma and Palermo; command-line provenance, no physical units.",
    assumptions_and_uncertainty="Adding SITE describes the necessary command syntax; it does not retroactively prove the historical command string or authorize overwriting frozen outputs. A safe separate-destination replay has not yet been shown.",
    counterevidence_alternative_explanation="Recorded source and hourly output hashes match, so this is a replay-documentation defect rather than evidence that v3 files were not produced.",
    falsification_or_weakening="A new manifest lacking its exact command/working directory or a replay that overwrites archived outputs leaves provenance incomplete.",
    dependencies_next_action_owner="Methods owner: create and test separate-output replay for new campaign, record executable/cwd/complete argv per site. Author: retain old manifests unchanged with labelled correction.",
    permitted_manuscript_wording="Historical v3 command records omit a required site argument; subsequent runs must include complete, safely replayable provenance.",
    related_claims_figures="MET-006; LIM-007")

add("LIM-006", "LIM-007",
    label="V3 weather-forcing outputs quarantined",
    exact_claim="The 20 v3 annual runs pass narrow state and arithmetic checks but all use EPWs whose opaque cloud-cover field is missing sentinel 99 and unfiltered in the model sky calculation; their annual loads, peaks, setback-energy partition and summer maximum are archived as defective-forcing diagnostics, not physical sensitivity evidence.",
    kind="limitation", evidence_class="derived estimate", status="supported",
    locators=weather_locators + [
        "RUN-CLIMATE-V3 | 05_simulations/climate_campaign_v3/{Padova,Bolzano,Roma,Palermo}/completion_manifest.json input hashes and run snapshots",
        "STATE-AUDIT-R2 | 07_reviews/reviewer_02/independent_audit.json narrow state passes despite invalid forcing",
    ],
    outputs=["07_reviews/reviewer_02/independent_audit.json", "07_reviews/reviewer_02/runtime_checks.json"],
    population_geography_period_comparator_units="20 archived v3 annual executions across four coordinate TMYs; 175,200 exported hours; numerical result magnitudes withheld.",
    assumptions_and_uncertainty="The observed defect lies in current weather preprocessing. It is not attributed to the published conference executable; valid acquisition provenance and network closure remain separate findings.",
    counterevidence_alternative_explanation="Fixed wrong forcing within pairs cannot be assumed to cancel under nonlinear thermostat switching and positive-load clipping.",
    falsification_or_weakening="A demonstrated valid alternative historical forcing path could narrow the quarantine; current frozen code and runtime weather checks show none.",
    dependencies_next_action_owner="Methods owner: preserve all v3 snapshots and create new versioned corrected-forcing runs; author: exclude v3 physical effect magnitudes from manuscript.",
    permitted_manuscript_wording="V3 physical load estimates are excluded because their sky forcing consumed a missing-data sentinel.",
    related_claims_figures="MET-009; RES-004; RES-005; RES-006; RES-007; INT-001")

patch = {
    "prior_draft_sha256": hashlib.sha256((here / "article_claims_v01.md").read_bytes()).hexdigest(),
    "review_sha256": hashlib.sha256((root / "07_reviews/review_02.md").read_bytes()).hexdigest(),
    "status_note": "Status: review-02 response, 20 October 2026. The source-to-schedule mapping and bounded single-commit state correction pass independent checks. Physical annual, peak, setback-partition and summer-temperature values from v1/v2 (state defect) and v3 (missing-cloud sky-forcing defect) are withheld. A documented longwave method, consumed-field QC, new versioned annual runs and independent review are required. No empirical validation or population inference is claimed. Source aliases are resolved in 02_evidence/source_ledger.md and local_source_ledger.md; newly introduced review-02 diagnostic/code aliases require registry extension.",
    "updates": updates,
    "new_claims": new,
}
target = here / "revision_02.json"
if target.exists():
    raise SystemExit(f"Refusing to overwrite {target}")
target.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"updates": len(updates), "new_claims": [c["id"] for c in new], "patch": str(target)}))
