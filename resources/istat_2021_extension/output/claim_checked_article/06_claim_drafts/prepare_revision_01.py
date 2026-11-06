"""Stage substantive response to independent review 01 as claim-field updates."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
prior = HERE / "article_claims_v00.md"
review = ROOT / "07_reviews/review_01.md"
out = HERE / "revision_01.json"
if out.exists():
    raise SystemExit(f"Refusing overwrite: {out}")
updates = {}

def u(cid, **kwargs):
    updates.setdefault(cid, {}).update(kwargs)

u("GAP-001",
  exact_claim="The selected respondent's reported mean winter-day system operating hours in three eight-hour bands do not identify unique within-band times. In this model those durations are encoded as hours at a 22°C high setpoint, not as a constraint on actual positive heating-load hours.",
  assumptions_and_uncertainty="Integer/coarsened reported means are mapped to a repeated two-level 22/19°C thermostat schedule; both contiguity and daily repetition are assumptions. Heating may be delivered at 19°C setback.",
  counterevidence_alternative_explanation="The survey asks mean system on-hours rather than thermostat levels; the simulation's temperature mapping is a separate assumption and does not reproduce observed burner duty.",
  permitted_manuscript_wording="Reported mean operating hours leave timing unidentified; our model encodes them as high-setpoint periods while allowing setback heating.")
u("RQ-001",
  exact_claim="For one illustrative dwelling, one selected Veneto survey response and Padova-coordinate TMY, compare annual useful sensible heating and maximum hourly load under three placements of the same 4/4/1 encoded 22°C high-setpoint hours; a 19°C setback permits heating in other hours.",
  status="partially supported",
  locators=["RQ-FREEZE | 01_plan/research_questions.md primary RQ and fixed-case boundary", "ISTAT-Q2013 | istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf pp41–42 reported operating-hour bands", "CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625, 643–656 high/setback mapping", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json candidate corrected outputs"],
  outputs=["05_simulations/climate_campaign_v3/Padova/summary.json"],
  assumptions_and_uncertainty="The 4/4/1 encoded high-setpoint hours are fixed; actual positive-load hours and duration are outputs and need not equal 9 h/day. The physical case remains illustrative.",
  dependencies_next_action_owner="Simulation owner completed corrected Padova run and full-step checks; reviewer 02 must independently inspect local correction and outputs.",
  permitted_manuscript_wording="We compare three placements of equal encoded high-setpoint periods under one assumed case.")
u("RQ-002",
  exact_claim="Compare the same encoded high-setpoint placement contrasts under homogeneous TMY weather for Padova, Bolzano, Roma and Palermo coordinates, holding building, respondent, thermostat levels and climate-E season fixed; corrected annual climate outputs require review.",
  locators=["RQ-FREEZE | 01_plan/research_questions.md Secondary RQ1", "CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json four PVGIS API requests", "CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv HDD18/CDD26/GHI/RH"],
  outputs=["04_analysis/climate_descriptors_v2.csv"],
  assumptions_and_uncertainty="Weather selection is based on computed descriptors, not climate labels; all other inputs are fixed. The old v2 climate loads are quarantined because they used the flawed dispatcher.",
  dependencies_next_action_owner="Simulation owner to complete corrected v3 Bolzano/Roma/Palermo runs and full-step checks; author/reviewer to inspect before result admission.",
  permitted_manuscript_wording="[CLIMATE COMPARISON PENDING: corrected matched runs and independent review.]" )
u("DAT-002",
  exact_claim="The Padova-coordinate EPW used in the initial and corrected pilot is byte-identical (SHA256 390ead08…) to a newly retrieved PVGIS TMY EPW for 45.408°N, 11.873°E; provider metadata specify ERA5 meteorology, SARAH3 radiation, 2005–2023 selection window and horizon inclusion.",
  locators=["CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json Padova JSON/EPW API URL, retrieval timestamp and hashes", "CLIMATE-METADATA-V2 | 05_simulations/climate_inputs_v2/audited_metadata.json Padova inputs.meteo_data and tmy_months_selected", "DATA-INPUT-AUDIT | 04_analysis/input_audit.json original local Padova EPW hash"],
  outputs=["05_simulations/climate_inputs_v2/Padova.epw", "05_simulations/climate_inputs_v2/Padova.json"],
  assumptions_and_uncertainty="The newly retrieved file is byte-identical to the earlier local input; this supports weather content/provenance now, not the original file's acquisition history. TMY selected months are not a historical measured year.",
  dependencies_next_action_owner="Reviewer to verify manifest/API metadata; retain grid/horizon and TMY limits.",
  permitted_manuscript_wording="The Padova TMY input matches a documented PVGIS retrieval with ERA5 meteorology and SARAH3 radiation for a 2005–2023 selection window.")
u("DAT-004",
  evidence_class="derived estimate",
  assumptions_and_uncertainty="This is a bounded inventory audit of inspected local GIS/directories, not an observed-measurement dataset and not proof that no compatible meter series exists elsewhere.",
  permitted_manuscript_wording="The inspected local files did not identify an independently documented meter series compatible with this pilot.")
u("MET-001",
  exact_claim="Current GRINS EndUse.py converts each reported band duration into one seeded contiguous binary profile segment, then maps binary 1/0 to assumed 22/19°C high/setback setpoints for this timer-selected respondent; positive heating is permitted at the setback setpoint.",
  locators=["CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625 and 643–656, SHA256 cf7442f0b2ddb39b338c2047a1b95016d3a4d246604da15cd49c9a823cc436d5", "ISTAT-Q2013 | questionnaire p43, question 3.13 option _2 timer", "REVIEW-01 | 07_reviews/review_01.md R1-02 and independent_checks.json positive heat at 19°C in archived outputs"],
  assumptions_and_uncertainty="The source 22/19°C values are not measured respondent setpoints. The binary profile represents the modeled high-setpoint choice, not equipment availability or actual burner runtime.",
  permitted_manuscript_wording="The inspected source places a contiguous high-setpoint segment in each reported time band and allows heating at the lower 19°C setting outside it.")
u("MET-002",
  exact_claim="The three modeled alternatives preserve the same 4/4/1 encoded daily high-setpoint hours in their respective bands, varying only contiguous placement; they do not preserve the duration of actual heating delivery at either setpoint.",
  locators=["SIM-PROTOCOL | 03_methods/simulation_protocol.md baseline and alternatives", "CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625 and 643–656", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json daily_on_profile", "REVIEW-01 | 07_reviews/review_01.md R1-02 distinction between high-setpoint profile and actual positive load"],
  outputs=["05_simulations/climate_campaign_v3/Padova/summary.json"],
  assumptions_and_uncertainty="22°C applies during encoded high periods and 19°C outside within modeled season; ideal heating can run under either setting. Earliest/latest are alternatives, not extrema.",
  permitted_manuscript_wording="The comparison fixes high-setpoint duration within each reported band while changing its placement; actual heating runtime remains free to vary.")
u("MET-003",
  exact_claim="A bounded local single-commit 1C adapter now uses unchanged EUReCA sensible_balance_1C algebra but evaluates free and constrained candidates from one prior state and commits only one accepted hourly state; this corrects the observed transition defect in the archived adapter without changing external checkouts.",
  status="partially supported",
  locators=["REVIEW-01 | 07_reviews/review_01.md R1-01 dispatcher diagnostic and source lines", "CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md scope and gates", "CORRECTION-CODE-V3 | 05_simulations/climate_campaign_v3/Padova/single_commit_1c.py select_solution/solve_1c_single_commit, SHA256 e549b4e5645ad8ae159fddf644d225d28622e1495f6fd64768c5242389b82c16", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/completion_manifest.json correction/source hashes"],
  outputs=["05_simulations/climate_campaign_v3/Padova/single_commit_1c.py", "05_simulations/climate_campaign_v3/Padova/completion_manifest.json"],
  assumptions_and_uncertainty="The correction supports only this ideal sensible 1C case with no mechanical/natural ventilation or latent gains; it is not a general EUReCA systems replacement, conference reproduction or empirical calibration.",
  counterevidence_alternative_explanation="Archived v1/v2 recursive dispatcher violated state balance at mode changes; direct pure-network algebra checks did not catch it.",
  dependencies_next_action_owner="Reviewer 02 to independently inspect local correction, transition tests and annual residual checker; external benchmark and measured accuracy remain absent.",
  permitted_manuscript_wording="A local study-specific adapter selects one hourly 1C solution from the same prior state and commits it once; its validity remains under independent review.")
u("MET-004",
  locators=["SIM-PROTOCOL | 03_methods/simulation_protocol.md Accounting and uncertainty", "CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md unchanged primary outputs", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json independently integrated corrected CSVs"],
  outputs=["06_claim_drafts/author_corrected_check_v01.json"],
  dependencies_next_action_owner="Author independently reintegrated corrected Padova CSVs; reviewer 02 to inspect meaning of sensible_zone_load and state residuals.")
u("MET-005",
  exact_claim="The archived v1/v2 technical screen passed but failed a subsequently applied full-step state-balance check; a new v3 Padova gate reports zero residual failures above 1e-6 W, one-state continuity, exact repeat, warm-up stability and 8760 finite rows, pending independent review of the correction/checker.",
  status="partially supported",
  locators=["REVIEW-01 | 07_reviews/review_01.md R1-01 and reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/transition_checks_pre_run.json five mode tests", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/full_step_balance_checks.json all five cases and maximum residuals", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/checks.json repeat/warm-up/finite screens", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json independent CSV arithmetic"],
  outputs=["05_simulations/climate_campaign_v3/Padova/transition_checks_pre_run.json", "05_simulations/climate_campaign_v3/Padova/full_step_balance_checks.json", "05_simulations/climate_campaign_v3/Padova/checks.json", "06_claim_drafts/author_corrected_check_v01.json"],
  assumptions_and_uncertainty="The declared 1e-6 W full-step screen checks numerical network consistency for this one ideal sensible adapter. The old gate omitted transition balance and is superseded; no observed accuracy follows.",
  counterevidence_alternative_explanation="Reviewer 01 found 898/846/766 failing seeded/early/late v1 hours and residuals up to 9.65 kW; old repeat/integration passes were inadequate.",
  falsification_or_weakening="Independent reviewer detection of state mismatch, incorrect forcing/state export, or non-matched run setup would withdraw the candidate gate.",
  dependencies_next_action_owner="Reviewer 02 must inspect v3 code, transition trials and full-step residual arithmetic; do not present v1/v2 effects as evidence.",
  permitted_manuscript_wording="The corrected Padova adapter passed its expanded internal numerical gates, subject to independent review; these checks do not establish measured accuracy.")
u("MET-006",
  exact_claim="Homogeneous PVGIS TMY inputs for Padova, Bolzano, Roma and Palermo coordinates have been retrieved and characterized consistently; only corrected v3 annual runs with full-step checks can answer the weather-interaction question.",
  status="partially supported",
  locators=["CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json four retrievals, URLs/hashes", "CLIMATE-METADATA-V2 | 05_simulations/climate_inputs_v2/audited_metadata.json same ERA5/SARAH3/window/horizon setup", "CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv HDD18/CDD26/GHI/RH", "REVIEW-01 | 07_reviews/review_01.md R1-01 invalidates original dispatcher-based climate outcomes"],
  outputs=["04_analysis/climate_descriptors_v2.csv", "05_simulations/climate_inputs_v2/acquisition_manifest.json"],
  assumptions_and_uncertainty="These are grid-coordinate TMYs, not historical measured weather; Bolzano/Roma/Palermo are controlled weather contrasts, not representative local household stocks. Fixed climate-E operating season deliberately holds a nonlocal assumption constant.",
  dependencies_next_action_owner="Simulation owner: finish corrected v3 climate runs and full-step checks; author/reviewer: inspect before scientific use.",
  permitted_manuscript_wording="The four coordinate TMY inputs are consistently sourced; corrected climate-response results remain under verification.")
u("RES-003",
  exact_claim="The corrected Padova v3 seeded, early and late binary profiles each encode 4/4/1 high-setpoint hours in the 05–13, 13–21 and 21–05 bands; this does not constrain positive heating-load hours at the 19°C setback.",
  locators=["RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json daily_on_profile", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json profile-count and CSV checks", "CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 599–608 high/setback mapping"],
  outputs=["05_simulations/climate_campaign_v3/Padova/summary.json", "06_claim_drafts/author_corrected_check_v01.json"],
  assumptions_and_uncertainty="Reported average hours are coarsened survey responses; the encoded schedule repeats them and maps them to 22°C rather than enforcing equipment availability.",
  permitted_manuscript_wording="The corrected run inputs retain the 4/4/1 encoded high-setpoint durations; heating can also occur at setback.")
u("RES-004",
  label="Corrected Padova candidate outputs",
  exact_claim="The corrected single-commit v3 Padova run reports seeded/early/late annual useful sensible heating of 5947.73/6078.71/5988.12 kWh and maximum hourly loads of 8.870/9.019/8.656 kW; relative to seeded, early changes annual demand by +2.20% and peak by +1.68%, while late changes them by +0.68% and −2.41%. These are candidate model outputs pending independent correction review, not accepted empirical findings.",
  status="partially supported",
  locators=["RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json corrected seeded/early/late", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/completion_manifest.json frozen hashes and imported EUReCA path", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/full_step_balance_checks.json all-hour network residual checks", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json independently reintegrated CSVs/deltas", "REVIEW-01 | 07_reviews/review_01.md R1-01 archived v1 values withdrawn"],
  outputs=["05_simulations/climate_campaign_v3/Padova/seeded_hourly.csv", "05_simulations/climate_campaign_v3/Padova/early_hourly.csv", "05_simulations/climate_campaign_v3/Padova/late_hourly.csv", "05_simulations/climate_campaign_v3/Padova/summary.json", "06_claim_drafts/author_corrected_check_v01.json"],
  assumptions_and_uncertainty="One deterministic assumed case under a TMY. Percent denominator is corrected seeded output. Values remain conditional on adapter correctness and independent review; no population or empirical confidence interval.",
  counterevidence_alternative_explanation="The original v1/v2 annual/peak figures were produced by a dispatcher that failed state balance and are quarantined. Corrected values differ materially, including reversal of annual-effect sign, demonstrating why old values cannot be used.",
  falsification_or_weakening="Independent review finding an incorrect single-commit transition, residual checker, source mapping, or run hash would withdraw the candidate numbers.",
  dependencies_next_action_owner="Reviewer 02 to independently verify the corrected code, hourly states/loads and gates; author to promote or withdraw after verdict.",
  permitted_manuscript_wording="[CANDIDATE CORRECTED RESULT, INDEPENDENT REVIEW PENDING: single-case Padova annual and peak schedule-placement contrasts.]" )
u("RES-005",
  exact_claim="No corrected, independently reviewed multi-weather result has yet established how high-setpoint placement sensitivity changes across the four coordinate TMYs; archived v2 climate loads are quarantined by the dispatcher defect.",
  locators=["REVIEW-01 | 07_reviews/review_01.md R1-01 same solver-path defect", "CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md quarantine and expansion gate", "CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv input contrast only"],
  outputs=["04_analysis/climate_descriptors_v2.csv"],
  assumptions_and_uncertainty="Homogeneous weather descriptors establish input contrast, not a valid model output. Corrected matched runs require full-step checks and independent review.",
  dependencies_next_action_owner="Simulation owner to finish v3 sites; author/reviewer to inspect correction and outputs.",
  permitted_manuscript_wording="[RESULT PENDING: corrected, checked multi-weather high-setpoint placement contrasts.]" )
u("INT-001",
  exact_claim="The archived v1/v2 effect signs and magnitudes cannot be interpreted as thermal sensitivity because the dispatcher violated hourly state balance at switching; corrected v3 Padova outputs are a candidate conditional response awaiting independent scientific review.",
  status="partially supported",
  locators=["REVIEW-01 | 07_reviews/review_01.md R1-01 state-balance diagnosis", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json candidate corrected contrasts", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json corrected arithmetic"],
  outputs=["06_claim_drafts/author_corrected_check_v01.json"],
  assumptions_and_uncertainty="No correct real schedule, population effect or predictive improvement follows. Candidate v3 results are conditional on one assumed case and still need independent implementation review.",
  counterevidence_alternative_explanation="A sign change between archived and corrected annual effects shows technical implementation mattered; it does not establish the corrected adapter against observations.",
  falsification_or_weakening="Failure of independent v3 state or source checks would remove the candidate response interpretation.",
  dependencies_next_action_owner="Reviewer 02 to verify corrected implementation; author then update interpretation status.",
  permitted_manuscript_wording="[INTERPRETATION PENDING: corrected model response must pass independent method review.]" )
u("LIM-005",
  locators=["RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json corrected maximum_air_C", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/design.json cooling_upper_C=50", "SIM-PROTOCOL | 03_methods/simulation_protocol.md cooling isolation"],
  outputs=["05_simulations/climate_campaign_v3/Padova/summary.json"],
  assumptions_and_uncertainty="The 42.81°C is an unconditioned modeled summer state under disabled cooling, not observed indoor temperature; correction review is pending.")

def new(cid, label, exact, kind, evidence_class, status, locators, outputs,
        population, assumptions, counter, falsifier, dependency, wording, related):
    return dict(id=cid, label=label, exact_claim=exact, kind=kind,
                evidence_class=evidence_class, status=status,
                source_ids=[p.split(" | ")[0] for p in locators], locators=locators,
                outputs=outputs, population_geography_period_comparator_units=population,
                assumptions_and_uncertainty=assumptions,
                counterevidence_alternative_explanation=counter,
                falsification_or_weakening=falsifier,
                dependencies_next_action_owner=dependency,
                permitted_manuscript_wording=wording,
                related_claims_figures=related)

new_claims = [
    new("MET-008", "Single-commit correction gate",
        "A study-specific local 1C adapter must select off/heating/cooling from candidates evaluated against the same beginning-of-hour state, commit once, and pass explicit mode-transition and every-hour air/surface/mass residual checks before annual outputs are admitted.",
        "method", "derived estimate", "partially supported",
        ["REVIEW-01 | 07_reviews/review_01.md R1-01 diagnostic and checkable resolution", "CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md pre-run and annual gates", "CORRECTION-CODE-V3 | 05_simulations/climate_campaign_v3/Padova/single_commit_1c.py source", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/transition_checks_pre_run.json and full_step_balance_checks.json"],
        ["05_simulations/climate_campaign_v3/Padova/single_commit_1c.py", "05_simulations/climate_campaign_v3/Padova/full_step_balance_checks.json"],
        "One ideal sensible 1C zone, hourly, no mechanical/natural ventilation or latent gains; 1e-6 W numerical residual tolerance.",
        "Adapter uses the unchanged pure EUReCA network solver with bounded local state selection; checker must be independent of engine and use exported old/current states.",
        "Passing network equations does not verify envelope truth, weather representativeness, software generality or measured accuracy.",
        "Reviewer-discovered state mismatch or checker dependence on incorrect saved states would defeat this gate.",
        "Reviewer 02: inspect candidate selection, one-commit behavior and independent full-step checker.",
        "The corrected pilot applies a one-state-commit hourly 1C selector and full-step balance checks; independent review remains pending.",
        "MET-003; MET-005; RES-004"),
    new("LIM-006", "Archived solver outputs quarantined",
        "The original Padova v1 and climate v2 annual loads are retained only as defective-run provenance because the recursive hourly dispatcher failed node balance at hundreds of mode transitions; they cannot support physical sensitivity estimates.",
        "limitation", "derived estimate", "supported",
        ["REVIEW-01 | 07_reviews/review_01.md R1-01 evidence and consequence", "REVIEW-DIAGNOSTIC | 07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json seeded/early/late failure counts and residuals", "CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md quarantine"],
        ["07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json"],
        "Archived 5 Padova v1 and 20 climate v2 annual executions; defect inspected directly on v1 with same dispatcher path for v2.",
        "V1 arithmetic and repeatability were valid as file properties, while thermal state transition was not. V2 path used the same defective dispatcher; no individual v2 residual figures are asserted here.",
        "The defect is in the current checkout's call path; no inference is made that the published conference executable shared it.",
        "A proven alternative historical v2 execution path could narrow quarantine, but current source/run snapshots identify the same dispatcher.",
        "Simulation owner: preserve archived files and labels; author: never reuse their effect magnitudes as scientific results.",
        "Archived dispatcher-based annual effects are excluded from scientific interpretation.",
        "RES-004; RES-005; MET-008"),
]

# The correction campaign completed while the review response was being staged.
# Keep review verdicts in 07_reviews; scientific claim locators point to source,
# diagnostic, code and run artifacts. All new numeric claims remain candidates
# until an independent reviewer checks the corrected implementation.
for change in updates.values():
    if "locators" in change:
        change["locators"] = [x for x in change["locators"] if not x.startswith("REVIEW-01 |")]
for c in new_claims:
    c["locators"] = [x for x in c["locators"] if not x.startswith("REVIEW-01 |")]
    c["source_ids"] = [x.split(" | ")[0] for x in c["locators"]]
u("MET-001", locators=updates["MET-001"]["locators"] + ["DISPATCH-DIAGNOSTIC | 07_reviews/reviewer_01/independent_checks.json positive heat at 19°C in archived outputs; figures quarantined"])
u("MET-005", locators=updates["MET-005"]["locators"] + ["DISPATCH-DIAGNOSTIC | 07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json original v1 transition residuals"])
u("MET-006", locators=[x for x in updates["MET-006"]["locators"] if not x.startswith("REVIEW-01")] + ["CORRECTION-PROTOCOL-V3 | 03_methods/correction_protocol_v3.md quarantine and corrected campaign gate"],
  status="partially supported",
  dependencies_next_action_owner="Twenty corrected v3 annual runs and their declared gates completed; author arithmetic checked. Reviewer 02 must independently assess the adapter and weather contrasts.",
  permitted_manuscript_wording="Four consistently sourced coordinate TMYs support a controlled weather comparison; corrected effect estimates await independent review.")
u("RQ-002", status="partially supported",
  locators=updates["RQ-002"]["locators"] + ["RUN-CLIMATE-V3 | 04_analysis/paired_schedule_contrasts_v3.csv all four-site candidate contrasts", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_result_check_v01.json independent integration and table checks"],
  outputs=["04_analysis/climate_descriptors_v2.csv", "04_analysis/paired_schedule_contrasts_v3.csv", "06_claim_drafts/author_result_check_v01.json"],
  dependencies_next_action_owner="Corrected v3 multi-weather runs and author arithmetic check completed; reviewer 02 to assess adapter, residual gate and conditional interpretation.",
  permitted_manuscript_wording="[CANDIDATE WEATHER COMPARISON, INDEPENDENT CORRECTION REVIEW PENDING.]" )
u("RES-005",
  label="Corrected weather-specific annual effects",
  exact_claim="In corrected v3 matched runs, early/late placements changed annual useful sensible heating relative to each site's seeded profile by +2.20/+0.68% Padova, +0.41/+0.18% Bolzano, +5.90/+1.44% Roma and +6.64/+1.81% Palermo; these conditional modeled effects await independent correction review.",
  evidence_class="simulation output", status="partially supported",
  locators=["RUN-CLIMATE-V3 | 04_analysis/paired_schedule_contrasts_v3.csv annual_sensible_heating_kWh and delta_annual_percent for 12 primary cases", "RUN-CLIMATE-V3 | 05_simulations/climate_campaign_v3/{Padova,Bolzano,Roma,Palermo}/completion_manifest.json and full_step_balance_checks.json per site", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_result_check_v01.json independent hourly CSV sums and paired table checks", "CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv consistent input contrast"],
  outputs=["04_analysis/paired_schedule_contrasts_v3.csv", "06_claim_drafts/author_result_check_v01.json"],
  population_geography_period_comparator_units="Same assumed dwelling and one Veneto response under four PVGIS TMY coordinate files; within-site early/late minus seeded, percent denominator same-site seeded kWh.",
  assumptions_and_uncertainty="All physical inputs and climate-E operating season are fixed; sites are weather exposures, not local household stock. One deterministic year per site; no population interval.",
  counterevidence_alternative_explanation="Weather descriptors co-vary in temperature, solar and humidity; this design cannot attribute between-site effect differences to one weather variable. Archived v2 climate outputs are excluded.",
  falsification_or_weakening="Independent reviewer finding a corrected-state, input equality, weather provenance, or arithmetic error would withdraw these candidate effects.",
  dependencies_next_action_owner="Reviewer 02 to inspect v3 local correction and all-site gates; author to promote or withdraw after verdict.",
  permitted_manuscript_wording="[CANDIDATE CORRECTED RESULT: annual placement effects differ across four fixed-input TMY exposures; independent review pending.]" )
u("INT-001",
  exact_claim="After correction, the candidate matched outputs vary with encoded high-setpoint placement in one assumed dwelling under each of four coordinate TMYs; archived v1/v2 effect signs and magnitudes remain invalid because of the old transition defect.",
  locators=["DISPATCH-DIAGNOSTIC | 07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json old state-balance failures", "RUN-CLIMATE-V3 | 04_analysis/paired_schedule_contrasts_v3.csv corrected four-site candidate outcomes", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_result_check_v01.json author arithmetic verification"],
  outputs=["04_analysis/paired_schedule_contrasts_v3.csv", "06_claim_drafts/author_result_check_v01.json"],
  assumptions_and_uncertainty="The pattern concerns one assumed zone/response and TMY coordinate contrasts; no schedule is proven true or more accurate. Scientific interpretation awaits independent correction review.",
  counterevidence_alternative_explanation="Different envelope, thermostat mapping, weather period or allowable placements could change effects; the original sign reversal shows implementation risk.",
  dependencies_next_action_owner="Reviewer 02 to audit corrected solver and full-step checks before author promotes interpretation.",
  permitted_manuscript_wording="[CONDITIONAL INTERPRETATION PENDING INDEPENDENT CORRECTION REVIEW.]" )
new_claims += [
    new("RES-006", "Post-hoc setback heating diagnostic",
        "In corrected v3 Padova seeded output, the ideal heater has positive sensible load in 671 of 2730 calendar hours at the modeled 19°C setback, delivering 893.22 kWh there; this is a post-hoc description showing encoded 4/4/1 high-setpoint hours do not fix actual heating-operation duration.",
        "result", "simulation output", "partially supported",
        ["POSTHOC-PARTITION-V3 | 04_analysis/posthoc_setpoint_partition_v3.csv Padova seeded 19°C row and tolerance 1e-6 W", "RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/seeded_hourly.csv sensible_load_W and heating_setpoint_C", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json corrected summary arithmetic"],
        ["04_analysis/posthoc_setpoint_partition_v3.csv", "05_simulations/climate_campaign_v3/Padova/seeded_hourly.csv"],
        "One assumed dwelling under Padova TMY; 19°C setpoint calendar hours 2730, positive-load hours 671, useful heating 893.22 kWh; not a predeclared endpoint.",
        "Setpoint partition uses >1e-6 W positive-load threshold. 4/4/1 is a model high-setpoint profile, not observed control or burner timing.",
        "A different thermostat interpretation or weather could change the diagnostic; the original v1 diagnostic is quarantined.",
        "Mismatched hourly partition or incorrect corrected source would withdraw the quantity.",
        "Reviewer 02 to inspect post-hoc computation and corrected hourly file; author to retain descriptive label.",
        "A post-hoc corrected-run check found heating at the 19°C modeled setback, so encoded high-setpoint hours do not equal active heating hours.",
        "MET-001; MET-002; RES-004"),
    new("RES-007", "Corrected weather-specific peak effects",
        "In corrected v3 matched runs, early/late placements changed maximum hourly sensible heating load relative to each site's seeded profile by +1.68/−2.41% Padova, −1.39/+2.29% Bolzano, +8.17/+2.70% Roma and +6.11/+0.34% Palermo; these deterministic modeled effects await independent correction review.",
        "result", "simulation output", "partially supported",
        ["RUN-CLIMATE-V3 | 04_analysis/paired_schedule_contrasts_v3.csv peak_sensible_heating_kW and delta_peak_percent for 12 primary cases", "RUN-CLIMATE-V3 | 05_simulations/climate_campaign_v3/{Padova,Bolzano,Roma,Palermo}/full_step_balance_checks.json each site", "AUTHOR-CHECK-V01 | 06_claim_drafts/author_result_check_v01.json independent peak and paired percentage checks"],
        ["04_analysis/paired_schedule_contrasts_v3.csv", "06_claim_drafts/author_result_check_v01.json"],
        "Same assumed dwelling and one Veneto response under four PVGIS TMY coordinates; within-site early/late minus seeded peak; percent denominator seeded peak kW.",
        "Hourly maximum is sensitive to 1 h resolution and fixed season; no observed peak or probabilistic uncertainty interval exists.",
        "Different weather timing, envelope, setpoint mapping or control could alter peak sign/magnitude; archived v2 peaks are excluded.",
        "Independent reviewer finding a corrected-state, peak extraction or input-pairing error would withdraw candidate effects.",
        "Reviewer 02 to check correction and peak outputs; author to promote or withdraw after verdict.",
        "[CANDIDATE CORRECTED RESULT: hourly peak placement effects vary across four TMY exposures; independent review pending.]",
        "RQ-002; RES-005; FIG-03"),
]
posthoc = next(c for c in new_claims if c["id"] == "RES-006")
posthoc["locators"].append("AUTHOR-POSTHOC-V01 | 06_claim_drafts/author_posthoc_check_v01.json independent 36-row CSV partition check")
posthoc["outputs"].append("06_claim_drafts/author_posthoc_check_v01.json")
posthoc["source_ids"] = [x.split(" | ")[0] for x in posthoc["locators"]]

patch = {
    "prior_draft_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
    "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    "status_note": "Status: review-01 response, 20 October 2026. Original Padova v1 and climate v2 load effects are quarantined after the confirmed hourly state-transition defect. Twenty corrected v3 annual runs provide candidate four-site effects with author arithmetic and full-step checks; scientific interpretation awaits independent review of the local correction. No empirical heating-accuracy claim is made.",
    "updates": updates, "new_claims": new_claims,
}
out.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "updates": len(updates), "new_claims": len(new_claims)}))
