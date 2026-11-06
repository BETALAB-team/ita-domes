"""Build review-03 claim patch from verified v5 primary artifacts."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
old = json.loads((HERE / "claim_ledger_v02.json").read_text(encoding="utf-8"))
by = {c["id"]: c for c in old["claims"]}
updates = {}


def u(cid, **fields):
    updates[cid] = fields


padova = [
    "RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/seeded_hourly.csv, early_hourly.csv, late_hourly.csv and summary.json; per-file SHA256 in summary/completion_manifest",
    "RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/full_step_balance_checks.json, surface_forcing_hourly.csv, surface_forcing_parameters.json and completion_manifest.json",
    "AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent EPW/CSV/paired/replay arithmetic",
    "INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json direct 20-case state/accounting/peak audit",
]
allsite = [
    "RUN-CLIMATE-V5 | 04_analysis/paired_schedule_contrasts_v5.csv twelve predeclared primary cases and same-site seeded denominators",
    "RUN-CLIMATE-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/summary.json, completion_manifest.json and hourly CSVs",
    "AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent integration/hash/paired table checks",
    "INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json independently reconstructed v5 rows, residuals and peak contexts",
]
surface = [
    "SURFACE-PROTOCOL-V5 | 03_methods/surface_correction_protocol_v5.md pre-run equations, unchanged parameter and test scope",
    "SURFACE-CODE-V5 | 04_analysis/surface_forcing_1c.py study-local per-surface SW/LW/glazing calculation, SHA256 24768556f49ef341791647502f0fe5d1f8baddc3e5c2fcf5df23a124f106883a",
    "SURFACE-FIXTURE-V5 | 05_simulations/surface_forcing_v5_fixture_checks.json analytic component and order tests",
    "SURFACE-AUDIT-V5 | 07_reviews/reviewer_03/v5_forcing_checks.json independent per-surface, perturbation, order and actual-solver forcing audit",
]

u("LIT-002",
  exact_claim="Prataviera et al. (2022) already reconstruct heating operation from three ISTAT time bands, use randomized profiles with EUReCA, and explicitly perturb HVAC start-up by a discrete uniform ±2 h shift because survey timing is unknown; their Verona hourly comparison has a district-heat measurement boundary distinct from this fixed-case experiment.",
  locators=["SRC-UA2022 | full primary article Section 3.1.1 p6, start-up-time bullet and Table 1 p7; Section 3.3.1 p11; Section 4.1.2 p12; DOI 10.1016/j.apenergy.2022.118691"],
  assumptions_and_uncertainty="The published shift is a whole-profile startup/timing perturbation within a broader UBEM uncertainty study, while the present experiment holds one respondent and each band's encoded high-setpoint duration fixed. No generic timing-method novelty is claimed.",
  permitted_manuscript_wording="The 2022 survey-informed EUReCA study already randomized three-band schedules and tested a ±2-hour start-up shift; this study isolates three placements for one fixed encoded-duration case.")

u("GAP-001",
  counterevidence_alternative_explanation="The survey asks mean system on-hours rather than thermostat levels. Prataviera et al. already addressed unknown timing with random profiles and ±2 h startup shifts, so the information gap is not a previously unrecognized one; our 22/19°C encoding is an assumption.",
  permitted_manuscript_wording="Mean reported hours do not identify within-band switching; prior work already studied timing uncertainty, and this fixed case encodes selected placements as high-setpoint periods.")

u("NOV-001",
  exact_claim="The candidate contribution is a bounded, reproducible comparison of three contiguous placements preserving each of one respondent's 4/4/1 encoded high-setpoint durations within their respective bands for one assumed dwelling and four coordinate weather exposures; novelty of this exact fixed-case contrast remains provisional in light of earlier three-band randomization and ±2 h startup perturbations.",
  locators=["SRC-UA2022 | Section 3.1.1 p6 startup shift and three-band reconstruction; Table 1 p7", "RQ-FREEZE | 01_plan/research_questions.md fixed respondent/placement contrast", "LIT-SEARCH | 02_evidence/literature_search_log.md targeted search limits"],
  counterevidence_alternative_explanation="Prataviera 2022 already tested schedule timing uncertainty in a related survey-informed EUReCA context; a broad first-method or first-timing claim would be false.",
  dependencies_next_action_owner="Literature owner: retain targeted rather than exhaustive novelty boundary; author: describe exact experimental contrast without first-use language.",
  permitted_manuscript_wording="We test a specific fixed-respondent, within-band placement contrast; prior studies already modeled survey-derived timing uncertainty.")

u("RQ-001",
  exact_claim="For one illustrative dwelling, selected Veneto response and Padova-coordinate TMY, the predeclared comparison is annual useful sensible heating and maximum hourly load under seeded, earliest and latest contiguous placements of the same 4/4/1 encoded high-setpoint hours; v5 now provides a checked conditional model comparison, pending a fresh fourth scientific review.",
  status="partially supported",
  locators=["RQ-FREEZE | 01_plan/research_questions.md primary question and fixed-case scope", "ISTAT-Q2013 | istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf pp41–43", "CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625, 643–656", *padova],
  outputs=["05_simulations/climate_campaign_v5/Padova/summary.json", "06_claim_drafts/author_v5_check_for_v03.json"],
  assumptions_and_uncertainty="Reported hours are converted to 22/19°C modeled setpoints; actual positive-load hours remain free. V5 is one assumed geometry/response/season under native annual-mean sky and solar-time conventions, not observed or representative heating.",
  dependencies_next_action_owner="Fresh reviewer 04: independently inspect the v03 claims, corrected forcing, source-to-profile mapping and v5 hourly outputs; author: retain conditional boundary.",
  permitted_manuscript_wording="We compare three equal-duration modeled high-setpoint placements in one illustrative Padova-coordinate case; v5 numerical contrasts are conditional model outputs under review.")

u("RQ-002",
  exact_claim="The same fixed dwelling, Veneto response, 22/19°C mapping and climate-E seasonal mask are compared under four consistently sourced coordinate TMYs (Padova, Bolzano, Roma, Palermo); v5 matched runs provide conditional weather-exposure contrasts, not local population or climate-variable causal effects, pending review 04.",
  status="partially supported",
  locators=["RQ-FREEZE | 01_plan/research_questions.md secondary weather question", "CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json acquired four-site weather hashes", "CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv descriptors", *allsite],
  outputs=["04_analysis/climate_descriptors_v2.csv", "04_analysis/paired_schedule_contrasts_v5.csv", "06_claim_drafts/author_v5_check_for_v03.json"],
  population_geography_period_comparator_units="One unchanged illustrative case under four coordinate TMYs; matched seeded/early/late contrasts in kWh, kW and same-site percent.",
  assumptions_and_uncertainty="The common climate-E operating season is imposed outside its local context. Temperature, solar and humidity descriptors co-vary, so site-to-site differences cannot be assigned to one weather variable; sky is annual-mean and native solar phase offset is unapplied.",
  dependencies_next_action_owner="Fresh reviewer 04: verify v5 forcing/output pairs and interpretation; author: retain fixed-input/weather-exposure framing.",
  permitted_manuscript_wording="The same illustrative case was exposed to four coordinate TMYs with paired schedules; differences are conditional model contrasts, not estimates for each city's housing stock.")

u("MET-003",
  locators=by["MET-003"]["locators"] + ["RUN-CLIMATE-V5 | 05_simulations/climate_campaign_v5/Padova/single_commit_1c.py unchanged single-commit source hash e549b4e5645ad8ae159fddf644d225d28622e1495f6fd64768c5242389b82c16"],
  dependencies_next_action_owner="The bounded state repair carries into v5 unchanged; fresh review 04 to inspect final pipeline without treating state closure as empirical validation.")

u("MET-005",
  exact_claim="The v5 20-case campaign passed direct transition, source-profile, repeat/warm-up and complete air/surface/mass state checks: all 175,200 exported hours have exact old-state continuity and no residual above 1e-6 W; independent maximum absolute node residual is 1.768×10⁻¹⁰ W. These tests verify this bounded numerical implementation, not measured accuracy.",
  locators=[*surface, *allsite, "STATE-CHECK-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/full_step_balance_checks.json and checks.json"],
  outputs=["04_analysis/independent_output_checks_v5.json", "07_reviews/reviewer_03/independent_audit_v5.json"],
  population_geography_period_comparator_units="Twenty v5 annual simulations (12 primary, 4 seeded repeats, 4 longer warm-up), 175,200 hourly records; ideal sensible 1C zone; residual tolerance 1e-6 W.",
  assumptions_and_uncertainty="Numerical state closure, repeatability and warm-up stability apply only to this fixed case and selected implementation; forcing and empirical validity are separate evidence layers. Earlier 66 executions remain quarantined.",
  counterevidence_alternative_explanation="Earlier v1/v2 recursive state transition, v3 sentinel-sky and v4 stale-surface forcing defects show why weaker gates were inadequate; the v5 actual forcing is separately checked.",
  dependencies_next_action_owner="Fresh reviewer 04: inspect immutable v5 source, actual forcing, state and output checks; author: keep numerical verification distinct from benchmark/measurement evidence.",
  permitted_manuscript_wording="The 20 v5 runs passed bounded source, forcing, state and repeat checks; these do not establish observed heating accuracy.")

u("MET-006",
  exact_claim="Four same-provider PVGIS coordinate TMYs were characterized consistently and used with unchanged building, response and seasonal mask in the v5 corrected-forcing campaign. The active model path guards missing cloud and invalid wind access; Palermo's one negative wind record remains an unused-field exception, not a universal weather-quality pass.",
  locators=["CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json and raw EPWs", "CLIMATE-METADATA-V2 | 05_simulations/climate_inputs_v2/audited_metadata.json", "CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv", "WEATHER-PROTOCOL-V4 | 03_methods/weather_correction_protocol_v4.md active-field, IR and time audit", "WEATHER-AUDIT-V4 | 05_simulations/weather_forcing_v4/weather_forcing_audit.json", *allsite],
  outputs=["04_analysis/climate_descriptors_v2.csv", "04_analysis/paired_schedule_contrasts_v5.csv"],
  population_geography_period_comparator_units="Four 8760-hour coordinate TMY files and 12 matched primary v5 cases; annual kWh and maximum hourly kW.",
  assumptions_and_uncertainty="TYMs are synthetic; common climate-E season is not location-specific. One Palermo −0.1 m/s wind value is retained but guarded and unused in fixed-ACH/no natural-mechanical ventilation path. Annual-mean sky and native solar-time conventions limit hourly fidelity.",
  counterevidence_alternative_explanation="Within-site matching controls design inputs but cannot isolate which co-varying meteorological field explains cross-site differences.",
  dependencies_next_action_owner="Fresh reviewer 04: inspect all-site active forcing and paired contrast provenance; any future wind-dependent model must resolve the Palermo value first.",
  permitted_manuscript_wording="Four consistently retrieved coordinate TMYs support conditional controlled weather exposures for this guarded fixed-ACH model.")

u("MET-007",
  locators=["VAL-PROTOCOL | 03_methods/validation_protocol.md current v5 candidate disposition and evidence layers", "VAL-MATRIX | 03_methods/validation_matrix.md current v5 output/validation matrix", "SIM-PROTOCOL | 03_methods/simulation_protocol.md canonical current status and frozen-version chronology"],
  dependencies_next_action_owner="Validation owner: maintain one current v5 candidate status pointer and the absent benchmark/meter layers; later prose must not translate numerical verification into accuracy.")

u("MET-008",
  exact_claim="The study-local 1C state stepper is independently checked for one-old-state/one-commit transitions and all-hour node balances in v5; a separate per-surface and weather-input forcing gate now also passes for this bounded geometry. Neither layer is an independent building benchmark or empirical accuracy test.",
  locators=["CORRECTION-CODE-V3 | 04_analysis/single_commit_1c.py unchanged candidate/commit logic", "STATE-CHECK-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/full_step_balance_checks.json", *surface],
  outputs=["07_reviews/reviewer_03/independent_audit_v5.json", "07_reviews/reviewer_03/v5_forcing_checks.json"],
  assumptions_and_uncertainty="Only this ideal sensible 1C scope was checked; native transposition and time convention are retained model assumptions.",
  dependencies_next_action_owner="Fresh reviewer 04: inspect both correction layers and final effects; author: avoid general EUReCA validation language.",
  permitted_manuscript_wording="The bounded v5 pilot separately checks its one-commit state solution and actual per-surface forcing.")

u("MET-009",
  exact_claim="For the bounded v5 path, EPW missing cloud cover is bypassed using valid hourly horizontal IR to form effective sky temperature; the native model's annual mean of air-minus-sky is then applied through an explicit per-surface longwave and shortwave calculation. Independent actual-forcing, zero-solar, perturbation and order tests pass, subject to annual-mean and solar-time approximation limits.",
  status="partially supported",
  evidence_class="derived estimate",
  locators=["EPW-SPEC | https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v24.1.0/AuxiliaryPrograms.pdf EPW missing codes", "WEATHER-PROTOCOL-V4 | 03_methods/weather_correction_protocol_v4.md IR conversion/annual-mean and time convention", "WEATHER-CODE-V4 | 04_analysis/ir_sky_forcing.py valid-IR scalar and active-path guard", "WEATHER-AUDIT-V4 | 05_simulations/weather_forcing_v4/weather_forcing_audit.json", *surface],
  outputs=["05_simulations/weather_forcing_v4/weather_forcing_audit.json", "07_reviews/reviewer_03/v5_forcing_checks.json"],
  assumptions_and_uncertainty="The one-hour EPW horizontal IR is converted to W/m²; the annual-mean sky difference, not hourly sky exchange, enters native 1C longwave. Solar position uses fixed UTC+1 interval-start labels without applying recorded irradiation offsets. One invalid Palermo wind field is guarded and unused.",
  counterevidence_alternative_explanation="V3's unfiltered cloud99 and v4's inherited roof area/irradiance invalidated their load values; the independent v5 per-surface actual-forcing check specifically tests both paths.",
  dependencies_next_action_owner="Fresh reviewer 04: inspect v5 code, raw IR, guards, per-surface all-hour forcing and retained time approximations.",
  permitted_manuscript_wording="The v5 pilot uses valid horizontal IR and checked per-surface forcing under the documented annual-mean sky and native solar-time conventions.")

u("MET-010",
  status="supported",
  exact_claim="The v3 manifests' historical SITE omission is preserved and annotated. The v5 runner records executable, launch working directory, full site/campaign argument vector, source/input/output hashes and refuses an existing completed destination; a separate seeded replay is byte-identical to the primary Padova hourly output and is not an additional scientific sample.",
  locators=["COMMAND-ADDENDUM | 05_simulations/recorded_command_addendum.md historical omission and non-overwrite rule", "RUNNER-V5 | 04_analysis/run_climate_campaign_v5.py SITE/--campaign/replay arguments and output guard", "RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/completion_manifest.json recorded command_argv/cwd/executable", "REPLAY-V5 | 05_simulations/climate_campaign_v5_replay/Padova/completion_manifest.json and seeded_hourly.csv", "REPLAY-CHECK-V5 | 05_simulations/v5_replay_interface_checks.json overwrite refusal and byte identity"],
  outputs=["05_simulations/climate_campaign_v5/Padova/completion_manifest.json", "05_simulations/climate_campaign_v5_replay/Padova/completion_manifest.json", "05_simulations/v5_replay_interface_checks.json"],
  assumptions_and_uncertainty="The historical v3 command string is not retroactively rewritten. Replay demonstrates deterministic production under the inspected current environment, not physical truth or an independent sample.",
  dependencies_next_action_owner="Methods owner: preserve complete replay/immutability discipline for any later run version; author: keep replay out of the 20-case scientific sample.",
  permitted_manuscript_wording="V5 commands and source hashes are recorded, and a separate Padova seeded replay reproduces its primary hourly file byte-for-byte.")

u("RES-003",
  exact_claim="The v5 seeded, earliest and latest Padova inputs each preserve 4/4/1 modeled 22°C high-setpoint hours in the 05–13, 13–21 and 21–05 bands; actual positive-load hours may occur at the 19°C setback and are not fixed by these profile counts.",
  locators=["CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625, 643–656", "RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/summary.json daily_on_profile and seeded/early/late hourly setpoints", "AUTHOR-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json source-setpoint partition arithmetic"],
  outputs=["05_simulations/climate_campaign_v5/Padova/summary.json", "06_claim_drafts/author_v5_posthoc_peak_check_v03.json"],
  permitted_manuscript_wording="All v5 alternatives keep the reported 4/4/1 durations as encoded high-setpoint periods, while heating may occur at setback.")

u("RES-004",
  label="Padova v5 conditional modeled contrasts", status="partially supported",
  exact_claim="In the checked v5 fixed Padova-coordinate case, seeded/early/late annual useful sensible heating is 9574.02/9659.63/9606.30 kWh and maximum hourly heating load is 10.050/10.253/9.838 kW. Relative to seeded, early changes annual heating by +0.894% and peak by +2.029%; late changes them by +0.337% and −2.103%. These are conditional model outputs awaiting fresh review 04, not observed energy or accuracy estimates.",
  locators=padova + surface,
  outputs=["04_analysis/paired_schedule_contrasts_v5.csv", "05_simulations/climate_campaign_v5/Padova/seeded_hourly.csv", "05_simulations/climate_campaign_v5/Padova/early_hourly.csv", "05_simulations/climate_campaign_v5/Padova/late_hourly.csv", "06_claim_drafts/author_v5_check_for_v03.json"],
  assumptions_and_uncertainty="Same illustrative dwelling, respondent, weather, 22/19°C mapping and season across placements. Annual-mean longwave, native unapplied solar offset, fixed gains/envelope and one TMY limit transfer. Percent denominator is same-site seeded. No measurement or confidence interval.",
  counterevidence_alternative_explanation="All 66 older annual executions (v1–v4 plus v4 replay) are quarantined for state/forcing defects. Different profile shapes, physical assumptions or hourly weather timing could alter v5 effects.",
  falsification_or_weakening="Fresh review finding source-to-forcing mismatch, state/summary error, unmatched run inputs or implausible retained approximations would withdraw or narrow these candidate values.",
  dependencies_next_action_owner="Fresh reviewer 04: independent full claim/forcing/output and scope audit; author: admit only bounded fixed-case modeled language if accepted.",
  permitted_manuscript_wording="[CANDIDATE V5 RESULT, FRESH REVIEW 04 PENDING.] In one fixed Padova-coordinate model case, placement changed annual useful heating and maximum hourly load by the stated paired amounts.")

u("RES-005",
  label="Four-site v5 annual modeled contrasts", status="partially supported",
  exact_claim="In v5 matched fixed-input runs, early/late placements change annual useful sensible heating relative to same-site seeded by +0.894/+0.337% Padova, −0.098/+0.027% Bolzano, +2.033/+0.500% Roma, and +3.868/+0.981% Palermo. These are deterministic conditional weather-exposure contrasts pending fresh review 04, not population effects or causal attribution to a weather variable.",
  locators=allsite + surface,
  outputs=["04_analysis/paired_schedule_contrasts_v5.csv", "06_claim_drafts/author_v5_check_for_v03.json", "07_reviews/reviewer_03/independent_audit_v5.json"],
  assumptions_and_uncertainty="One fixed illustrative building/response/setpoint/season across coordinate TMYs. Temperature, radiation and humidity vary together. Native annual-mean sky and solar-phase approximation remain; no statistical interval.",
  counterevidence_alternative_explanation="The small Bolzano differences could be sensitive to model/forcing/timing assumptions; no single weather mechanism can be isolated. Older 66 annual executions remain invalid for physical effects.",
  falsification_or_weakening="A fresh v5 forcing, matched-input, integration or interpretation failure would withdraw or narrow these candidate effects.",
  dependencies_next_action_owner="Fresh reviewer 04: verify all-site forcing and direct CSV contrasts; author: retain conditional and nonrepresentative scope.",
  permitted_manuscript_wording="[CANDIDATE V5 RESULT, FRESH REVIEW 04 PENDING.] Same-case annual placement contrasts differ across four coordinate TMYs under the documented model convention.")

u("RES-006",
  label="V5 post-hoc setback heating diagnostic", status="partially supported",
  exact_claim="In the v5 Padova seeded output, the ideal heater has positive sensible load during 1323 of 2730 modeled 19°C setback calendar hours, delivering 2375.62 kWh there. This post-hoc descriptive partition illustrates that the fixed 4/4/1 high-setpoint duration does not fix positive-heating runtime; it is neither an observed operation record nor a primary endpoint.",
  locators=["POSTHOC-V5 | 04_analysis/posthoc_setpoint_partition_v5.csv Padova seeded19°C row", "RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/seeded_hourly.csv hourly setpoints and sensible loads", "AUTHOR-POSTHOC-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json 36 partition rows recalculated", "INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json independently checked post-hoc partition"],
  outputs=["04_analysis/posthoc_setpoint_partition_v5.csv", "06_claim_drafts/author_v5_posthoc_peak_check_v03.json"],
  population_geography_period_comparator_units="One assumed Padova-coordinate TMY case; 19°C modeled setback; 2730 calendar hours, 1323 positive-load hours, 2375.62 useful kWh; post-hoc.",
  assumptions_and_uncertainty="Positive load threshold is >1e-6 W. The control is modeled 22/19°C, not the respondent's observed thermostat or burner timing. The quantity is conditional on v5 forcing and one case.",
  counterevidence_alternative_explanation="Changing weather, season, thermostat interpretation or envelope can change counts; source code alone establishes only that setback heating is allowed.",
  dependencies_next_action_owner="Fresh reviewer 04: inspect partition against v5 raw hourly file and ensure post-hoc/nonobserved label is retained.",
  permitted_manuscript_wording="[CANDIDATE POST-HOC V5 DIAGNOSTIC.] The modeled heater also delivers heat at the 19°C setback; encoded high-setpoint hours do not equal positive-load hours.")

u("RES-007",
  label="Four-site v5 hourly-maximum contrasts", status="partially supported",
  exact_claim="In v5 matched runs, early/late placements change maximum hourly sensible heating load relative to same-site seeded by +2.029/−2.103% Padova, 0/0% Bolzano, +8.326/+3.381% Roma and +5.969/+1.524% Palermo. The Bolzano maxima coincide at the modeled season-onset hour, so their zero paired peak differences do not demonstrate no effect of intraday timing.",
  locators=allsite + ["PEAK-CONTEXT-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json annual peak hour and adjacent setpoints", "INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json peak rows and preceding states"],
  outputs=["04_analysis/paired_schedule_contrasts_v5.csv", "06_claim_drafts/author_v5_posthoc_peak_check_v03.json"],
  assumptions_and_uncertainty="Maximum is a predeclared 1-hour resolution endpoint, not system capacity or robust design peak. It can be dominated by seasonal startup and native solar-phase/sky approximations; no peak timing sensitivity sweep is completed.",
  counterevidence_alternative_explanation="At Bolzano the prescribed 0→19°C season-start recovery, common to all profiles, drives the maximum; this masks possible differences at other hours. Across sites, no one weather variable is isolated.",
  dependencies_next_action_owner="Fresh reviewer 04: inspect each peak timestamp/prior state and paired extraction, especially Bolzano season onset; author: do not omit inconvenient maxima or infer equipment sizing.",
  permitted_manuscript_wording="[CANDIDATE V5 RESULT, FRESH REVIEW 04 PENDING.] Hourly maxima differ in three sites; Bolzano's equal maxima occur at the shared modeled season-start recovery.")

u("INT-001",
  label="Conditional fixed-case interpretation", status="partially supported",
  exact_claim="Under one assumed dwelling, one respondent, three encoded placements and the declared native weather-time/annual-mean-sky convention, v5 outputs show placement-dependent annual useful-heating and maximum-hourly-load contrasts that differ by coordinate TMY. The comparison is descriptive and model-conditional; it does not identify a causal weather variable, an observed schedule, population effect, savings or measured accuracy.",
  locators=allsite + ["RQ-FREEZE | 01_plan/research_questions.md predeclared estimands", "PEAK-CONTEXT-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json seasonal-start maximum diagnostic"],
  outputs=["04_analysis/paired_schedule_contrasts_v5.csv", "06_claim_drafts/author_v5_check_for_v03.json"],
  assumptions_and_uncertainty="Only the encoded choices, fixed geometry/response/season and native forcing convention are tested. Small effects and hourly maxima may change with physical/timing assumptions; no ensemble or empirical validation quantifies that uncertainty.",
  counterevidence_alternative_explanation="Bolzano maxima are shared seasonal recovery events; weather descriptors co-vary; prior defective campaigns show implementation sensitivity and remain excluded.",
  dependencies_next_action_owner="Fresh reviewer 04: evaluate final central effect claims and limits; author: preserve descriptive wording and avoid robust mechanism or transfer claims.",
  permitted_manuscript_wording="[CANDIDATE CONDITIONAL INTERPRETATION.] These v5 paired contrasts describe this specified model case, not measured households or general schedule accuracy.")

u("LIM-005",
  exact_claim="The v5 no-cooling case reaches a maximum modeled Padova zone air temperature of 41.05°C, so it cannot support claims of summer comfort, realistic whole-year conditioned operation, or an appropriate cooling design.",
  locators=["RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/summary.json maximum_air_C and design.json cooling_upper_C=50", "INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json saved temperature maxima", "SIM-PROTOCOL | 03_methods/simulation_protocol.md cooling-isolation boundary"],
  outputs=["05_simulations/climate_campaign_v5/Padova/summary.json"],
  population_geography_period_comparator_units="One assumed uncooled zone under Padova-coordinate TMY; simulated air temperature °C, not a measured indoor series.",
  assumptions_and_uncertainty="The maximum is conditional on the v5 fixed model and native weather forcing; no cooling/ventilation/comfort intervention is evaluated.",
  permitted_manuscript_wording="Cooling is disabled; the modeled summer air temperature reaches about 41°C, so the analysis addresses heating only.")

u("LIM-007",
  label="All pre-v5 physical outputs quarantined",
  exact_claim="All 66 annual executions before the v5 campaign, including the separately identified v4 seeded replay, are retained as diagnostic provenance but excluded from physical energy-effect interpretation: v1/v2 use the recursive state path, v3 consumes missing cloud cover, and v4 uses stale roof area/irradiance in native envelope preprocessing.",
  locators=["RUN-INVENTORY | 05_simulations/run_inventory.csv version/status inventory", "DISPATCH-DIAGNOSTIC | 07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json", "WEATHER-DIAGNOSTIC-R2 | 07_reviews/reviewer_02/independent_audit.json sentinel/sky finding", "SURFACE-DIAGNOSTIC-R3 | 07_reviews/reviewer_03/runtime_forcing_checks.json actual preprocessing/order defects", "SIM-RESULTS | 04_analysis/simulation_results.md current quarantine and v5 status"],
  outputs=["05_simulations/run_inventory.csv", "07_reviews/reviewer_03/runtime_forcing_checks.json"],
  population_geography_period_comparator_units="66 archived pre-v5 annual executions: 25 v1/v2, 20 v3, 20 v4 primary/repeat/warm-up, one v4 seeded replay; no admitted physical effect magnitude.",
  assumptions_and_uncertainty="Defects are shown in current local execution paths, not attributed to the published conference executable. Archived arithmetic/state passes remain narrow historical records, not physical effect evidence.",
  counterevidence_alternative_explanation="Repeatability of a defective run does not imply correct envelope forcing; the exact v5 campaign uses new versioned outputs.",
  dependencies_next_action_owner="Methods/author: preserve archives and explicit version labels; fresh reviewer 04: verify v5-only numerical provenance.",
  permitted_manuscript_wording="Earlier local run versions were excluded after distinct state, sky-input and surface-forcing defects; the current conditional estimates come from v5 only.")

new = []
def add(base, cid, **fields):
    c = copy.deepcopy(by[base]); c.update(fields); c["id"] = cid
    c["source_ids"] = [x.split(" | ")[0] for x in c["locators"]]
    new.append(c)

add("MET-009", "MET-011",
  label="Explicit per-surface 1C preprocessing",
  exact_claim="A bounded local v5 preprocessing routine computes each external wall/roof's opaque shortwave and longwave terms from its own area, orientation, irradiance, material and radiative fields, and computes glazing shortwave only where glazed area is positive. Independent per-surface, zero-solar/zero-longwave, order-invariance and actual-solver-forcing tests pass for this case without changing physical/network parameters or the external checkout.",
  kind="method", evidence_class="derived estimate", status="partially supported",
  locators=surface + ["SURFACE-DIAGNOSTIC-R3 | 07_reviews/reviewer_03/runtime_forcing_checks.json v4 inherited roof values and order failure", "PROVENANCE-V5 | 07_reviews/reviewer_03/v5_provenance_comparison.json unchanged design/network/inputs"],
  outputs=["07_reviews/reviewer_03/v5_forcing_checks.json", "04_analysis/independent_surface_checks_v5.json", "05_simulations/climate_campaign_v5/Padova/surface_forcing_hourly.csv"],
  population_geography_period_comparator_units="Five exterior opaque surfaces in one assumed 1C zone; hourly SW/LW/node forcing W under four coordinate TMYs; tests cover 20 v5 annual files.",
  assumptions_and_uncertainty="This fixes the demonstrated local zero-glazed roof/stale-local defect for the inspected geometry and native radiation formula. Solar-plane transposition and annual-mean sky remain model conventions; no general package certification follows.",
  counterevidence_alternative_explanation="V4's expected-coefficient check passed without comparing implemented forcing; review03 found 129–179 W longwave response errors and surface-order dependence. V5 independently checks actual solver forcing.",
  falsification_or_weakening="A per-surface/solver mismatch, order dependence, altered frozen physical parameter or failed active field guard would invalidate the bounded correction.",
  dependencies_next_action_owner="Fresh reviewer 04: inspect v5 source/fixtures/hourly exports and unchanged-input comparison before accepting modeled effects.",
  permitted_manuscript_wording="A study-local per-surface preprocessing correction passed direct forcing and order checks for the tested 1C geometry.",
  related_claims_figures="MET-009; MET-005; RES-004; LIM-007")

add("LIM-002", "LIM-008",
  label="Sky aggregation and solar-time approximation",
  exact_claim="V5 replaces invalid missing-cloud input with valid hourly horizontal IR but retains the native annual mean of hourly air-minus-effective-sky temperature for longwave exchange, and native solar position at fixed UTC+1 interval-start labels without applying the EPW/JRC irradiation time offset. Hourly peak phase/ranking sensitivity to these approximations is not quantified.",
  kind="limitation", evidence_class="derived estimate", status="supported",
  locators=["WEATHER-PROTOCOL-V4 | 03_methods/weather_correction_protocol_v4.md annual-mean method and index/offset audit", "WEATHER-CODE-V4 | 04_analysis/ir_sky_forcing.py scalar construction", "CODE-EURECA-WEATHER | C:/Works/EUReCA/EUReCA/eureca_building/weather.py lines 195–206 native solar-position indexing", "JRC-TMY | https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/pvgis-typical-meteorological-year-tmy-generator_en irradiance offsets/local-hour EPW convention"],
  outputs=["05_simulations/weather_forcing_v4/weather_forcing_audit.json"],
  population_geography_period_comparator_units="Four 8760-hour TMY exposures, 1 h model step; sky-temperature K, solar-time offset h, annual useful kWh and peak kW endpoints.",
  assumptions_and_uncertainty="Air and IR are paired by EPW row; this does not establish exact solar phase or hourly longwave physics. No alternative timing/longwave sensitivity campaign has quantified possible changes to small effects or peak rankings.",
  counterevidence_alternative_explanation="JRC documents irradiation offsets; native pvlib index does not apply them. Valid IR handling and per-surface algebra alone cannot prove accurate hourly environmental forcing.",
  falsification_or_weakening="A benchmark or declared sensitivity experiment could quantify or narrow this limit; no current measurement or alternate-forcing run does.",
  dependencies_next_action_owner="Author/reviewer 04: retain this limitation with every interpretation of small annual and peak differences; future methods work could test phase/longwave alternatives without altering predeclared outputs here.",
  permitted_manuscript_wording="The pilot uses annual-mean sky exchange and the native solar-time convention; sensitivity of hourly peaks to those approximations remains untested.",
  related_claims_figures="MET-009; RES-007; INT-001")

add("RES-007", "RES-008",
  label="Bolzano season-onset maximum context",
  exact_claim="As a post-hoc diagnostic of the v5 Bolzano hourly files, seeded, early and late annual maxima are identical at 17.8086 kW in hour 6912 (16 October 00:00), when the imposed lower setpoint changes from 0°C in the preceding hour to 19°C and old zone air is 9.641°C. The common season-start recovery can dominate this annual-maximum endpoint; zero paired peaks do not show zero intraday timing response or equipment sizing need.",
  kind="result", evidence_class="simulation output", status="partially supported",
  locators=["RUN-BOLZANO-V5 | 05_simulations/climate_campaign_v5/Bolzano/seeded_hourly.csv hours 6911–6912", "RUN-BOLZANO-V5 | 05_simulations/climate_campaign_v5/Bolzano/early_hourly.csv hours 6911–6912", "RUN-BOLZANO-V5 | 05_simulations/climate_campaign_v5/Bolzano/late_hourly.csv hours 6911–6912", "AUTHOR-POSTHOC-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json all 12 peak rows/prior setpoints", "INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json Bolzano peak rows and preceding states"],
  outputs=["06_claim_drafts/author_v5_posthoc_peak_check_v03.json", "05_simulations/climate_campaign_v5/Bolzano/summary.json"],
  population_geography_period_comparator_units="One assumed zone under Bolzano-coordinate TMY; three schedules; hour 6912, kW and °C; post-hoc explanatory diagnostic of predeclared annual maximum.",
  assumptions_and_uncertainty="The common seasonal mask is deliberate but not a local regulatory/occupancy estimate. No peak was excluded; an alternate-season or sizing analysis was not performed.",
  counterevidence_alternative_explanation="Other hours may show profile-specific differences, but the prescribed season restart sets the annual maximum in this case.",
  falsification_or_weakening="A different raw peak row, setpoint transition or old air temperature would invalidate this diagnostic; future alternate-season modeling could change its relevance.",
  dependencies_next_action_owner="Fresh reviewer 04: verify peak context directly; author: keep primary annual maximum and label this as post-hoc explanation rather than a new endpoint.",
  permitted_manuscript_wording="[POST-HOC V5 PEAK CONTEXT.] Bolzano's identical annual maxima occur at the shared imposed heating-season restart, not at a profile-specific daytime switch.",
  related_claims_figures="RES-007; LIM-008")

patch = {
  "prior_draft_sha256": hashlib.sha256((HERE / "article_claims_v02.md").read_bytes()).hexdigest(),
  "review_sha256": hashlib.sha256((ROOT / "07_reviews/review_03.md").read_bytes()).hexdigest(),
  "status_note": "Status: review-03 author response, 20 October 2026. All 66 pre-v5 annual executions are quarantined; v5's 20 annual runs plus one separate non-sample replay provide conditional fixed-case model contrasts after independent source/forcing/state/arithmetic verification. Physical results remain candidate pending fresh independent review 04. No empirical validation, population inference, system sizing or broad method novelty is claimed. Exact source aliases/locators appear in 02_evidence/source_ledger.md and local_source_ledger.md; new v5 aliases require registry extension.",
  "updates": updates,
  "new_claims": new,
}
target = HERE / "revision_03.json"
if target.exists():
    raise SystemExit(f"Refusing overwrite: {target}")
target.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"updates": len(updates), "new_claims": [c["id"] for c in new], "patch": str(target)}))
