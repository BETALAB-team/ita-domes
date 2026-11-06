# Article claim checker v01

Status: review-01 response, 20 October 2026. Original Padova v1 and climate v2 load effects are quarantined after the confirmed hourly state-transition defect. Twenty corrected v3 annual runs provide candidate four-site effects with author arithmetic and full-step checks; scientific interpretation awaits independent review of the local correction. No empirical heating-accuracy claim is made.

## Literature review

### LIT-001 — Conference foundation

- Exact claim: Vivian et al. published ita-domes as a survey-informed Italian residential-stock model with regional resolution in 2026 CLIMA 2025 proceedings.
- Kind: literature
- Evidence class: published finding
- Status: supported
- Provenance: CONF-001 | printed p1302; PDF p2; title/authors/abstract; DOI 10.1007/978-3-032-10546-2_123; SRC-CLIMA2026 | Springer chapter landing page, Cite this paper, First Online 1 April 2026
- Population, geography, period, comparator, and units: Italian residential stock; publication 2026; CLIMA 2025 event; no numerical comparator.
- Assumptions and uncertainty: This is the chapter's scope and bibliographic record, not new work.
- Counterevidence / alternative explanation: The chapter discusses predecessors MOIRAE and MODENA.
- What would falsify or weaken the claim: A discrepant publisher record or chapter text would require correction.
- Required next action / dependency / owner: None; author verified target pages through conference evidence inventory.
- Permitted manuscript wording: The 2026 CLIMA chapter presents ita-domes for survey-informed Italian residential-stock modelling at regional resolution.
- Related claims / proposed tables and figures: LIT-002; RES-001

### LIT-002 — Prior ISTAT schedule uncertainty

- Exact claim: Prataviera et al. (2022) already reconstruct heating operation from the three ISTAT time bands, use randomized profiles, propagate operational input uncertainty with EUReCA, and compare selected Verona hourly heating with district-heat measurements.
- Kind: literature
- Evidence class: published finding
- Status: supported
- Provenance: SRC-UA2022 | full primary article Section 3.1.1 p6/Fig.2; Section 3.3.1 p11; Section 4.1.2 p12; DOI 10.1016/j.apenergy.2022.118691
- Population, geography, period, comparator, and units: Italian ISTAT-informed UBEM; Verona mixed building sample and selected 2019/20 winter periods; no result transferred numerically.
- Assumptions and uncertainty: Measurement boundary excludes upstream generation/distribution; their schedule procedure differs in aggregation and subsequent parameter sampling.
- Counterevidence / alternative explanation: The paper's abstract/body disagree on attribution of one 18%→10% example, so it is excluded.
- What would falsify or weaken the claim: Direct reinspection of cited primary pages or a corrected article could narrow this summary.
- Required next action / dependency / owner: None; maintain this precedent in novelty wording.
- Permitted manuscript wording: Three-band random reconstruction and EUReCA operational uncertainty analysis were already reported in 2022.
- Related claims / proposed tables and figures: NOV-001; GAP-001

### LIT-003 — Earlier stock scenarios

- Exact claim: MODENA used ISTAT 2013 and EUReCA and examined weather, envelope, and heating-technology scenarios with aggregate comparisons before the selected extension.
- Kind: literature
- Evidence class: published finding
- Status: supported
- Provenance: SRC-MODENA2024 | full primary BSA 2024 chapter Sections 2.2–3.2 pp158–163
- Population, geography, period, comparator, and units: Italian residential-stock modelling; scenario and aggregate scales; no current-pilot numerical comparator.
- Assumptions and uncertainty: Code lineage and simulation configuration are not assumed identical to current GRINS or the conference chapter.
- Counterevidence / alternative explanation: Aggregate comparisons do not establish individual or hourly accuracy.
- What would falsify or weaken the claim: Contrary original-method text or repository provenance could revise the lineage description.
- Required next action / dependency / owner: None; prevent climate/stock scenario novelty overclaim.
- Permitted manuscript wording: Earlier MODENA work had already examined national residential-stock scenarios using ISTAT and EUReCA.
- Related claims / proposed tables and figures: NOV-001; LIM-004

### LIT-004 — Existing Italian archetypes

- Exact claim: Khajedehi et al. (2025) identified Italian demographic user archetypes and probabilistic dwelling links from survey evidence; archetype discovery is not a new contribution of this study.
- Kind: literature
- Evidence class: published finding
- Status: partially supported
- Provenance: SRC-ARCH2025 | publisher indexed full-text Sections 2, 3.1, 4.2–4.4, 6; DOI 10.1016/j.enbuild.2025.115924
- Population, geography, period, comparator, and units: Italian national survey; demographic archetypes; no numerical archetype result imported.
- Assumptions and uncertainty: Direct full PDF fetch was unavailable; search-indexed publisher sections were inspected.
- Counterevidence / alternative explanation: Archetype heating hours are survey responses and electricity is reconstructed, not measured thermal demand.
- What would falsify or weaken the claim: Full primary PDF could reveal qualifications omitted in indexed sections.
- Required next action / dependency / owner: Literature owner: obtain complete primary text before submission; current manuscript may use bounded statement.
- Permitted manuscript wording: Italian survey-derived user archetypes had already been identified.
- Related claims / proposed tables and figures: NOV-001

## Gap and novelty

### GAP-001 — Coarse hours leave placement unresolved

- Exact claim: The selected respondent's reported mean winter-day system operating hours in three eight-hour bands do not identify unique within-band times. In this model those durations are encoded as hours at a 22°C high setpoint, not as a constraint on actual positive heating-load hours.
- Kind: gap
- Evidence class: derived estimate
- Status: supported
- Provenance: CONF-M04 | printed p1304; PDF p4; final paragraph on morning/afternoon/evening hours; CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 643–656, create_random_prof/create_prof; DATA-INPUT-AUDIT | 04_analysis/input_audit.json selected_survey_response q_3_11A/B/C=4/4/1; ISTAT-Q2013 | istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf pp41–42, questions 3.11_A/B/C, mean winter-day hours and band endpoints; RAW-VERIFY | 00_context/survey_schedule_verification.json records.raw nprog3523 and bands
- Population, geography, period, comparator, and units: One 2013 Veneto response; three daily 8 h blocks; count of admissible contiguous positions per block is 8−n+1.
- Assumptions and uncertainty: Integer/coarsened reported means are mapped to a repeated two-level 22/19°C thermostat schedule; both contiguity and daily repetition are assumptions. Heating may be delivered at 19°C setback.
- Counterevidence / alternative explanation: The survey asks mean system on-hours rather than thermostat levels; the simulation's temperature mapping is a separate assumption and does not reproduce observed burner duty.
- What would falsify or weaken the claim: Official questionnaire fields specifying exact within-band switching times would weaken the stated information gap; inspected questions 3.11_A/B/C do not.
- Required next action / dependency / owner: None for this identifiability statement; preserve raw/compact response provenance and avoid claims of observed timing.
- Permitted manuscript wording: Reported mean operating hours leave timing unidentified; our model encodes them as high-setpoint periods while allowing setback heating.
- Related claims / proposed tables and figures: MET-001; MET-002; RES-003

### GAP-002 — No established pilot-scale accuracy

- Exact claim: The CLIMA national annual final-energy comparison does not establish hourly or Padova household heating accuracy for this illustrative sensitivity pilot.
- Kind: gap
- Evidence class: published finding
- Status: supported
- Provenance: CONF-V01 | printed pp1306–1307; PDF pp6–7; Section 4.1/Table 3; CONF-W01 | printed p1306; PDF p6; Section 3.3 partial historic weather coverage; VAL-PROTOCOL | 03_methods/validation_protocol.md Sections 3–4
- Population, geography, period, comparator, and units: Published national carrier totals for 2013/14 versus proposed one-dwelling hourly sensible heating output; unlike scale and energy boundary.
- Assumptions and uncertainty: Potential compensating errors in a national aggregate; pilot uses TMY rather than matched measured year.
- Counterevidence / alternative explanation: National aggregates may hide compensating building/end-use errors, while the current pilot has a different year/weather and energy boundary.
- What would falsify or weaken the claim: Independent Padova heating observations matched in building, weather, interval and end-use boundary could support a separate accuracy claim; the chapter national totals alone cannot.
- Required next action / dependency / owner: Validation owner: identify compatible independent meter series; otherwise retain missing-validation marker.
- Permitted manuscript wording: The published aggregate check cannot validate the present dwelling-scale hourly output.
- Related claims / proposed tables and figures: RES-001; LIM-002

### NOV-001 — Bounded candidate contribution

- Exact claim: The proposed contribution is to quantify sensitivity to three specified duration-preserving contiguous placements within a fixed illustrative EUReCA case; novelty of the exact comparison remains provisional.
- Kind: novelty
- Evidence class: hypothesis
- Status: unverified
- Provenance: SRC-UA2022 | Section 3.1.1 p6, prior three-band random reconstruction; RQ-FREEZE | 01_plan/research_questions.md primary RQ and contribution map; LIT-SEARCH | 02_evidence/literature_search_log.md batches 11–12 and search limits
- Population, geography, period, comparator, and units: One illustrative Veneto 2013 response and fixed dwelling; Padova TMY first, further climates gated; kWh and kW endpoints.
- Assumptions and uncertainty: Targeted narrative search is neither exhaustive nor proof that no exact prior comparison exists.
- Counterevidence / alternative explanation: Prior work already combines ISTAT time bands and randomization; generic method novelty is false.
- What would falsify or weaken the claim: A primary study of this exact contrast or a failed technical pilot would narrow or negate contribution.
- Required next action / dependency / owner: Literature owner: targeted exact-comparison search before submission; methods owner: run paired experiment.
- Permitted manuscript wording: We test the consequences of specified within-block placement choices under controlled assumptions.
- Related claims / proposed tables and figures: LIT-002; RQ-001; RQ-002

## Research questions

### RQ-001 — Primary sensitivity question

- Exact claim: For one illustrative dwelling, one selected Veneto survey response and Padova-coordinate TMY, compare annual useful sensible heating and maximum hourly load under three placements of the same 4/4/1 encoded 22°C high-setpoint hours; a 19°C setback permits heating in other hours.
- Kind: method
- Evidence class: planned test
- Status: partially supported
- Provenance: RQ-FREEZE | 01_plan/research_questions.md primary RQ and fixed-case boundary; ISTAT-Q2013 | istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf pp41–42 reported operating-hour bands; CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625, 643–656 high/setback mapping; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json candidate corrected outputs
- Population, geography, period, comparator, and units: One illustrative case; 8760 hourly steps; Padova-coordinate TMY; baseline seeded versus early/late; kWh and kW.
- Assumptions and uncertainty: The 4/4/1 encoded high-setpoint hours are fixed; actual positive-load hours and duration are outputs and need not equal 9 h/day. The physical case remains illustrative.
- Counterevidence / alternative explanation: A null difference is possible; neither direction is presupposed.
- What would falsify or weaken the claim: Run failure or violation of paired invariants makes comparison uninterpretable.
- Required next action / dependency / owner: Simulation owner completed corrected Padova run and full-step checks; reviewer 02 must independently inspect local correction and outputs.
- Permitted manuscript wording: We compare three placements of equal encoded high-setpoint periods under one assumed case.
- Related claims / proposed tables and figures: MET-002; RES-004; FIG-02

### RQ-002 — Conditional weather question

- Exact claim: Compare the same encoded high-setpoint placement contrasts under homogeneous TMY weather for Padova, Bolzano, Roma and Palermo coordinates, holding building, respondent, thermostat levels and climate-E season fixed; corrected annual climate outputs require review.
- Kind: method
- Evidence class: planned test
- Status: partially supported
- Provenance: RQ-FREEZE | 01_plan/research_questions.md Secondary RQ1; CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json four PVGIS API requests; CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv HDD18/CDD26/GHI/RH; RUN-CLIMATE-V3 | 04_analysis/paired_schedule_contrasts_v3.csv all four-site candidate contrasts; AUTHOR-CHECK-V01 | 06_claim_drafts/author_result_check_v01.json independent integration and table checks
- Population, geography, period, comparator, and units: One unchanged case across candidate sites; 2–4 further files only if selected; paired kWh/kW effects.
- Assumptions and uncertainty: Weather selection is based on computed descriptors, not climate labels; all other inputs are fixed. The old v2 climate loads are quarantined because they used the flawed dispatcher.
- Counterevidence / alternative explanation: Weather can change annual load without changing relative placement sensitivity; no trend assumed.
- What would falsify or weaken the claim: If comparable weather files cannot be sourced or Padova checks fail, this question remains unanswered.
- Required next action / dependency / owner: Corrected v3 multi-weather runs and author arithmetic check completed; reviewer 02 to assess adapter, residual gate and conditional interpretation.
- Permitted manuscript wording: [CANDIDATE WEATHER COMPARISON, INDEPENDENT CORRECTION REVIEW PENDING.]
- Related claims / proposed tables and figures: RES-005; FIG-03

## Data

### DAT-001 — Illustrative survey response

- Exact claim: Raw Istat 2013 response nprog 3523 and compact row id 3523 agree on Veneto code 5 and mean winter-day morning/afternoon/night reported heating hours 4/4/1; the compact row is the lowest eligible id under the declared pilot filter.
- Kind: result
- Evidence class: survey response
- Status: supported
- Provenance: DATA-INPUT-AUDIT | 04_analysis/input_audit.json selected_survey_response, eligible_veneto_count=372, selection, survey SHA256 a7d6b19540b5b68505843494abca4c4575c00e00684d2c5a71c00e692963f40f; DATA-AUDIT-SCRIPT | 04_analysis/audit_inputs.py survey filter/selection; RAW-VERIFY | 00_context/survey_schedule_verification.json records.raw/compact, field_match_checks, bands; raw SHA256 500d80dbbcdab6b4384a9e8547167855d3c9bd0bd7909b8f8be3bc4f21212e37; ISTAT-Q2013 | istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf pp40–43, questions 3.10–3.13; ISTAT-REG2013 | istat/ECH_2013_IT/METADATI/Classificazioni/ECH_Classificazione_Anno 2013_var89.html, code 5 Veneto
- Population, geography, period, comparator, and units: One raw/compact matched 2013 release response; reported mean winter-day on-hours, not a named winter or measured daily schedule; unweighted selection.
- Assumptions and uncertainty: Raw q_2_32C/D are blank while compact fields are zero; this is preprocessing, although neither representation triggers the source fireplace branch. q_3_13_2=1 denotes timer, not thermostat or valves; source-model temperatures are assumptions.
- Counterevidence / alternative explanation: There are 372 eligible Veneto rows under filter; row choice can influence output and is not a regional estimate.
- What would falsify or weaken the claim: Mismatch on script rerun, record hash, or codebook wording would require correction.
- Required next action / dependency / owner: Simulation owner: document raw-blank to compact-zero preprocessing and verify it does not affect this run; author: retain reported-mean wording.
- Permitted manuscript wording: An illustrative Veneto survey response reports 4/4/1 mean winter-day on-hours in the three bands; the hourly repetition is modeled.
- Related claims / proposed tables and figures: GAP-001; MET-002; LIM-001

### DAT-002 — Padova-coordinate weather input

- Exact claim: The Padova-coordinate EPW used in the initial and corrected pilot is byte-identical (SHA256 390ead08…) to a newly retrieved PVGIS TMY EPW for 45.408°N, 11.873°E; provider metadata specify ERA5 meteorology, SARAH3 radiation, 2005–2023 selection window and horizon inclusion.
- Kind: result
- Evidence class: derived estimate
- Status: supported
- Provenance: CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json Padova JSON/EPW API URL, retrieval timestamp and hashes; CLIMATE-METADATA-V2 | 05_simulations/climate_inputs_v2/audited_metadata.json Padova inputs.meteo_data and tmy_months_selected; DATA-INPUT-AUDIT | 04_analysis/input_audit.json original local Padova EPW hash
- Population, geography, period, comparator, and units: Padova-coordinate typical-year candidate, 8760 h; not a historical 2013/14 meter-matched weather series.
- Assumptions and uncertainty: The newly retrieved file is byte-identical to the earlier local input; this supports weather content/provenance now, not the original file's acquisition history. TMY selected months are not a historical measured year.
- Counterevidence / alternative explanation: Weather mismatch may affect results; TMY is unsuitable for historical empirical validation.
- What would falsify or weaken the claim: A corrupt EPW, duplicate/missing timestamps or contrary origin metadata would invalidate or relabel this input.
- Required next action / dependency / owner: Reviewer to verify manifest/API metadata; retain grid/horizon and TMY limits.
- Permitted manuscript wording: The Padova TMY input matches a documented PVGIS retrieval with ERA5 meteorology and SARAH3 radiation for a 2005–2023 selection window.
- Related claims / proposed tables and figures: MET-003; LIM-002

### DAT-003 — Declared illustrative physical case

- Exact claim: The planned dwelling is an assumed 80 m², 240 m³ rectangular zone with predeclared envelope, glazing, infiltration and internal gains, not a sampled or measured Padova dwelling.
- Kind: method
- Evidence class: planned test
- Status: supported
- Provenance: SIM-PROTOCOL | 03_methods/simulation_protocol.md Physical model and departures; 10×8×3 m, U-values, gains and ACH
- Population, geography, period, comparator, and units: Single idealized dwelling; Padova-coordinate exposure; floor area m² and volume m³.
- Assumptions and uncertainty: Assumed geometry and thermal parameters; no uncertainty distribution assigned in the minimum pilot.
- Counterevidence / alternative explanation: Different envelope, gains or infiltration could change schedule sensitivity.
- What would falsify or weaken the claim: Run configuration differing from protocol would require amendment and new claim wording.
- Required next action / dependency / owner: Simulation owner froze design.json and completion manifest; reviewer to compare adapter implementation with declared assumptions.
- Permitted manuscript wording: An illustrative single-zone dwelling is used to isolate schedule placement under fixed physical assumptions.
- Related claims / proposed tables and figures: MET-003; LIM-001

### DAT-004 — Measurement inventory boundary

- Exact claim: The inspected Padova GIS and related local files have not yielded a compatible independently documented building meter time series for this pilot.
- Kind: limitation
- Evidence class: derived estimate
- Status: partially supported
- Provenance: DATA-INPUT-AUDIT | 04_analysis/input_audit.json GIS_inventory property keys for LivingLab and PaduaDigital files; VAL-PROTOCOL | 03_methods/validation_protocol.md Section 3 empirical validation
- Population, geography, period, comparator, and units: Inspected local directories only; no inference that all Padova measurements are nonexistent.
- Assumptions and uncertainty: This is a bounded inventory audit of inspected local GIS/directories, not an observed-measurement dataset and not proof that no compatible meter series exists elsewhere.
- Counterevidence / alternative explanation: An external owner may have compatible monitored data outside inventory.
- What would falsify or weaken the claim: Receipt of a documented independent time series with matching building/period/end use would overturn the current availability statement.
- Required next action / dependency / owner: Validation owner: seek meter provenance/access if empirical validation becomes feasible.
- Permitted manuscript wording: The inspected local files did not identify an independently documented meter series compatible with this pilot.
- Related claims / proposed tables and figures: GAP-002; LIM-002

## Methodological claims

### MET-001 — Source reconstruction

- Exact claim: Current GRINS EndUse.py converts each reported band duration into one seeded contiguous binary profile segment, then maps binary 1/0 to assumed 22/19°C high/setback setpoints for this timer-selected respondent; positive heating is permitted at the setback setpoint.
- Kind: method
- Evidence class: derived estimate
- Status: supported
- Provenance: CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625 and 643–656, SHA256 cf7442f0b2ddb39b338c2047a1b95016d3a4d246604da15cd49c9a823cc436d5; ISTAT-Q2013 | questionnaire p43, question 3.13 option _2 timer; DISPATCH-DIAGNOSTIC | 07_reviews/reviewer_01/independent_checks.json positive heat at 19°C in archived outputs; figures quarantined
- Population, geography, period, comparator, and units: Source implementation; daily 8-hour bands 05–13, 13–21, 21–05; integer hours 0–8.
- Assumptions and uncertainty: The source 22/19°C values are not measured respondent setpoints. The binary profile represents the modeled high-setpoint choice, not equipment availability or actual burner runtime.
- Counterevidence / alternative explanation: This local implementation is not proven bit-identical to the 2026 published pipeline.
- What would falsify or weaken the claim: Different repository revision or source behavior on execution would require method revision.
- Required next action / dependency / owner: Simulation owner: test extracted routine and record exact resulting 24 h profiles.
- Permitted manuscript wording: The inspected source places a contiguous high-setpoint segment in each reported time band and allows heating at the lower 19°C setting outside it.
- Related claims / proposed tables and figures: GAP-001; MET-002; LIM-004

### MET-002 — Duration-preserving paired alternatives

- Exact claim: The three modeled alternatives preserve the same 4/4/1 encoded daily high-setpoint hours in their respective bands, varying only contiguous placement; they do not preserve the duration of actual heating delivery at either setpoint.
- Kind: method
- Evidence class: planned test
- Status: supported
- Provenance: SIM-PROTOCOL | 03_methods/simulation_protocol.md baseline and alternatives; CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625 and 643–656; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json daily_on_profile
- Population, geography, period, comparator, and units: One response, same 24 h daily block durations; source seeded versus earliest/latest; hours.
- Assumptions and uncertainty: 22°C applies during encoded high periods and 19°C outside within modeled season; ideal heating can run under either setting. Earliest/latest are alternatives, not extrema.
- Counterevidence / alternative explanation: Other admissible schedules exist; early/late are not global extrema of energy response.
- What would falsify or weaken the claim: A block-total mismatch, date-rotation bug or hidden physical-input difference invalidates pairing.
- Required next action / dependency / owner: Three profile outputs and block checks saved; reviewer to inspect night rotation and source equivalence.
- Permitted manuscript wording: The comparison fixes high-setpoint duration within each reported band while changing its placement; actual heating runtime remains free to vary.
- Related claims / proposed tables and figures: RQ-001; RES-003; LIM-003

### MET-003 — Explicit EUReCA 1C adapter

- Exact claim: A bounded local single-commit 1C adapter now uses unchanged EUReCA sensible_balance_1C algebra but evaluates free and constrained candidates from one prior state and commits only one accepted hourly state; this corrects the observed transition defect in the archived adapter without changing external checkouts.
- Kind: method
- Evidence class: planned test
- Status: partially supported
- Provenance: CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md scope and gates; CORRECTION-CODE-V3 | 05_simulations/climate_campaign_v3/Padova/single_commit_1c.py select_solution/solve_1c_single_commit, SHA256 e549b4e5645ad8ae159fddf644d225d28622e1495f6fd64768c5242389b82c16; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/completion_manifest.json correction/source hashes
- Population, geography, period, comparator, and units: One assumed zone, 8760 h, ideal sensible load; no stock aggregation.
- Assumptions and uncertainty: The correction supports only this ideal sensible 1C case with no mechanical/natural ventilation or latent gains; it is not a general EUReCA systems replacement, conference reproduction or empirical calibration.
- Counterevidence / alternative explanation: Archived v1/v2 recursive dispatcher violated state balance at mode changes; direct pure-network algebra checks did not catch it.
- What would falsify or weaken the claim: If adapter imports unexpected checkout, model or config, methodological link fails.
- Required next action / dependency / owner: Reviewer 02 to independently inspect local correction, transition tests and annual residual checker; external benchmark and measured accuracy remain absent.
- Permitted manuscript wording: A local study-specific adapter selects one hourly 1C solution from the same prior state and commits it once; its validity remains under independent review.
- Related claims / proposed tables and figures: LIM-004; RES-004

### MET-004 — Output accounting

- Exact claim: Annual useful sensible heating is defined as Σ_t max(P_t,0)Δt/1000 kWh at Δt=1 h and peak as max_t max(P_t,0)/1000 kW, with P_t in W.
- Kind: method
- Evidence class: planned test
- Status: supported
- Provenance: SIM-PROTOCOL | 03_methods/simulation_protocol.md Accounting and uncertainty; CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md unchanged primary outputs; AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json independently integrated corrected CSVs
- Population, geography, period, comparator, and units: One simulated zone, hourly 8760-step year; W→kWh and W→kW conversion.
- Assumptions and uncertainty: Ideal unlimited heating, constant gains; demand excludes system efficiency, fuel, DHW and appliance electricity.
- Counterevidence / alternative explanation: Peak depends on temporal resolution; annual useful demand cannot be compared directly to final-energy BEN totals.
- What would falsify or weaken the claim: Recalculation from saved hourly W differing from summary invalidates result.
- Required next action / dependency / owner: Author independently reintegrated corrected Padova CSVs; reviewer 02 to inspect meaning of sensible_zone_load and state residuals.
- Permitted manuscript wording: Useful sensible heating demand and maximum hourly load are the only primary energy/load endpoints.
- Related claims / proposed tables and figures: RES-004; LIM-002

### MET-005 — Technical pilot checks

- Exact claim: The archived v1/v2 technical screen passed but failed a subsequently applied full-step state-balance check; a new v3 Padova gate reports zero residual failures above 1e-6 W, one-state continuity, exact repeat, warm-up stability and 8760 finite rows, pending independent review of the correction/checker.
- Kind: method
- Evidence class: simulation output
- Status: partially supported
- Provenance: RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/transition_checks_pre_run.json five mode tests; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/full_step_balance_checks.json all five cases and maximum residuals; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/checks.json repeat/warm-up/finite screens; AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json independent CSV arithmetic; DISPATCH-DIAGNOSTIC | 07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json original v1 transition residuals
- Population, geography, period, comparator, and units: Five planned annual runs; 168 h baseline/early/late/repeat and 336 h baseline; prescribed numerical tolerances.
- Assumptions and uncertainty: The declared 1e-6 W full-step screen checks numerical network consistency for this one ideal sensible adapter. The old gate omitted transition balance and is superseded; no observed accuracy follows.
- Counterevidence / alternative explanation: Reviewer 01 found 898/846/766 failing seeded/early/late v1 hours and residuals up to 9.65 kW; old repeat/integration passes were inadequate.
- What would falsify or weaken the claim: Independent reviewer detection of state mismatch, incorrect forcing/state export, or non-matched run setup would withdraw the candidate gate.
- Required next action / dependency / owner: Reviewer 02 must inspect v3 code, transition trials and full-step residual arithmetic; do not present v1/v2 effects as evidence.
- Permitted manuscript wording: The corrected Padova adapter passed its expanded internal numerical gates, subject to independent review; these checks do not establish measured accuracy.
- Related claims / proposed tables and figures: RES-003; RES-004; LIM-002

### MET-006 — Climate expansion gate

- Exact claim: Homogeneous PVGIS TMY inputs for Padova, Bolzano, Roma and Palermo coordinates have been retrieved and characterized consistently; only corrected v3 annual runs with full-step checks can answer the weather-interaction question.
- Kind: method
- Evidence class: planned test
- Status: partially supported
- Provenance: CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json four retrievals, URLs/hashes; CLIMATE-METADATA-V2 | 05_simulations/climate_inputs_v2/audited_metadata.json same ERA5/SARAH3/window/horizon setup; CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv HDD18/CDD26/GHI/RH; CORRECTION-PROTOCOL-V3 | 03_methods/correction_protocol_v3.md quarantine and corrected campaign gate
- Population, geography, period, comparator, and units: Candidate Bolzano/Roma/Palermo coordinates are provisional; 2–4 sites maximum initial plan; Kd, kWh/m² and % indicators.
- Assumptions and uncertainty: These are grid-coordinate TMYs, not historical measured weather; Bolzano/Roma/Palermo are controlled weather contrasts, not representative local household stocks. Fixed climate-E operating season deliberately holds a nonlocal assumption constant.
- Counterevidence / alternative explanation: Heterogeneous TMY source/vintage could confound weather contrast.
- What would falsify or weaken the claim: If same-source metadata unavailable or descriptors fail to provide contrast, no climate expansion claim.
- Required next action / dependency / owner: Twenty corrected v3 annual runs and their declared gates completed; author arithmetic checked. Reviewer 02 must independently assess the adapter and weather contrasts.
- Permitted manuscript wording: Four consistently sourced coordinate TMYs support a controlled weather comparison; corrected effect estimates await independent review.
- Related claims / proposed tables and figures: RQ-002; RES-005

### MET-007 — Validation separation

- Exact claim: Implementation checks, model benchmarks, empirical meter comparison, aggregate external checks and robustness must be reported as distinct evidentiary layers; no new empirical validation is part of the current pilot.
- Kind: method
- Evidence class: planned test
- Status: supported
- Provenance: VAL-PROTOCOL | 03_methods/validation_protocol.md Sections 1–5; VAL-MATRIX | 03_methods/validation_matrix.md full claim/output rows
- Population, geography, period, comparator, and units: Current illustrative hourly useful-demand pilot; no independent observed Padova heating reference identified.
- Assumptions and uncertainty: If observations arise, calibration and untouched validation sets must be separated and weather/end-use boundaries matched.
- Counterevidence / alternative explanation: Published national BEN table and internal checks are different evidence classes.
- What would falsify or weaken the claim: A claimed observed-accuracy metric without independent aligned observations violates this method.
- Required next action / dependency / owner: Validation owner: preserve missing layers and data-request specification.
- Permitted manuscript wording: The analysis reports verification separately from empirical validation; the latter remains unavailable for this case.
- Related claims / proposed tables and figures: GAP-002; LIM-002

### MET-008 — Single-commit correction gate

- Exact claim: A study-specific local 1C adapter must select off/heating/cooling from candidates evaluated against the same beginning-of-hour state, commit once, and pass explicit mode-transition and every-hour air/surface/mass residual checks before annual outputs are admitted.
- Kind: method
- Evidence class: derived estimate
- Status: partially supported
- Provenance: CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md pre-run and annual gates; CORRECTION-CODE-V3 | 05_simulations/climate_campaign_v3/Padova/single_commit_1c.py source; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/transition_checks_pre_run.json and full_step_balance_checks.json
- Population, geography, period, comparator, and units: One ideal sensible 1C zone, hourly, no mechanical/natural ventilation or latent gains; 1e-6 W numerical residual tolerance.
- Assumptions and uncertainty: Adapter uses the unchanged pure EUReCA network solver with bounded local state selection; checker must be independent of engine and use exported old/current states.
- Counterevidence / alternative explanation: Passing network equations does not verify envelope truth, weather representativeness, software generality or measured accuracy.
- What would falsify or weaken the claim: Reviewer-discovered state mismatch or checker dependence on incorrect saved states would defeat this gate.
- Required next action / dependency / owner: Reviewer 02: inspect candidate selection, one-commit behavior and independent full-step checker.
- Permitted manuscript wording: The corrected pilot applies a one-state-commit hourly 1C selector and full-step balance checks; independent review remains pending.
- Related claims / proposed tables and figures: MET-003; MET-005; RES-004

## Results

### RES-001 — Published national comparison

- Exact claim: The CLIMA paper's Table 3 reports 2013/2014 national final-energy comparisons for six carriers; for example natural gas is 18,090 versus 18,073 ktoe in 2013 and 17,115 versus 15,151 ktoe in 2014 (model versus BEN).
- Kind: result
- Evidence class: published finding
- Status: supported
- Provenance: CONF-V01 | printed p1307; PDF p7; Table 3 national final-energy totals; CONF-NUM-CHECK | 00_context/conference_evidence.md Table 3; recalculated deviations +0.0941% and +12.9628%
- Population, geography, period, comparator, and units: Italian national annual final energy by carrier; model versus BEN; 2013 and 2014; ktoe.
- Assumptions and uncertainty: Published numbers are transcribed and arithmetic-checked, not newly reproduced; paper's matching of carrier boundaries requires more detail.
- Counterevidence / alternative explanation: 2014 gas discrepancy is substantial; aggregate agreement in some carriers can conceal offsetting component errors.
- What would falsify or weaken the claim: A table transcription or original-pdf discrepancy would alter the example, not establish new validation.
- Required next action / dependency / owner: None for citation; no inheritance of these values as pilot results.
- Permitted manuscript wording: The chapter reports national carrier comparisons, including a larger modeled natural-gas total than BEN in 2014.
- Related claims / proposed tables and figures: GAP-002; LIM-002; TAB-01

### RES-002 — Published regional maps

- Exact claim: The CLIMA paper presents modeled regional annual end-use and carrier totals as GWh maps, without regional measured comparator or uncertainty intervals in the figures.
- Kind: result
- Evidence class: published finding
- Status: supported
- Provenance: CONF-R01 | printed p1308; PDF p8; Fig.1 four end-use panels and captions; CONF-R02 | printed p1309; PDF p9; Fig.2 four carrier panels and captions
- Population, geography, period, comparator, and units: Italian regions, annual modeled GWh totals; panel colour scales differ.
- Assumptions and uncertainty: Maps show totals rather than per-household intensity or measured consumption.
- Counterevidence / alternative explanation: Regional distribution may reflect household population as well as climate/stock; map pattern does not isolate cause.
- What would falsify or weaken the claim: A supplementary original numerical table/validation source could expand support; none in inspected chapter.
- Required next action / dependency / owner: None; keep as conference context only.
- Permitted manuscript wording: The chapter maps modeled regional GWh totals by end use and carrier.
- Related claims / proposed tables and figures: LIT-001; LIM-001

### RES-003 — Checked profile invariants

- Exact claim: The corrected Padova v3 seeded, early and late binary profiles each encode 4/4/1 high-setpoint hours in the 05–13, 13–21 and 21–05 bands; this does not constrain positive heating-load hours at the 19°C setback.
- Kind: result
- Evidence class: simulation output
- Status: supported
- Provenance: RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json daily_on_profile; AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json profile-count and CSV checks; CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 599–608 high/setback mapping
- Population, geography, period, comparator, and units: One illustrative response; three 8 h blocks; hours; 8760 h annual sequence.
- Assumptions and uncertainty: Reported average hours are coarsened survey responses; the encoded schedule repeats them and maps them to 22°C rather than enforcing equipment availability.
- Counterevidence / alternative explanation: Different contiguous placements and noncontiguous real schedules remain admissible.
- What would falsify or weaken the claim: A disagreement in raw response, profile counts, CSV row count, or frozen run hashes would invalidate this result.
- Required next action / dependency / owner: Reviewer to verify source-to-run mapping; author to preserve encoded-versus-observed distinction.
- Permitted manuscript wording: The corrected run inputs retain the 4/4/1 encoded high-setpoint durations; heating can also occur at setback.
- Related claims / proposed tables and figures: MET-002; FIG-01

### RES-004 — Corrected Padova candidate outputs

- Exact claim: The corrected single-commit v3 Padova run reports seeded/early/late annual useful sensible heating of 5947.73/6078.71/5988.12 kWh and maximum hourly loads of 8.870/9.019/8.656 kW; relative to seeded, early changes annual demand by +2.20% and peak by +1.68%, while late changes them by +0.68% and −2.41%. These are candidate model outputs pending independent correction review, not accepted empirical findings.
- Kind: result
- Evidence class: simulation output
- Status: partially supported
- Provenance: RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json corrected seeded/early/late; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/completion_manifest.json frozen hashes and imported EUReCA path; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/full_step_balance_checks.json all-hour network residual checks; AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json independently reintegrated CSVs/deltas
- Population, geography, period, comparator, and units: One dwelling/response/Padova-coordinate TMY; seeded baseline versus early/late; kWh and kW.
- Assumptions and uncertainty: One deterministic assumed case under a TMY. Percent denominator is corrected seeded output. Values remain conditional on adapter correctness and independent review; no population or empirical confidence interval.
- Counterevidence / alternative explanation: The original v1/v2 annual/peak figures were produced by a dispatcher that failed state balance and are quarantined. Corrected values differ materially, including reversal of annual-effect sign, demonstrating why old values cannot be used.
- What would falsify or weaken the claim: Independent review finding an incorrect single-commit transition, residual checker, source mapping, or run hash would withdraw the candidate numbers.
- Required next action / dependency / owner: Reviewer 02 to independently verify the corrected code, hourly states/loads and gates; author to promote or withdraw after verdict.
- Permitted manuscript wording: [CANDIDATE CORRECTED RESULT, INDEPENDENT REVIEW PENDING: single-case Padova annual and peak schedule-placement contrasts.]
- Related claims / proposed tables and figures: RQ-001; MET-004; FIG-02

### RES-005 — Corrected weather-specific annual effects

- Exact claim: In corrected v3 matched runs, early/late placements changed annual useful sensible heating relative to each site's seeded profile by +2.20/+0.68% Padova, +0.41/+0.18% Bolzano, +5.90/+1.44% Roma and +6.64/+1.81% Palermo; these conditional modeled effects await independent correction review.
- Kind: result
- Evidence class: simulation output
- Status: partially supported
- Provenance: RUN-CLIMATE-V3 | 04_analysis/paired_schedule_contrasts_v3.csv annual_sensible_heating_kWh and delta_annual_percent for 12 primary cases; RUN-CLIMATE-V3 | 05_simulations/climate_campaign_v3/{Padova,Bolzano,Roma,Palermo}/completion_manifest.json and full_step_balance_checks.json per site; AUTHOR-CHECK-V01 | 06_claim_drafts/author_result_check_v01.json independent hourly CSV sums and paired table checks; CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv consistent input contrast
- Population, geography, period, comparator, and units: Same assumed dwelling and one Veneto response under four PVGIS TMY coordinate files; within-site early/late minus seeded, percent denominator same-site seeded kWh.
- Assumptions and uncertainty: All physical inputs and climate-E operating season are fixed; sites are weather exposures, not local household stock. One deterministic year per site; no population interval.
- Counterevidence / alternative explanation: Weather descriptors co-vary in temperature, solar and humidity; this design cannot attribute between-site effect differences to one weather variable. Archived v2 climate outputs are excluded.
- What would falsify or weaken the claim: Independent reviewer finding a corrected-state, input equality, weather provenance, or arithmetic error would withdraw these candidate effects.
- Required next action / dependency / owner: Reviewer 02 to inspect v3 local correction and all-site gates; author to promote or withdraw after verdict.
- Permitted manuscript wording: [CANDIDATE CORRECTED RESULT: annual placement effects differ across four fixed-input TMY exposures; independent review pending.]
- Related claims / proposed tables and figures: RQ-002; MET-006; FIG-03

### RES-006 — Post-hoc setback heating diagnostic

- Exact claim: In corrected v3 Padova seeded output, the ideal heater has positive sensible load in 671 of 2730 calendar hours at the modeled 19°C setback, delivering 893.22 kWh there; this is a post-hoc description showing encoded 4/4/1 high-setpoint hours do not fix actual heating-operation duration.
- Kind: result
- Evidence class: simulation output
- Status: partially supported
- Provenance: POSTHOC-PARTITION-V3 | 04_analysis/posthoc_setpoint_partition_v3.csv Padova seeded 19°C row and tolerance 1e-6 W; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/seeded_hourly.csv sensible_load_W and heating_setpoint_C; AUTHOR-CHECK-V01 | 06_claim_drafts/author_corrected_check_v01.json corrected summary arithmetic; AUTHOR-POSTHOC-V01 | 06_claim_drafts/author_posthoc_check_v01.json independent 36-row CSV partition check
- Population, geography, period, comparator, and units: One assumed dwelling under Padova TMY; 19°C setpoint calendar hours 2730, positive-load hours 671, useful heating 893.22 kWh; not a predeclared endpoint.
- Assumptions and uncertainty: Setpoint partition uses >1e-6 W positive-load threshold. 4/4/1 is a model high-setpoint profile, not observed control or burner timing.
- Counterevidence / alternative explanation: A different thermostat interpretation or weather could change the diagnostic; the original v1 diagnostic is quarantined.
- What would falsify or weaken the claim: Mismatched hourly partition or incorrect corrected source would withdraw the quantity.
- Required next action / dependency / owner: Reviewer 02 to inspect post-hoc computation and corrected hourly file; author to retain descriptive label.
- Permitted manuscript wording: A post-hoc corrected-run check found heating at the 19°C modeled setback, so encoded high-setpoint hours do not equal active heating hours.
- Related claims / proposed tables and figures: MET-001; MET-002; RES-004

### RES-007 — Corrected weather-specific peak effects

- Exact claim: In corrected v3 matched runs, early/late placements changed maximum hourly sensible heating load relative to each site's seeded profile by +1.68/−2.41% Padova, −1.39/+2.29% Bolzano, +8.17/+2.70% Roma and +6.11/+0.34% Palermo; these deterministic modeled effects await independent correction review.
- Kind: result
- Evidence class: simulation output
- Status: partially supported
- Provenance: RUN-CLIMATE-V3 | 04_analysis/paired_schedule_contrasts_v3.csv peak_sensible_heating_kW and delta_peak_percent for 12 primary cases; RUN-CLIMATE-V3 | 05_simulations/climate_campaign_v3/{Padova,Bolzano,Roma,Palermo}/full_step_balance_checks.json each site; AUTHOR-CHECK-V01 | 06_claim_drafts/author_result_check_v01.json independent peak and paired percentage checks
- Population, geography, period, comparator, and units: Same assumed dwelling and one Veneto response under four PVGIS TMY coordinates; within-site early/late minus seeded peak; percent denominator seeded peak kW.
- Assumptions and uncertainty: Hourly maximum is sensitive to 1 h resolution and fixed season; no observed peak or probabilistic uncertainty interval exists.
- Counterevidence / alternative explanation: Different weather timing, envelope, setpoint mapping or control could alter peak sign/magnitude; archived v2 peaks are excluded.
- What would falsify or weaken the claim: Independent reviewer finding a corrected-state, peak extraction or input-pairing error would withdraw candidate effects.
- Required next action / dependency / owner: Reviewer 02 to check correction and peak outputs; author to promote or withdraw after verdict.
- Permitted manuscript wording: [CANDIDATE CORRECTED RESULT: hourly peak placement effects vary across four TMY exposures; independent review pending.]
- Related claims / proposed tables and figures: RQ-002; RES-005; FIG-03

## Interpretation

### INT-001 — Permitted sensitivity interpretation

- Exact claim: After correction, the candidate matched outputs vary with encoded high-setpoint placement in one assumed dwelling under each of four coordinate TMYs; archived v1/v2 effect signs and magnitudes remain invalid because of the old transition defect.
- Kind: interpretation
- Evidence class: simulation output
- Status: partially supported
- Provenance: DISPATCH-DIAGNOSTIC | 07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json old state-balance failures; RUN-CLIMATE-V3 | 04_analysis/paired_schedule_contrasts_v3.csv corrected four-site candidate outcomes; AUTHOR-CHECK-V01 | 06_claim_drafts/author_result_check_v01.json author arithmetic verification
- Population, geography, period, comparator, and units: Single illustrative case, specific placements, Padova TMY, useful heating and hourly peak.
- Assumptions and uncertainty: The pattern concerns one assumed zone/response and TMY coordinate contrasts; no schedule is proven true or more accurate. Scientific interpretation awaits independent correction review.
- Counterevidence / alternative explanation: Different envelope, thermostat mapping, weather period or allowable placements could change effects; the original sign reversal shows implementation risk.
- What would falsify or weaken the claim: Failure of independent v3 state or source checks would remove the candidate response interpretation.
- Required next action / dependency / owner: Reviewer 02 to audit corrected solver and full-step checks before author promotes interpretation.
- Permitted manuscript wording: [CONDITIONAL INTERPRETATION PENDING INDEPENDENT CORRECTION REVIEW.]
- Related claims / proposed tables and figures: RES-004; LIM-003

## Limitations

### LIM-001 — Spatial and survey inference

- Exact claim: One unweighted 2013 Veneto response combined with Padova-coordinate weather does not represent Padova households, Veneto population outcomes, or the national residential stock.
- Kind: limitation
- Evidence class: derived estimate
- Status: supported
- Provenance: DATA-INPUT-AUDIT | 04_analysis/input_audit.json selection and selected_survey_response; RQ-FREEZE | 01_plan/research_questions.md spatial inference boundary; SIM-PROTOCOL | 03_methods/simulation_protocol.md survey response and physical model
- Population, geography, period, comparator, and units: One selected row, one assumed zone; no population weighting or sampled Padova dwelling.
- Assumptions and uncertainty: Weather location is a boundary condition, not a city household sampling frame.
- Counterevidence / alternative explanation: A representative weighted local sample would require distinct design and evidence.
- What would falsify or weaken the claim: Population inference claimed from this one case would falsify manuscript scope.
- Required next action / dependency / owner: None; ensure abstract/conclusions state case boundary.
- Permitted manuscript wording: Findings, when available, are conditional on one illustrative case and do not estimate a population mean.
- Related claims / proposed tables and figures: DAT-001; DAT-003; RES-004

### LIM-002 — No empirical validation

- Exact claim: The proposed TMY pilot lacks an independently documented matching meter series, so technical pass or a modeled effect cannot establish measured heating accuracy.
- Kind: limitation
- Evidence class: derived estimate
- Status: supported
- Provenance: DAT-INPUT-AUDIT | 04_analysis/input_audit.json GIS_inventory; VAL-PROTOCOL | 03_methods/validation_protocol.md empirical and aggregate layers; CONF-V01 | printed p1307; PDF p7; Table 3 national aggregate benchmark
- Population, geography, period, comparator, and units: Illustrative zone and TMY; no matched historical meter/end-use boundary.
- Assumptions and uncertainty: Inventory is limited; compatible observations might exist elsewhere.
- Counterevidence / alternative explanation: A full independent meter series with matching building/weather/boundary could support later evaluation.
- What would falsify or weaken the claim: Claim would need revision only after provenance-qualified empirical data and untouched evaluation.
- Required next action / dependency / owner: Validation owner: keep minimum dataset specification; author retain missing-validation language.
- Permitted manuscript wording: Empirical heating accuracy has not been established for this pilot.
- Related claims / proposed tables and figures: GAP-002; MET-007; RES-004

### LIM-003 — Placement alternatives are not bounds

- Exact claim: Earliest and latest contiguous placements are constructed alternatives, neither observed schedule estimates nor proven upper/lower bounds on heating demand or peak across all admissible profiles.
- Kind: limitation
- Evidence class: derived estimate
- Status: supported
- Provenance: SIM-PROTOCOL | 03_methods/simulation_protocol.md Accounting and uncertainty; CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 643–656
- Population, geography, period, comparator, and units: Three placements within fixed 8 h blocks; single physical case and weather.
- Assumptions and uncertainty: Thermal inertia and weather timing can make energy response nonmonotonic in start time.
- Counterevidence / alternative explanation: Other admissible starts and non-contiguous schedules remain untested.
- What would falsify or weaken the claim: An exhaustive search with proved monotonic response could establish extrema, but is absent.
- Required next action / dependency / owner: Author: avoid min/max uncertainty or confidence-bound language.
- Permitted manuscript wording: Early and late placements sample two alternatives; they are not uncertainty bounds.
- Related claims / proposed tables and figures: MET-002; INT-001

### LIM-004 — Code lineage unresolved

- Exact claim: The current GRINS and EUReCA checkouts have recorded revisions, but neither has been shown to be the exact conference executable/configuration; the pilot must be labelled a documented extension.
- Kind: limitation
- Evidence class: derived estimate
- Status: supported
- Provenance: CONF-A01 | printed p1310; PDF p10; chapter repository availability statement; SIM-PROTOCOL | 03_methods/simulation_protocol.md Physical model and departures; DATA-INPUT-AUDIT | 04_analysis/input_audit.json revisions and source hashes; CODE-GRINS-MAIN | C:/Works/GRINS/GRINS-Tool/funcs/main_functions.py lines 792–806
- Population, geography, period, comparator, and units: Current local GRINS HEAD a129009e; EUReCA HEAD 009d2af8; conference run revision/config unspecified.
- Assumptions and uncertainty: Current GRINS quasi-steady 2C differs from conference-described 1C; exact published code lineage unavailable.
- Counterevidence / alternative explanation: Published outputs may have used another version or configuration.
- What would falsify or weaken the claim: A verified release tag plus config and run provenance could resolve equivalence; absent now.
- Required next action / dependency / owner: Simulation owner: freeze new adapter identity; author avoid reproduction wording.
- Permitted manuscript wording: The new pilot extends the conference question under documented code and assumptions; it is not a reproduced conference run.
- Related claims / proposed tables and figures: MET-003; RES-001

### LIM-005 — Unconditioned summer temperature

- Exact claim: The pilot's no-cooling configuration allowed a maximum modeled zone air temperature of 42.81°C, so the case cannot support comfort or whole-year conditioned-building conclusions.
- Kind: limitation
- Evidence class: simulation output
- Status: supported
- Provenance: RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/summary.json corrected maximum_air_C; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/design.json cooling_upper_C=50; SIM-PROTOCOL | 03_methods/simulation_protocol.md cooling isolation
- Population, geography, period, comparator, and units: Single assumed zone, Padova-coordinate TMY, no active cooling; modeled air temperature °C.
- Assumptions and uncertainty: The 42.81°C is an unconditioned modeled summer state under disabled cooling, not observed indoor temperature; correction review is pending.
- Counterevidence / alternative explanation: An active cooling/ventilation or shading design could lower temperature but would be a distinct experiment.
- What would falsify or weaken the claim: Any claim of realistic annual comfort or cooling behavior is unsupported by this design.
- Required next action / dependency / owner: Simulation owner: retain no-cooling boundary; author: avoid comfort inference.
- Permitted manuscript wording: Cooling was disabled and modeled summer air temperature reached 42.81°C; the reported results concern useful heating only.
- Related claims / proposed tables and figures: DAT-003; RES-004

### LIM-006 — Archived solver outputs quarantined

- Exact claim: The original Padova v1 and climate v2 annual loads are retained only as defective-run provenance because the recursive hourly dispatcher failed node balance at hundreds of mode transitions; they cannot support physical sensitivity estimates.
- Kind: limitation
- Evidence class: derived estimate
- Status: supported
- Provenance: REVIEW-DIAGNOSTIC | 07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json seeded/early/late failure counts and residuals; CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md quarantine
- Population, geography, period, comparator, and units: Archived 5 Padova v1 and 20 climate v2 annual executions; defect inspected directly on v1 with same dispatcher path for v2.
- Assumptions and uncertainty: V1 arithmetic and repeatability were valid as file properties, while thermal state transition was not. V2 path used the same defective dispatcher; no individual v2 residual figures are asserted here.
- Counterevidence / alternative explanation: The defect is in the current checkout's call path; no inference is made that the published conference executable shared it.
- What would falsify or weaken the claim: A proven alternative historical v2 execution path could narrow quarantine, but current source/run snapshots identify the same dispatcher.
- Required next action / dependency / owner: Simulation owner: preserve archived files and labels; author: never reuse their effect magnitudes as scientific results.
- Permitted manuscript wording: Archived dispatcher-based annual effects are excluded from scientific interpretation.
- Related claims / proposed tables and figures: RES-004; RES-005; MET-008
