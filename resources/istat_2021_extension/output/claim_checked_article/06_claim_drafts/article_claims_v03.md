# Article claim checker v03

Status: review-03 author response, 20 October 2026. All 66 pre-v5 annual executions are quarantined; v5's 20 annual runs plus one separate non-sample replay provide conditional fixed-case model contrasts after independent source/forcing/state/arithmetic verification. Physical results remain candidate pending fresh independent review 04. No empirical validation, population inference, system sizing or broad method novelty is claimed. Exact source aliases/locators appear in 02_evidence/source_ledger.md and local_source_ledger.md; new v5 aliases require registry extension.

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

- Exact claim: Prataviera et al. (2022) already reconstruct heating operation from three ISTAT time bands, use randomized profiles with EUReCA, and explicitly perturb HVAC start-up by a discrete uniform ±2 h shift because survey timing is unknown; their Verona hourly comparison has a district-heat measurement boundary distinct from this fixed-case experiment.
- Kind: literature
- Evidence class: published finding
- Status: supported
- Provenance: SRC-UA2022 | full primary article Section 3.1.1 p6, start-up-time bullet and Table 1 p7; Section 3.3.1 p11; Section 4.1.2 p12; DOI 10.1016/j.apenergy.2022.118691
- Population, geography, period, comparator, and units: Italian ISTAT-informed UBEM; Verona mixed building sample and selected 2019/20 winter periods; no result transferred numerically.
- Assumptions and uncertainty: The published shift is a whole-profile startup/timing perturbation within a broader UBEM uncertainty study, while the present experiment holds one respondent and each band's encoded high-setpoint duration fixed. No generic timing-method novelty is claimed.
- Counterevidence / alternative explanation: The paper's abstract/body disagree on attribution of one 18%→10% example, so it is excluded.
- What would falsify or weaken the claim: Direct reinspection of cited primary pages or a corrected article could narrow this summary.
- Required next action / dependency / owner: None; maintain this precedent in novelty wording.
- Permitted manuscript wording: The 2022 survey-informed EUReCA study already randomized three-band schedules and tested a ±2-hour start-up shift; this study isolates three placements for one fixed encoded-duration case.
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
- Counterevidence / alternative explanation: The survey asks mean system on-hours rather than thermostat levels. Prataviera et al. already addressed unknown timing with random profiles and ±2 h startup shifts, so the information gap is not a previously unrecognized one; our 22/19°C encoding is an assumption.
- What would falsify or weaken the claim: Official questionnaire fields specifying exact within-band switching times would weaken the stated information gap; inspected questions 3.11_A/B/C do not.
- Required next action / dependency / owner: None for this identifiability statement; preserve raw/compact response provenance and avoid claims of observed timing.
- Permitted manuscript wording: Mean reported hours do not identify within-band switching; prior work already studied timing uncertainty, and this fixed case encodes selected placements as high-setpoint periods.
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

- Exact claim: The candidate contribution is a bounded, reproducible comparison of three contiguous placements preserving each of one respondent's 4/4/1 encoded high-setpoint durations within their respective bands for one assumed dwelling and four coordinate weather exposures; novelty of this exact fixed-case contrast remains provisional in light of earlier three-band randomization and ±2 h startup perturbations.
- Kind: novelty
- Evidence class: hypothesis
- Status: unverified
- Provenance: SRC-UA2022 | Section 3.1.1 p6 startup shift and three-band reconstruction; Table 1 p7; RQ-FREEZE | 01_plan/research_questions.md fixed respondent/placement contrast; LIT-SEARCH | 02_evidence/literature_search_log.md targeted search limits
- Population, geography, period, comparator, and units: One illustrative Veneto 2013 response and fixed dwelling; Padova TMY first, further climates gated; kWh and kW endpoints.
- Assumptions and uncertainty: Targeted narrative search is neither exhaustive nor proof that no exact prior comparison exists.
- Counterevidence / alternative explanation: Prataviera 2022 already tested schedule timing uncertainty in a related survey-informed EUReCA context; a broad first-method or first-timing claim would be false.
- What would falsify or weaken the claim: A primary study of this exact contrast or a failed technical pilot would narrow or negate contribution.
- Required next action / dependency / owner: Literature owner: retain targeted rather than exhaustive novelty boundary; author: describe exact experimental contrast without first-use language.
- Permitted manuscript wording: We test a specific fixed-respondent, within-band placement contrast; prior studies already modeled survey-derived timing uncertainty.
- Related claims / proposed tables and figures: LIT-002; RQ-001; RQ-002

## Research questions

### RQ-001 — Primary sensitivity question

- Exact claim: For one illustrative dwelling, selected Veneto response and Padova-coordinate TMY, the predeclared comparison is annual useful sensible heating and maximum hourly load under seeded, earliest and latest contiguous placements of the same 4/4/1 encoded high-setpoint hours; v5 now provides a checked conditional model comparison, pending a fresh fourth scientific review.
- Kind: method
- Evidence class: planned test
- Status: partially supported
- Provenance: RQ-FREEZE | 01_plan/research_questions.md primary question and fixed-case scope; ISTAT-Q2013 | istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf pp41–43; CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625, 643–656; RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/seeded_hourly.csv, early_hourly.csv, late_hourly.csv and summary.json; per-file SHA256 in summary/completion_manifest; RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/full_step_balance_checks.json, surface_forcing_hourly.csv, surface_forcing_parameters.json and completion_manifest.json; AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent EPW/CSV/paired/replay arithmetic; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json direct 20-case state/accounting/peak audit
- Population, geography, period, comparator, and units: One illustrative case; 8760 hourly steps; Padova-coordinate TMY; baseline seeded versus early/late; kWh and kW.
- Assumptions and uncertainty: Reported hours are converted to 22/19°C modeled setpoints; actual positive-load hours remain free. V5 is one assumed geometry/response/season under native annual-mean sky and solar-time conventions, not observed or representative heating.
- Counterevidence / alternative explanation: A null difference is possible; neither direction is presupposed.
- What would falsify or weaken the claim: Run failure or violation of paired invariants makes comparison uninterpretable.
- Required next action / dependency / owner: Fresh reviewer 04: independently inspect the v03 claims, corrected forcing, source-to-profile mapping and v5 hourly outputs; author: retain conditional boundary.
- Permitted manuscript wording: We compare three equal-duration modeled high-setpoint placements in one illustrative Padova-coordinate case; v5 numerical contrasts are conditional model outputs under review.
- Related claims / proposed tables and figures: MET-002; RES-004; FIG-02

### RQ-002 — Conditional weather question

- Exact claim: The same fixed dwelling, Veneto response, 22/19°C mapping and climate-E seasonal mask are compared under four consistently sourced coordinate TMYs (Padova, Bolzano, Roma, Palermo); v5 matched runs provide conditional weather-exposure contrasts, not local population or climate-variable causal effects, pending review 04.
- Kind: method
- Evidence class: planned test
- Status: partially supported
- Provenance: RQ-FREEZE | 01_plan/research_questions.md secondary weather question; CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json acquired four-site weather hashes; CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv descriptors; RUN-CLIMATE-V5 | 04_analysis/paired_schedule_contrasts_v5.csv twelve predeclared primary cases and same-site seeded denominators; RUN-CLIMATE-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/summary.json, completion_manifest.json and hourly CSVs; AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent integration/hash/paired table checks; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json independently reconstructed v5 rows, residuals and peak contexts
- Population, geography, period, comparator, and units: One unchanged illustrative case under four coordinate TMYs; matched seeded/early/late contrasts in kWh, kW and same-site percent.
- Assumptions and uncertainty: The common climate-E operating season is imposed outside its local context. Temperature, solar and humidity descriptors co-vary, so site-to-site differences cannot be assigned to one weather variable; sky is annual-mean and native solar phase offset is unapplied.
- Counterevidence / alternative explanation: Weather can change annual load without changing relative placement sensitivity; no trend assumed.
- What would falsify or weaken the claim: If comparable weather files cannot be sourced or Padova checks fail, this question remains unanswered.
- Required next action / dependency / owner: Fresh reviewer 04: verify v5 forcing/output pairs and interpretation; author: retain fixed-input/weather-exposure framing.
- Permitted manuscript wording: The same illustrative case was exposed to four coordinate TMYs with paired schedules; differences are conditional model contrasts, not estimates for each city's housing stock.
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
- Assumptions and uncertainty: Retrieval metadata and byte identity remain supported, but all 8,760 opaque-cover values are missing sentinel 99 in each EPW; provenance does not establish compatibility with the unguarded model sky routine. TMY is not a historical meter-matched year.
- Counterevidence / alternative explanation: The weather reader consumes missing cloud cover as a physical value; the resulting v3 load simulations are not scientifically admissible. The acquired EPW's horizontal IR field offers a possible documented alternative after checks.
- What would falsify or weaken the claim: A corrupt EPW, duplicate/missing timestamps or contrary origin metadata would invalidate or relabel this input.
- Required next action / dependency / owner: Methods owner: retain acquired hashes; document an IR-based or otherwise justified longwave treatment with consumed-field QC and an independent forcing audit.
- Permitted manuscript wording: Padova's TMY matches the documented PVGIS acquisition; its opaque sky-cover field is missing and requires explicit handling before load simulation.
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

- Exact claim: For the bounded ideal sensible 1C case, the local single-commit adapter evaluates candidates from one previous state and commits one accepted hourly state. Independent direct-transition and 175,200-hour equation checks support this narrow state correction; they do not validate the weather forcing or physical annual effects.
- Kind: method
- Evidence class: derived estimate
- Status: supported
- Provenance: CORRECTION-PROTOCOL-V3 | 05_simulations/climate_campaign_v3/Padova/correction_protocol_v3.md scope and gates; CORRECTION-CODE-V3 | 05_simulations/climate_campaign_v3/Padova/single_commit_1c.py select_solution/solve_1c_single_commit, SHA256 e549b4e5645ad8ae159fddf644d225d28622e1495f6fd64768c5242389b82c16; RUN-PADOVA-V3 | 05_simulations/climate_campaign_v3/Padova/completion_manifest.json correction/source hashes; STATE-AUDIT-R2 | 07_reviews/reviewer_02/independent_audit.json 175,200 air/surface/mass residual and continuity checks; STATE-RUNTIME-R2 | 07_reviews/reviewer_02/runtime_checks.json direct actual stepper transition checks; RUN-CLIMATE-V5 | 05_simulations/climate_campaign_v5/Padova/single_commit_1c.py unchanged single-commit source hash e549b4e5645ad8ae159fddf644d225d28622e1495f6fd64768c5242389b82c16
- Population, geography, period, comparator, and units: One assumed zone, 8760 h, ideal sensible load; no stock aggregation.
- Assumptions and uncertainty: Only the declared ideal sensible 1C network was tested; no general EUReCA implementation, conference replication, measured accuracy or weather-input correctness follows.
- Counterevidence / alternative explanation: Archived v1/v2 recursive dispatcher violated state balance at mode changes; direct pure-network algebra checks did not catch it.
- What would falsify or weaken the claim: If adapter imports unexpected checkout, model or config, methodological link fails.
- Required next action / dependency / owner: The bounded state repair carries into v5 unchanged; fresh review 04 to inspect final pipeline without treating state closure as empirical validation.
- Permitted manuscript wording: The study-local single-commit 1C stepper passed independent state-equation checks; weather forcing still requires correction.
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

- Exact claim: The v5 20-case campaign passed direct transition, source-profile, repeat/warm-up and complete air/surface/mass state checks: all 175,200 exported hours have exact old-state continuity and no residual above 1e-6 W; independent maximum absolute node residual is 1.768×10⁻¹⁰ W. These tests verify this bounded numerical implementation, not measured accuracy.
- Kind: method
- Evidence class: simulation output
- Status: supported
- Provenance: SURFACE-PROTOCOL-V5 | 03_methods/surface_correction_protocol_v5.md pre-run equations, unchanged parameter and test scope; SURFACE-CODE-V5 | 04_analysis/surface_forcing_1c.py study-local per-surface SW/LW/glazing calculation, SHA256 24768556f49ef341791647502f0fe5d1f8baddc3e5c2fcf5df23a124f106883a; SURFACE-FIXTURE-V5 | 05_simulations/surface_forcing_v5_fixture_checks.json analytic component and order tests; SURFACE-AUDIT-V5 | 07_reviews/reviewer_03/v5_forcing_checks.json independent per-surface, perturbation, order and actual-solver forcing audit; RUN-CLIMATE-V5 | 04_analysis/paired_schedule_contrasts_v5.csv twelve predeclared primary cases and same-site seeded denominators; RUN-CLIMATE-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/summary.json, completion_manifest.json and hourly CSVs; AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent integration/hash/paired table checks; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json independently reconstructed v5 rows, residuals and peak contexts; STATE-CHECK-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/full_step_balance_checks.json and checks.json
- Population, geography, period, comparator, and units: Twenty v5 annual simulations (12 primary, 4 seeded repeats, 4 longer warm-up), 175,200 hourly records; ideal sensible 1C zone; residual tolerance 1e-6 W.
- Assumptions and uncertainty: Numerical state closure, repeatability and warm-up stability apply only to this fixed case and selected implementation; forcing and empirical validity are separate evidence layers. Earlier 66 executions remain quarantined.
- Counterevidence / alternative explanation: Earlier v1/v2 recursive state transition, v3 sentinel-sky and v4 stale-surface forcing defects show why weaker gates were inadequate; the v5 actual forcing is separately checked.
- What would falsify or weaken the claim: A new state discrepancy would narrow the solver finding; no physical effect may be admitted without a separate weather-forcing gate and fresh annual runs.
- Required next action / dependency / owner: Fresh reviewer 04: inspect immutable v5 source, actual forcing, state and output checks; author: keep numerical verification distinct from benchmark/measurement evidence.
- Permitted manuscript wording: The 20 v5 runs passed bounded source, forcing, state and repeat checks; these do not establish observed heating accuracy.
- Related claims / proposed tables and figures: RES-003; RES-004; LIM-002

### MET-006 — Climate expansion gate

- Exact claim: Four same-provider PVGIS coordinate TMYs were characterized consistently and used with unchanged building, response and seasonal mask in the v5 corrected-forcing campaign. The active model path guards missing cloud and invalid wind access; Palermo's one negative wind record remains an unused-field exception, not a universal weather-quality pass.
- Kind: method
- Evidence class: planned test
- Status: partially supported
- Provenance: CLIMATE-INPUTS-V2 | 05_simulations/climate_inputs_v2/acquisition_manifest.json and raw EPWs; CLIMATE-METADATA-V2 | 05_simulations/climate_inputs_v2/audited_metadata.json; CLIMATE-DESCRIPTORS | 04_analysis/climate_descriptors_v2.csv; WEATHER-PROTOCOL-V4 | 03_methods/weather_correction_protocol_v4.md active-field, IR and time audit; WEATHER-AUDIT-V4 | 05_simulations/weather_forcing_v4/weather_forcing_audit.json; RUN-CLIMATE-V5 | 04_analysis/paired_schedule_contrasts_v5.csv twelve predeclared primary cases and same-site seeded denominators; RUN-CLIMATE-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/summary.json, completion_manifest.json and hourly CSVs; AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent integration/hash/paired table checks; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json independently reconstructed v5 rows, residuals and peak contexts
- Population, geography, period, comparator, and units: Four 8760-hour coordinate TMY files and 12 matched primary v5 cases; annual kWh and maximum hourly kW.
- Assumptions and uncertainty: TYMs are synthetic; common climate-E season is not location-specific. One Palermo −0.1 m/s wind value is retained but guarded and unused in fixed-ACH/no natural-mechanical ventilation path. Annual-mean sky and native solar-time conventions limit hourly fidelity.
- Counterevidence / alternative explanation: Within-site matching controls design inputs but cannot isolate which co-varying meteorological field explains cross-site differences.
- What would falsify or weaken the claim: If same-source metadata unavailable or descriptors fail to provide contrast, no climate expansion claim.
- Required next action / dependency / owner: Fresh reviewer 04: inspect all-site active forcing and paired contrast provenance; any future wind-dependent model must resolve the Palermo value first.
- Permitted manuscript wording: Four consistently retrieved coordinate TMYs support conditional controlled weather exposures for this guarded fixed-ACH model.
- Related claims / proposed tables and figures: RQ-002; RES-005

### MET-007 — Validation separation

- Exact claim: Implementation checks, model benchmarks, empirical meter comparison, aggregate external checks and robustness must be reported as distinct evidentiary layers; no new empirical validation is part of the current pilot.
- Kind: method
- Evidence class: planned test
- Status: supported
- Provenance: VAL-PROTOCOL | 03_methods/validation_protocol.md current v5 candidate disposition and evidence layers; VAL-MATRIX | 03_methods/validation_matrix.md current v5 output/validation matrix; SIM-PROTOCOL | 03_methods/simulation_protocol.md canonical current status and frozen-version chronology
- Population, geography, period, comparator, and units: Current illustrative hourly useful-demand pilot; no independent observed Padova heating reference identified.
- Assumptions and uncertainty: If observations arise, calibration and untouched validation sets must be separated and weather/end-use boundaries matched.
- Counterevidence / alternative explanation: Published national BEN table and internal checks are different evidence classes.
- What would falsify or weaken the claim: A claimed observed-accuracy metric without independent aligned observations violates this method.
- Required next action / dependency / owner: Validation owner: maintain one current v5 candidate status pointer and the absent benchmark/meter layers; later prose must not translate numerical verification into accuracy.
- Permitted manuscript wording: The analysis reports verification separately from empirical validation; the latter remains unavailable for this case.
- Related claims / proposed tables and figures: GAP-002; LIM-002

### MET-008 — Single-commit correction gate

- Exact claim: The study-local 1C state stepper is independently checked for one-old-state/one-commit transitions and all-hour node balances in v5; a separate per-surface and weather-input forcing gate now also passes for this bounded geometry. Neither layer is an independent building benchmark or empirical accuracy test.
- Kind: method
- Evidence class: derived estimate
- Status: supported
- Provenance: CORRECTION-CODE-V3 | 04_analysis/single_commit_1c.py unchanged candidate/commit logic; STATE-CHECK-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/full_step_balance_checks.json; SURFACE-PROTOCOL-V5 | 03_methods/surface_correction_protocol_v5.md pre-run equations, unchanged parameter and test scope; SURFACE-CODE-V5 | 04_analysis/surface_forcing_1c.py study-local per-surface SW/LW/glazing calculation, SHA256 24768556f49ef341791647502f0fe5d1f8baddc3e5c2fcf5df23a124f106883a; SURFACE-FIXTURE-V5 | 05_simulations/surface_forcing_v5_fixture_checks.json analytic component and order tests; SURFACE-AUDIT-V5 | 07_reviews/reviewer_03/v5_forcing_checks.json independent per-surface, perturbation, order and actual-solver forcing audit
- Population, geography, period, comparator, and units: One ideal sensible 1C zone, hourly, no mechanical/natural ventilation or latent gains; 1e-6 W numerical residual tolerance.
- Assumptions and uncertainty: Only this ideal sensible 1C scope was checked; native transposition and time convention are retained model assumptions.
- Counterevidence / alternative explanation: Passing network equations does not verify envelope truth, weather representativeness, software generality or measured accuracy.
- What would falsify or weaken the claim: Reviewer-discovered state mismatch or checker dependence on incorrect saved states would defeat this gate.
- Required next action / dependency / owner: Fresh reviewer 04: inspect both correction layers and final effects; author: avoid general EUReCA validation language.
- Permitted manuscript wording: The bounded v5 pilot separately checks its one-commit state solution and actual per-surface forcing.
- Related claims / proposed tables and figures: MET-003; MET-005; RES-004

### MET-009 — Weather-forcing admission gate

- Exact claim: For the bounded v5 path, EPW missing cloud cover is bypassed using valid hourly horizontal IR to form effective sky temperature; the native model's annual mean of air-minus-sky is then applied through an explicit per-surface longwave and shortwave calculation. Independent actual-forcing, zero-solar, perturbation and order tests pass, subject to annual-mean and solar-time approximation limits.
- Kind: method
- Evidence class: derived estimate
- Status: partially supported
- Provenance: EPW-SPEC | https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v24.1.0/AuxiliaryPrograms.pdf EPW missing codes; WEATHER-PROTOCOL-V4 | 03_methods/weather_correction_protocol_v4.md IR conversion/annual-mean and time convention; WEATHER-CODE-V4 | 04_analysis/ir_sky_forcing.py valid-IR scalar and active-path guard; WEATHER-AUDIT-V4 | 05_simulations/weather_forcing_v4/weather_forcing_audit.json; SURFACE-PROTOCOL-V5 | 03_methods/surface_correction_protocol_v5.md pre-run equations, unchanged parameter and test scope; SURFACE-CODE-V5 | 04_analysis/surface_forcing_1c.py study-local per-surface SW/LW/glazing calculation, SHA256 24768556f49ef341791647502f0fe5d1f8baddc3e5c2fcf5df23a124f106883a; SURFACE-FIXTURE-V5 | 05_simulations/surface_forcing_v5_fixture_checks.json analytic component and order tests; SURFACE-AUDIT-V5 | 07_reviews/reviewer_03/v5_forcing_checks.json independent per-surface, perturbation, order and actual-solver forcing audit
- Population, geography, period, comparator, and units: Four 8760-hour PVGIS EPWs; opaque cover in tenths, horizontal IR W/m², sky/air temperatures °C or K, zone forcing W.
- Assumptions and uncertainty: The one-hour EPW horizontal IR is converted to W/m²; the annual-mean sky difference, not hourly sky exchange, enters native 1C longwave. Solar position uses fixed UTC+1 interval-start labels without applying recorded irradiation offsets. One invalid Palermo wind field is guarded and unused.
- Counterevidence / alternative explanation: V3's unfiltered cloud99 and v4's inherited roof area/irradiance invalidated their load values; the independent v5 per-surface actual-forcing check specifically tests both paths.
- What would falsify or weaken the claim: A missing/invalid consumed field, timestamp mismatch, implausible night forcing, non-reproducing independent sky calculation or failed annual gate prevents admission.
- Required next action / dependency / owner: Fresh reviewer 04: inspect v5 code, raw IR, guards, per-surface all-hour forcing and retained time approximations.
- Permitted manuscript wording: The v5 pilot uses valid horizontal IR and checked per-surface forcing under the documented annual-mean sky and native solar-time conventions.
- Related claims / proposed tables and figures: MET-003; MET-005; RES-004; LIM-007

### MET-010 — Executable command provenance

- Exact claim: The v3 manifests' historical SITE omission is preserved and annotated. The v5 runner records executable, launch working directory, full site/campaign argument vector, source/input/output hashes and refuses an existing completed destination; a separate seeded replay is byte-identical to the primary Padova hourly output and is not an additional scientific sample.
- Kind: method
- Evidence class: derived estimate
- Status: supported
- Provenance: COMMAND-ADDENDUM | 05_simulations/recorded_command_addendum.md historical omission and non-overwrite rule; RUNNER-V5 | 04_analysis/run_climate_campaign_v5.py SITE/--campaign/replay arguments and output guard; RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/completion_manifest.json recorded command_argv/cwd/executable; REPLAY-V5 | 05_simulations/climate_campaign_v5_replay/Padova/completion_manifest.json and seeded_hourly.csv; REPLAY-CHECK-V5 | 05_simulations/v5_replay_interface_checks.json overwrite refusal and byte identity
- Population, geography, period, comparator, and units: Archived v3 run manifests for Padova, Bolzano, Roma and Palermo; command-line provenance, no physical units.
- Assumptions and uncertainty: The historical v3 command string is not retroactively rewritten. Replay demonstrates deterministic production under the inspected current environment, not physical truth or an independent sample.
- Counterevidence / alternative explanation: Recorded source and hourly output hashes match, so this is a replay-documentation defect rather than evidence that v3 files were not produced.
- What would falsify or weaken the claim: A new manifest lacking its exact command/working directory or a replay that overwrites archived outputs leaves provenance incomplete.
- Required next action / dependency / owner: Methods owner: preserve complete replay/immutability discipline for any later run version; author: keep replay out of the 20-case scientific sample.
- Permitted manuscript wording: V5 commands and source hashes are recorded, and a separate Padova seeded replay reproduces its primary hourly file byte-for-byte.
- Related claims / proposed tables and figures: MET-006; LIM-007

### MET-011 — Explicit per-surface 1C preprocessing

- Exact claim: A bounded local v5 preprocessing routine computes each external wall/roof's opaque shortwave and longwave terms from its own area, orientation, irradiance, material and radiative fields, and computes glazing shortwave only where glazed area is positive. Independent per-surface, zero-solar/zero-longwave, order-invariance and actual-solver-forcing tests pass for this case without changing physical/network parameters or the external checkout.
- Kind: method
- Evidence class: derived estimate
- Status: partially supported
- Provenance: SURFACE-PROTOCOL-V5 | 03_methods/surface_correction_protocol_v5.md pre-run equations, unchanged parameter and test scope; SURFACE-CODE-V5 | 04_analysis/surface_forcing_1c.py study-local per-surface SW/LW/glazing calculation, SHA256 24768556f49ef341791647502f0fe5d1f8baddc3e5c2fcf5df23a124f106883a; SURFACE-FIXTURE-V5 | 05_simulations/surface_forcing_v5_fixture_checks.json analytic component and order tests; SURFACE-AUDIT-V5 | 07_reviews/reviewer_03/v5_forcing_checks.json independent per-surface, perturbation, order and actual-solver forcing audit; SURFACE-DIAGNOSTIC-R3 | 07_reviews/reviewer_03/runtime_forcing_checks.json v4 inherited roof values and order failure; PROVENANCE-V5 | 07_reviews/reviewer_03/v5_provenance_comparison.json unchanged design/network/inputs
- Population, geography, period, comparator, and units: Five exterior opaque surfaces in one assumed 1C zone; hourly SW/LW/node forcing W under four coordinate TMYs; tests cover 20 v5 annual files.
- Assumptions and uncertainty: This fixes the demonstrated local zero-glazed roof/stale-local defect for the inspected geometry and native radiation formula. Solar-plane transposition and annual-mean sky remain model conventions; no general package certification follows.
- Counterevidence / alternative explanation: V4's expected-coefficient check passed without comparing implemented forcing; review03 found 129–179 W longwave response errors and surface-order dependence. V5 independently checks actual solver forcing.
- What would falsify or weaken the claim: A per-surface/solver mismatch, order dependence, altered frozen physical parameter or failed active field guard would invalidate the bounded correction.
- Required next action / dependency / owner: Fresh reviewer 04: inspect v5 source/fixtures/hourly exports and unchanged-input comparison before accepting modeled effects.
- Permitted manuscript wording: A study-local per-surface preprocessing correction passed direct forcing and order checks for the tested 1C geometry.
- Related claims / proposed tables and figures: MET-009; MET-005; RES-004; LIM-007

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

- Exact claim: The v5 seeded, earliest and latest Padova inputs each preserve 4/4/1 modeled 22°C high-setpoint hours in the 05–13, 13–21 and 21–05 bands; actual positive-load hours may occur at the 19°C setback and are not fixed by these profile counts.
- Kind: result
- Evidence class: simulation output
- Status: supported
- Provenance: CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 593–625, 643–656; RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/summary.json daily_on_profile and seeded/early/late hourly setpoints; AUTHOR-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json source-setpoint partition arithmetic
- Population, geography, period, comparator, and units: One illustrative response; three 8 h blocks; hours; 8760 h annual sequence.
- Assumptions and uncertainty: Reported average hours are coarsened survey responses; the encoded schedule repeats them and maps them to 22°C rather than enforcing equipment availability.
- Counterevidence / alternative explanation: Different contiguous placements and noncontiguous real schedules remain admissible.
- What would falsify or weaken the claim: A disagreement in raw response, profile counts, CSV row count, or frozen run hashes would invalidate this result.
- Required next action / dependency / owner: Reviewer to verify source-to-run mapping; author to preserve encoded-versus-observed distinction.
- Permitted manuscript wording: All v5 alternatives keep the reported 4/4/1 durations as encoded high-setpoint periods, while heating may occur at setback.
- Related claims / proposed tables and figures: MET-002; FIG-01

### RES-004 — Padova v5 conditional modeled contrasts

- Exact claim: In the checked v5 fixed Padova-coordinate case, seeded/early/late annual useful sensible heating is 9574.02/9659.63/9606.30 kWh and maximum hourly heating load is 10.050/10.253/9.838 kW. Relative to seeded, early changes annual heating by +0.894% and peak by +2.029%; late changes them by +0.337% and −2.103%. These are conditional model outputs awaiting fresh review 04, not observed energy or accuracy estimates.
- Kind: result
- Evidence class: simulation output
- Status: partially supported
- Provenance: RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/seeded_hourly.csv, early_hourly.csv, late_hourly.csv and summary.json; per-file SHA256 in summary/completion_manifest; RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/full_step_balance_checks.json, surface_forcing_hourly.csv, surface_forcing_parameters.json and completion_manifest.json; AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent EPW/CSV/paired/replay arithmetic; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json direct 20-case state/accounting/peak audit; SURFACE-PROTOCOL-V5 | 03_methods/surface_correction_protocol_v5.md pre-run equations, unchanged parameter and test scope; SURFACE-CODE-V5 | 04_analysis/surface_forcing_1c.py study-local per-surface SW/LW/glazing calculation, SHA256 24768556f49ef341791647502f0fe5d1f8baddc3e5c2fcf5df23a124f106883a; SURFACE-FIXTURE-V5 | 05_simulations/surface_forcing_v5_fixture_checks.json analytic component and order tests; SURFACE-AUDIT-V5 | 07_reviews/reviewer_03/v5_forcing_checks.json independent per-surface, perturbation, order and actual-solver forcing audit
- Population, geography, period, comparator, and units: One dwelling/response/Padova-coordinate TMY; seeded baseline versus early/late; kWh and kW.
- Assumptions and uncertainty: Same illustrative dwelling, respondent, weather, 22/19°C mapping and season across placements. Annual-mean longwave, native unapplied solar offset, fixed gains/envelope and one TMY limit transfer. Percent denominator is same-site seeded. No measurement or confidence interval.
- Counterevidence / alternative explanation: All 66 older annual executions (v1–v4 plus v4 replay) are quarantined for state/forcing defects. Different profile shapes, physical assumptions or hourly weather timing could alter v5 effects.
- What would falsify or weaken the claim: Fresh review finding source-to-forcing mismatch, state/summary error, unmatched run inputs or implausible retained approximations would withdraw or narrow these candidate values.
- Required next action / dependency / owner: Fresh reviewer 04: independent full claim/forcing/output and scope audit; author: admit only bounded fixed-case modeled language if accepted.
- Permitted manuscript wording: [CANDIDATE V5 RESULT, FRESH REVIEW 04 PENDING.] In one fixed Padova-coordinate model case, placement changed annual useful heating and maximum hourly load by the stated paired amounts.
- Related claims / proposed tables and figures: RQ-001; MET-004; FIG-02

### RES-005 — Four-site v5 annual modeled contrasts

- Exact claim: In v5 matched fixed-input runs, early/late placements change annual useful sensible heating relative to same-site seeded by +0.894/+0.337% Padova, −0.098/+0.027% Bolzano, +2.033/+0.500% Roma, and +3.868/+0.981% Palermo. These are deterministic conditional weather-exposure contrasts pending fresh review 04, not population effects or causal attribution to a weather variable.
- Kind: result
- Evidence class: simulation output
- Status: partially supported
- Provenance: RUN-CLIMATE-V5 | 04_analysis/paired_schedule_contrasts_v5.csv twelve predeclared primary cases and same-site seeded denominators; RUN-CLIMATE-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/summary.json, completion_manifest.json and hourly CSVs; AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent integration/hash/paired table checks; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json independently reconstructed v5 rows, residuals and peak contexts; SURFACE-PROTOCOL-V5 | 03_methods/surface_correction_protocol_v5.md pre-run equations, unchanged parameter and test scope; SURFACE-CODE-V5 | 04_analysis/surface_forcing_1c.py study-local per-surface SW/LW/glazing calculation, SHA256 24768556f49ef341791647502f0fe5d1f8baddc3e5c2fcf5df23a124f106883a; SURFACE-FIXTURE-V5 | 05_simulations/surface_forcing_v5_fixture_checks.json analytic component and order tests; SURFACE-AUDIT-V5 | 07_reviews/reviewer_03/v5_forcing_checks.json independent per-surface, perturbation, order and actual-solver forcing audit
- Population, geography, period, comparator, and units: Same assumed dwelling and one Veneto response under four PVGIS TMY coordinate files; within-site early/late minus seeded, percent denominator same-site seeded kWh.
- Assumptions and uncertainty: One fixed illustrative building/response/setpoint/season across coordinate TMYs. Temperature, radiation and humidity vary together. Native annual-mean sky and solar-phase approximation remain; no statistical interval.
- Counterevidence / alternative explanation: The small Bolzano differences could be sensitive to model/forcing/timing assumptions; no single weather mechanism can be isolated. Older 66 annual executions remain invalid for physical effects.
- What would falsify or weaken the claim: A fresh v5 forcing, matched-input, integration or interpretation failure would withdraw or narrow these candidate effects.
- Required next action / dependency / owner: Fresh reviewer 04: verify all-site forcing and direct CSV contrasts; author: retain conditional and nonrepresentative scope.
- Permitted manuscript wording: [CANDIDATE V5 RESULT, FRESH REVIEW 04 PENDING.] Same-case annual placement contrasts differ across four coordinate TMYs under the documented model convention.
- Related claims / proposed tables and figures: RQ-002; MET-006; FIG-03

### RES-006 — V5 post-hoc setback heating diagnostic

- Exact claim: In the v5 Padova seeded output, the ideal heater has positive sensible load during 1323 of 2730 modeled 19°C setback calendar hours, delivering 2375.62 kWh there. This post-hoc descriptive partition illustrates that the fixed 4/4/1 high-setpoint duration does not fix positive-heating runtime; it is neither an observed operation record nor a primary endpoint.
- Kind: result
- Evidence class: simulation output
- Status: partially supported
- Provenance: POSTHOC-V5 | 04_analysis/posthoc_setpoint_partition_v5.csv Padova seeded19°C row; RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/seeded_hourly.csv hourly setpoints and sensible loads; AUTHOR-POSTHOC-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json 36 partition rows recalculated; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json independently checked post-hoc partition
- Population, geography, period, comparator, and units: One assumed Padova-coordinate TMY case; 19°C modeled setback; 2730 calendar hours, 1323 positive-load hours, 2375.62 useful kWh; post-hoc.
- Assumptions and uncertainty: Positive load threshold is >1e-6 W. The control is modeled 22/19°C, not the respondent's observed thermostat or burner timing. The quantity is conditional on v5 forcing and one case.
- Counterevidence / alternative explanation: Changing weather, season, thermostat interpretation or envelope can change counts; source code alone establishes only that setback heating is allowed.
- What would falsify or weaken the claim: Mismatched hourly partition or incorrect corrected source would withdraw the quantity.
- Required next action / dependency / owner: Fresh reviewer 04: inspect partition against v5 raw hourly file and ensure post-hoc/nonobserved label is retained.
- Permitted manuscript wording: [CANDIDATE POST-HOC V5 DIAGNOSTIC.] The modeled heater also delivers heat at the 19°C setback; encoded high-setpoint hours do not equal positive-load hours.
- Related claims / proposed tables and figures: MET-001; MET-002; RES-004

### RES-007 — Four-site v5 hourly-maximum contrasts

- Exact claim: In v5 matched runs, early/late placements change maximum hourly sensible heating load relative to same-site seeded by +2.029/−2.103% Padova, 0/0% Bolzano, +8.326/+3.381% Roma and +5.969/+1.524% Palermo. The Bolzano maxima coincide at the modeled season-onset hour, so their zero paired peak differences do not demonstrate no effect of intraday timing.
- Kind: result
- Evidence class: simulation output
- Status: partially supported
- Provenance: RUN-CLIMATE-V5 | 04_analysis/paired_schedule_contrasts_v5.csv twelve predeclared primary cases and same-site seeded denominators; RUN-CLIMATE-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/summary.json, completion_manifest.json and hourly CSVs; AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent integration/hash/paired table checks; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json independently reconstructed v5 rows, residuals and peak contexts; PEAK-CONTEXT-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json annual peak hour and adjacent setpoints; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json peak rows and preceding states
- Population, geography, period, comparator, and units: Same assumed dwelling and one Veneto response under four PVGIS TMY coordinates; within-site early/late minus seeded peak; percent denominator seeded peak kW.
- Assumptions and uncertainty: Maximum is a predeclared 1-hour resolution endpoint, not system capacity or robust design peak. It can be dominated by seasonal startup and native solar-phase/sky approximations; no peak timing sensitivity sweep is completed.
- Counterevidence / alternative explanation: At Bolzano the prescribed 0→19°C season-start recovery, common to all profiles, drives the maximum; this masks possible differences at other hours. Across sites, no one weather variable is isolated.
- What would falsify or weaken the claim: Independent reviewer finding a corrected-state, peak extraction or input-pairing error would withdraw candidate effects.
- Required next action / dependency / owner: Fresh reviewer 04: inspect each peak timestamp/prior state and paired extraction, especially Bolzano season onset; author: do not omit inconvenient maxima or infer equipment sizing.
- Permitted manuscript wording: [CANDIDATE V5 RESULT, FRESH REVIEW 04 PENDING.] Hourly maxima differ in three sites; Bolzano's equal maxima occur at the shared modeled season-start recovery.
- Related claims / proposed tables and figures: RQ-002; RES-005; FIG-03

### RES-008 — Bolzano season-onset maximum context

- Exact claim: As a post-hoc diagnostic of the v5 Bolzano hourly files, seeded, early and late annual maxima are identical at 17.8086 kW in hour 6912 (16 October 00:00), when the imposed lower setpoint changes from 0°C in the preceding hour to 19°C and old zone air is 9.641°C. The common season-start recovery can dominate this annual-maximum endpoint; zero paired peaks do not show zero intraday timing response or equipment sizing need.
- Kind: result
- Evidence class: simulation output
- Status: partially supported
- Provenance: RUN-BOLZANO-V5 | 05_simulations/climate_campaign_v5/Bolzano/seeded_hourly.csv hours 6911–6912; RUN-BOLZANO-V5 | 05_simulations/climate_campaign_v5/Bolzano/early_hourly.csv hours 6911–6912; RUN-BOLZANO-V5 | 05_simulations/climate_campaign_v5/Bolzano/late_hourly.csv hours 6911–6912; AUTHOR-POSTHOC-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json all 12 peak rows/prior setpoints; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json Bolzano peak rows and preceding states
- Population, geography, period, comparator, and units: One assumed zone under Bolzano-coordinate TMY; three schedules; hour 6912, kW and °C; post-hoc explanatory diagnostic of predeclared annual maximum.
- Assumptions and uncertainty: The common seasonal mask is deliberate but not a local regulatory/occupancy estimate. No peak was excluded; an alternate-season or sizing analysis was not performed.
- Counterevidence / alternative explanation: Other hours may show profile-specific differences, but the prescribed season restart sets the annual maximum in this case.
- What would falsify or weaken the claim: A different raw peak row, setpoint transition or old air temperature would invalidate this diagnostic; future alternate-season modeling could change its relevance.
- Required next action / dependency / owner: Fresh reviewer 04: verify peak context directly; author: keep primary annual maximum and label this as post-hoc explanation rather than a new endpoint.
- Permitted manuscript wording: [POST-HOC V5 PEAK CONTEXT.] Bolzano's identical annual maxima occur at the shared imposed heating-season restart, not at a profile-specific daytime switch.
- Related claims / proposed tables and figures: RES-007; LIM-008

## Interpretation

### INT-001 — Conditional fixed-case interpretation

- Exact claim: Under one assumed dwelling, one respondent, three encoded placements and the declared native weather-time/annual-mean-sky convention, v5 outputs show placement-dependent annual useful-heating and maximum-hourly-load contrasts that differ by coordinate TMY. The comparison is descriptive and model-conditional; it does not identify a causal weather variable, an observed schedule, population effect, savings or measured accuracy.
- Kind: interpretation
- Evidence class: simulation output
- Status: partially supported
- Provenance: RUN-CLIMATE-V5 | 04_analysis/paired_schedule_contrasts_v5.csv twelve predeclared primary cases and same-site seeded denominators; RUN-CLIMATE-V5 | 05_simulations/climate_campaign_v5/{Padova,Bolzano,Roma,Palermo}/summary.json, completion_manifest.json and hourly CSVs; AUTHOR-V5 | 06_claim_drafts/author_v5_check_for_v03.json independent integration/hash/paired table checks; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json independently reconstructed v5 rows, residuals and peak contexts; RQ-FREEZE | 01_plan/research_questions.md predeclared estimands; PEAK-CONTEXT-V5 | 06_claim_drafts/author_v5_posthoc_peak_check_v03.json seasonal-start maximum diagnostic
- Population, geography, period, comparator, and units: Single illustrative case, specific placements, Padova TMY, useful heating and hourly peak.
- Assumptions and uncertainty: Only the encoded choices, fixed geometry/response/season and native forcing convention are tested. Small effects and hourly maxima may change with physical/timing assumptions; no ensemble or empirical validation quantifies that uncertainty.
- Counterevidence / alternative explanation: Bolzano maxima are shared seasonal recovery events; weather descriptors co-vary; prior defective campaigns show implementation sensitivity and remain excluded.
- What would falsify or weaken the claim: Failure of independent v3 state or source checks would remove the candidate response interpretation.
- Required next action / dependency / owner: Fresh reviewer 04: evaluate final central effect claims and limits; author: preserve descriptive wording and avoid robust mechanism or transfer claims.
- Permitted manuscript wording: [CANDIDATE CONDITIONAL INTERPRETATION.] These v5 paired contrasts describe this specified model case, not measured households or general schedule accuracy.
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

- Exact claim: The v5 no-cooling case reaches a maximum modeled Padova zone air temperature of 41.05°C, so it cannot support claims of summer comfort, realistic whole-year conditioned operation, or an appropriate cooling design.
- Kind: limitation
- Evidence class: simulation output
- Status: supported
- Provenance: RUN-PADOVA-V5 | 05_simulations/climate_campaign_v5/Padova/summary.json maximum_air_C and design.json cooling_upper_C=50; INDEPENDENT-V5 | 07_reviews/reviewer_03/independent_audit_v5.json saved temperature maxima; SIM-PROTOCOL | 03_methods/simulation_protocol.md cooling-isolation boundary
- Population, geography, period, comparator, and units: One assumed uncooled zone under Padova-coordinate TMY; simulated air temperature °C, not a measured indoor series.
- Assumptions and uncertainty: The maximum is conditional on the v5 fixed model and native weather forcing; no cooling/ventilation/comfort intervention is evaluated.
- Counterevidence / alternative explanation: An active cooling/ventilation or shading design could lower temperature but would be a distinct experiment.
- What would falsify or weaken the claim: Any claim of realistic annual comfort or cooling behavior is unsupported by this design.
- Required next action / dependency / owner: Methods owner: recompute maximum zone air temperature after corrected forcing; author: restrict any future statement to a model state, not observed comfort.
- Permitted manuscript wording: Cooling is disabled; the modeled summer air temperature reaches about 41°C, so the analysis addresses heating only.
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

### LIM-007 — All pre-v5 physical outputs quarantined

- Exact claim: All 66 annual executions before the v5 campaign, including the separately identified v4 seeded replay, are retained as diagnostic provenance but excluded from physical energy-effect interpretation: v1/v2 use the recursive state path, v3 consumes missing cloud cover, and v4 uses stale roof area/irradiance in native envelope preprocessing.
- Kind: limitation
- Evidence class: derived estimate
- Status: supported
- Provenance: RUN-INVENTORY | 05_simulations/run_inventory.csv version/status inventory; DISPATCH-DIAGNOSTIC | 07_reviews/reviewer_01/dispatcher_diagnostic/state_balance_diagnostic.json; WEATHER-DIAGNOSTIC-R2 | 07_reviews/reviewer_02/independent_audit.json sentinel/sky finding; SURFACE-DIAGNOSTIC-R3 | 07_reviews/reviewer_03/runtime_forcing_checks.json actual preprocessing/order defects; SIM-RESULTS | 04_analysis/simulation_results.md current quarantine and v5 status
- Population, geography, period, comparator, and units: 66 archived pre-v5 annual executions: 25 v1/v2, 20 v3, 20 v4 primary/repeat/warm-up, one v4 seeded replay; no admitted physical effect magnitude.
- Assumptions and uncertainty: Defects are shown in current local execution paths, not attributed to the published conference executable. Archived arithmetic/state passes remain narrow historical records, not physical effect evidence.
- Counterevidence / alternative explanation: Repeatability of a defective run does not imply correct envelope forcing; the exact v5 campaign uses new versioned outputs.
- What would falsify or weaken the claim: A demonstrated valid alternative historical forcing path could narrow the quarantine; current frozen code and runtime weather checks show none.
- Required next action / dependency / owner: Methods/author: preserve archives and explicit version labels; fresh reviewer 04: verify v5-only numerical provenance.
- Permitted manuscript wording: Earlier local run versions were excluded after distinct state, sky-input and surface-forcing defects; the current conditional estimates come from v5 only.
- Related claims / proposed tables and figures: MET-009; RES-004; RES-005; RES-006; RES-007; INT-001

### LIM-008 — Sky aggregation and solar-time approximation

- Exact claim: V5 replaces invalid missing-cloud input with valid hourly horizontal IR but retains the native annual mean of hourly air-minus-effective-sky temperature for longwave exchange, and native solar position at fixed UTC+1 interval-start labels without applying the EPW/JRC irradiation time offset. Hourly peak phase/ranking sensitivity to these approximations is not quantified.
- Kind: limitation
- Evidence class: derived estimate
- Status: supported
- Provenance: WEATHER-PROTOCOL-V4 | 03_methods/weather_correction_protocol_v4.md annual-mean method and index/offset audit; WEATHER-CODE-V4 | 04_analysis/ir_sky_forcing.py scalar construction; CODE-EURECA-WEATHER | C:/Works/EUReCA/EUReCA/eureca_building/weather.py lines 195–206 native solar-position indexing; JRC-TMY | https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/pvgis-typical-meteorological-year-tmy-generator_en irradiance offsets/local-hour EPW convention
- Population, geography, period, comparator, and units: Four 8760-hour TMY exposures, 1 h model step; sky-temperature K, solar-time offset h, annual useful kWh and peak kW endpoints.
- Assumptions and uncertainty: Air and IR are paired by EPW row; this does not establish exact solar phase or hourly longwave physics. No alternative timing/longwave sensitivity campaign has quantified possible changes to small effects or peak rankings.
- Counterevidence / alternative explanation: JRC documents irradiation offsets; native pvlib index does not apply them. Valid IR handling and per-surface algebra alone cannot prove accurate hourly environmental forcing.
- What would falsify or weaken the claim: A benchmark or declared sensitivity experiment could quantify or narrow this limit; no current measurement or alternate-forcing run does.
- Required next action / dependency / owner: Author/reviewer 04: retain this limitation with every interpretation of small annual and peak differences; future methods work could test phase/longwave alternatives without altering predeclared outputs here.
- Permitted manuscript wording: The pilot uses annual-mean sky exchange and the native solar-time convention; sensitivity of hourly peaks to those approximations remains untested.
- Related claims / proposed tables and figures: MET-009; RES-007; INT-001
