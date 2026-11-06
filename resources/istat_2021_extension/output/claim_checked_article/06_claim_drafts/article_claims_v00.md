# Article claim checker v00

Status: initial scientific draft, 20 October 2026. Historical published results and new illustrative EUReCA outputs are distinguished claim by claim. This is not article prose. Source IDs resolve through `../02_evidence/source_ledger.md`, `../00_context/conference_evidence.md`, and the exact paths below. No empirical heating-accuracy claim is asserted.

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

- Exact claim: The selected respondent's reported mean winter-day heating hours in three eight-hour bands do not identify a unique hourly on/off sequence; contiguous placement alone admits multiple starts for each 4/4/1 band.
- Kind: gap
- Evidence class: derived estimate
- Status: supported
- Provenance: CONF-M04 | printed p1304; PDF p4; final paragraph on morning/afternoon/evening hours; CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 643–656, create_random_prof/create_prof; DATA-INPUT-AUDIT | 04_analysis/input_audit.json selected_survey_response q_3_11A/B/C=4/4/1; ISTAT-Q2013 | istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf pp41–42, questions 3.11_A/B/C, mean winter-day hours and band endpoints; RAW-VERIFY | 00_context/survey_schedule_verification.json records.raw nprog3523 and bands
- Population, geography, period, comparator, and units: One 2013 Veneto response; three daily 8 h blocks; count of admissible contiguous positions per block is 8−n+1.
- Assumptions and uncertainty: Released integer answers coarsen reported mean hours; daily repetition and contiguous placement are simulation choices, not observed duty cycles.
- Counterevidence / alternative explanation: The survey may encode rounded mean on-hours rather than a literal schedule repeated daily; this strengthens the need to distinguish model construction from response.
- What would falsify or weaken the claim: Official questionnaire fields specifying exact within-band switching times would weaken the stated information gap; inspected questions 3.11_A/B/C do not.
- Required next action / dependency / owner: None for this identifiability statement; preserve raw/compact response provenance and avoid claims of observed timing.
- Permitted manuscript wording: Reported mean on-hours leave within-band timing unidentified for the selected response.
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

- Exact claim: For one illustrative dwelling, one selected Veneto survey response and Padova-coordinate TMY weather, compare annual sensible heating demand and maximum hourly sensible heating load under seeded source, earliest, and latest contiguous block placements.
- Kind: method
- Evidence class: planned test
- Status: supported
- Provenance: RQ-FREEZE | 01_plan/research_questions.md Primary RQ and Secondary RQ2; SIM-PROTOCOL | 03_methods/simulation_protocol.md Question and estimands
- Population, geography, period, comparator, and units: One illustrative case; 8760 hourly steps; Padova-coordinate TMY; baseline seeded versus early/late; kWh and kW.
- Assumptions and uncertainty: Fixed physical inputs and reported durations; no measured timing or population estimand.
- Counterevidence / alternative explanation: A null difference is possible; neither direction is presupposed.
- What would falsify or weaken the claim: Run failure or violation of paired invariants makes comparison uninterpretable.
- Required next action / dependency / owner: Pilot run and author arithmetic check completed; Astra reviewer to inspect implementation and result boundary.
- Permitted manuscript wording: We compare three duration-preserving placements for one illustrative case under Padova-coordinate weather.
- Related claims / proposed tables and figures: MET-002; RES-004; FIG-02

### RQ-002 — Conditional weather question

- Exact claim: After the Padova technical gate, repeat the same fixed dwelling/response/placement contrasts under consistently sourced contrasting weather files to test whether modeled sensitivity differs by weather.
- Kind: method
- Evidence class: planned test
- Status: planned
- Provenance: RQ-FREEZE | 01_plan/research_questions.md Secondary RQ1; CLIMATE-PLAN | 03_methods/climate_selection.md selection gate
- Population, geography, period, comparator, and units: One unchanged case across candidate sites; 2–4 further files only if selected; paired kWh/kW effects.
- Assumptions and uncertainty: Keep season, physics and occupancy assumptions fixed; weather source metadata must be homogeneous.
- Counterevidence / alternative explanation: Weather can change annual load without changing relative placement sensitivity; no trend assumed.
- What would falsify or weaken the claim: If comparable weather files cannot be sourced or Padova checks fail, this question remains unanswered.
- Required next action / dependency / owner: Padova technical gate passed; methods/simulation owners still need same-provider weather provenance, frozen site selection and matched runs.
- Permitted manuscript wording: [PLANNED TEST: fixed-input climate contrasts, contingent on input provenance and pilot checks.]
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

- Exact claim: The initial local EPW has coordinates 45.408°N, 11.873°E, 8760 hourly records and an ECMWF/ERA-labelled header; its filename-suggested 2005–2023 provenance is unverified.
- Kind: result
- Evidence class: derived estimate
- Status: supported
- Provenance: DATA-INPUT-AUDIT | 04_analysis/input_audit.json weather and SHA256 390ead08c3e32d34a874c564b57c28ef1b7275682c7a749477208f5b6846adca; EPW-LOCAL | C:/Works/EUReCA/Fork-Macovei/EUReCA-maco/eureca_ubem/Input/tmy_45.408_11.873_2005_2023.epw header lines 1–8
- Population, geography, period, comparator, and units: Padova-coordinate typical-year candidate, 8760 h; not a historical 2013/14 meter-matched weather series.
- Assumptions and uncertainty: Retrieval, generator, selected-month and underlying-period documentation absent; header label is not full provenance.
- Counterevidence / alternative explanation: Weather mismatch may affect results; TMY is unsuitable for historical empirical validation.
- What would falsify or weaken the claim: A corrupt EPW, duplicate/missing timestamps or contrary origin metadata would invalidate or relabel this input.
- Required next action / dependency / owner: Simulation owner: run integrity checks and preserve checksum; methods owner: seek complete provenance.
- Permitted manuscript wording: An 8760-hour Padova-coordinate EPW supplies the initial controlled weather exposure; source-period metadata remain incomplete.
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
- Evidence class: observed measurement
- Status: partially supported
- Provenance: DATA-INPUT-AUDIT | 04_analysis/input_audit.json GIS_inventory property keys for LivingLab and PaduaDigital files; VAL-PROTOCOL | 03_methods/validation_protocol.md Section 3 empirical validation
- Population, geography, period, comparator, and units: Inspected local directories only; no inference that all Padova measurements are nonexistent.
- Assumptions and uncertainty: GIS energy-looking fields may be modeled; their measurement provenance and boundary are unverified.
- Counterevidence / alternative explanation: An external owner may have compatible monitored data outside inventory.
- What would falsify or weaken the claim: Receipt of a documented independent time series with matching building/period/end use would overturn the current availability statement.
- Required next action / dependency / owner: Validation owner: seek meter provenance/access if empirical validation becomes feasible.
- Permitted manuscript wording: No compatible independent meter series was identified in the inspected local assets.
- Related claims / proposed tables and figures: GAP-002; LIM-002

## Methodological claims

### MET-001 — Source reconstruction

- Exact claim: Current GRINS EndUse.py builds each reported block as one contiguous on segment at a seeded start and concatenates night, morning and afternoon blocks with night wrapping.
- Kind: method
- Evidence class: derived estimate
- Status: supported
- Provenance: CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 643–656; SHA256 cf7442f0b2ddb39b338c2047a1b95016d3a4d246604da15cd49c9a823cc436d5; CODE-GRINS-ENDUSE | same file lines 593–625; thermostat and season mapping; ISTAT-Q2013 | istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf p43, question 3.13: option _2 is timer
- Population, geography, period, comparator, and units: Source implementation; daily 8-hour bands 05–13, 13–21, 21–05; integer hours 0–8.
- Assumptions and uncertainty: random.seed(int(id)) is reinvoked for each band. For selected q_3_13_2=1 (timer), source code assigns modeled 22/19°C on/off setpoints; the respondent did not provide these values. Raw q_2_32C/D blanks were compacted to zeros; both fail the source's equality-to-one fireplace test.
- Counterevidence / alternative explanation: This local implementation is not proven bit-identical to the 2026 published pipeline.
- What would falsify or weaken the claim: Different repository revision or source behavior on execution would require method revision.
- Required next action / dependency / owner: Simulation owner: test extracted routine and record exact resulting 24 h profiles.
- Permitted manuscript wording: The inspected GRINS routine uses seeded contiguous placement within each reported eight-hour block.
- Related claims / proposed tables and figures: GAP-001; MET-002; LIM-004

### MET-002 — Duration-preserving paired alternatives

- Exact claim: The planned alternatives place the same contiguous 4/4/1 on-hours at the earliest or latest admissible start in each respective eight-hour band while preserving all three survey block totals.
- Kind: method
- Evidence class: planned test
- Status: supported
- Provenance: SIM-PROTOCOL | 03_methods/simulation_protocol.md Question and estimands; baseline and alternatives; CODE-GRINS-ENDUSE | C:/Works/GRINS/GRINS-Tool/classes/EndUse.py lines 643–656
- Population, geography, period, comparator, and units: One response, same 24 h daily block durations; source seeded versus earliest/latest; hours.
- Assumptions and uncertainty: Block totals and thermal/physical parameters fixed; schedules are constructed possibilities, not observed patterns.
- Counterevidence / alternative explanation: Other admissible schedules exist; early/late are not global extrema of energy response.
- What would falsify or weaken the claim: A block-total mismatch, date-rotation bug or hidden physical-input difference invalidates pairing.
- Required next action / dependency / owner: Three profile outputs and block checks saved; reviewer to inspect night rotation and source equivalence.
- Permitted manuscript wording: We compare three admissible contiguous placements with equal reported block durations.
- Related claims / proposed tables and figures: RQ-001; RES-003; LIM-003

### MET-003 — Explicit EUReCA 1C adapter

- Exact claim: The proposed dynamic 1C EUReCA pilot is a new explicit adapter; current GRINS simulate_hvac uses 2C with quasi-steady simulation, so the pilot is not a reproduction of the local full pipeline or published national run.
- Kind: method
- Evidence class: planned test
- Status: supported
- Provenance: SIM-PROTOCOL | 03_methods/simulation_protocol.md Physical model and departures; EUReCA revision 009d2af866f143d37aecf2a934adbee3b0f304aa; CODE-GRINS-MAIN | C:/Works/GRINS/GRINS-Tool/funcs/main_functions.py lines 792–806; SHA256 4b90297a4f22ae35a406d3a01be7db98cef3767c4a212ae49ac67362cc92a3bf; CONF-M02 | printed p1304; PDF p4; Section 3.1 conference 1C description
- Population, geography, period, comparator, and units: One assumed zone, 8760 h, ideal sensible load; no stock aggregation.
- Assumptions and uncertainty: Chapter describes 1C, but complete conference software/config hash is unavailable; executable equivalence unresolved.
- Counterevidence / alternative explanation: The current local GRINS function differs explicitly; a 1C dynamic adapter has its own boundary.
- What would falsify or weaken the claim: If adapter imports unexpected checkout, model or config, methodological link fails.
- Required next action / dependency / owner: Simulation owner: record exact imports, hashes, config, command and run manifest.
- Permitted manuscript wording: We execute a documented 1C EUReCA sensitivity adapter related to the chapter's formulation, without claiming conference reproduction.
- Related claims / proposed tables and figures: LIM-004; RES-004

### MET-004 — Output accounting

- Exact claim: Annual useful sensible heating is defined as Σ_t max(P_t,0)Δt/1000 kWh at Δt=1 h and peak as max_t max(P_t,0)/1000 kW, with P_t in W.
- Kind: method
- Evidence class: planned test
- Status: supported
- Provenance: SIM-PROTOCOL | 03_methods/simulation_protocol.md Accounting and uncertainty; primary outputs; VAL-MATRIX | 03_methods/validation_matrix.md Annual sensible heating integration row
- Population, geography, period, comparator, and units: One simulated zone, hourly 8760-step year; W→kWh and W→kW conversion.
- Assumptions and uncertainty: Ideal unlimited heating, constant gains; demand excludes system efficiency, fuel, DHW and appliance electricity.
- Counterevidence / alternative explanation: Peak depends on temporal resolution; annual useful demand cannot be compared directly to final-energy BEN totals.
- What would falsify or weaken the claim: Recalculation from saved hourly W differing from summary invalidates result.
- Required next action / dependency / owner: Hourly CSV saved and independently integrated in author_result_check_v00.json; reviewer to check solver output semantics.
- Permitted manuscript wording: Useful sensible heating demand and maximum hourly load are the only primary energy/load endpoints.
- Related claims / proposed tables and figures: RES-004; LIM-002

### MET-005 — Technical pilot checks

- Exact claim: The completed five-run Padova pilot passed the declared internal checks for profile totals, 8760 finite records, zero cooling, temperature/load screens, active setpoint adherence, exact repeat and 168/336 h warm-up comparison.
- Kind: method
- Evidence class: simulation output
- Status: supported
- Provenance: RUN-PADOVA-V1 | 05_simulations/padova_pilot_v1/checks.json full checks; completion_manifest.json status complete/checks_pass; RUN-PADOVA-V1 | 05_simulations/padova_pilot_v1/summary.json five cases and hourly hashes; AUTHOR-CHECK-V00 | 06_claim_drafts/author_result_check_v00.json independent CSV integration/profile-count checks
- Population, geography, period, comparator, and units: Five planned annual runs; 168 h baseline/early/late/repeat and 336 h baseline; prescribed numerical tolerances.
- Assumptions and uncertainty: Numerical gates passed within predeclared tolerance; 42.81°C maximum air temperature occurs under no-cooling assumption and is not comfort validation. Full benchmark and observed validation remain absent.
- Counterevidence / alternative explanation: Identical repeat and algebraic consistency cannot detect all physical-model or boundary errors.
- What would falsify or weaken the claim: Any failed numerical check blocks admitting associated result until diagnosed.
- Required next action / dependency / owner: Reviewer to inspect run script and hashes; validation owner to retain absent benchmark/empirical layers.
- Permitted manuscript wording: All declared Padova pilot technical screens passed; these are implementation checks, not measured accuracy.
- Related claims / proposed tables and figures: RES-003; RES-004; LIM-002

### MET-006 — Climate expansion gate

- Exact claim: Additional climate runs now depend on acquisition of comparable provider/database weather files whose HDD18, CDD26, GHI and RH are computed consistently; the Padova pilot technical gate has passed.
- Kind: method
- Evidence class: planned test
- Status: planned
- Provenance: CLIMATE-PLAN | 03_methods/climate_selection.md selection gate and controlled test; SIM-PROTOCOL | 03_methods/simulation_protocol.md Expansion gate
- Population, geography, period, comparator, and units: Candidate Bolzano/Roma/Palermo coordinates are provisional; 2–4 sites maximum initial plan; Kd, kWh/m² and % indicators.
- Assumptions and uncertainty: Keep dwelling, response, schedule alternatives, season and physical inputs fixed; no geographic household representativeness.
- Counterevidence / alternative explanation: Heterogeneous TMY source/vintage could confound weather contrast.
- What would falsify or weaken the claim: If same-source metadata unavailable or descriptors fail to provide contrast, no climate expansion claim.
- Required next action / dependency / owner: Methods/simulation owners: source, hash, characterize and select comparable weather files, then execute matched runs.
- Permitted manuscript wording: [CLIMATE EXPANSION PENDING: comparable weather files and matched runs.]
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

- Exact claim: Saved seeded, early and late daily binary profiles each contain 4/4/1 encoded on-hours in the 05–13, 13–21 and 21–05 bands, respectively, and each run has 8760 finite hourly output rows.
- Kind: result
- Evidence class: simulation output
- Status: supported
- Provenance: RUN-PADOVA-V1 | 05_simulations/padova_pilot_v1/summary.json daily_on_profile and rows per case; RUN-PADOVA-V1 | 05_simulations/padova_pilot_v1/checks.json block_totals_exact/all8760_finite; AUTHOR-CHECK-V00 | 06_claim_drafts/author_result_check_v00.json cases block_counts_h and CSV arithmetic
- Population, geography, period, comparator, and units: One illustrative response; three 8 h blocks; hours; 8760 h annual sequence.
- Assumptions and uncertainty: These are encoded integer-hour repetitions of coarsened mean winter-day survey answers; they do not recover observed switching.
- Counterevidence / alternative explanation: Different contiguous placements and noncontiguous real schedules remain admissible.
- What would falsify or weaken the claim: A disagreement in raw response, profile counts, CSV row count, or frozen run hashes would invalidate this result.
- Required next action / dependency / owner: Reviewer to verify source-to-run mapping; author to preserve encoded-versus-observed distinction.
- Permitted manuscript wording: The three constructed profiles preserve the reported 4/4/1 band totals in the checked pilot outputs.
- Related claims / proposed tables and figures: MET-002; FIG-01

### RES-004 — Padova paired output

- Exact claim: For the single assumed dwelling under the Padova-coordinate TMY, seeded, earliest and latest placements produced annual useful sensible heating of 6521.19, 6376.55 and 6412.56 kWh and maximum hourly loads of 9.005, 9.151 and 8.881 kW, respectively; relative to seeded, earliest changed annual demand by −2.22% and peak by +1.62%, while latest changed them by −1.67% and −1.39%.
- Kind: result
- Evidence class: simulation output
- Status: supported
- Provenance: RUN-PADOVA-V1 | 05_simulations/padova_pilot_v1/summary.json seeded/early/late annual_sensible_heating_kWh and peak_sensible_heating_kW; RUN-PADOVA-V1 | 05_simulations/padova_pilot_v1/completion_manifest.json script/config/design/protocol/input hashes and checks_pass; AUTHOR-CHECK-V00 | 06_claim_drafts/author_result_check_v00.json cases/deltas, independently summed positive hourly W and recomputed percentages
- Population, geography, period, comparator, and units: One dwelling/response/Padova-coordinate TMY; seeded baseline versus early/late; kWh and kW.
- Assumptions and uncertainty: Deterministic contrast only; no sampling interval, population uncertainty or measured error metric. Denominator for percentages is seeded output; ideal sensible load excludes final-energy conversion.
- Counterevidence / alternative explanation: Results depend on assumed zone, setpoint mapping, weather, fixed season, model and only three placements; the selected duration is a coarsened mean response.
- What would falsify or weaken the claim: Failed independent CSV integration, schedule invariant, or full-run technical check would withdraw these values.
- Required next action / dependency / owner: Astra reviewer to inspect script/solver boundary and independent checks; future climate and empirical validation remain separate.
- Permitted manuscript wording: In this one illustrative TMY case, early and late encoded placements reduced modeled annual useful heating by 2.22% and 1.67% relative to the seeded source placement; the early peak rose 1.62% and late peak fell 1.39%.
- Related claims / proposed tables and figures: RQ-001; MET-004; FIG-02

### RES-005 — Weather interaction pending

- Exact claim: No verified comparable-climate run currently establishes how the schedule-placement sensitivity changes across weather exposures.
- Kind: result
- Evidence class: planned test
- Status: planned
- Provenance: CLIMATE-PLAN | 03_methods/climate_selection.md source/selection gate; RQ-FREEZE | 01_plan/research_questions.md Secondary RQ1
- Population, geography, period, comparator, and units: Potentially 2–4 consistently sourced weather files beyond Padova; same case; paired kWh/kW differences.
- Assumptions and uncertainty: Named candidate locations are not selected sites; no climatic class or transfer claim is inferred from names.
- Counterevidence / alternative explanation: Site differences may also reflect TMY provenance if inputs are heterogeneous.
- What would falsify or weaken the claim: Without comparable files and checked matched runs, the interaction question remains unanswered.
- Required next action / dependency / owner: Methods/simulation owners: pass gate, freeze selection, execute matched runs and checks.
- Permitted manuscript wording: [RESULT PENDING: conditional weather-by-placement contrasts.]
- Related claims / proposed tables and figures: RQ-002; MET-006; FIG-03

## Interpretation

### INT-001 — Permitted sensitivity interpretation

- Exact claim: The checked pilot shows different model outputs for three equal-duration placements in this assumed Padova-weather case, with annual and peak changes of differing sign for the early placement; it does not identify a correct real schedule or predictive improvement.
- Kind: interpretation
- Evidence class: simulation output
- Status: supported
- Provenance: RUN-PADOVA-V1 | 05_simulations/padova_pilot_v1/summary.json seeded/early/late; AUTHOR-CHECK-V00 | 06_claim_drafts/author_result_check_v00.json deltas; ISTAT-Q2013 | questionnaire pp41–42 mean winter-day duration questions without within-band timing
- Population, geography, period, comparator, and units: Single illustrative case, specific placements, Padova TMY, useful heating and hourly peak.
- Assumptions and uncertainty: Case-conditional deterministic inference. This does not quantify uncertainty over all admissible profiles, responses, buildings or weather years.
- Counterevidence / alternative explanation: Different admissible starts, envelope settings or weather could change sign/magnitude.
- What would falsify or weaken the claim: A source-to-input or solver defect would weaken this interpretation; observed accuracy would need separate data.
- Required next action / dependency / owner: Reviewer to assess scientific inference and code boundary; author retain case qualifier.
- Permitted manuscript wording: Equal reported band totals yielded different modeled annual and peak heating responses for this illustrative case.
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
- Provenance: RUN-PADOVA-V1 | 05_simulations/padova_pilot_v1/summary.json maximum_air_C in all five cases; RUN-PADOVA-V1 | 05_simulations/padova_pilot_v1/design.json cooling_upper_C=50 and energy_boundary; SIM-PROTOCOL | 03_methods/simulation_protocol.md cooling isolation and technical screen
- Population, geography, period, comparator, and units: Single assumed zone, Padova-coordinate TMY, no active cooling; modeled air temperature °C.
- Assumptions and uncertainty: Maximum is a solver output under intentionally disabled cooling; it is not observed indoor temperature.
- Counterevidence / alternative explanation: An active cooling/ventilation or shading design could lower temperature but would be a distinct experiment.
- What would falsify or weaken the claim: Any claim of realistic annual comfort or cooling behavior is unsupported by this design.
- Required next action / dependency / owner: Simulation owner: retain no-cooling boundary; author: avoid comfort inference.
- Permitted manuscript wording: Cooling was disabled and modeled summer air temperature reached 42.81°C; the reported results concern useful heating only.
- Related claims / proposed tables and figures: DAT-003; RES-004
