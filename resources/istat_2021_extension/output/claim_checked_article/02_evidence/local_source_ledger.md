# Local source and artifact registry — v04

Owner: `context_evidence`. Access: 20 October 2026. Requested model: `gpt-6.1-sol`; confirmed backend metadata: unknown. Frozen v04 manuscript SHA256 `8a641ab68bd80fb96aad3016f3deea37ba0dbd4ab50ce9d984d8b59cfe50da7e`; 41 claims.

**55 active supplemental aliases resolve.** All 67 entries retain historical aliases, upstream survey inputs and primary-document supplements. Archived v01/v03 registries are preserved under `registry_history`. The machine index contains exact paths, hashes, schemas, current and historical claim mappings, locators, access depth, creators where known and evidence boundaries.

**365 distinct files inspected; 205 expected-hash comparisons, all matched.** Missing files: 0. Unresolved aliases: 0. Unmapped manuscript alias candidates: 0.

Details: [local_sources.json](local_sources.json). Final audit: [local_registry_audit_v04.md](local_registry_audit_v04.md).

## Final source correspondence

- Every changed source/locator claim was inspected: MET-002, MET-003, MET-004, MET-005 and MET-009. Reviewer04 now appears in `INDEPENDENT-V5` through its actual technical JSON and producer. The unchanged review report is separately identified as procedural evidence.
- Reviewer04 initially inspected v03 numerical inputs. Its final v04 audit verifies unchanged result numbers, claim correspondence and bounded issue closure. That distinction is preserved; no new simulation or scientific result is implied.
- Only v5 energy-effect outputs remain current. Earlier v1–v4 results stay quarantined, even where limited arithmetic/state checks passed. The v5 replay is a technical check, not a new scientific sample.
- The eight canonical summaries were hashed **after** the orchestrator froze their status-only closure update. Their preserved prior copies match the before hashes. These mutable current documents are distinct from immutable per-run protocol/config/code/input snapshots.
- The actual v5 reproduction uses `run_climate_campaign_v5.py SITE --campaign NEW_ID`, followed by `check_full_step_balances_v4.py` and `check_surface_forcing_v5.py` for that campaign. No model/checker was rerun for this registry. Completion summary hashes target **summary.csv**; JSON identities are independently recorded.

## Primary documentation and access

V04 repaired EPW-SPEC to the inspected [EnergyPlus24.2 field dictionary](https://bigladdersoftware.com/epx/docs/24-2/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html). Opaque-cover99 and horizontal-IR9999 are documented missing codes. The earlier inaccessible24.1PDF target remains historical provenance; there is no current unresolved URL dependency.

The [JRC TMY documentation](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/pvgis-typical-meteorological-year-tmy-generator_en) supports the stated IR/time conventions; actual weather window/database identification comes from the acquired JSON. The [EnergyPlus climate reference](https://energyplus.readthedocs.io/en/v26.2.0/guides/engineering-reference/5.1-climate-calculations.html) supports the effective-sky equation. Documentation does not validate model predictions.

## Alias index

| Alias | Current claim IDs | Source class | Precise locator |
|---|---|---|---|
|`CONF-M04`|GAP-001|published chapter child alias|printed1304/PDF4 finalparagraph; Section3.1|
|`CONF-V01`|GAP-002, LIM-002, RES-001|published chapter child alias|printed1306-1307/PDF6-7 Section3.3/4.1/Table3|
|`CONF-W01`|GAP-002|published chapter child alias|printed1306/PDF6 Section3.3 weather overwrite coverage|
|`CONF-R01`|RES-002|published chapter child alias|printed1308/PDF8 Fig1 and caption|
|`CONF-R02`|RES-002|published chapter child alias|printed1309/PDF9 Fig2 and caption|
|`CONF-A01`|LIM-004|published chapter child alias|printed1310/PDF10 DataAvailability|
|`CONF-NUM-CHECK`|RES-001|derived arithmetic verification|Table3 published final-energy values and independently recalculated gas deviations|
|`CODE-GRINS-ENDUSE`|GAP-001, LIM-003, MET-001, MET-002, RES-003, RQ-001|external executable source|lines593-625:temperature/season mapping;643-656:seeded contiguous placement and night rotation|
|`CODE-GRINS-MAIN`|LIM-004|external executable source|lines792-806:current model construction/solver invocation|
|`DATA-INPUT-AUDIT`|DAT-001, DAT-002, DAT-004, GAP-001, LIM-001, LIM-004|derived local input inventory|selected_survey_response;eligible_veneto_count;selection;revisions;sha256;GIS_inventory|
|`DAT-INPUT-AUDIT`|LIM-002|alternate alias of derived local input inventory|GIS_inventory|
|`DATA-AUDIT-SCRIPT`|DAT-001|analysis executable code|survey filter reg5,strict0<hours<8,and lowest eligibleid;input/source hashes|
|`ISTAT-Q2013`|DAT-001, GAP-001, MET-001, RQ-001|official questionnaire|questionnaire pp40-43Q3.10-3.13;pp41-42band endpoints/coarsening;trace orders175-182|
|`ISTAT-REG2013`|DAT-001|official code classification|reg code5=Veneto|
|`RAW-VERIFY`|DAT-001, GAP-001|derived source-to-input verification|records.raw/compact;field_match_checks;bands;control_interpretation;blank_to_zero_conversion|
|`RQ-FREEZE`|INT-001, LIM-001, NOV-001, RQ-001, RQ-002|local research design|primaryRQ;secondaryRQ1/2;contribution-to-evidence map|
|`LIT-SEARCH`|NOV-001|derived literature search log|batches11-12 and scope/access limits|
|`SIM-PROTOCOL`|DAT-003, LIM-001, LIM-003, LIM-004, LIM-005, MET-002, MET-004, MET-007|local analysis assumptions/protocol|Physical model/departures;baseline/alternatives;Accounting/uncertainty;cooling isolation|
|`VAL-PROTOCOL`|DAT-004, GAP-002, LIM-002, MET-007|local validation design|Sections1-5;empirical and aggregate validation layers|
|`VAL-MATRIX`|MET-007|local validation design matrix|full claim/output target rows|
|`CLIMATE-INPUTS-V2`|DAT-002, MET-006, RQ-002|official API weather inputs and acquisition record|sites[].files[]:API URL,status,retrievedUTC,hash;TMY2005-2023,usehorizon1|
|`CLIMATE-METADATA-V2`|DAT-002, MET-006|derived weather metadata audit|site.inputs.meteo_data;meta;tmy_months_selected;numericQC|
|`CLIMATE-DESCRIPTORS`|MET-006, RQ-002|derived weather descriptor table|fourcoordinate rows;n,HDD18_daily_Kd,CDD26_daily_Kd,meanRH,annualGHI,EPWhash|
|`RUN-PADOVA-V3`|historical / procedural / auxiliary|corrected simulation output bundle|summary percase;daily_on_profile;checks;all-hour balances;completion hashes;hourly sensible_load_W/setpoint;completion summary_sha256 targets summary.csv per runnerline144|
|`RUN-CLIMATE-V3`|historical / procedural / auxiliary|corrected multi-weather derived output bundle|12primary site×schedule rows;4site summaries/completion records;full-step balances;each completion summary_sha256 targets site summary.csv|
|`CORRECTION-PROTOCOL-V3`|LIM-006, MET-003|local correction and quarantine protocol|scope;unchanged endpoints;pre-run and annual gates;quarantine|
|`CORRECTION-CODE-V3`|MET-003, MET-008|adapter-local executable correction|select_solution/solve_1c_single_commit;network residual and transition-check definitions|
|`AUTHOR-CHECK-V01`|historical / procedural / auxiliary|derived author arithmetic checks|source_tablehash;sites[].runs;Padova case integration,deltas,completion checks|
|`AUTHOR-POSTHOC-V01`|historical / procedural / auxiliary|derived author post-hoc arithmetic check|tablehash;36rows;max discrepancy;Padovaseeded19C|
|`POSTHOC-PARTITION-V3`|historical / procedural / auxiliary|derived exploratory simulation partition|36site×schedule×setpointrows;positive_load_tolerance_W1e-6;Padovaseeded19C|
|`DISPATCH-DIAGNOSTIC`|LIM-007, MET-001|derived diagnostic of quarantined run implementation|original v1case residual/failure counts;selected-record and old-hourly arithmetic fields|
|`REVIEW-DIAGNOSTIC`|LIM-006|derived diagnostic of quarantined run implementation|seeded/early/late failure counts,max node residuals,first/largest failinghours|
|`LOCAL-RAW-ISTAT2013`|historical / procedural / auxiliary|official public-use survey input|nprog3523;source record line3524;release descriptionpp1,3,6|
|`LOCAL-COMPACT-ISTAT2013`|historical / procedural / auxiliary|local transformed survey input|id3523;line3524;selected fields and routed zero conversion|
|`RUN-PADOVA-V5`|LIM-005, MET-002, MET-003, MET-010, RES-003, RES-004, RES-006, RQ-001|corrected candidate simulation bundle|5Padovacases;summaryJSON+CSV;hourly states;surface_forcing_hourly/parameters;completion/frozen manifests|
|`RUN-CLIMATE-V5`|INT-001, MET-004, MET-005, MET-006, RES-005, RES-007, RQ-002|corrected multi-weather campaign and paired table|12primarysite×placementrows;20campaignannualruns;per-siteforcing/state/summary/manifests|
|`RUN-BOLZANO-V5`|RES-008|selected corrected run peak context|hours6911-6912;load,air/massstates,setpoints and adjacent previous state|
|`AUTHOR-V5`|INT-001, MET-004, MET-005, MET-006, RES-003, RES-004, RES-005, RES-007, RQ-001, RQ-002|author arithmetic/source audit|sites;case_checks;paired_table_max_abs_error;replayidentity;partitionandpeakcontexts|
|`AUTHOR-POSTHOC-V5`|RES-006, RES-008|author exploratory partition/peak check|36partitionrows;12peakrows;adjacentsetpoints|
|`PEAK-CONTEXT-V5`|INT-001, RES-007|author peak context diagnostics|annualpeakhour;previoussetpoints/states;seasonalstartmaxima|
|`POSTHOC-V5`|RES-006|corrected setpoint partition table|36site×placement×setpointrows;Padovaseeded19C;positive-loadtolerance1e-6W|
|`INDEPENDENT-V5`|INT-001, LIM-005, MET-002, MET-003, MET-004, MET-005, MET-006, RES-004, RES-005, RES-006, RES-007, RES-008, RQ-001, RQ-002|independent direct artifact audit|Reviewer03 independent_audit_v5.json and producer;Reviewer04 independent_recheck.json source_schedule_max_error/source_profile/profile_checks/sites.*.cases/summary_comparison/weather_mapping/replay_identical;recheck.py lines10-71. review_04.md is procedural identity only.|
|`STATE-AUDIT-R2`|historical / procedural / auxiliary|historicalv3independent state audit|175200node-residual/continuitychecks|
|`STATE-RUNTIME-R2`|historical / procedural / auxiliary|historicalstepper runtime tests|directsingle-committransitiontests|
|`STATE-CHECK-V5`|MET-005, MET-008|corrected per-hour state verification|all175200hours;air/surface/massnodes;continuity;repeat/warmup/finitechecks|
|`SURFACE-PROTOCOL-V5`|MET-005, MET-008, MET-009, MET-011, RES-004, RES-005|frozen local preprocessing design|per-surfaceSW/LW/glazing equations;unchangedphysicalparameter/schedule/endpointscope;gates|
|`SURFACE-CODE-V5`|MET-005, MET-008, MET-009, MET-011, RES-004, RES-005|study-local surface preprocessing|per-surfacesolar/longwave/glazing;nodeweights;orderingguards;actualforcingchecker|
|`SURFACE-FIXTURE-V5`|MET-005, MET-008, MET-009, MET-011, RES-004, RES-005|analytic component/order fixtures|analyticcomponentexpectations;standalone/order/nodedistributionerrors|
|`SURFACE-AUDIT-V5`|MET-005, MET-008, MET-009, MET-011, RES-004, RES-005|independent surface forcing audit|componentperturbation/order;actualannualsolverforcingcomparison|
|`SURFACE-DIAGNOSTIC-R3`|LIM-007, MET-011|historicalv4forcing diagnostic|nativezero-glazedroofinheritedarea/irradiance;surfaceorderfailure|
|`WEATHER-PROTOCOL-V4`|LIM-008, MET-006, MET-009|retained IR/weather correction protocol|validIR→effectivesky;annualmeanscalar;timeconvention;unusednegative-windexception|
|`WEATHER-AUDIT-V4`|MET-006, MET-009|active-field weather audit and derived sky|rawEPW/JSONmapping;finite/range/sentinelcounts;sigmaTsky^4;Palermonegativewind|
|`WEATHER-CODE-V4`|LIM-008, MET-009|retained IR-to-sky scalar code|applyIRcorrection;activeruntimewind/ventilationguard;annualmeanscalar|
|`WEATHER-DIAGNOSTIC-R2`|LIM-007|historical sentinel/sky diagnostic|opaquecloud99;invalidoriginalskyscalar|
|`CODE-EURECA-WEATHER`|LIM-008|external weather parser source|lines195-206;nativesolarpositiontimeindex|
|`PROVENANCE-V5`|MET-011|independent run-version comparison|protocol/fixhashes;per-sitedesign/network/inputcomparison|
|`COMMAND-ADDENDUM`|MET-010|historical command-provenance addendum|omittedSITEhistoricalrecords;explicitv5SITE--campaign;actualcheckernames|
|`RUNNER-V5`|MET-010|corrected campaign executable interface|SITE--campaign;--replay-only-seeded;path/overwriteguards;frozencommandargv;summaryCSVhash|
|`REPLAY-V5`|MET-010|separate seeded annual replay|completionargv/cwd/executable;8760seededrows;separatecampaign|
|`REPLAY-CHECK-V5`|MET-010|overwrite refusal and replay check|allbefore/afterhashesequal;returncode1;byteidentity;statecheck|
|`RUN-INVENTORY`|LIM-007|all-version execution inventory|campaign/site/case/purpose;scientific_disposition;hourlyhashes|
|`SIM-RESULTS`|LIM-007|current simulation status/result summary|quarantinechronology;currentv5results;limits|
|`EPW-SPEC`|MET-009|primary project documentation (manual reproduction)|N12horizontalIRunits/missing9999;N23opaquecloudmissing99;Field:HorizontalInfraredRadiationIntensity;Field:OpaqueSkyCover|
|`JRC-TMY`|LIM-008|official primary documentation|Overviewirradiancetimeoffset;Outputs2.1IR(h);Outputs2.3EPWlocaltime/hour-endingandcommentsoffset|
|`LOCAL-ENERGYPLUS-SKY`|historical / procedural / auxiliary|official primary project documentation|SkyRadiationModelingEq5.2;SkyTemperatureCalculationEq5.5;sigma5.6697e-8|
|`LOCAL-FINAL-CLOSURE-V04`|historical / procedural / auxiliary|independent final review closure and artifact correspondence|final_closure.md Decision/Frozeninputs/Issueclosure/Administrativeannotations;final_audit.json hashes,same_ids,exact_claims_match_markdown,result_numbers_same,cycles|
|`LOCAL-CANONICAL-STATUS-V04`|historical / procedural / auxiliary|mutable canonical status refresh with preserved prior copies|canonical_status_closure_update.json files[].before_sha256/after_sha256/change_scope andclosure_sha256|

## Boundaries

Reported survey hours remain distinct from encoded high-setpoint periods and positive heating-load hours. The selected record identifies Veneto, not a Padova household sample. Geometry, gains, equipment and schedules remain illustrative assumptions; outputs are useful sensible thermal demand. Model/replay/state/forcing checks, author calculations and review closure are not empirical validation. Missing whole-building benchmark, compatible observations, demonstrated novelty, national transfer and model-sensitivity evidence remain explicit.

6 file identities changed from the v03 registry snapshot; their identities and procedural revision scope are recorded. The eight canonical before/after pairs and all frozen producer-target hashes were checked. No scientific manuscript, review, worker data, external checkout or frozen run file was edited. No redistribution, publication, submission or contact is authorized by this index.
