# Independent scientific review 03

Date: 20 October 2026. Reviewer task: `/root/review_03`; requested model `gpt-6-astra`; actual runtime model metadata not exposed, therefore unconfirmed.

Reviewed immutable draft: `06_claim_drafts/article_claims_v02.md`, SHA-256 **91373b60fe5f574057f2e2fdfc597cf9e373c090ae4d2aba4757682b183c0045**. All 38 claim blocks were read. This is an independent first-pass critique, saved before access to earlier review files, author responses, issue registers, closure records, author packets/build generators, or `01_plan/decisions.md`. The supplied correction/weather method excerpts were used. Ledgers were used to locate primary material, not as proof.

## Recommendation

The frozen v02 is scientifically appropriate as an **incomplete claim checker that withholds physical results**. Its distinctions between survey hours, encoded high-setpoint duration, actual heating delivery, useful demand, and measured accuracy are supported. However, **do not admit the newly completed v4 physical outputs into v03**: my independent runtime experiment finds an additional material roof/solar/longwave preprocessing defect, despite valid IR conversion and closed hourly network equations. This is R3-01 below. A new run version and fresh forcing checks are necessary. The existing v4 output remains useful only as diagnostic provenance.

This review does not count a completed draft–review–revision cycle by itself, establish novelty, or validate empirical accuracy. It does not assess any later corrected campaign.

## Primary inspection and reproducible calculations

The reviewer wrote and executed only `07_reviews/reviewer_03/independent_audit.py` and `runtime_forcing_check.py`, with outputs in that directory. Failed preliminary audit attempts were due to raw-field whitespace and the compact file's semicolon delimiter; the final script handles both and completed successfully. No manuscript, simulation, external checkout, or environment was modified.

* Chapter: read the complete extracted target text for PDF pp2–11/printed pp1302–1311, and visually inspected the source-page renderings of Table 3 and both map figures (PDF pp7–9). Adjacent papers were excluded. The title, 2026 publication versus CLIMA 2025 event, 1C description, coarse heating schedule construction, partial historical weather replacement, national comparison, regional totals and repository statement support their bounded uses in the draft.
* Survey: opened the original tab-delimited 2013 microdata, compact semicolon-delimited input, official questionnaire PDF pp41–43, and official region classification HTML line7. Independently obtained Veneto code5, respondent3523, 4/4/1, timer option2, 372 eligible rows and minimum eligible ID3523. Raw fireplace fields are blank, compact values zero. Neither activates the inspected fireplace increment. Questionnaire wording confirms mean winter-day operating hours and coarsening of fractional answers, not observed thermostat settings.
* Source mapping: inspected `C:/Works/GRINS/GRINS-Tool/classes/EndUse.py` lines593–656, including same-ID reseeding, night rotation, 22/19°C mapping, and seasonal zero lower setpoint. The three exported profiles retain counts4/4/1. Equality of these counts does not imply equal heater runtime.
* Literature: opened the complete primary [Prataviera et al. 2022 article](https://www.research.unipd.it/bitstream/11577/3417286/9/1-s2.0-S0306261922001568-main_compressed.pdf), particularly Section3.1.1/p6 and its Verona measurement boundary, and the primary [MODENA BSA2024 chapter](https://pro.unibz.it/library/bupress/publications/fulltext/9788860462022_21.pdf), Sections2.2–3.2. These support LIT-002/003 and materially constrain novelty. The archetype DOI endpoint failed in this review; LIT-004's qualified access status should remain.
* Physics and forcing: inspected the actual `thermal_zone.py` preprocessing and pure-network algebra, `weather.py` reader/solar-position pathway, `single_commit_1c.py`, `ir_sky_forcing.py`, frozen v4 runner/setup, protocols, raw EPWs/JSON metadata and campaign files. Checked the primary [EPW field dictionary](https://bigladdersoftware.com/epx/docs/24-2/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html), [sky-temperature reference](https://energyplus.readthedocs.io/en/v26.2.0/guides/engineering-reference/5.1-climate-calculations.html), and [JRC TMY documentation](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/pvgis-typical-meteorological-year-tmy-generator_en). The IR fourth-root relation and one-hour energy/rate conversion are supported.
* Numerical audit: independently integrated all **40 v3/v4 annual CSVs, 350,400 hourly rows**, checked all three node balances, continuity, finiteness, sequence, cooling, active setpoints, profile counts, repeated outputs, warm-up contrasts, hourly hashes and source-package hashes. The v3 maximum absolute node residual is 2.0236257114e-10W, reproducing the draft's rounded value. The v4 maximum is 1.7462298274e-10W. These are equation-closure results only; both campaigns have invalid preprocessing for physical interpretation.
* Raw weather: independently recomputed HDD18/CDD26 from daily mean temperatures, annual horizontal irradiation and mean RH; they agree with `climate_descriptors_v2.csv`. All four files have 8760 opaque-cover99 entries. The one negative wind is Palermo zero-based row3631, −0.1m/s. Independently derived mean air-minus-effective-sky values are Padova12.775656K, Bolzano17.620782K, Roma13.551708K, Palermo14.073727K. Actual WeatherFile cloud-derived scalars before replacement are approximately −81.51, −95.81, −81.01 and −79.76K.
* Validation evidence: directly opened `PaduaDigitalModel/ANACI-processed/AB_GB_MERGED_NO Duplicates BASE.geojson`, SHA-256 a2f5103dc6f3e44bc8379627792ce33f14f13e62c6bbad96cfec79d669113863. Confirmed16117features, 1001 populated STDMC_2021/TSet_cal/ExtWallCoef_cal records, 1000 positive STDMC_2021 entries and505 populated Verified flags. Names/counts do not establish meter definition, period, thermal/final-energy boundary or independence from calibration. The proposed access plan appropriately requests those missing details.

Exact inspected v4 completion-manifest hashes:

| Run directory | SHA-256 of completion_manifest.json |
|---|---|
| climate_campaign_v4/Padova | 55d825acccd2e300b4da675687bfa2e5149ea4b2cb869a75a73ee237dc971ab9 |
| climate_campaign_v4/Bolzano | 4558b22ffaf4cde87e18d2e376807ac85f743d37cbd6ea958a61e2436e6b3ab6 |
| climate_campaign_v4/Roma | 189562ab243200ae316742bf01a5e99601a0299ccc9d771a8962aef5e7aedddb |
| climate_campaign_v4/Palermo | 6c0c0177f70917e72ab389c081341b9740804614a7d15c08eb0fccad70ae5d80 |

All inspected frozen runner snapshots hash to32a29f2af3f478b0ff5936900fb8d690e3de0298c2e26aabdf1567ea9491cd96. Single-commit module: e549b4e5645ad8ae159fddf644d225d28622e1495f6fd64768c5242389b82c16. IR correction: d2e09d070e8e2635adc171022c0df7f3e545a5e107811d41ea39ea3db775eb8d. Per-hour-file hashes are in the reviewer audit JSON.

## Findings

### R3-01 — Critical: candidate v4 envelope forcing is still wrong, and its saved longwave gate does not test the implemented forcing

**Claims:** MET-003, MET-005, MET-008, MET-009; prospective admission of RES-004–007 and INT-001. Frozen v02 correctly withholds these physical results; this finding rejects their replacement by v4.

**Precise locations:** `C:/Works/EUReCA/EUReCA/eureca_building/thermal_zone.py` lines803–813 and839–841; `04_analysis/ir_sky_forcing.py` lines39–50; `05_simulations/climate_campaign_v4/Padova/seeded_longwave_check.json`; reviewer `runtime_forcing_checks.json`, every site.

Inside `calculate_zone_loads_ISO13790`, irradiance, radiative coefficient and opaque area are assigned inside the window-presence branch. The case's80m² roof has no `window` attribute. Its opaque forcing consequently reuses locals from the preceding wall:19.2m² opaque area and that wall's irradiance plane. The saved longwave checker computes the expected coefficient using each surface's own area, but never compares it to the actual distributed node forcing.

I instantiated the frozen zone setup, independently recomputed the IR scalar, regenerated its actual node forcing, and matched the exported CSV forcing within9.1e-13W. Then I changed only the sky scalar to zero and recovered the aggregate radiation term from `phi_m/(Am/Atot)`. In Padova the saved checker asserts−289.993840W, whereas actual preprocessing changes by−160.549424W: a129.444416W discrepancy. The corresponding discrepancies are178.535791W Bolzano,137.307464W Roma and142.596620W Palermo. These quantities concern the preprocessing forcing before node distribution, not annual heating errors.

The stronger invariant test confirms the cause: swapping two walls in the same physical surface list changes Padova zero-solar forcing by10.219296W and the maximum full-solar forcing by596.254333W. Putting the roof first raises `UnboundLocalError` for `TRV`. Equivalent full-solar order discrepancies occur at every site. A physically unchanged surface ordering must not change forcing.

**Consequence:** the IR conversion itself passes, but the envelope-forcing admission condition in MET-009 fails. The problem affects both shortwave and longwave terms. Closed node balances simply propagate these incorrect inputs. All v4 annual/peak/partition/summer-temperature effects remain inadmissible; paired errors cannot be presumed to cancel.

**Checkable remedy:** implement a bounded local preprocessing correction that assigns area, radiative coefficient and irradiance independently for every exposed surface, with glazing contributions separately conditional on glazing. Preserve the external checkout and frozen outputs. Before reruns, independently verify per-surface and total shortwave/longwave forcing against actual `phi_st`/`phi_m`, zero-solar and zero-longwave perturbations, surface-order invariance, and roof-first/opaque-only cases. Preserve all state checks. Freeze new code/protocol/input hashes, run a new Padova gate then the remaining matched sites under a new version, and recompute summaries from new hourly files. An updated expected-coefficient JSON alone is insufficient.

### R3-02 — Minor, but required for later interpretation: carry the retained sky/time approximations into the claim manuscript

**Claims:** MET-009, RQ-001/002, RES-004/005/007, INT-001, LIM-002.

**Locations:** weather-method excerpt sections Evidence and method/Field and time audit; `weather.py` solar-position calculation around lines195–206; JRC TMY documentation sections1 and2.3; candidate v4 forcing records.

The new method uses the annual mean of hourly air-minus-sky temperature, not hourly IR-derived longwave exchange. Native solar position is evaluated at pvlib's fixed-UTC+1 interval-start labels without applying the stated irradiance offset. JRC explicitly instructs applying the irradiance offset when calculating sun height. Row correspondence establishes correct pairing, not exact astronomical timing. These approximations are documented in the candidate protocol but are not yet explicit scientific limitations in frozen v02's claim blocks.

**Consequence:** a future generic statement that “weather forcing is validated” could imply a stronger timing/physics result than the checks establish, particularly for the small placement effects and hourly maxima. This is not evidence that a specific effect changes sign, and it does not by itself invalidate a clearly conditional comparison using a declared model convention.

**Remedy:** add atomic method/limitation wording giving the annual-mean approximation, index/offset convention and unquantified timing sensitivity; retain the distinction between valid IR input handling and accurate hourly environmental forcing. Any claim of robust physical peak ranking or general climate mechanism would additionally require a declared timing/forcing sensitivity comparison. Do not infer an undocumented UTC shift merely from matching labels.

### R3-03 — Minor: state the closest prior schedule-timing experiment explicitly

**Claims:** LIT-002, GAP-001, NOV-001.

**Location:** Prataviera2022 Section3.1.1 p6, bullet “Start-up time of the HVAC system”; Table1 p7.

The closest precedent does more than random reconstruction: it introduces a uniform discrete ±2h start-up shift because the survey does not identify startup time. The draft's broad summary is true, and its novelty remains properly provisional, but it omits the feature closest to the selected question.

**Consequence:** describing only prior randomization understates the established timing-sensitivity precedent and makes the proposed contribution appear broader.

**Remedy:** explicitly distinguish the earlier whole-profile shift/uncertainty experiment from this fixed respondent's three placements preserving each band's encoded high-setpoint duration. Retain “provisional” and avoid claiming the first study of survey-derived timing uncertainty.

### R3-04 — Minor: remove contradictory current-status pointers from canonical methods records

**Claims:** MET-007, MET-010, LIM-006/007 and manuscript provenance generally.

**Locations:** `03_methods/simulation_protocol.md` opening Correction disposition says current results use v3; its newly appended weather status points to v4. The validation protocol/matrix likewise contain successive post-run descriptions requiring a single current disposition. Immutable run snapshots must remain unchanged.

**Consequence:** a reader following a provenance link can encounter an obsolete positive current-result statement even though v02 withholds v3 and this review rejects v4.

**Remedy:** retain historical chronology but clearly label each historical status and add one authoritative present disposition. Mark v4 physics as withheld for R3-01, preserve its numerical consistency findings, and only point to a subsequent campaign after it passes its explicitly scoped checks.

## Full-rubric assessment and interpretation limits

| Dimension | Assessment |
|---|---|
| Source correspondence | Conference examples and scope checked. Published gas deviations independently recompute to+0.094063% and+12.962841%. No inherited household/hourly accuracy is warranted. LIT-004 remains access-limited. |
| Extension and novelty | The narrow conditional experiment is distinguishable from a national-stock model and legitimate even if differences are null. Three-band reconstruction, survey-informed EUReCA uncertainty and timing perturbation already exist. No proven novelty claim. |
| Methods and inference | Fixed dwelling, fixed response, fixed22/19°C mapping and climate-E seasonal mask across weather files constitute a controlled weather-exposure comparison. They do not represent local populations or local operating regulations. Earliest/latest are not extrema; the200 possible contiguous start combinations (5×5×8) are not exhaustively sampled. |
| Numerical provenance | All v3/v4 hourly arithmetic and network balances were independently checked; current package hashes match their manifests. Arithmetic correctness does not rescue the preprocessing error. Exact candidate run hashes are above. |
| Simulation validity | Single-commit candidate evaluations preserve the old state in direct heating/off/cooling trials. All-hour equations/continuity pass. IR handling and active-path guards pass their limited tests. R3-01 blocks physical admission. |
| Validation independence | Technical checks, independent-model benchmarks, measured validation and official aggregates remain different layers. No matched independent observed pilot reference was found. ANACI annual attributes require definition and calibration-history checks. No calibration was identified in this pilot. |
| Uncertainty | The design estimates deterministic contrasts, not a probability distribution. Single response/dwelling, envelope, gains, fixed seasonal operation, annual-mean sky, solar phase, model structure and weather vintage limit generality. Repeats/warm-up quantify numerical repeatability and initialization only. |
| Overstatement | v02's withheld effects, no-runtime-equivalence, no-empirical-validation and no-population-inference wording are appropriate. Preserve them. Do not promote v4 numerical directions as scientific findings. |

One additional diagnostic for the eventual interpretation: all three v4 Bolzano peaks occur at hour6912,16 October00:00, immediately after the imposed seasonal lower-setpoint change from0 to19°C. The previous air temperature is about9.742°C. Thus their identical maxima reflect this common season-start recovery in these defective-forcing files, rather than demonstrating that ordinary within-day timing has no peak influence. After correction, report each maximum's hour, preceding/current setpoints and state; distinguish seasonal recovery if it persists. Keep the predeclared annual maximum as the primary endpoint rather than deleting inconvenient peaks. This is a diagnostic to recheck, not an admissible physical result.

## Review limits and artifacts

No new full annual campaign or isolated replay was executed by this reviewer. The20 completed v4 campaign cases and20 v3 cases were recalculated from saved hourly records; the separately reported seeded replay was not independently re-executed. The runtime test reused the frozen geometry setup intentionally, then used independent perturbation/recovery equations; it is implementation verification, not an independent building model.

I did not independently rerun v1/v2 transition diagnostics, authenticate historical conference executable/configuration, obtain the archetype full text, perform a systematic novelty search, reproduce every literature result, acquire a standard benchmark specification, quantify envelope/weather/timing/model-form uncertainty, or obtain independently documented Padova meter data. I checked the candidate active path and dictionary-access guards, not every possible future EUReCA configuration or dictionary access API. The unused invalid wind is an explicit exception for this bounded path, not a whole-weather-file QC pass.

`independent_audit.json` SHA-256: a098b37b6150e9f81519d78c988c13f491ad5bde6e7a5c4555c2b65ba0bcb7af.

`runtime_forcing_checks.json` SHA-256: 08d093075ffc06c1ffc33bdf9f5bd8da6b3dfbe1f14f76931518fe50e66d6509.

This independent critique is now complete. A later, explicitly labelled closure pass may inspect earlier issue history; it must not overwrite this first-pass diagnosis.
