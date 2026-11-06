# Independent scientific review 02

Date: 20 October 2026. Reviewer role requested: `gpt-6-astra`; runtime-confirmed model metadata: unknown. Fresh-context independent pass, with emphasis on methods, comparability and reproducibility.

Reviewed input: `06_claim_drafts/article_claims_v01.md`, SHA-256 `5e35bc02e77d6345a9c82be51a8a405d5d4ba3e8490f6c24d037b8488a04bab1`. The matching `claim_ledger_v01.json` contains 35 claims, matches this hash, and reproduces all exact claim texts.

I read the human request, research questions, full rubric and correction-method excerpt. I did not read review 01, author responses, issue registers, author build generators/private reasoning, or `01_plan/decisions.md` before saving this critique. References to the earlier defect in the current manuscript and protocol excerpt were necessarily visible. I did not treat those references as independent proof.

## Recommendation

**Keep scientific admission of the v3 result values blocked.** The new single-commit selector passes the independent state/equation checks performed here. All 175,200 exported hourly states close the declared network equations, and the reported annual and peak values reproduce arithmetically. However, I found a separate critical weather-forcing error: the code consumes the EPW missing opaque-cloud-cover sentinel `99` as a physical cloud amount, causing very large nonphysical sky-temperature forcing in all four sites. The numerical balance tests faithfully balance that incorrect forcing.

Preserve the valid solver correction, provenance, survey distinctions and constrained question. Correct and verify weather-to-load preprocessing before rerunning and reconsidering the result claims. No empirical validation or general software validation follows from this review.

## Findings requiring disposition

### R2-01 — Critical: missing cloud-cover sentinel becomes spurious longwave heat gain

**Claims:** DAT-002, MET-003, MET-005, MET-006, MET-008, RES-004, RES-005, RES-006, RES-007, INT-001, LIM-005. DAT-002's acquisition provenance remains valid; the defect is use of those files by the model.

**Primary evidence and independent calculations:**

1. In each of `05_simulations/climate_inputs_v2/{Padova,Bolzano,Roma,Palermo}.epw`, all 8,760 records have `99` in the opaque sky-cover field, zero-based column 23. The [EnergyPlus EPW specification, Opaque Sky Cover](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v24.1.0/AuxiliaryPrograms.pdf) identifies physical values as 0–10 tenths and `99` as missing. The official indexed text was accessible; direct official PDF opening failed. The same specification text was readable in the [EnergyPlus 24.2 documentation mirror](https://bigladdersoftware.com/epx/docs/24-2/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html#field-opaque-sky-cover).
2. `C:/Works/EUReCA/EUReCA/eureca_building/weather.py`, WeatherFile initialization around lines 120–127, sends that unfiltered column to `_TskyCalc`. Lines 383–390 calculate `nopaque = 99*0.1 = 9.9`, then infrared cloud fraction `C = 8.91`, and `eps_sky = eps_clear + (1-eps_clear)*C`. This is outside the formula's intended physical cloud domain.
3. The independent standard-library calculation and an actual `WeatherFile` execution agree:

| Site | Invalid opaque-cloud hours | Calculated sky-emissivity range | Engine annual mean air minus sky, K | Diagnostic mean air minus sky from supplied horizontal IR, K |
|---|---:|---:|---:|---:|
| Padova | 8760 | 1.659–3.709 | −81.506 | +12.784 |
| Bolzano | 8760 | 2.371–4.128 | −95.815 | +17.629 |
| Roma | 8760 | 1.769–3.720 | −81.013 | +13.560 |
| Palermo | 8760 | 1.983–3.523 | −79.757 | +14.082 |

The last column is a diagnostic only, using the EPW's nonmissing horizontal IR and `T_sky = (IR/σ)^(1/4) − 273.15`; it is not a replacement campaign or a fully adjudicated repair. It establishes that useful alternative input exists and illustrates the scale of the sentinel error.

4. `thermal_zone.py`, `calculate_zone_loads_ISO13790`, around lines 837–842, subtracts a positive envelope coefficient times `weather.general_data['average_dt_air_sky']`. A large negative air-minus-sky value therefore adds heat instead of the expected net sky-cooling contribution. This term propagates into `phi_surface_W` and `phi_mass_W`. At Padova hour 0, when EPW solar irradiance is zero, the saved forcing is `phi_air_W=190.08`, `phi_surface_W=32.8468`, `phi_mass_W=1030.3358 W`; the declared internal radiant source is only `240*0.99*0.2=47.52 W`. The excessive environmental contribution is already present in the exported forcing that the residual checker accepts.
5. The campaign weather QC in `04_analysis/run_climate_campaign_v3.py` checks columns 6, 8 and 13–15, but omits the cloud field actually consumed by this preprocessing path. Hash matching and homogeneous provider metadata cannot detect this semantic missing-value error.

**Consequence:** The v3 annual demand, peaks, setback partition and maximum summer temperature are accurately reported properties of a run with invalid weather forcing. They cannot yet substantiate physical schedule-placement sensitivity. Holding the erroneous forcing fixed within each site does not make paired effects safe: heating clips at zero, controls switch by temperature, and climate-specific erroneous gains differ. No cancellation or preserved sign/magnitude has been demonstrated. The pending-review labels are appropriately cautious but should now become an explicit scientific-use hold/withdrawal of the candidate values until correction.

**Checkable remedy:**

- Preserve v3 files as diagnostic provenance and mark all dependent numerical results as withheld.
- Declare a local, documented longwave treatment before new results: for example, derive effective sky temperature from the provided valid horizontal IR with explicit units, timestamp convention and temporal aggregation, or obtain valid cloud inputs and justify the alternative correlation. Do not silently clamp `99` to a physical cloud value.
- Keep external checkouts unchanged if using a study-local adapter. Record every preprocessing departure and input checksum.
- Add a missing-value/range guard for every weather field consumed by the chosen calculation, plus independent night/zero-solar tests of sky temperature and opaque radiative forcing. Export enough intermediate longwave information to audit it; equation residuals alone are insufficient.
- Recompute the Padova gate first, then all affected matched weather cases and derived tables/partitions, preserving the same predeclared schedules and endpoints. Recheck node balances, continuity, summaries, repeat/warm-up behavior and physical forcing. Update claims and validation records with the new run version.

**Reproduction artifacts:** `07_reviews/reviewer_02/independent_audit.py` and `.json`; `runtime_checks.py` and `.json`. The former independently reads raw EPW/CSV values without importing any author checker; the latter confirms the actual weather routine and exercises the complete new stepper.

### R2-02 — Minor: recorded run command omits its required site argument

**Claims:** MET-003, MET-005, MET-006 and the run-provenance portion of RES-004/RES-005/RES-007.

**Evidence:** Each v3 completion/frozen manifest records a command ending in `04_analysis/run_climate_campaign_v3.py`, with no site argument. The script reads `SITE=sys.argv[1]` at line 8. Replaying the recorded command stops with `IndexError: list index out of range`, before simulation. Exact command and captured traceback are in `07_reviews/reviewer_02/recorded_command_replay.txt`. Site-specific designs and outputs make the missing argument inferable, so this is a documentation/replay defect rather than evidence that the recorded outputs were not produced. The source and output hashes inspected match.

**Consequence:** The advertised exact execution command is not executable as recorded, impeding independent replay and obscuring whether command, environment and site were actually bound together.

**Checkable remedy:** Record the full argument vector including site, working directory and executable for each future run. Provide one tested replay entry point with a separate output destination, so verification cannot overwrite archived evidence. Retain historical manifests immutably and add a clearly labelled provenance correction rather than rewriting them as if the original field had been correct.

## Independent checks and findings that passed

### Thermal state and bookkeeping

I inspected `single_commit_1c.py`, the actual `sensible_balance_1C` implementation at `thermal_zone.py:1011–1110`, the frozen Padova adapter, the author's full-step checker and transition-test source, and per-case static network parameters. For this sensible-only, ideal-load scope, the selector evaluates free and constrained states against the same beginning-of-hour air/mass temperatures and commits the accepted state once. The old mode flag does not force another physical timestep. The lack of moisture stepping is expressly scoped out and does not enter the declared sensible network.

I independently formed the three backward-Euler node equations and recomputed every saved row from exported states and forcing:

- 20 annual files × 8760 rows = 175,200 checked hours.
- Maximum absolute residual over all air/surface/mass balances: `2.0236257114447653e-10 W`; no residual exceeds `1e-6 W`.
- Old-state continuity errors are exactly zero for all adjacent exported hours.
- Heating setpoint adherence/off-state inequality passed; all paired primary cases have equal weather/load forcing and network parameters within site.
- Identical seeded repeats are exactly equal. Warm-up annual relative differences are approximately `2.79e-12`, `2.94e-12`, `3.18e-9`, and `8.16e-5` for Padova, Bolzano, Roma and Palermo; peaks are unchanged.
- Runtime tests call the actual `solve_1c_single_commit`, not just its selector, for off→heat, heat→off, off→cool, cool→heat and equilibrium. An independently assembled linear system solved with `numpy.linalg.solve` agrees within `4.55e-13` in the mixed W/°C output vector. These are targeted implementation checks, not an independent building benchmark.

The following reproduces the stored arithmetic, **not admitted physical findings after R2-01**:

| Site | Seeded useful heating, kWh | Early/late annual change, % | Seeded hourly peak, kW | Early/late peak change, % |
|---|---:|---:|---:|---:|
| Padova | 5947.725173 | +2.20233 / +0.67922 | 8.869599 | +1.67913 / −2.40776 |
| Bolzano | 10613.613573 | +0.41233 / +0.17664 | 10.779184 | −1.39433 / +2.28653 |
| Roma | 3325.654150 | +5.90200 / +1.43857 | 7.704409 | +8.16727 / +2.70309 |
| Palermo | 2321.090972 | +6.64083 / +1.80878 | 7.078467 | +6.11035 / +0.33720 |

Annual integration uses positive W times 3600/3,600,000; peaks use positive W/1000. The baseline denominator is consistently same-site seeded. Padova's post-hoc 19°C partition reproduces 2730 calendar hours, 671 positive-load hours and `893.21783218 kWh`. Its maximum air temperature reproduces `42.81476827°C`; R2-01 also affects that physical interpretation.

I verified all source/input paths and hashes listed in the v3 completion manifests, adapter/config/design/correction snapshots, and hourly hashes in every summary. There were no hash mismatches. The environment package inventory is present, including Python, NumPy, pandas, scipy and pvlib; this does not replace the command correction in R2-02.

### Survey, schedules, comparability and estimands

Directly reading the 20,000-record raw release and the compact input confirms selected ID 3523, region 5 and 4/4/1 reported hours. The filter has 372 eligible records and minimum ID 3523. Raw fireplace fields are blanks; compact fields are zero. The current manuscript preserves that difference and does not describe these as explicit negative raw answers.

I opened the original questionnaire PDF, extracted pages 40–43, and visually inspected page 41. Questions 3.11_A/B/C ask average winter-day hours in 05–13, 13–21 and 21–05, including an instruction to average occupied/unoccupied days and coarsening of fractional hours. Page 43 option 3.13_2 is a timer. The official classification HTML maps 5 to Veneto. No observed thermostat temperatures or exact switching times follow.

I independently extracted the actual three GRINS `User` methods and the seasonal assignment, executed them, and matched all 8760 setpoints in each Padova primary CSV. The high periods are:

| Case | Morning high period | Afternoon high period | Night high period |
|---|---|---|---|
| Seeded | 09–13 | 17–21 | 23–24 |
| Early | 05–09 | 13–17 | 21–22 |
| Late | 09–13 | 17–21 | 04–05 |

This is an asymmetric comparison: late versus seeded changes only the night placement, whereas early versus seeded changes all three blocks. That is consistent with the defined contrasts; any later mechanism discussion should make it visible. The three alternatives are neither population samples nor all 200 contiguous combinations (5×5×8). Current claims correctly avoid probabilities, confidence intervals and extrema. The design is descriptive deterministic sensitivity, without a fitted model or causal population estimand.

The same source-derived climate-E season and illustrative physical case are held fixed across coordinate weather exposures. I confirmed the weather reader produces fixed UTC+1 indices from 2023-01-01 00:00 through 2023-12-31 23:00; I did not independently establish every subhourly irradiation-offset convention. Provider ERA5/SARAH3/window/horizon metadata and all reported climate descriptors reproduce from the source JSON/EPWs. Homogeneous metadata are supported; model compatibility fails separately under R2-01.

### Conference source, nearest literature and extension

I independently opened the actual proceedings PDF, SHA-256 `51ce8fb615c15cf34087682eafbc7beb4267dab4e20a78bb49a844ccffa3450b`, extracted only target pages 2–11 and reviewed their text. I rendered and visually checked Table 3 and Figures 1–2 on pages 7–9. No adjacent-paper fragments were used.

- The target chapter and [publisher record](https://link.springer.com/chapter/10.1007/978-3-032-10546-2_123) support the 2026 publication/CLIMA 2025 distinction and ita-domes scope.
- Table 3 values in RES-001 are correct; independently calculated natural-gas deviations are +0.09406297% and +12.96284074%. These remain published results, not new replication.
- Figures 1–2 present modeled regional GWh totals without measured regional comparator panels or uncertainty bars. The chapter's partial 2013/14 weather replacement and national comparison do not validate this hourly illustrative case.
- The actual GRINS `main_functions.py:792–806` constructs a 2C building and calls `simulate_quasi_steady_state`; the dynamic call is commented. The manuscript appropriately avoids exact conference-executable equivalence.

The independently reopened [Prataviera et al. 2022 full article](https://www.research.unipd.it/bitstream/11577/3417286/9/1-s2.0-S0306261922001568-main_compressed.pdf), especially §3.1.1 and Table 1, supports LIT-002. In addition to random three-band profiles, it explicitly introduces a ±2-hour startup shift; schedule timing sensitivity is therefore an especially close precedent. The current claim of a particular conditional comparison remains appropriately provisional. Its aggregation/sampling differs from this fixed-respondent design.

The [BSA 2024 primary chapter](https://pro.unibz.it/library/bupress/publications/fulltext/9788860462022_21.pdf), §§2.2–3.2, supports LIT-003's ISTAT/EUReCA and climate/envelope/technology precedent. The [Khajedehi et al. publisher text](https://doi.org/10.1016/j.enbuild.2025.115924) supports LIT-004's restrained archetype statement; direct full-PDF access failed here too. [Koene and Eslami-Mossallam's publisher abstract/introduction](https://www.sciencedirect.com/science/article/abs/pii/S0360132322010691) is an additional close thermostat-timing/RC-model precedent, with inferred settings and a different district-level scope. Its full methods were not audited.

A limited independent search used `"heating" "schedule" "ISTAT" "sensitivity"`, `"within" "thermostat" "temporal dispersion" Koene 2023`, and `"heating" "duration" "timing" "EUReCA"`. It does not establish an exhaustive novelty gap. The exact proposed contribution is a bounded audit/sensitivity extension beyond the conference's aggregate and regional results, not invention of survey-based schedules, archetypes, climate scenarios or hourly EUReCA simulation. No new broad novelty claim is warranted.

### Validation independence and uncertainty

The validation protocol/matrix correctly distinguish numerical checks, a matched model benchmark, empirical evaluation, aggregate checking and robustness. Missing compatible measurements are not manufactured from GIS attributes, expenditure or the national BEN table. I inspected the local inventory/ANACI audit records and the minimum-data specification; I did not independently inventory every external GIS file. Thus DAT-004 remains a qualified inventory claim.

No empirical dataset was used for calibration, selection or held-out accuracy in this experiment. There is no supported empirical confidence interval or acceptable-error threshold to report. The absence of a matched independent building benchmark and measured validation is explicitly acknowledged, rather than disguised as a passing technical test. These limitations should remain even after the new preprocessing error is corrected.

## Coverage of all current claims

| Claim IDs | Independent review disposition |
|---|---|
| LIT-001–LIT-004 | Bounded primary-source correspondence supported at the access depths above; LIT-004 full PDF remains unavailable. |
| GAP-001, GAP-002 | Supported: hours do not identify switching times; national comparison does not establish pilot hourly accuracy. |
| NOV-001 | Appropriate provisional status; exact novelty unproven and timing precedent is close. |
| RQ-001, RQ-002 | Coherent fixed-case estimands; completed numerical comparison cannot be admitted before R2-01 correction. |
| DAT-001 | Raw/compact selection and declared preprocessing distinction independently checked. |
| DAT-002 | Acquisition metadata and hash correspondence supported; use of missing cloud values is R2-01. |
| DAT-003 | Explicitly assumed physical case, not observed or representative. |
| DAT-004 | Qualified inventory boundary reasonable; external measurement inventory not exhaustively repeated. |
| MET-001, MET-002 | Source-to-profile/setpoint mapping independently executed; heating runtime distinction correct. |
| MET-003, MET-004, MET-008 | Single-step sensible network and accounting verified within declared scope; preprocessing remains invalid. |
| MET-005, MET-006 | Numerical gates reproduce, but incomplete weather QC cannot support physical admission; R2-01 and command R2-02. |
| MET-007 | Appropriate separation of evidence layers; retain missing benchmark/empirical layers. |
| RES-001, RES-002 | Original chapter/table/figures independently checked; no new replication implied. |
| RES-003 | All source-derived profile/setpoint invariants reproduce. |
| RES-004–RES-007 | Arithmetic reproduces; scientific-use hold under R2-01. |
| INT-001 | Must remain withheld/reframed until forcing correction; no current valid physical effect estimate. |
| LIM-001–LIM-004 | Appropriate spatial, validation, schedule-bound and code-lineage limits. |
| LIM-005 | Numeric maximum reproduces but is affected by R2-01; no-cooling/comfort limitation remains necessary. |
| LIM-006 | Archived old call path uses the recursive dispatcher; quarantine is justified. I did not rerun old cases or independently confirm the exact earlier failure counts before this independent critique. |

## Unperformed checks and review limits

No complete independent annual rerun was made: I recomputed the exported annual files and ran independent transition/weather/source-schedule checks. I did not validate geometry/envelope construction against measurements, every transposed solar-gain term, subhourly solar timestamp conventions, a standard building benchmark, all past campaign outputs, or complete novelty literature. I did not contact data holders or obtain new measurements. Runtime tests and the primary data files were read in place; outputs were restricted to this reviewer's directory. The unrelated official PDF and archetype direct-fetch failures are recorded above.

The appropriate next step is a new documented forcing correction and rerun, followed by author revision and the next independent review. A successful author response alone cannot close R2-01 without new primary evidence. This review is an independent critique, not proof of empirical truth and not completion of the four-cycle prose gate.
