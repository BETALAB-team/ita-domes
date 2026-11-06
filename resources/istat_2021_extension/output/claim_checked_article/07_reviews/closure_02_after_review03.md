# Closure of review-02 issues after independent review03

20 October 2026. Task `/root/review_03`; requested model `gpt-6-astra`; actual runtime model metadata remains unexposed. This is a **second pass**, performed only after the independent full critique was saved. `review_03.md` remains unchanged, SHA-256 `dda6e7288e34af7368f458cc40390297efa7b3da0ab097cabb1cb606e9e77e34`.

Inspected historical records:

| Record | SHA-256 |
|---|---|
| review_02.md | 3e258716d7516c05d8b78c306b358ff5651d204ed39110b831652877fa761fd4 |
| response_02.md | 1f6069e8dbfcdae9692199c759ec55293e83a98871aff1e19b28bcfaa86796d2 |
| issue_register_02.json | 50f28e39ffb1867b063e333be8dfe991d04abd1f2b6b76b9477853affb3b14c9 |
| article_claims_v02.md | 91373b60fe5f574057f2e2fdfc597cf9e373c090ae4d2aba4757682b183c0045 |

These records were compared against the independent primary-source/code/runtime/CSV inspection already documented in review03 and the additional replay artifacts inspected below. No new annual simulation was executed. No later v5 correction or campaign is assessed here.

## R2-01 — Specific missing-cloud treatment conditionally closed; composite physical-admission gate remains open

The author correctly accepted the issue and withdrew v3 physical magnitudes in v02. RES-004/005/006/007 and INT-001 remain withheld, with LIM-007 recording the forcing defect. DAT-002 appropriately separates source provenance from model compatibility. The raw EPWs still contain the original missing opaque-cover99 values; their preservation is correct.

The **narrow corrective method is independently supported**:

* `04_analysis/ir_sky_forcing.py`, SHA-256 `d2e09d070e8e2635adc171022c0df7f3e545a5e107811d41ea39ea3db775eb8d`, derives effective sky temperature from valid horizontal IR using the documented Stefan–Boltzmann constant and replaces the scalar before zone-load preprocessing. It does not clip or impute cloud99.
* The weather-method excerpt explicitly records one-hour units, paired air/IR records, annual-mean aggregation, pvlib fixed-UTC+1 indexing, unapplied irradiance offsets and the limited active-path QC exception.
* Independent raw-file calculations and actual WeatherFile execution reproduce the corrected mean differences of12.775656,17.620782,13.551708 and14.073727K for Padova/Bolzano/Roma/Palermo. The valid IR-derived scalar eliminates the original negative-sky-difference defect.
* The runtime access guards reject `wind_speed` and `opaque_sky_coverage`; the inspected active case uses fixed ACH, no natural/mechanical ventilation and no PV. The single invalid Palermo wind record is retained and explicitly excluded from claims of universal weather quality. This supports only the bounded path; it is not a declaration that every loaded weather field is valid.
* The twenty v4 annual files reproduce numerically with state continuity, node balances, repeatability, warm-up and accounting checks. Their exact source/run/output hashes are recorded in `reviewer_03/independent_audit.json`.

However, R2-01's original closure condition also required **correct actual envelope forcing**, not simply a valid sky scalar and expected sign. That part fails under **R3-01**. In the frozen native preprocessing, the roof lacks a window attribute and therefore reuses opaque area/irradiance locals from the preceding wall. The independent actual-forcing perturbations and surface-order checks in `reviewer_03/runtime_forcing_checks.json` establish the failure; the expected-coefficient checker does not detect it. This is a distinct downstream defect, not failure of the IR fourth-root transformation.

**Recommended register disposition:** “Conditionally closed for the specific missing-cloud-to-sky transformation and bounded active-field handling; physical result admission remains open and is additionally blocked by R3-01.” If the register supports only one `still_open` flag for the full original composite issue, retain `still_open: true` until the actual envelope-forcing correction and subsequent versioned campaign are independently checked. Do not classify the supported scalar repair itself as unverified simply because the downstream pipeline still fails.

No v3 or v4 physical effect magnitude, direction, setback-energy fraction or summer-temperature result is admitted by this closure. Annual-mean sky and solar-timing approximations remain explicit limitations under R3-02 even after the downstream defect is corrected.

## R2-02 — Closed for recorded command provenance and tested isolated replay

The historical response clearly annotates the missing SITE argument without rewriting v3 manifests. I independently rehashed all four v3 completion manifests against the values recorded during my first-pass audit: all remain unchanged.

The v4 runner now:

1. Parses an explicit `SITE`, required `--campaign` and optional `--replay-only-seeded`.
2. Restricts campaign names and resolves output directories inside `05_simulations`.
3. Refuses existing completed/hourly-output destinations before simulation.
4. Records executable, working directory, complete `command_argv`, source/input hashes and separate destination.

Primary replay evidence inspected:

* `05_simulations/climate_campaign_v4_replay/Padova/completion_manifest.json`, SHA-256 `712e8c1d1c22a933ec7645c765bbafca12c45de823e7ff155716ad5ea3aa8591`.
* That directory's frozen adapter, design, summary, hourly file and full-step check.
* `05_simulations/climate_campaign_v4_replay.log`, containing the imported path, weather index and completed annual case.
* `05_simulations/v4_replay_interface_checks.json`, documenting rejection of the completed primary destination and equality of33before/after file hashes.
* Direct inspection of the runner's overwrite guard and argument handling.

The recorded replay command vector specifies:

```text
C:\Works\EUReCA\EUReCA\.venv\Scripts\python.exe
-B
C:\Works\GRINS\GRINS-Tool\resources\istat_data\output\claim_checked_article\04_analysis\run_climate_campaign_v4.py
Padova
--campaign
climate_campaign_v4_replay
--replay-only-seeded
```

Recorded launch directory: `C:\Works\GRINS\GRINS-Tool\resources\istat_data`.

I directly compared the replay and primary seeded hourly bytes: both SHA-256 `2ba2e81abb095ff47eee0bbdbd4f8b969972d9558db67e6b7119eb88d33c0fcc`, byte-identical. Their frozen runner sources and input hashes also agree; replay source and summary match their manifest hashes. This is concrete evidence of a tested isolated replay, even though I did not execute another annual run. The saved overwrite-rejection test was inspected, not independently re-executed.

**Disposition:** close R2-02 (`still_open: false`) for this provenance/interface defect. Future scientific run versions must retain the same complete command and immutable-output discipline. Reproducibility of a defective physical calculation does not imply physical correctness, so R3-01 remains unaffected. The extra seeded replay is a reproducibility check, not a new household, climate, or independent scientific sample.

## Earlier issue dispositions carried in register02

| Issue | Second-pass disposition |
|---|---|
| R1-01, bounded single-commit state repair | Retain conditional closure for the declared ideal sensible1C state transition. My independent all-hour residual/continuity checks corroborate the narrow numerical conclusion. Neither R2-01 nor R3-01 reopens this specific state-update repair; both concern forcing. |
| R1-02, reported hours versus encoded duration/runtime | Retain closure. Independent raw questionnaire/response/source inspection supports v02's explicit distinction. Old or current defective-forcing partition quantities remain withheld. |
| R1-03, measurement-inventory evidence class | Retain closure. DAT-004 is a bounded derived-estimate availability statement, not an observed heating dataset. Direct ANACI inspection confirms candidate annual attributes requiring provenance and calibration-independence checks; it does not establish a compatible meter series. |

Additional verification metadata are saved in `07_reviews/reviewer_03/closure_checks.json`. This pass closes only the issues and subcomponents stated above; it does not modify the independent review03, revise the manuscript, admit physical results, or assess the forthcoming preprocessing correction.
