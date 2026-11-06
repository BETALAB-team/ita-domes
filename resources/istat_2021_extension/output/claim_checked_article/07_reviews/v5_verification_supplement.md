# Independent v5 source, forcing and output verification supplement

20 October 2026. Reviewer task `/root/review_03`; requested model `gpt-6-astra`; runtime model metadata unexposed. This supplement follows the saved independent review03 and the separately authorized prior-issue closure pass. It does not replace either record or constitute independent review04. No annual simulation was executed by this reviewer.

**Finding:** the specific surface-preprocessing defect R3-01 is corrected in the inspected v5 implementation. Independent per-surface, actual-forcing perturbation and order-invariance tests pass for all four weather exposures. All twenty completed v5 annual cases also pass the independent numerical checks performed here. Their results may support **conditional fixed-case modeled contrasts** in the next claim revision, subject to the retained limitations and the planned fresh fourth review. This is not empirical validation, a general certification of EUReCA, or proof of robust physical peak rankings.

The original review03 remains unchanged, SHA-256 `dda6e7288e34af7368f458cc40390297efa7b3da0ab097cabb1cb606e9e77e34`. All earlier v1–v4 physical outputs remain quarantined.

## Inspected implementation and exact run identity

Read `03_methods/surface_correction_protocol_v5.md`, pre-run readiness, `04_analysis/surface_forcing_1c.py`, runner and checker source, then independently tested the frozen correction imported from the Padova run snapshot. The author's saved pass flags were not used as verification.

| Item | SHA-256 |
|---|---|
| surface_forcing_1c.py, identical across four v5 run snapshots | 24768556f49ef341791647502f0fe5d1f8baddc3e5c2fcf5df23a124f106883a |
| run_climate_campaign_v5.py / frozen runner | 48b3175c67eb3983f2059f8de32f24480a3374090572b59e7c2e37c7f3ff8f2f |
| surface_correction_protocol_v5.md | 2860820cd4a3f1600dd517bc38a38dd11e6098f61435ff54a72304ec9b767c7c |
| climate_campaign_v5/Padova/completion_manifest.json | 9dba850692e4b0048f6ca4b8a3d92c0352f45ecd221f6484ed2560f6ef4289b5 |
| climate_campaign_v5/Bolzano/completion_manifest.json | 0dd8c29bd8dd018235ae30355cfd9ac34e9bab0751104bd602d09e552719181e |
| climate_campaign_v5/Roma/completion_manifest.json | 06a6b478127c1cc009e59639ef18a5b786b0f965a567b67894aca4000cdb46e3 |
| climate_campaign_v5/Palermo/completion_manifest.json | cdbf8e3b58553f20d80a6f4e9e9cbbdaf58c1a41f7832b036eb6cd059082d926 |

Direct comparisons confirm unchanged response, building design, cases/warm-ups, cooling bound, calendar and energy boundary. Every case retains identical hourly heating-setpoint and outdoor-temperature arrays relative to v4. Static network parameters, single-commit stepping and IR correction are unchanged. Frozen source hashes agree with manifests; current external package sources agree with their recorded hashes. The changes are local preprocessing and its verification/export path.

## Independent forcing tests

Reviewer script: `07_reviews/reviewer_03/verify_v5_forcing.py`. This reuses the reviewer's earlier frozen-geometry fixture to instantiate the same physical surfaces, opens each raw EPW through WeatherFile, independently computes the valid-IR scalar, imports the frozen v5 correction and tests its actual output. It does not call the author's `check_surface_forcing_v5.py`.

For each of the five exposed surfaces at each site, the reviewer independently reconstructs opaque shortwave, opaque longwave and glazed shortwave contributions from the live surface's own area, orientation, material and window parameters and its own weather irradiation plane. Window angular response is independently evaluated using `scipy.interpolate.BSpline`, rather than the author's `splev` call. The reconstructed surface sum is distributed to nodes using independently formed weights and the declared190.08W convective/47.52W radiant internal gains.

Checks executed:

* Each exposed surface alone reproduces its own contribution when all surfaces are combined, including the zero-glazed80m² roof.
* Reversed order, rotated order and roof-first order reproduce the same node forcing. Roof-first no longer throws an exception.
* Setting only the air-minus-sky scalar to zero and recalculating actual node forcing recovers the expected longwave term.
* Setting direct/global solar irradiation to zero leaves exactly internal radiant gain plus the expected longwave loss before node distribution.
* Independent reconstructed node forcing matches the actual exported solver forcing in **all twenty completed annual cases**, not just the preprocessing export or an expected coefficient.

Across sites, maximum order/node reconstruction discrepancies are approximately2.73e-12W, well below the declared1e-8W tolerance. The actual longwave perturbation errors are at most2.73e-12W; zero-solar errors are at most5.69e-14W. These floating-point residuals establish the corrected implementation for the tested geometry and native radiation convention.

| Site | Correct total opaque longwave contribution, W |
|---|---:|
| Padova | −289.993840 |
| Bolzano | −399.973064 |
| Roma | −307.609399 |
| Palermo | −319.458673 |

These are the implemented annual-mean longwave contributions before distribution to the thermal nodes. They are not measurements or annual heating corrections. In particular, the Padova implementation now uses the expected−289.993840W, instead of the erroneous−160.549424W diagnosed in v4.

## Independent annual-output audit

Reviewer script: `07_reviews/reviewer_03/independent_audit_v5.py`, using the same independently written standard-library balance/accounting equations as review03, pointed only at v5.

All20annual CSVs were complete when inspected: five cases at each of Padova, Bolzano, Roma and Palermo, **175,200 rows** in total. Every file has8760finite sequential rows; stored hashes agree; block counts remain4/4/1. All current/previous state continuity errors are zero. The maximum absolute air/surface/mass residual is **1.7681145437e-10W**. Active heating setpoint errors and cooling demand are zero. All four repeated seeded outputs are exactly identical. Maximum annual-summary integration discrepancy is1.82e-12kWh; peak summaries match exactly.

Warm-up annual relative changes are tiny: the largest is approximately1.08249e-8 as a fraction (Palermo), below0.005; every maximum hourly load is unchanged. This establishes initialization stability for the tested seeded cases, not structural or statistical uncertainty.

The following values were independently derived directly from the hourly files. Percent denominators are the same site's seeded annual demand or seeded hourly maximum; they are not population estimates or uncertainty intervals.

| Site | Seeded heating, kWh | Early annual change, % | Late annual change, % | Seeded hourly maximum, kW | Early peak change, % | Late peak change, % |
|---|---:|---:|---:|---:|---:|---:|
| Padova | 9574.018168 | +0.894239 | +0.337144 | 10.049540 | +2.029262 | −2.103222 |
| Bolzano | 15851.213775 | −0.097977 | +0.026815 | 17.808621 | 0 | 0 |
| Roma | 6602.228771 | +2.032903 | +0.500007 | 8.909927 | +8.325986 | +3.381353 |
| Palermo | 5078.837200 | +3.868129 | +0.981468 | 8.292527 | +5.969112 | +1.524234 |

The Padova seeded descriptive partition independently gives1323positive-load hours at19°C and2375.617488kWh delivered during those setback hours. Its maximum simulated air temperature is41.046114°C. These are post-hoc diagnostic/model-state quantities, distinct from the predeclared primary contrasts; they must retain that label if included. The temperature is not evidence of measured summer comfort.

All three Bolzano maxima still occur at hour6912,16October00:00, at the common transition from the0°C off-season lower setpoint to19°C. Report this seasonal recovery origin with any zero peak difference. It does not imply that ordinary daily schedule timing has no peak effect. The fixed season is deliberate for the weather-exposure comparison, but does not represent each location's actual regulations or household practice.

## Issue disposition and permitted use

**R3-01: conditionally closed for this exact local preprocessing correction and inspected v5 campaign.** The fix addresses the demonstrated missing-per-surface-initialization cause, and the required actual-forcing/order/opaque-only tests and new versioned annual checks are now independently verified. No correction to the external EUReCA checkout is inferred or requested.

**R2-01:** the previously verified IR transformation plus the now verified downstream surface correction satisfy the specific sky-to-envelope implementation requirements for the bounded model. This supplement updates the earlier closure pass's still-open physical-admission condition with new evidence. The underlying v3 missing-cloud simulations and v4 stale-surface simulations remain inadmissible.

**R2-02:** its closure remains limited to the documented/tested command and isolated replay interface. This supplement does not independently re-execute any v5 replay.

**R3-02–04:** manuscript treatment of annual-mean sky/solar timing, closest ±2h startup literature, and canonical status consistency still requires author revision. This supplement does not close those editorial/scientific-scope requirements.

The author may now replace appropriate withheld placeholders with v5 numerical contrasts, exact provenance and bounded language such as “under this fixed illustrative geometry, response, seasonal mask and documented native weather-time convention.” It would remain unsupported to describe these as observed behaviour, accurate household prediction, improved reconstruction, representative climate/population effects, savings, uncertainty bounds or universally robust peak rankings. One early/late pair across four coordinate TMYs does not isolate which weather variable causes the differences.

## Limits and reproducibility artifacts

This is a targeted source/forcing and saved-output verification. The weather-to-plane transposition algorithm and its native solar-time convention were reused, not independently validated against astronomical/irradiation observations. Annual-mean longwave, fixed gains/envelope/season, absence of measured thermostat schedules, one response/dwelling and missing empirical reference remain limitations. No new standard benchmark, independent building model, uncertainty ensemble, observation dataset or annual rerun was produced by this reviewer. The full chronology of orchestration was not independently reconstructed from system event logs.

The supplemental checks do not alter frozen v02, the original critique, or historical outputs. Exact per-file hashes, peak rows, partitions and residuals are retained in:

| Reviewer artifact | SHA-256 |
|---|---|
| reviewer_03/v5_forcing_checks.json | cea309f33597f20f7e78d3d9a8e65b2f383ab7e7ea8133dd6a243267c104b1d6 |
| reviewer_03/independent_audit_v5.json | 1f20eed3570602f43ac547693549c303c288e3af20a788d680f2d650704e2c22 |
| reviewer_03/verify_v5_forcing.py | 016be5f34091b9ea4a937ec65e78fae549fd82910a9eaaf15cf112d8a61d1147 |
| reviewer_03/independent_audit_v5.py | efefcf16d6e79278448d4d5bb9cf147b70060544639f1f62f75fd74b05be3815 |

`reviewer_03/v5_provenance_comparison.json` separately records unchanged design/setpoint/weather/network inputs and correction-source identity. Fresh review04 should inspect the resulting v03 claim text against these primary artifacts and retain all unperformed-validation limits.
