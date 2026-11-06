# Pre-run climate campaign addendum v2 — 20 October 2026

This addendum is frozen after the successful new local-weather Padova pilot and its analytical network checks, before any official-weather campaign result. The original v1 protocol is archived unchanged in the pilot directory. The following is a documented change in weather provenance and scope, not an unrecorded replacement of pilot inputs.

## Inputs and coordinate selection

Fresh official JRC PVGIS API v5_3 TMY EPW and JSON files were requested with the same2005–2023window and horizon enabled. JSON reports SARAH3 solar radiation, ERA5 meteorology, DEM horizon/elevation and selected TMY months. Acquisition URLs, date, response status, metadata and SHA256 are stored in `05_simulations/climate_inputs_v2/acquisition_manifest.json` and `audited_metadata.json`. These are TMYs, not contemporaneous measurement-year files. Both input metadata and EPW time convention must be retained: JSON irradiance-time-offset and the EPW offset differ by local-standard-time convention; neither is silently used to shift the series. pvlib reads the delivered EPW hourly series into fixedUTC+1, then coerces2023 for a shared calendar, without DST.

Select Padova45.408N11.873E and three additional coordinate cases Bolzano46.500N11.350E, Roma41.900N12.500E, Palermo38.120N13.360E. Names identify coordinates; they do not make these files representative of the whole municipality, climate class or building stock. Provider elevation and horizon are properties of the retrieved cells/locations and must not be treated as surveyed building altitude. Selection is based on inspected descriptors before model runs:

| Coordinate case | Daily HDD18 Kd | Daily CDD26 Kd | Annual GHI kWh/m² | Mean RH% |
|---|---:|---:|---:|---:|
| Padova |1983.168|94.525|1507.689|76.104|
| Bolzano |3521.419|0.000|1455.099|67.386|
| Roma |1459.450|55.513|1642.606|75.264|
| Palermo |1140.930|22.982|1727.514|74.669|

These provide3.09-foldHDD18range and a1455–1728kWh/m²solar range, relevant to a heating schedule experiment. CDD26 is largest in Padova among these files; do not claim a simple north–south cooling gradient. Legal climate classifications are neither needed nor asserted. Candidate metadata, numeric QC, calculation code and descriptor table were saved before selection. No model outputs informed which three additions were retained.

## Controlled design, counts and scope

All five configurations from v1 (seeded/early/late at168h warm-up; repeated seeded168h; seeded336h) will be run anew under each of four official TMYs:20annual runs. Repeat/warm-up are technical checks, leaving12scientific case outputs. No population/stock sampling or retrofit scenario is added. Same2013 Veneto responseid3523, source22/19°Ccontrol mapping, climate-E seasonindices105/288,80m²assumed dwelling, geometry/envelope/gains/infiltration and1Cmodel are held fixed across all sites. Climate-E season is a controlled assumption outside Padova; no claim of local operating regulation is made. Use independent fresh zone/schedule objects; no alias sharing.

Primary outcomes and checks remain v1. Additional outputs: fixed-input schedule paired contrasts for each site and interaction as difference of schedule deltas relative to Padova. These are deterministic conditional sensitivity outputs. Ratios/percentages use the seeded demand at the same site as denominator. Early/late are not minimum/maximum over every feasible schedule. A site with low annual heating may have proportionally larger percentage change; report absolute values too. Annualthermal demand is useful sensible heat; no final fuel/electricity, empirical accuracy, confidence interval or comfort result is inferred. Zero active cooling allows high summer free-floating temperatures and does not establish acceptable comfort.

## Technical clarifications

The source-selected indicator`q_3_13_2=1` denotes a timer/schedule control in the official questionnaire, not a measured thermostat temperature. The exact source indicator-based22/19°Cmapping is retained; the questionnaire supplies average winter-day on-hours including occupied/unoccupied cases, not monitored schedules. Some ancillary fireplace/control blanks are coded0in the compact model input. Their treatment is an explicit preprocessing assumption; only the directly verified fields may be described as matching the raw survey. This wording corrects imprecision in v1 documentation after its run; it changes no numeric pilot input.

EUReCA's weather reader warns atRH=100%because it tests strictly<1after converting percent. All downloaded files pass independent0–100%rangeQC;100%RHhours are52/16/15/18. The warning is retained, input humidity is not altered. The analytical network check passes zero-gradient equilibrium, steady convective heating and independently recomputed transient air/surface/mass residuals; it verifies network algebra only, not empirical validity or an external benchmark. No full campaign heat-balance residual has yet been recomputed from exported states.

Before each campaign execution, archive immutable adapter, original protocol, this addendum and config/design snapshots with hashes, actual EPW+JSON checksums, source-package hashes, import path, environment and command. A completion flag requires all numerical gates and independent postprocessing integration checks. Any failures remain visible and block scientific use for the affected case.
