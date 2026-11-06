# Selected coordinate climates — current v5 candidate

**Current scientific status: v5 — reviewed conditional model illustration.** All 66 earlier annual executions are quarantined. The current campaign has 20 annual cases (12 primary, 4 repeat and 4 longer-warm-up checks), plus one separate genuine seeded replay that is not another scientific sample. Total execution count is 87. Independent technical review 03 supplement passed; fresh full-claim review 04 and final v04 closure passed for the bounded conditional claims (07_reviews/final_closure.md); no empirical validation is claimed. Exact frozen pre-run protocols/configs/source snapshots remain unchanged under each run directory. This canonical index was updated after execution and does not replace those pre-run records.

Sites were selected from source/QC and degree/solar/humidity descriptors before the original expanded thermal runs. Subsequent implementation corrections did not change the sites, TMY data or selection. Shared JRC PVGIS v5_3 inputs use SARAH3 radiation, ERA5 meteorology, 2005–2023 and a DEM horizon. The immutable pre-run choice is in `simulation_protocol_v2_addendum.md`; complete acquisition metadata and QC are in `05_simulations/climate_inputs_v2` and `weather_forcing_v4`.

| Coordinate label | Latitude | Longitude | Provider elevation m | HDD18 Kd | CDD26 Kd | GHI kWh/m² | Mean RH % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Padova |45.408|11.873|25|1983.168|94.525|1507.689|76.104|
| Bolzano |46.500|11.350|272|3521.419|0.000|1455.099|67.386|
| Roma |41.900|12.500|72|1459.450|55.513|1642.606|75.264|
| Palermo |38.120|13.360|29|1140.930|22.982|1727.514|74.669|

HDD18 and CDD26 use daily mean temperatures and fixed bases 18/26°C; they are not legally defined degree days. This coordinate set spans 3.09-fold HDD18 and annual GHI 1,455–1,728 kWh/m². Padova has the largest CDD26 among the files, so no simple north–south cooling ranking is asserted. Provider grid/elevation/horizon values do not establish complete municipal climates or local building-stock representativeness.

The controlled comparison holds the assumed dwelling, one Veneto response, high-setpoint durations, schedule alternatives, equipment boundary and climate-E season fixed while changing weather. It is not a locally representative case comparison or empirical climate-transfer validation. Native annual-mean longwave, irradiance-time interpretation and the explicitly guarded unused invalid wind value remain limitations. Separate official-source municipal climate-class context is in `01_plan/climate_classification_context.md`; it does not change timing or establish legal compliance.

Post-review status annotation by the orchestrator: the saved final closure supersedes procedural pending-review labels in historical sections. Frozen protocols and run snapshots remain unchanged.
