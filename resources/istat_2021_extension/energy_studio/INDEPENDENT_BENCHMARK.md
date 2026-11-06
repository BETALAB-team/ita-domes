# Independent household schedule benchmark

Version 0.5 separates the EnergyPlus household inputs from the ISTAT predictor. The earlier shared-input workflow remains available and its results are unchanged. The new built-in cases are synthetic experiments, not measured Italian households or a normative schedule set.

## Run it

The v0.5 packaged application opens with the independent benchmark configuration. Click Run experiment. To run only one reference and make predictions, uncheck “Simulate every scenario for benchmarking”. That does not evaluate target accuracy.

From the repository IDE or terminal:

```powershell
.\.venv\Scripts\python.exe run_energy_studio.py --write-independent-config independent_config.json
.\.venv\Scripts\python.exe run_energy_studio.py --config independent_config.json --run
```

The EXE accepts the same arguments. EnergyPlus 25.1 and EPW paths are configurable. PowerPoint export requires the separately installed report runtime described in README.md; scientific CSV, PNG, SVG and HTML outputs do not require it.

## Two distinct tests

| Test | EnergyPlus inputs | Hybrid target inputs | Interpretation |
| --- | --- | --- | --- |
| Known aggregates | Independently specified 8,760-hour schedules | Member count, mean presence, equivalent thermostat hours, annual appliance kWh | Thermal transfer error when aggregate behaviour is known |
| ISTAT profiles | The same independent schedules | Demographics assign the archetype; frozen survey cohorts supply behaviour indices | Thermal transfer plus survey-profile prediction error |

Both tests fit the same single reference EP result. Target results do not enter fitting or prediction. Known-aggregate appliance agreement is excluded from accuracy metrics because annual appliance energy is supplied directly. The ISTAT appliance prediction is evaluated separately.

The known-aggregate test intentionally receives more information than the ISTAT-profile test. Their error difference is not a causal estimate of the effect of using ISTAT. It includes uncertainty in members, hours, appliances and the assumed mean presence. The synthetic profiles were chosen for transparent contrasts, not statistical representativeness.

## How the benchmark is independent

1. Build or import household schedules without accessing ISTAT.
2. Save every input hour and its SHA-256 checksum before simulation.
3. If requested, fit energy groups using only the survey training split. Existing archetypes use the saved demographic model, after checking that it reproduces the survey labels exactly. Refit mode saves its own demographic transform and centres.
4. Assign independent households using demographics only. Heating and appliance behaviour are not assignment features.
5. Run the independent reference case in EnergyPlus.
6. Fit its daily energy signature. Save annual target predictions and checksums before running any target simulation.
7. Run target EP cases for evaluation, without refitting. Exclude the reference from accuracy statistics.

Turn off `benchmark.compare_istat` to run the thermal-transfer test without reading any household CSV or demographic model. The automated tests enforce this separation.

No target 1R1C model, indoor-temperature integration or target hourly hybrid prediction is used. The underlying EnergyPlus benchmark is still the same simplified one-zone model. Geometry, material assumptions and weather are shared experimental controls, so this is behavioural-input independence, not independence of all modelling assumptions.

## Where each input acts

For case i, the independent hourly inputs are member count X_i, presence fraction p_i(t), appliance power P_i(t), heating setpoint T_H,i(t) and cooling setpoint T_C,i(t).

- EP People: X_i × p_i(t), with the configured sensible watts/person.
- EP ElectricEquipment: the supplied P_i(t) watts. Annual appliance electricity is e_i = sum(P_i(t))/1000 kWh, with hourly intervals.
- EP thermostat: the supplied hourly temperature schedules. The ideal system remains available at setback temperatures; “inactive” hours do not imply zero heating/cooling energy.
- Weekday/weekend inputs expand to 365 days beginning on a Monday, with no daylight-saving changes or holidays. Imported hourly data use the same explicit calendar.

The hybrid compresses these inputs:

```
pbar_i = mean(p_i(t))
h_i = 24 × (mean(T_H,i) − T_H,setback) / (T_H,active − T_H,setback)
c_i = 24 × (mean(T_C,i) − T_C,setback) / (T_C,active − T_C,setback)
G_i = X_i × W_person × pbar_i + e_i × 1000/8760 × appliance_heat_fraction
H_i = exposed-envelope conductance + infiltration conductance + ground conductance
```

The equivalent h_i and c_i are mean-setting summaries, not measured hours with positive load. Binary active/setback schedules give ordinary active hours/day averaged over the year.

For either heating or cooling:

```
B_target = B_reference + (Tbar_target − Tbar_reference)
                        − (G_target/H_target − G_reference/H_reference)

D_H,i = sum_days max(B_i − T_equivalent,i − alpha_H × S_i, 0)
D_C,i = sum_days max(T_equivalent,i + alpha_C × S_i − B_i, 0)

Q_target = Q_reference × (H_target/H_reference) × (D_target/D_reference)
```

The reference EP result identifies B_reference and alpha. The target EPW supplies daily temperature and the solar feature. The existing geometry and ground-coupling definitions in HYBRID_METHOD.md are unchanged.

In the known-aggregate test, the target X, pbar, h, c and e come from the independent input summaries. In the ISTAT-profile test, the supported survey indices replace X, h, c and e; pbar retains the application's configured assumption because ISTAT does not supply an hourly presence profile. The reference features always describe the actual reference simulation.

For appliance electricity, the known-aggregate transfer reproduces the supplied e_target. The ISTAT version uses Q_appliance,reference × e_ISTAT,target / e_ISTAT,reference. This is an anchor-normalised prediction, not a direct meter reading or a validation of the original appliance conversion model.

## Edit profiles without changing code

Export the editable synthetic specification:

```powershell
.\.venv\Scripts\python.exe run_energy_studio.py --write-benchmark-profiles my_profiles.json
```

Select that JSON under Data & execution → Independent profiles JSON. Set the reference profile ID separately from the survey archetype number.

Each profile has a unique ID, label, member count and demographics. Education uses the existing 0–1 attainment proxy, not a qualification code. For existing labels, the model uses PCA of minimum age, maximum age and member count, followed by scaling of the first two PCs plus education and nearest frozen centre. No arbitrary manual A-label assignment is used.

For generated schedules, each `weekday` and `weekend` contains:

- `occupancy`: 24 fractions between zero and one.
- `heating_on_hours` and `cooling_on_hours`: unique hour-beginning integers 0–23. Active and setback temperatures come from Building & controls.
- `appliance_w`: 24 nonnegative watt values. The appliance demand is not rescaled to an ISTAT annual total.

Alternatively set a profile's `hourly_csv` path. Relative paths resolve beside the profile JSON. Its required columns are:

| Column | Meaning |
| --- | --- |
| hour | Ordered integers 1–8760, with no gaps or duplicates |
| occupancy | Fraction of household present, 0–1 |
| heat | Heating setpoint, °C |
| cool | Cooling setpoint, °C |
| appliance_w | Appliance electricity demand, W |

All values must be finite. Heating must remain below cooling, and temperature schedules must lie within the configured active/setback bounds so that the aggregate-hour definition remains comparable. Changing the building setpoints does not rewrite an imported CSV. Measured HVAC on/off availability needs a separate modelling extension; do not disguise it as a thermostat temperature.

The document requires `evidence_kind` (`synthetic`, `external_schedule` or `measured_schedule`) and a nonempty `source`. These are provenance declarations, not automatic certification. Even measured schedules would validate the behavioural inputs only; EP thermal outputs are not measured building demand.

## Current experimental evidence

The completed 45-case run combines Milano, Roma and Palermo with apartment, multi-family and single-family exposures and five profiles. The reference is Milano / Multi-family / P2. All EP cases completed without warnings.

| Test | Heating CVRMSE | Cooling CVRMSE |
| --- | --- | --- |
| Known aggregates | 4.90% | 32.97% |
| ISTAT profiles | 11.34% | 37.68% |

These metrics exclude the reference and use 44 target cases. P2_shift moves P2's occupancy, thermostat and appliance timing by eight hours without changing its aggregate inputs. The known-aggregate hybrid therefore gives the same estimate, while EP gives different thermal loads. The timing figure compares only EP and the known-aggregate test, avoiding the ISTAT test's special observed-reference information advantage.

Cooling errors are retained; the reference fit was not tuned against these target results. No empirical accuracy claim follows from this synthetic experiment.

## Saved files and source code

Each run preserves configuration, schedule source declarations, input-hour CSVs, demographic assignments, survey model files, frozen predictions, reference fit, per-term transfer audits, case errors, metrics and all EP input/output files. Plots include PNG, SVG and the plotted CSV data. Report refresh creates a new revision and does not rerun simulations or overwrite the original results.

Implementation: `independent_inputs.py` defines/imports schedules and assigns demographics; `independent.py` enforces the reference-first information boundary; `energyplus.py` builds the detailed hourly schedules; `signature.py` performs the annual transfer; `independent_report.py` and `independent_deck.mjs` produce plots and reports.

The repository workbook `resources/user_data/schedules_EN16798_Residential.xlsx` was inspected but not adopted as independent empirical evidence. It contains residential schedule data, not measured household records; its filename alone does not verify normative conformity. The default benchmark explicitly uses synthetic assumptions instead.

The schedule and internal-gain implementation follows the [EnergyPlus 25.1 schedule documentation](https://bigladdersoftware.com/epx/docs/25-1/input-output-reference/group-schedules.html) and [internal-gain object documentation](https://bigladdersoftware.com/epx/docs/25-1/input-output-reference/group-internal-gains-people-lights-other.html). These references describe EP behaviour; they do not justify the chosen synthetic household schedules.
