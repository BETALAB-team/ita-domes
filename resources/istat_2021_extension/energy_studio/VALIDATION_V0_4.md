# Energy Studio 0.4 acceptance record

Checked on this Windows workstation, 30 September 2026. This records software checks and an exploratory model benchmark, not measured-building validation.

## Scientific benchmark

- Completed run: energy_studio_runs/20260929_225633_26db04.
- One Milan / Multi-family / A2 reference, 100 m²; 36 total cases across Milano, Roma and Palermo, three dwelling contexts and four household archetypes.
- EnergyPlus 25.1: 36 successful simulations, zero reported warnings.
- Predictions and sequential component variants were frozen before target simulations. Reference daily outputs alone determined balance temperatures and solar coefficients. Neither parameter was at a grid boundary.
- No predictor change was made in response to these target results. The scenario matrix had been examined during method development, so this is not an untouched external validation set.

| Method | End use | Non-reference cases | MAE kWh | RMSE kWh | CVRMSE % |
| --- | --- | --- | --- | --- | --- |
| ISTAT-ratio baseline | Heating | 35 | 2003.149 | 2215.822 | 102.863 |
| Climate/exposure/gains hybrid | Heating | 35 | 78.025 | 101.126 | 4.694 |
| ISTAT-ratio baseline | Cooling | 35 | 440.913 | 545.307 | 45.424 |
| Climate/exposure/gains hybrid | Cooling | 35 | 176.592 | 223.927 | 18.653 |

Palermo's 12 cases: heating RMSE 100.579 kWh, CVRMSE 19.608%; cooling RMSE 306.963 kWh, CVRMSE 20.256%. The overall heating percentage is not a Palermo accuracy claim. These are annual sensible ideal loads, not purchased energy.

## Software and reuse

- 22 unit tests passed across test_energy_studio, test_energy_studio_plots, test_istat_transfer and test_energy_signature.
- Source reuse integration: energy_studio_runs/20260930_002817_8b3d36. Zero new EP simulations; all 36 hybrid predictions match the full benchmark within rtol 1e-12 / atol 1e-9 kWh. A mock prohibited any EnergyPlus or RC execution.
- Built EXE GUI self-test passed.
- Packaged fresh-reference run: energy_studio_runs/packaged_v0_4/fresh/20260930_003505_23a178. Exactly one EP simulation; all predictions match source benchmark within rtol 1e-10 / atol 1e-7 kWh.
- Packaged reuse run: energy_studio_runs/packaged_v0_4/reuse/20260930_003531_1f582d. Zero new EP simulations; same prediction agreement.
- Packaged acceptance details: energy_studio_runs/packaged_v0_4/acceptance.json.
- Default execution is reference-only. Benchmarking must be enabled explicitly.

## Report

Packaged EXE refreshed the full benchmark with PowerPoint enabled; exit code 0. Revision: energy_studio_runs/20260929_225633_26db04/reports/20260930_004155_67d01a.

35 slides and 53 native charts. Package integrity, layout and embedded chart-data checks passed. Speaker-note text is empty. All initial rendered slides were reviewed individually; the changed ratio-axis slide and added climate-accuracy slide were reviewed after revision, and unchanged slide bodies were compared pixel-for-pixel. Native PowerPoint UI rendering was not tested.

## Remaining limits

One configured reference and one fixed scenario matrix are not broad external validation. Constant ground temperature, assumed daily profiles, approximate solar projection and the mean-control/gain transfer remain modelling assumptions. Cooling error remains appreciable. No target hourly/monthly predictions are supplied. Appliance agreement shares the same reconstructed input and is not independent validation.

EnergyPlus and the local presentation-export runtime are separate dependencies. The scientific engine and GUI are bundled, but the package is not a fully self-contained PowerPoint-export installation for another computer. Existing outputs and the main branch were preserved; no commit was made.
