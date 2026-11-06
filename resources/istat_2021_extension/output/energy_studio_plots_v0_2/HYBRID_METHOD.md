# The implemented hybrid model

This version uses a **survey-driven hourly thermal model calibrated with one
EnergyPlus reference simulation**. The calibration consists of two annual
multipliers, one for sensible heating and one for sensible cooling. The same
multipliers apply to every scenario. No other EnergyPlus simulation enters the
hybrid prediction.

## 1. Define the scenario matrix

The default is a 100 m² dwelling in three contexts (apartment, multifamily,
single-family), three climates (Milano, Roma, Palermo) and four household
archetypes: 36 scenarios. Geometry, exposed walls, roof/floor boundaries,
glazing, U-values, infiltration and thermostat settings come from the saved
configuration. Each climate uses an EnergyPlus-database EPW reference year.

Changing dwelling context changes both the chosen physical exposure and the
corresponding survey cohort. A difference between contexts therefore cannot
be attributed to exposure alone. Holding the household profile constant via
the per-archetype overrides supports a more controlled exposure experiment.

## 2. Derive household inputs from the survey

The application uses training households only to fit group means and scenario
profiles. Existing archetypes retain their saved labels. Optional refits use
weighted standardisation, optional PCA and weighted k-means. Heating and
electricity group counts are separate settings.

For each climate-region, dwelling context and archetype, the profile contains:

- Survey-weighted household size.
- Heating hours and appliance kWh pooled from supported, fitted energy-group
  cell predictions over that cohort.
- Survey-weighted cooling hours, with the configured fallback when unavailable.

A small cohort falls back to the archetype-wide training cohort. The profile
CSV records the source and support count. Overrides can replace household
size, heating/cooling hours and appliance kWh.

Reported daily hours constrain **assumed** hourly thermostat schedules.
Heating prioritises morning/evening hours and cooling afternoon/evening hours;
fractional hours interpolate occupied and setback temperatures. The source
does not identify the actual hourly schedule. Daily occupancy and appliance
profile shapes are also configurable assumptions. The appliance profile is
normalised to the annual household appliance estimate, which itself was
modelled from the survey, not read from a meter.

## 3. Run the simplified thermal model

Every scenario has a single-zone hourly 1R1C model: one effective heat-transfer
resistance and one lumped thermal capacity. Its heat balance is

`C × dT/dt = H_out × (T_out − T) + H_ground × (T_ground − T) + G + Q_heat − Q_cool`.

Here, `C` is the configured capacity per floor area times the floor area.
`H_out` combines exposed-envelope and infiltration conductances. `H_ground`
represents the ground-contact floor, where present. Adiabatic boundaries
contribute no external heat transfer. `G` includes sensible occupant gains,
the configured appliance heat fraction, and an approximate window-solar gain
calculation from EPW direct and diffuse radiation.

With `H = H_out + H_ground`, an hourly step uses exponential decay
`d = exp(−H × 3600 / C)` to calculate the free-running temperature. The target
temperature clips this result to the assumed heating/cooling setpoints.
The required constant hourly ideal thermal power is
`Q = (T_target − T_free) × H / (1 − d)`.
Positive values represent heating; negative values represent cooling.
Four repetitions of the first weather week provide spin-up, excluded from
annual demand. These ideal sensible loads exclude HVAC efficiency and latent
conditioning. EnergyPlus resolves a more detailed surface heat balance.

## 4. Calibrate against one EnergyPlus reference

The default reference is **Roma / Apartment / A1**. Run both EnergyPlus and the
1R1C model with the same reference inputs. Let `E` denote annual sensible
demand in kWh, `r` the reference, and `u` heating or cooling:

`alpha_u = E_EnergyPlus(r,u) / E_1R1C(r,u)`.

In the completed default run `20260929_140859_a59dda`:

| End use | Reference EnergyPlus, kWh | Reference 1R1C, kWh | Multiplier |
| --- | ---: | ---: | ---: |
| Heating | 889.138 | 1410.541 | 0.630352 |
| Cooling | 1052.320 | 1827.109 | 0.575948 |

These values describe that configuration and reference weather only. A new
configuration recalculates them. A reference 1R1C annual demand at or below
`1e-6 kWh` leaves the factor unavailable. The code does not invent a division
by zero or borrow a factor from another end use.

## 5. Transfer the correction to each scenario

For every scenario `s` and hour `t`:

`q_hybrid(s,t,heating) = alpha_heating × q_1R1C(s,t,heating)`

`q_hybrid(s,t,cooling) = alpha_cooling × q_1R1C(s,t,cooling)`.

The scenario-specific 1R1C run supplies the response to weather, exposure,
household gains and schedules. The fixed reference factors rescale that
response. Sum the hourly values for monthly or annual results. Appliance
electricity remains the shared survey-derived input and receives no hybrid
calibration.

The hybrid therefore still calculates the simplified model for all scenarios.
Its saving is in avoiding additional **EnergyPlus** simulations. In
reference-only mode, it runs exactly one EnergyPlus case. Full-benchmark mode
runs all cases for evaluation, but the hybrid factors still use only the
reference case.

## 6. Evaluate the transfer, separately from calibration

The reference annual hybrid totals match EnergyPlus by construction, when
the factors are available. Exclude that reference from error metrics and
scatter plots. The default comparison therefore has 35 non-reference cases.
The error is `hybrid − EnergyPlus`: positive means overestimation.

This is a model-to-model assessment, not validation against metered buildings.
One global multiplier assumes that the reference correction transfers to
different climates, exposures and household profiles. The ratio may vary in
reality. Annual calibration does not establish an hourly match, correct peak
loads, or accuracy for a new building. Archetypes represent survey patterns;
they do not remove these physical-model assumptions.

## Source and outputs

- `analysis.py`: household/group fitting and scenario profile selection.
- `physics.py`: schedules, geometry and hourly thermal balance.
- `energyplus.py`: detailed reference/benchmark simulation.
- `engine.py`: reference ratios, transfer, error metrics and report generation.
- Run `config.json`, `scenario_profiles.csv` and `manifest.json`: exact inputs,
  profile support and calibration factors.
- Run `hourly/<scenario>/hybrid.csv`: transferred hourly predictions.
- Run `energyplus/<scenario>/hourly.csv`: independent benchmark output.

The no-EnergyPlus comparison uses the uncalibrated 1R1C output. Strict data-only
mode instead leaves unsupported heating/cooling kWh blank; survey hours alone
do not supply thermal energy.
