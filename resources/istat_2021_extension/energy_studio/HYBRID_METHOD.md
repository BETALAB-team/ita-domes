# Milan reference to Palermo: hybrid method 0.4

## Question and data boundary

Given one simulated reference space in Milan, estimate the same floor area under another climate, dwelling exposure and household archetype. The default is Milano / Multi-family / A2. Target EnergyPlus outputs are not inputs; they are optional evaluation evidence.

The hybrid is an empirical reference energy signature with static corrections. It has no indoor-temperature state, thermal capacitance, differential equation or target thermal simulation. It estimates annual totals only.

## 1. Survey profiles

Use training households, survey weights and the supported macro-region × dwelling × archetype cohort. By default, heating hours and appliance electricity pool fitted H/E group means over this cohort. Cooling hours use valid reporting users. The cohort_means option uses direct weighted heating/appliance means instead. Household size is a separate weighted supported cohort mean.

H/E cells use the selected factors. Missing selected factors and unsupported cells are excluded, not filled. Require the configured minimum valid training count (default 30) and retain counts/coverage. Cooling presence remains a grouping factor for electricity, independently of excluding insulation. Survey indices are not changed by manual simulation overrides.

## 2. Run or reuse the reference

EnergyPlus simulates the reference's annual sensible heating/cooling using configured exposure, construction, schedules and weather. Its 8760 hourly records are aggregated into 365 daily totals for fitting.

An optional completed Energy Studio run can supply the reference. Reuse requires exact generated epJSON equality, matching reference weather SHA-256, finite nonnegative hourly values and annual/hourly reconciliation. Only the matching reference model and output are copied. Target results in that folder are not read by the predictor.

An annual reference total without daily or hourly information is insufficient to fit the weather response. Arbitrary external EP exports are not imported by this release.

## 3. Static weather and building features

For each case:

- H = H_out + H_ground, in W/K, from nominal envelope conduction and infiltration.
- T_eq,d = (H_out × T_out,d + H_ground × T_ground) / H.
- S_d = mean projected window solar gain on day d / H, in K.
- G = members × sensible W/person × mean occupancy fraction + appliance kWh × 1000/8760 × heat fraction, in W.
- Mean thermostat = setback + (reported operating hours / 24) × (setpoint − setback), separately for heating and cooling.

T_out,d is the mean EPW outdoor temperature. The floor-ground temperature is constant. Solar projection is a geometric approximation, not EnergyPlus's full solar/shading calculation. These averages cannot capture thermal storage or the timing of gains relative to operation.

## 4. Fit the reference weather response

Fit heating and cooling independently against the reference daily EnergyPlus totals:

~~~text
Heating: y_d ≈ slope_h × max(B_h − T_eq,d − alpha_h × S_d, 0)
Cooling: y_d ≈ slope_c × max(T_eq,d + alpha_c × S_d − B_c, 0)
~~~

Search configured balance-temperature B and solar-coefficient alpha grids. For each pair, fit the positive slope by least squares; select the smallest reference daily mean squared error. Record fit error, active days and search-boundary flags. Default grids: heating −5 to 30°C; cooling 5 to 45°C; 0.25°C steps; alpha 0 to 1.5 by 0.1.

Require at least 20 active reference days per end use. Insufficient activity produces an unavailable estimate, not an invented slope. All these fit choices use the reference series only. Reference daily fit error is in-sample, not validation.

## 5. Transfer the balance point and annual energy

For each end use u, reference r and target t:

~~~text
B_t,u = B_r,u
        + (mean_thermostat_t,u − mean_thermostat_r,u)
        − (G_t/H_t − G_r/H_r)

D_t,heat = sum_d max(B_t,heat − T_eq,t,d − alpha_heat × S_t,d, 0)
D_t,cool = sum_d max(T_eq,t,d + alpha_cool × S_t,d − B_t,cool, 0)

E_t,u = E_EP,r,u × (H_t/H_r) × (D_t,u/D_r,u)
~~~

The same reference alpha and balance fit serve every target. The annual normalization ensures reference identity. Increased internal gains lower the heating threshold and increase cooling degree sums. Exposure affects conductance, ground coupling and irradiation. ISTAT hours shift mean thermostat settings; **they do not multiply a degree-day ratio a second time**.

This is an explicit approximation. For example, changing daily hours may have effects that a mean-setpoint shift misses. Balance points may not transfer across very different envelopes or operating patterns. Weather outside the reference temperature range is recorded in the audit.

Missing/unsupported required ISTAT inputs produce unavailable values. No target EP energy enters the computation. The switches adjust_exposure, adjust_operating_hours, adjust_internal_gains and adjust_solar allow sensitivity checks. Disabling corrections does not make unsupported survey cohorts valid.

## 6. Appliances and comparison baseline

Appliance transfer retains:

~~~text
E_t,appliance = E_EP,r,appliance × ISTAT_appliance_t / ISTAT_appliance_r
~~~

The original baseline uses this ratio form separately for heating hours, cooling hours and appliance kWh. It has no target climate or exposure response and is labelled ISTAT-ratio baseline. Selecting method istat_ratio disables the new signature model.

Appliance agreement with EnergyPlus is not independent evidence: its annual simulation input uses the same reconstructed survey quantity.

## 7. Freeze before evaluation

Save the reference fit, all annual predictions and sequential component variants before starting target EnergyPlus runs. Store/check SHA-256 hashes. The fixed sequence is weather only; weather plus exposure; plus hours; plus gains. Its apparent contributions depend on that order.

Only after freezing may target EnergyPlus scenarios be simulated. Exclude the reference from error metrics. No benchmark error feeds back into balance temperatures, solar coefficients, coefficients or frozen predictions.

Report MAE, RMSE, bias and CVRMSE = RMSE / mean benchmark energy. Also report pooled signed error and WAPE to expose cancellation. These are descriptive errors over the configured unweighted scenarios, not sampling confidence intervals. Relative errors are sensitive to Palermo's small heating denominator; include kWh errors.

This experiment is exploratory. The scenario matrix informed method development, although target energy is not used in parameter fitting. An untouched scenario set and measured-building validation are still needed.

## Three methods, simply

1. EnergyPlus: simulate a virtual home in each setting.
2. Hybrid: observe how one simulated home responds to weather, then use climate, exposure and survey information to estimate another setting.
3. ISTAT only: report household hours and estimated appliance electricity. Do not turn uncalibrated hours into absolute thermal kWh.

Only EnergyPlus has hourly/monthly/load-duration outputs. The hybrid's daily climate features are not claimed as target daily load predictions.

## Reproduction and references

The implementation is in signature.py, transfer.py, analysis.py, reuse.py and engine.py. Exact settings and data hashes are saved per run. Unit tests are in tests/test_energy_signature.py and tests/test_istat_transfer.py; check_signature_workflow.py verifies reuse and prediction equivalence to a completed benchmark without running EnergyPlus.

[CIBSE TM41, Degree days: theory and application (2006)](https://cibse.org/knowledge-research/knowledge-portal/technical-memorandum-41-degree-days-theory-and-application-2006-pdf/?id=a0q20000008I73TAAS) provides the building-energy-balance basis for building-specific degree-day base temperatures. The transfer above is this project's empirical extension, not a claim to implement or validate TM41 exactly. EPW source URLs and checksums are recorded in each manifest.
