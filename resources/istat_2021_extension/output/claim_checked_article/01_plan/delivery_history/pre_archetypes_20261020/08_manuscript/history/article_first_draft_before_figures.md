# Within-band heating schedule placement in one survey-informed dwelling: a conditional four-weather simulation

**First manuscript draft — validation incomplete.** The results describe a single assumed dwelling. An independent building benchmark and a compatible independently documented meter comparison remain missing.

## Abstract

<!-- P01 -->
Coarse survey reports of heating-use duration leave within-band switching times unresolved. We compare three contiguous placements of the same encoded high-setpoint hours for one assumed 80 m² dwelling and one unweighted 2013 Veneto survey response. Reported mean winter-day hours of 4/4/1 in three eight-hour bands are encoded as repeated 22°C periods with a 19°C setback; these durations do not constrain actual heater runtime. A study-local ideal-sensible EUReCA 1C implementation first tests Padova-coordinate typical meteorological year (TMY) weather, then holds the dwelling, response and seasonal mask fixed under Bolzano, Roma and Palermo coordinate TMYs.

<!-- P02 -->
For Padova, annual useful sensible heating is 9574.02, 9659.63 and 9606.30 kWh for seeded, early and late placement, respectively. Early/late annual changes relative to the same-site seeded case are +0.894/+0.337% in Padova, −0.098/+0.027% in Bolzano, +2.033/+0.500% in Roma and +3.868/+0.981% in Palermo. Maximum hourly loads respond differently: Bolzano has zero paired differences because all three annual maxima occur at the common imposed seasonal restart. These deterministic comparisons are conditional on fixed physical inputs, a common heating season, annual-mean sky exchange and the native solar-time convention. They establish neither measured accuracy nor population effects. Sensitivity to physical and forcing assumptions remains unquantified.

## 1. Introduction

<!-- P03 -->
The CLIMA 2025 proceedings chapter by Vivian et al., published in 2026, presents ita-domes for survey-informed modelling of the Italian residential stock at regional resolution [1]. Its national annual final-energy comparisons for 2013 and 2014 and regional modeled end-use and carrier maps provide the foundation for this extension. Those comparisons do not establish hourly accuracy for an individual dwelling or Padova households.

<!-- P04 -->
Schedule uncertainty is already part of related research. Prataviera et al. reconstructed three-band ISTAT schedules, randomized profiles and tested a discrete uniform ±2-hour start-up shift in a survey-informed EUReCA study [2]. Their Verona comparison used district-heating measurements with a boundary distinct from the present ideal zone load. Earlier MODENA work examined weather, envelope and heating-technology scenarios [3], and Italian survey-derived user archetypes have also been identified [4]. Neither schedule randomization, stock scenarios nor archetype discovery is claimed here as a new method.

<!-- P05 -->
The present question is narrower: for one fixed respondent and dwelling, how do annual useful sensible heating and maximum hourly heating load change when equal-duration high-setpoint segments move within their reported time bands? Padova is the initial exposure. Three further coordinate TMYs test the same fixed case under different weather. The contribution is a reproducible, bounded placement comparison; novelty of this exact contrast remains provisional. The literature assessment is targeted rather than exhaustive.

## 2. Methods

### 2.1 Survey response and constructed schedules

<!-- P06 -->
The selected 2013 ISTAT response, identifier 3523, reports Veneto residence and mean winter-day heating-system operating hours of 4, 4 and 1 in the 05–13, 13–21 and 21–05 bands. Raw and compact records agree on these fields. It is the lowest eligible identifier under the declared pilot filter, which yields 372 eligible Veneto records. Selection is unweighted; it does not define a sample representative of Padova or Veneto. The questionnaire and response verification are documented in the [local survey audit](../00_context/survey_schedule_verification.json).

<!-- P07 -->
The inspected GRINS reconstruction places one contiguous binary segment in each band and maps its values to assumed 22°C high and 19°C setback setpoints for this timer-selected response. The alternatives retain the seeded placement or place each segment at its earliest or latest admissible position. All preserve 4/4/1 daily high-setpoint hours within the respective bands. Contiguity, repeated daily timing and the thermostat temperatures are modeling assumptions. The survey reports mean system on-hours; heating can occur at the modeled 19°C setback, so the encoding does not preserve positive-load duration.

### 2.2 Physical model and weather

<!-- P08 -->
The physical case is an assumed rectangular single zone with 80 m² floor area and 240 m³ volume, with fixed envelope, glazing, infiltration and internal-gain inputs. It is not a measured dwelling. The ideal-sensible 1C calculation uses a study-local adapter that evaluates each hourly solution from one previous state and commits one accepted state. Its per-surface preprocessing uses each external surface's own area, orientation, irradiance, construction and radiative fields. The [simulation protocol](../03_methods/simulation_protocol.md) and [Padova design](../05_simulations/climate_campaign_v5/Padova/design.json) document the case. Current GRINS and EUReCA revisions have not been established as the exact conference executable and configuration; this is a documented extension.

<!-- P09 -->
Four consistently retrieved PVGIS coordinate TMYs provide the weather exposures. Padova's input matches the documented acquisition at 45.408°N, 11.873°E, with ERA5 meteorology, SARAH3 radiation and a 2005–2023 selection window. The same dwelling, response, 22/19°C mapping and climate-E seasonal mask are retained for Padova, Bolzano, Roma and Palermo. This design changes weather while holding those inputs fixed. It does not represent each city's building stock, and the imposed season is not a local occupancy or regulatory estimate for the other sites.

<!-- P10 -->
The active calculation bypasses missing opaque sky cover using valid hourly horizontal infrared radiation to derive effective sky temperature. It retains the native annual mean of hourly air-minus-sky temperature for longwave exchange. Solar position uses native fixed UTC+1 interval-start labels without applying the EPW/JRC irradiation time offset. These choices limit interpretation of small annual differences and hourly peak rankings; sensitivity to alternative sky and solar-time treatments has not been quantified. The guarded fixed-ACH model does not use Palermo's one negative wind record, which remains an unused-field exception rather than a general weather-quality pass.

### 2.3 Endpoints, verification and validation

<!-- P11 -->
For hourly sensible zone load Pₜ in W and a one-hour timestep, annual useful sensible heating is Q = Σₜ max(Pₜ, 0)Δt/1000 in kWh. The annual maximum hourly load is H = maxₜ max(Pₜ, 0)/1000 in kW. For either endpoint X, each paired percentage is 100(Xplacement − Xseeded)/Xseeded, using the seeded value at the same site. These outputs are useful sensible heat, with no equipment-efficiency or final-energy conversion. The three deterministic alternatives receive no statistical confidence intervals and are not proven bounds over admissible schedules.

<!-- P12 -->
The admitted numerical results come exclusively from campaign v5. Its twelve primary cases comprise three placements under four weather exposures. Four repeat and four longer-warm-up cases check implementation behavior; a separate seeded Padova replay reproduces its primary hourly file byte-for-byte. These checks add no independent dwellings or household samples. All 175,200 exported hours in the twenty-case campaign pass the declared 10⁻⁶ W node-balance tolerance and old-state continuity checks. State solution and actual per-surface forcing were checked separately. Earlier local annual executions are excluded from physical interpretation after state, sky-input or surface-forcing defects; these findings are not attributed to the published conference executable.

<!-- P13 -->
Implementation verification, independent model benchmarking, empirical meter comparison, aggregate external checks and robustness assessment are distinct evidence layers. The present checks establish bounded numerical consistency. No independent whole-building benchmark or compatible independently documented meter series has been supplied. The conference national comparison cannot fill these gaps, and evidence review itself does not validate the model. The [validation protocol](../03_methods/validation_protocol.md) and [validation matrix](../03_methods/validation_matrix.md) specify the remaining evaluation work.

## 3. Results

### 3.1 Primary annual and maximum-hourly endpoints

<!-- P14 -->
In the fixed Padova-coordinate case, seeded, early and late placements produce 9574.02, 9659.63 and 9606.30 kWh annual useful sensible heating. Their maximum hourly loads are 10.050, 10.253 and 9.838 kW. Relative to seeded placement, early increases annual heating by 0.894% and the maximum hourly load by 2.029%; late increases annual heating by 0.337% while reducing the maximum by 2.103%. These are conditional model contrasts under the declared sky, solar-time, season and physical assumptions.

<!-- P15 -->
Annual placement differences vary across exposures. Early/late changes relative to the same-site seeded case are −0.098/+0.027% in Bolzano, +2.033/+0.500% in Roma and +3.868/+0.981% in Palermo, compared with +0.894/+0.337% in Padova. Maximum-hourly changes are 0/0% in Bolzano, +8.326/+3.381% in Roma and +5.969/+1.524% in Palermo, compared with +2.029/−2.103% in Padova. The results describe the same assumed case under coordinate weather exposures. Covarying weather fields prevent attribution to an individual meteorological variable.

<!-- FIG-01 | Brief: ../09_figures/figure_briefs.md#fig-01 -->
[FIG-01 PLACEHOLDER: Annual useful sensible heating and maximum-hourly-load contrasts. Ready v5 source data; see the linked brief.](../09_figures/figure_briefs.md#fig-01)

<!-- CAP01 -->
*Figure 1. Paired early and late placement changes relative to seeded placement at each coordinate TMY exposure. Annual useful sensible heating and maximum hourly load are separate primary endpoints. One assumed dwelling and one selected response are held fixed. Points are deterministic alternatives without confidence intervals. Bolzano's zero maximum differences are retained. Annual-mean sky exchange, native solar time, the common season and fixed physical inputs constrain interpretation.*

### 3.2 Post-hoc diagnostics

<!-- P16 -->
A post-hoc partition of the Padova seeded output finds positive sensible heating during 1323 of 2730 calendar hours at the 19°C setback, delivering 2375.62 kWh. This illustrates the distinction between encoded high-setpoint hours and heating runtime. It is a descriptive partition of modeled output, rather than an observed controller record or an additional primary endpoint.

<!-- FIG-02 | Brief: ../09_figures/figure_briefs.md#fig-02 -->
[FIG-02 PLACEHOLDER: Padova seeded setback heating diagnostic. Ready v5 partition; see the linked brief.](../09_figures/figure_briefs.md#fig-02)

<!-- CAP02 -->
*Figure 2. Post-hoc Padova seeded diagnostic at the modeled 19°C setback: 1323 positive-load hours among 2730 calendar hours and 2375.62 kWh useful sensible heat. Positive-load classification uses the saved 10⁻⁶ W tolerance. Encoded high-setpoint durations do not constrain heater runtime. This is one modeled case, without a measured-operation comparator.*

<!-- P17 -->
The Bolzano annual maximum is 17.8086 kW for all placements at hour 6912, corresponding to 16 October 00:00. A post-hoc inspection shows the imposed lower setpoint changing from 0°C in the preceding hour to 19°C, with old zone air at 9.641°C. The common season-start recovery can dominate the annual-maximum endpoint. The maximum is retained as predeclared; its equality does not show an absence of intraday timing response and does not establish equipment-sizing requirements.

### 3.3 Incomplete benchmark and empirical results

<!-- P18 -->
[RESULT PENDING: MET-007, LIM-002, GAP-002; validation task. An independent documented whole-building benchmark with matched inputs and accounting boundaries is required before reporting a benchmark comparison. A compatible independently documented meter series, building description, measurement period, matched weather, interval, end-use boundary and an independent evaluation design are required before reporting measured error. No benchmark score, bias or temporal accuracy result is available.]

<!-- FIG-03 | Brief: ../09_figures/figure_briefs.md#fig-03 -->
[FIG-03 BLOCKED: Independent meter comparison. Compatible measured heating and matched historical-weather simulation are missing; see the dependency brief.](../09_figures/figure_briefs.md#fig-03)

<!-- CAP03 -->
*Figure 3. Reserved for an independent empirical heating comparison after compatible measurements and a matched evaluation design become available. No empirical curve or error estimate is presently shown.*

## 4. Discussion

<!-- P19 -->
The fixed-case comparison shows that preserving encoded band durations does not preserve annual useful heating or the maximum hourly load. Their responses also differ: Padova's late placement increases annual heating while lowering its maximum, whereas Bolzano retains identical annual maxima despite small annual differences. These outcomes support a descriptive sensitivity statement for the declared model and exposures. They do not establish energy savings, an improved schedule estimate, measured accuracy or a causal weather mechanism.

<!-- P20 -->
Earlier three-band randomization and ±2-hour start-up perturbation work [2] limits the novelty claim. Here, the specific contrast holds one selected response and each band's encoded duration fixed while comparing three contiguous placements. Earliest and latest placement are constructed alternatives, not observed schedules or demonstrated energy extrema. Other admissible start times and non-contiguous schedules remain outside the experiment; no probability distribution over possible household behavior is estimated.

## 5. Limitations and remaining validation

<!-- P21 -->
One unweighted 2013 Veneto response combined with one assumed dwelling cannot estimate Padova, Veneto or national population outcomes. TMY scenario comparisons do not establish historical-year accuracy. Physical-input uncertainty, household assignment, alternative seasonal masks and sky/solar-time sensitivity have not been quantified; small annual differences and hourly rankings may depend on these assumptions. Cooling is disabled, and modeled Padova summer air temperature reaches 41.05°C, excluding conclusions about summer comfort or realistic whole-year conditioned operation.

<!-- P22 -->
The inspected local files did not identify a compatible independently documented meter time series; this limited inventory does not show that no suitable observations exist elsewhere. Empirical evaluation requires an independent reference aligned with the dwelling, weather, period and useful-heating boundary. Aggregate national final-energy agreement, implementation checks and repeated deterministic runs cannot substitute for that evaluation. The exact conference executable lineage also remains unresolved. The manuscript therefore remains an incomplete simulation illustration rather than a validated results paper.

## 6. Conclusions

<!-- P23 -->
Under the declared single-dwelling, respondent, seasonal and weather-forcing assumptions, three placements preserving 4/4/1 encoded high-setpoint hours produce different annual useful sensible heating and maximum-hourly-load contrasts across four coordinate TMY exposures. Heating at setback prevents interpreting these encoded durations as heater runtime; Bolzano's shared seasonal-restart maximum also limits what its zero peak contrasts imply. Independent benchmark and empirical validation, together with physical and forcing sensitivity work, remain necessary before accuracy, transfer or decision-use claims can be assessed.

## References

1. Vivian, J., Prataviera, E., Carnieletto, L., Pasut, W., and Zarrella, A. (2026). [A model to Simulate the Energy Consumption of the Italian Residential Building Stock with Regional Resolution](https://link.springer.com/chapter/10.1007/978-3-032-10546-2_123). *Proceedings of the 15th REHVA HVAC World Congress – CLIMA 2025*, pp. 1302–1311. DOI: 10.1007/978-3-032-10546-2_123.
2. Prataviera, E., Vivian, J., Lombardo, G., and Zarrella, A. (2022). [Evaluation of the impact of input uncertainty on urban building energy simulations using uncertainty and sensitivity analysis](https://www.research.unipd.it/bitstream/11577/3417286/9/1-s2.0-S0306261922001568-main_compressed.pdf). *Applied Energy*, 311, 118691. DOI: 10.1016/j.apenergy.2022.118691.
3. Prataviera, E., Vivian, J., Gaudino, F., and Zarrella, A. (2024). [Analysis of Energy Consumption Scenarios of the Italian Residential Building Stock](https://pro.unibz.it/library/bupress/publications/fulltext/9788860462022_21.pdf). *Building Simulation Applications BSA 2024*, pp. 157–164.
4. Khajedehi, M. H., Vivian, J., and Zarrella, A. (2025). [Identification of Italian user archetypes for building energy simulations based on a national survey](https://doi.org/10.1016/j.enbuild.2025.115924). *Energy and Buildings*, 344, 115924. Publisher-indexed sections inspected; complete primary PDF access remains a submission dependency.
