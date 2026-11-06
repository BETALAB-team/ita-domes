# Implementing Italian demographic user archetypes as operational mixtures: a held-out model-reference comparison under four weather exposures

<!-- P00 -->
**Incomplete first manuscript.** This study evaluates a documented operational-input representation against detailed-input outputs from the same thermal model. Independent whole-building benchmarking, empirical heating accuracy and broader uncertainty remain unresolved.

## Abstract

<!-- P01 -->
Published demographic household archetypes require an explicit bridge to building-simulation inputs. We implement four Italian survey-derived groups through archived household assignments joined to real reported heating-duration and control responses. The eligible domain contains 13,624 unweighted records; 13,592 train group-conditional or pooled probabilities over the same four real source-control components, and 32 held-out respondents provide detailed-input model references. Each component is simulated separately for one assumed 80 m² dwelling under Padova, Bolzano, Roma and Palermo coordinate typical meteorological years (TMYs). Mixtures are formed from the resulting endpoints rather than from averaged temperature inputs. Physical inputs, internal gains and the seasonal mask remain fixed.

<!-- P02 -->
Eligible-stratum-weighted annual useful-heating RMSE decreases from pooled to group-conditional representation from 1828.20 to 1787.94 kWh in Padova, 2633.86 to 2565.53 kWh in Bolzano, 1437.68 to 1411.16 kWh in Roma and 1215.81 to 1196.74 kWh in Palermo. Overall RMSE for expected individual annual maximum hourly load is also lower at each exposure, but Bolzano groups 2 and 4 worsen. Post hoc, 27/32 Bolzano held-out cases and 2/4 shared components attain their maxima on the imposed seasonal-restart day. The conditional representation uses twelve free probabilities versus three pooled and retains limited within-branch variability. These results describe one split and one physical case against a same-model reference; they establish neither measured predictive accuracy nor a general demographic advantage. Native annual-mean sky exchange, solar-time convention and unquantified physical uncertainty constrain interpretation.

## 1. Introduction

<!-- P03 -->
The ita-domes chapter in the CLIMA 2025 proceedings, published in 2026, describes survey-informed Italian residential-stock modeling with regional resolution [[1]](https://link.springer.com/chapter/10.1007/978-3-032-10546-2_123). The present extension connects that modeling foundation with the distinct four-group demographic representation published by Khajedehi et al. in 2025 [[2]](https://doi.org/10.1016/j.enbuild.2025.115924). It implements a specific household-assignment-to-operational-input bridge and tests its fidelity to detailed-input model outputs. It does not reproduce the complete published stock model or introduce new demographic clusters.

<!-- P04 -->
The principal question is whether four shared real operating components, mixed using group-conditional training frequencies, retain the annual heating and individual-maximum outputs of held-out respondent inputs better than the same components mixed using pooled frequencies. Demographic labels affect the learned probabilities, while source responses determine the executable profiles. The original one-respondent timing experiment is retained as secondary evidence. Independent building benchmarks and compatible measurements are unavailable, so the present comparison concerns representation within the model rather than real-world prediction accuracy.

## 2. Literature review and scientific positioning

### 2.1 Italian stock-model lineage and the represented population

<!-- P05 -->
Brøgger and Wittchen's review identifies sample representativeness as a requirement for extrapolating a modeled building sample to a population [[3]](https://www.sciencedirect.com/science/article/abs/pii/S1364032117308857). This is a useful starting point for reading the CLIMA chapter: regional resolution describes the level of its stock outputs, whereas representativeness concerns the households and buildings behind those outputs. The chapter's national annual final-energy comparison and regional modeled GWh maps are different evidence from dwelling-scale hourly heating observations. A new user representation must therefore be assessed at its actual sampling, physical and temporal scope rather than inheriting accuracy from a national aggregate.

<!-- P06 -->
The CLIMA chapter explicitly places ita-domes after MOIRAE, the RSE MODENA report and the BSA 2024 scenario study, describing ita-domes as almost completely rewritten [[1]](https://link.springer.com/chapter/10.1007/978-3-032-10546-2_123). MOIRAE and MODENA already use Italian household-survey information for bottom-up energy modeling [[4]](https://www.researchgate.net/publication/344001872_MOIRAE_-_bottom-up_MOdel_to_compute_the_energy_consumption_of_the_Italian_REsidential_sector_Model_design_validation_and_evaluation_of_electrification_pathways)[[5]](https://www.rse-web.it/rapporti/applicativo-per-la-valutazione-dei-consumi-energetici-degli-edifici-residenziali-italiani/). The official RSE report description includes building, sociodemographic, equipment and use information in hourly simplified residential modeling. The subsequent BSA study examines weather, envelope and heating-technology scenarios using ISTAT and EUReCA [[6]](https://pro.unibz.it/library/bupress/publications/fulltext/9788860462022_21.pdf). Household descriptors and scenario calculations thus precede this experiment. The stated lineage does not establish identical executable versions, input transformations or validation boundaries across these models.

<!-- P07 -->
The shared data foundation is specific: CLIMA reference 9 and the demographic-archetype study cite the Istat 2013 public-use release, published on 30 August 2017 [[7]](https://www.istat.it/microdati/indagine-sui-consumi-energetici-delle-famiglie-anno-2013-2/)[[2]](https://doi.org/10.1016/j.enbuild.2025.115924). Puglisi et al. also use Italian survey dwelling classes to construct energy profiles [[8]](https://www.iieta.org/journals/ijht/paper/10.18280/ijht.340320). These precedents matter because a common survey can support different modeled objects: dwelling-type profiles, demographic classes and stock energy outputs. Declared heating-system on-hours and appliance electricity calculated from survey answers remain distinct from measured thermal demand. Reusing the same survey does not independently validate the subsequent thermal calculation.

### 2.2 Physical, demographic and probabilistic archetypes

<!-- P08 -->
Italian physical prototype work characterizes buildings by use, construction period, envelope and HVAC properties [[9]](https://escholarship.org/content/qt8q57r16d/qt8q57r16d_noSplash_a473a55b40e402f3c45f423bc4dbdf0c.pdf). The URBEM reference-building library develops traceable, harmonized input scorecards with uncertainty labels [[10]](https://www.mdpi.com/1996-1073/18/22/6026). These organize the physical description of a building. The four groups of Khajedehi et al. instead summarize derived minimum and maximum age, household size and education [[2]](https://doi.org/10.1016/j.enbuild.2025.115924). A demographic label supplies neither a wall construction nor a measured intraday schedule. Combining these forms of archetype consequently requires an explicit link; substituting one term for the other hides rather than resolves missing inputs.

<!-- P09 -->
Probability also has different roles. Sokol et al. update uncertain residential archetype-parameter distributions from training homes and evaluate on a separate larger set [[11]](https://www.sciencedirect.com/science/article/pii/S037877881631372X). The Italian demographic study estimates probabilities of user classes conditional on dwelling properties [[2]](https://doi.org/10.1016/j.enbuild.2025.115924). In the former, probability describes uncertainty in model parameters; in the latter it describes household–dwelling linkage. Neither is a certain household identity at a specific address. Ali et al.'s comparison of physical archetypes at different Irish spatial scales further shows that applying a national representation locally is itself a modeling choice [[12]](https://researchrepository.ucd.ie/rest/bitstreams/42898/retrieve). Coordinate weather does not turn the present fixed dwelling into a representative local stock.

### 2.3 What surveys, activities and sensors observe

<!-- P10 -->
Activity-based models begin from different information. Richardson et al. combine active occupancy and daily activities to generate minute-resolution appliance electricity, comparing simulated statistics with recorded dwelling electricity [[13]](https://pureportal.strath.ac.uk/en/publications/domestic-electricity-use-a-high-resolution-energy-demand-model/). Widén and Wäckelgård generate stochastic activities and household electricity from time-use-trained Markov processes [[14]](https://www.sciencedirect.com/science/article/pii/S0306261909004930). These studies represent activity sequences explicitly. The Italian demographic groups and coarse winter heating bands do not supply the same sequences. Their use for thermal simulation requires documented control and timing assumptions rather than an inferred archetype-specific presence curve.

<!-- P11 -->
Occupancy-integrated archetypes already make such links explicit. Buttitta et al. embed clustered time-use profiles within physical dwelling archetypes and map occupancy to assumed heating controls and internal gains [[15]](https://researchrepository.ucd.ie/server/api/core/bitstreams/d1e15168-1930-4c63-ade3-d9f66969c0ca/content). Buttitta and Finn extend this approach with stochastic multi-day occupancy categories for heating simulations [[16]](https://www.sciencedirect.com/science/article/abs/pii/S0378778819322170). Chen et al. couple stochastic occupant and end-use events with ResStock, reporting checks against time-use data and measured end-use electricity [[17]](https://www.sciencedirect.com/science/article/pii/S0306261922011540). These are substantive implementation precedents, preventing a claim that the present study is the first occupant-aware or stochastic simulation. Here the narrower implemented channel is response-linked heating control; demographic occupancy, gain and ventilation trajectories are not invented.

<!-- P12 -->
Sensor-derived information also needs careful naming. Jung et al.'s thermostat-based schedules infer occupancy status rather than occupant count and identify sensor and ground-truth limitations [[18]](https://escholarship.org/content/qt4wd5w010/qt4wd5w010.pdf). Hanmer et al. distinguish programmed occupancy states and call-for-heat signals from actual presence and directly metered heating energy [[19]](https://link.springer.com/article/10.1007/s12053-018-9632-x). Huebner et al. infer heterogeneous English heating patterns from indoor temperatures and compare them with common modeling schedules [[20]](https://discovery.ucl.ac.uk/id/eprint/1402869/). Together these studies show why the observed or inferred variable must remain attached to a schedule claim. A survey on-hour, a modeled high-setpoint hour and a positive heating-load hour are not interchangeable quantities.

### 2.4 Assignment and actual simulation precedents

<!-- P13 -->
The Italian demographic study's dwelling linkage is probabilistic, not an observed household–building pairing [[2]](https://doi.org/10.1016/j.enbuild.2025.115924). Latifi et al. link population, building, activity and weather sources while acknowledging absent individual linkage ground truth [[21]](https://doi.org/10.1016/j.buildenv.2026.114872). Mosteiro-Romero et al. compare register-based individual daily schedules with standard-based deterministic and stochastic alternatives in district simulations [[22]](https://research.tudelft.nl/en/publications/a-novel-population-based-occupancy-modeling-approach-for-district/). These approaches address assignment and schedule construction with different source support. Our archived-ID join avoids making a new synthetic dwelling assignment, but it does not implement the published probabilistic allocation or verify identity to its inaccessible remote assignment branch.

<!-- P14 -->
The closest Italian timing precedent is Prataviera et al.'s EUReCA uncertainty study: it reconstructs three-band ISTAT schedules, randomizes profiles and applies a discrete uniform ±2-hour start-up shift because timing is unknown [[23]](https://www.research.unipd.it/bitstream/11577/3417286/9/1-s2.0-S0306261922001568-main_compressed.pdf). Its selected Verona district-heating evaluation and unmeasured Milan application have different evidentiary status. Koene and Eslami-Mossallam likewise examine temporal thermostat-setting dispersion with an RC district-heating model [[24]](https://repository.tno.nl/SingleDoc?docId=55394). Rugani et al. compare occupancy-profile databases for a standard building under three city climates in EnergyPlus [[25]](https://www.sciencedirect.com/science/article/pii/S0378778824002305). Timing uncertainty and multi-weather profile sensitivity therefore motivate, rather than establish novelty for, the present controlled comparison.

<!-- P15 -->
The physical-method foundation also separates endpoints. The CLIMA references include EUReCA and Vivian et al.'s evaluation of 5R1C and 7R2C formulations against TRNSYS for a room and apartment, examining seasonal needs and dynamic or peak response separately [[26]](https://github.com/BETALAB-team/EUReCA)[[27]](https://www.sciencedirect.com/science/article/pii/S0378778816314244). That published comparison motivates explicit annual and maximum-hourly endpoints here. It is not evidence that the current study-local adapter has passed the same independent benchmark. Our useful sensible heating also excludes the heat-pump efficiency and DHW/system conversions for which CLIMA cites Staffell and UNI/TS 11300-2 [[28]](https://pubs.rsc.org/en/content/articlehtml/2012/ee/c2ee22653g)[[29]](https://store.uni.com/uni-ts-11300-2-2019). A thermal-load contrast cannot inherit final-energy meaning from those citations.

### 2.5 Aggregation and representation capacity

<!-- P16 -->
De Jaeger et al. examine alternative aggregation of occupant setpoint profiles because represented dwellings have different thermal responses [[30]](https://www.mech.kuleuven.be/en/tme/research/thermal_systems/publications/publications/view?fromnr=21&pubsonpage=20&pubtype=journal-article&sortby=scdate). Ferrando et al. compare smart-meter-informed randomized schedules with a baseline at multiple temporal and spatial aggregation levels [[31]](https://re.public.polimi.it/bitstream/11311/1220975/2/Manuscript.pdf). Conversely, Gilani et al. find convergence of stochastic office-lighting predictions at larger simulated aggregation scales [[32]](https://www.sciencedirect.com/science/article/abs/pii/S0360132318300581). These findings concern different end uses and endpoints; they do not supply a universal rule that additional schedule detail is either necessary or dispensable. Our mixtures are therefore defined after separate component simulations, and an expectation of individual annual maxima is kept distinct from a coincident aggregate peak.

<!-- P17 -->
Between-household heterogeneity and within-process stochasticity are also different. Widén's aggregated occupancy-model comparison explicitly leaves heterogeneity and end-use/building interactions for future work [[33]](https://www.tandfonline.com/doi/full/10.1080/19401493.2024.2445134). Sood et al. apply time-use-trained probabilistic occupancy to Irish residential archetypes across scales [[34]](https://pure.ul.ie/en/publications/quantifying-uncertainty-in-multi-scale-energy-analysis-of-residen/), while Dabirian et al. compare deterministic and measured-data-derived stochastic schedules in Montreal institutional offices [[35]](https://www.frontiersin.org/journals/sustainable-cities/articles/10.3389/frsc.2026.1771845/full). Neither the stochastic label nor a favorable result in another setting proves that four Italian demographic groups will improve heating representation. The relevant test must specify what those groups change, how much variation is compressed, and whether the comparator has the same statistical capacity.

### 2.6 Validation boundary and the present contribution

<!-- P18 -->
Validation targets in the inherited literature are specific. Calama-González et al. combine socioeconomic indicators with parametric comfort simulations for southern-Spain social housing and describe checks against monitored hourly indoor temperatures before scaling [[36]](https://idus.us.es/server/api/core/bitstreams/d437d2eb-631e-4c9e-9d1f-5f19c3e4287b/content). Richardson et al. compare electricity statistics with recordings; Prataviera et al. use selected district-heating periods; the CLIMA chapter compares national annual carrier totals. These are not interchangeable tests. Similarly, the official ISO 15927-4 abstract concerns reference years for average annual heating and cooling energy and excludes extreme or semi-extreme-year construction [[37]](https://www.iso.org/standard/41371.html). Neither a standard's scope nor a typical-year maximum establishes design-extreme or measured heating performance here.

<!-- P19 -->
The review is a targeted narrative synthesis anchored in the original chapter's bibliography and relevant primary studies. Complete local primary texts support the CLIMA, demographic-archetype, uncertainty and selected physical/occupancy methods; other records were accessible only as abstracts, indexed sections or official catalog descriptions. For example, MOIRAE and RSE support bounded lineage statements, while Sokol, Ali, Richardson, Sood and several aggregation studies support the inspected method descriptions rather than imported quantitative accuracy claims. The resulting contribution is deliberately specific: an auditable archived-demographic-assignment-to-operational-mixture implementation and a held-out comparison within one model. Literature motivates this experiment; it does not predetermine a favorable result or demonstrate measured accuracy.

## 3. Data and methods

### 3.1 Published demographic groups and recovered assignment

<!-- P20 -->
The four published groups use derived age, household-member and education features. Their printed means and shares are summarized in Table 1. Descriptive examples are not threshold rules for assigning a new household, and none prescribes hourly presence or heating control. The source paper reports declared heating on-hours and modeled appliance electricity, with individual-building simulation consequences identified as further work [[2]](https://doi.org/10.1016/j.enbuild.2025.115924).

<!-- TAB-01 | Brief: ../09_figures/archetype_extension/figure_briefs.md#tab-01 -->
<!-- CAPT01 -->
*Table 1. Published demographic group summaries from Khajedehi et al. (2025), Table 3 and Figure 5. Ages are derived class-based means; education is a dimensionless index. Household and person shares use distinct unweighted denominators. Descriptions are interpretations, not membership rules or hourly schedules.*

| Group | Published description | Mean min/max age (years) | Mean members | Mean education index | Household/person shares (%) |
|---|---|---:|---:|---:|---:|
| 1 | Few older members; mid-level education | 70.2 / 78.3 | 1.8 | 0.5 | 40.1 / 29.9 |
| 2 | More members; commonly younger and middle-aged members | 21.2 / 57.2 | 3.7 | 0.7 | 26.9 / 40.2 |
| 3 | Few middle-aged members; medium education | 46.0 / 63.0 | 2.3 | 0.5 | 22.9 / 21.3 |
| 4 | Few middle-/older-aged members; higher education | 55.6 / 72.2 | 2.1 | 0.8 | 10.1 / 8.6 |

<!-- P21 -->
We use the archived lowercase four-group assignment joined by exact household ID, without clustering again. Its 17,855 unique IDs join the 20,000-row official release and compact input one-to-one. Rounded demographic and share summaries match 39 of 40 printed cells; the exception is group 4 maximum-age mean, 72.1 locally versus 72.2 printed. Aggregate correspondence does not establish identical per-household membership to the inaccessible published branch. The remaining 2145 release IDs have no archived label and remain outside the assignment domain. The [assignment specification and audit](../00_context/archetype_extension/published_archetype_specification.md) preserve this distinction.

### 3.2 Operational domain and fixed holdout

<!-- P22 -->
Eligibility requires central or individual main heating, three valid observed integer winter-day band responses between zero and eight hours, and a positive daily total. The resulting 13,624 records include 5430, 3693, 2991 and 1510 in groups 1–4, with all four modeled source-control branches supported in every group. These are unweighted eligible survey records. The analysis does not use national expansion weights or equate its eligible group proportions with the published whole-sample shares.

<!-- TAB-02 | Brief: ../09_figures/archetype_extension/figure_briefs.md#tab-02 -->
<!-- CAPT02 -->
*Table 2. Domains of the response-linked experiment. Counts describe unweighted archived survey records, not a national expansion or 152 independent households. Eligibility and the fixed stratified holdout are distinct from the published whole-sample group shares.*

| Domain | Records | Role or restriction |
|---|---:|---|
| Official survey release | 20,000 | Source ID domain |
| Archived four-group assignment | 17,855 | Exact-ID join; remote published membership not proved |
| Eligible operational-input cohort | 13,624 | Group counts 5430 / 3693 / 2991 / 1510 |
| Training subset | 13,592 | Component selection and probability estimation |
| Held-out subset | 32 | Two respondents in each of 16 group × control-branch strata |

<!-- P23 -->
The frozen PCG64(20261020) split selects two held-out IDs in each group-by-control-branch stratum, retaining 32 test respondents and excluding them from component fitting and probability estimation. The other 13,592 records train the representation. The initial input-only design selected four group medoids and one pooled medoid, but all five roles produced an identical complete temperature profile. No annual extension simulation was run for that design. A documented post-input, pre-output amendment retained the split and replaced singleton prototypes with the shared branch mixture below. This is an explicitly recorded design amendment, not untouched public preregistration.

### 3.3 Response-to-simulator bridge and common physics

<!-- P24 -->
Archived demographic labels are joined to each respondent's real heating-band and control fields. The inspected source rules produce four assumed high/setback branches: 22/11, 22/19, 24/13 and 24/21°C. Timer, valve, retrofit-control and fireplace routing enter this source mapping; they are not observations of indoor temperature or biomass consumption. Source-ID-seeded contiguous placement converts band durations to repeated high-setpoint periods. Heating remains possible at setback, so reported hours do not constrain positive-load runtime. Demographic ages and education do not directly generate thermal schedules, occupancy, gains or ventilation.

<!-- P25 -->
The common physical case is one assumed 80 m², 240 m³ single-zone dwelling, with fixed envelope, glazing, internal gains, infiltration and ideal sensible heating. The existing study-local 1C adapter commits one accepted state from a single old state per hour and initializes each surface's own forcing. Internal gains and physical inputs are held fixed across groups; the published probabilistic dwelling allocation and appliance model are not implemented in this experiment. The current GRINS/EUReCA configuration is documented but has not been established as the exact conference executable. The [executed mixture protocol](../03_methods/archetype_extension/mixture_protocol_v2.md) records the retained physical inputs and corrections.

<!-- P26 -->
The same four coordinate PVGIS TMYs and common climate-E seasonal mask are used for all profiles. The hourly nonleap calendar uses fixed UTC+1 without daylight saving. Valid horizontal infrared radiation supplies effective sky temperature, while the native annual-mean air-minus-sky approximation remains in longwave exchange. Native solar-position timing does not apply the irradiation-time offset. These forcing approximations, the imposed common season and fixed physical assumptions limit peak and small-effect interpretation. Changing weather exposures does not establish locally representative households or isolate an individual causal weather variable. The guarded unused Palermo wind exception remains a scoped input limitation.

### 3.4 Shared components, mixture probabilities and endpoints

<!-- P27 -->
For each control branch, the amended design selects one real training respondent nearest the branch's 24-hour input-temperature centroid, with smallest-ID tie resolution. Centroids are never simulated and no thermal output selects the inputs. The four shared component respondent IDs are 2941, 981, 5303 and 9574 for branches 22/11, 22/19, 24/13 and 24/21°C, respectively. Their complete source-seeded profiles are simulated separately. Both representations therefore have the same component inputs, while differing in probabilities learned before outcomes.

<!-- P28 -->
For branch b and group g, the conditional weights are p(b|g)=Ntrain,g,b/Ntrain,g and pooled weights are p(b)=Ntrain,b/13,592. For site s and endpoint k, predictions are Σb p(b|g) fₖ,ₛ(xb) and Σb p(b) fₖ,ₛ(xb), where fₖ,ₛ(xb) is the endpoint from the separately simulated real component. No averaged temperature input is passed through the simulator. Four probabilities sum to one in each distribution: group conditioning uses twelve free probabilities versus three pooled. The comparison holds component count fixed, not total statistical parameter count.

<!-- P29 -->
The training groups have distinct branch frequencies. For the 22/11°C branch, conditional proportions are 63.04%, 51.51%, 57.46% and 55.73% for groups 1–4; for 24/21°C they are 4.81%, 8.93%, 7.11% and 7.19%. These eligible-cohort frequencies supply actual group-specific mixture inputs. They are associations within the archived training domain under the source mapping, rather than observed thermostat preferences or causal demographic effects. One shared component per branch compresses remaining duration and timing differences within that branch.

<!-- FIG-04 | Brief: ../09_figures/archetype_extension/figure_briefs.md#fig-04 -->
[FIG-04 PLACEHOLDER: Actual shared component profiles and pooled/group training probabilities; ready frozen inputs.](../09_figures/archetype_extension/figure_briefs.md#fig-04)

<!-- CAP04 -->
*Figure 4. Actual shared source-control inputs and training mixture probabilities. Four real training respondents supply source-ID-seeded temperature profiles; both representations simulate these same components separately. Archetypes enter through P_train(branch|group), compared with pooled P_train(branch), learned from 13,592 training records. The conditional representation has twelve free probabilities and the pooled representation three. Setpoints are model assumptions based on reported heating/control responses, not measured occupancy, temperatures or heater runtime. Dwelling, gains and season are fixed.*

<!-- P30 -->
The primary endpoints are annual useful sensible heating Q=Σt max(Pt,0)Δt/1000 in kWh and individual annual maximum hourly load H=maxt max(Pt,0)/1000 in kW, for hourly sensible load Pt in W and Δt=1 h. The mixture's maximum-load prediction is the expectation of separately computed individual annual maxima. It is neither the maximum of an averaged hourly curve nor a coincident or diversified urban peak. There is no conversion to final energy, equipment efficiency, fuel or DHW demand.

### 3.5 Detailed-input references, loss and verification

<!-- P31 -->
Each held-out respondent is simulated using its complete response-conditioned input under the same dwelling and weather to provide a detailed-input model reference yi,s. For either endpoint, primary loss is MSE=Σi wi(pi,s−yi,s)² and RMSE=√MSE. The paired loss contrast is Σi wi[(pconditional−yi,s)²−(ppooled−yi,s)²]; a negative value favors the conditional representation within this experiment. Units remain kWh or kW for RMSE and squared units for MSE. No percentage normalization, reference-value denominator or zero-endpoint replacement is used. These deviations measure compression fidelity to the same model, not measured prediction error.

<!-- P32 -->
With eligible stratum size Ngb, each of its two test respondents receives overall evaluation weight Ngb/(2×13,624); within a group of eligible size Ng the weight is Ngb/(2Ng). The weights sum to one for the relevant overall or group domain. They account descriptively for the disproportionate holdout selection across the eligible group-by-branch domain, and are distinct from training mixture probabilities. They are not national expansion weights, survey-design confidence intervals or independent observations. The one frozen split provides eight held-out respondents per group; no statistical confidence interval is assigned to these deterministic alternatives.

<!-- P33 -->
The v2 archetype campaign completed 144 scientific annual executions: 32 detailed-input references and four shared components under each of four weather exposures. Eight repeat or longer-warm-up checks bring the execution count to 152 and the exported hourly-row count to 1,331,520. All four site gates and independent integration, source-hash and mixture-arithmetic checks passed for the bounded sensible-only path. These are numerical verification checks, not empirical validation. The [executed-v2 validation matrix](../03_methods/archetype_extension/validation_matrix.md) distinguishes this design from the historical unexecuted singleton design; [frozen protocols and manifests](../05_simulations/archetype_campaign_v2/pre_run/selection_protocol.md) preserve the actual inputs and commands.

## 4. Results

### 4.1 Overall model-reference fidelity

<!-- P34 -->
For annual useful sensible heating, pooled-to-conditional overall RMSE is 1828.20→1787.94 kWh in Padova, 2633.86→2565.53 kWh in Bolzano, 1437.68→1411.16 kWh in Roma and 1215.81→1196.74 kWh in Palermo. Paired weighted MSE differences, conditional minus pooled, are −145596.59, −355285.79, −75546.73 and −46011.95 kWh² in the same order. Thus conditional weighting lowers overall annual-heating loss for this held-out eligible cohort under each fixed weather exposure. It does not establish measured heating accuracy.

<!-- P35 -->
For expected individual annual maximum hourly load, pooled-to-conditional RMSE is 4.7155→4.6015 kW in Padova, 2.3658→2.3571 kW in Bolzano, 3.5514→3.4742 kW in Roma and 3.2389→3.1945 kW in Palermo. Corresponding paired MSE differences are −1.0627, −0.0413, −0.5421 and −0.2855 kW². These are expectations of individual annual maxima mixed from the same four components. The lower overall deviations do not demonstrate a better coincident load forecast, an equipment-sizing result or universal superiority.

### 4.2 Adverse subgroup results and post-hoc peak context

<!-- P36 -->
The predeclared Bolzano subgroup comparisons contain counterexamples. For individual annual maxima, group 2 RMSE increases from 1.965770 kW pooled to 2.008767 kW conditional, and group 4 increases from 2.929172 to 2.941238 kW. These adverse held-out comparisons remain alongside the lower overall Bolzano RMSE. Group conditioning therefore does not benefit every subgroup, even within the same physical case and weather.

<!-- P37 -->
A post-hoc diagnostic of the new v2 Bolzano files locates 27 of 32 held-out respondent maxima and two of four shared-component maxima on the imposed season-restart day, model hours 6912–6935 (16 October). Eighteen of 32 respondents and those two components peak exactly at hour 6912; the remaining five respondents and two components peak in January. The two denominators describe different roles and are not combined into 36 people. This is a descriptive calendar coincidence under an imposed common season, not a causal restart test. No maximum or primary loss is excluded or replaced, and no sizing or observed-operation inference follows.

<!-- FIG-05 | Brief: ../09_figures/archetype_extension/figure_briefs.md#fig-05 -->
[FIG-05 PLACEHOLDER: Overall two-endpoint RMSE and complete Bolzano subgroup comparison, with post-hoc peak context; ready reviewed results.](../09_figures/archetype_extension/figure_briefs.md#fig-05)

<!-- CAP05 -->
*Figure 5. Held-out fidelity to detailed-input outputs from the same model: annual useful-heating RMSE and expected-individual-annual-maximum RMSE overall, and the complete Bolzano maximum-load subgroup comparison. Overall metrics use eligible-stratum weights for 32 held-out respondents; group metrics use within-group weights for eight respondents each. Group-conditional weights lower overall RMSE at each exposure but worsen Bolzano groups 2 and 4. This is one split, one dwelling and fixed gains/season, with twelve conditional versus three pooled free probabilities. Post hoc, 27/32 Bolzano test cases and 2/4 shared components peak on the imposed restart day. No measured-accuracy, coincident-peak or sizing inference follows; native annual-mean sky and solar-time sensitivity remain unquantified.*

### 4.3 Residual compression and secondary timing evidence

<!-- P38 -->
The Padova detailed-input model outcomes for the 32 held-out respondents span 5538.60–13145.26 kWh annual useful heating. The four conditional mixture predictions span only 7736.55–7953.89 kWh, while the pooled prediction is 7834.86 kWh. The mixtures retain much less of the detailed-input respondent variation than remains in those references. Lower overall squared loss consequently does not show that the representation resolves individual operating-input heterogeneity. The complete [weighted metrics](../04_analysis/archetype_extension/results_v2/weighted_metrics.csv) and [individual deviations](../04_analysis/archetype_extension/results_v2/individual_deviations.csv) accompany these summary results.

<!-- P39 -->
The earlier v5 pilot is a separate one-respondent timing comparison: a selected unweighted Veneto response with 4/4/1 reported band hours, encoded as 22°C high and 19°C setback periods for the same assumed dwelling. It tests seeded, earliest and latest contiguous placement rather than demographic mixtures. In Padova, annual useful heat is 9574.02/9659.63/9606.30 kWh and maximum hourly load 10.050/10.253/9.838 kW for seeded/early/late. A post-hoc partition finds 1323 positive-heating hours among 2730 modeled setback hours, delivering 2375.62 kWh. This illustrates why encoded high-setpoint duration is not heater runtime; it supplies no four-group or population result.

<!-- P40 -->
The secondary pilot's same-site early/late annual changes are +0.894/+0.337% in Padova, −0.098/+0.027% in Bolzano, +2.033/+0.500% in Roma and +3.868/+0.981% in Palermo. Maximum-hourly changes are +2.029/−2.103%, 0/0%, +8.326/+3.381% and +5.969/+1.524%, respectively. Its three equal Bolzano maxima, 17.8086 kW at hour 6912, coincide with a common 0→19°C seasonal restart and old air at 9.641°C. This older post-hoc context is distinct from the new 32-respondent/four-component count. The original [timing figures, plotting code and provenance](../09_figures/figure_manifest.md) retain their one-response scope; earliest/latest placements are not proven bounds.

### 4.4 Independent validation remains incomplete

<!-- P41 -->
[RESULT PENDING: GAP-005, MET-007, LIM-002, DAT-004; validation task. No independent matched whole-building benchmark score is available. Reporting physical or predictive accuracy requires a provenance-qualified independent reference, compatible building and system data, energy boundary and units, measured period and coverage, matched historical weather, time alignment and an independent evaluation design. The same-model held-out references and the conference national annual carrier comparison do not supply this evidence. No empirical bias, hourly error score or measured savings is reported.]

<!-- FIG-06 | Brief: ../09_figures/archetype_extension/figure_briefs.md#fig-06 -->
[FIG-06 BLOCKED: Independent meter comparison requires compatible measurements, matched historical-weather runs and an independent evaluation design.](../09_figures/archetype_extension/figure_briefs.md#fig-06)

<!-- CAP06 -->
*Figure 6. Independent empirical comparison remains unavailable. Compatible measurements, matched historical weather and an independent evaluation design are required before this panel can report heating accuracy.*

## 5. Discussion

<!-- P42 -->
The archetypes materially enter this experiment through different training probabilities over four shared, real operational inputs. This is more than attaching demographic labels to the earlier timing pilot. It also differs from assigning four invented occupancy or gain curves: the household ID links archived membership to reported heating/control fields, while explicit source assumptions supply the temperature schedule. The result supports a bounded input-representation finding: conditional branch frequencies lower overall model-reference loss in this split, with adverse Bolzano maximum-load subgroups and substantial remaining detailed-input spread.

<!-- P43 -->
The comparison does not isolate a general demographic advantage. Conditional weights have twelve free probability parameters against three pooled, and within-branch duration and timing differences remain compressed into one shared real respondent. The component inputs are held common, but representation capacity is unequal. A small single-split holdout cannot quantify how selection, branch choice or parameter count would affect broader performance. The favorable overall metrics should therefore be read together with the adverse subgroups and the narrow prediction ranges, rather than as a universal improvement.

<!-- P44 -->
Peak interpretation requires particular restraint. The mixture endpoint concerns expected individual annual maxima, and many new Bolzano maxima coincide with the fixed restart day. That post-hoc observation does not identify the cause of the deviations or justify removing the peaks. The imposed seasonal mask, annual-mean longwave treatment and native solar-time convention remain part of the experiment; their influence on phase, magnitude and ranking has not been quantified. Weather fields covary across exposures, preventing attribution of their contrasts to a single climatic variable.

## 6. Limitations and unfinished work

<!-- P45 -->
The archived assignment is article-consistent but not verified identical to the published per-household branch, and one rounded demographic cell differs. Excluded release records are not assigned from verbal examples. Eligibility further narrows the household domain, and its descriptive weights do not make it a nationally representative stock. Only response-linked heating inputs vary: dwelling, gains, ventilation and season are fixed, and the published probabilistic dwelling allocation, physical-stock variation and appliance coupling remain outside this implementation. The four weather exposures are simulation conditions, not four local population validations.

<!-- P46 -->
Physical credibility remains distinct from numerical consistency and model-reference fidelity. The study-local state and forcing path passes bounded implementation checks, while exact conference executable lineage is unresolved. Earlier defective original annual runs remain quarantined; no correction finding is attributed to the conference release. Sensitivity to envelope, gains, controls, season, sky/solar timing, assignment and train/test split is unquantified. Cooling and comfort are outside the study; the earlier uncooled Padova pilot reaches 41.05°C modeled summer air temperature and cannot establish realistic annual conditioned operation.

<!-- P47 -->
The local inventory has not yielded a compatible independently documented meter series, which is a bounded inventory result rather than proof that none exists. An independent whole-building benchmark and matched measured heating remain missing. Evidence review, repeated annual runs, state balances and a same-model holdout do not establish empirical accuracy. These dependencies leave the manuscript incomplete for claims about real household prediction, population transfer, measured savings, coincident system peaks or equipment sizing.

## 7. Conclusions

<!-- P48 -->
An archived four-group demographic assignment has been implemented as group-conditional probabilities over four shared real source-control components in a documented conference-related GRINS/EUReCA configuration. Against detailed-input outputs from the same model, conditional weighting lowers overall annual-heating and expected-individual-maximum RMSE in one 32-respondent split under four coordinate TMYs. Bolzano groups 2 and 4 worsen on the maximum endpoint; many Bolzano maxima coincide post hoc with the imposed restart day, and substantial respondent variation remains compressed. The result supports a specific operational-mixture fidelity comparison under fixed physical assumptions. Independent benchmarking, compatible measurements and broader uncertainty analysis are still required before predictive or transfer claims can be assessed.

[Figure and table briefs](../09_figures/archetype_extension/figure_briefs.md) specify every new visual and its scientific limits. Rendered source files, editable code and QA will be linked after the separate visualization handoff.

## References

1. Jacopo Vivian; Enrico Prataviera; Laura Carnieletto; Wilmer Pasut; Angelo Zarrella. (2026). [A model to Simulate the Energy Consumption of the Italian Residential Building Stock with Regional Resolution](https://link.springer.com/chapter/10.1007/978-3-032-10546-2_123). *Proceedings of the 15th REHVA HVAC World Congress - CLIMA 2025*, 788, 1302–1311. DOI: 10.1007/978-3-032-10546-2_123.

2. Mohamad Hasan Khajedehi; Jacopo Vivian; Angelo Zarrella. (2025). [Identification of Italian user archetypes for building energy simulations based on a national survey](https://doi.org/10.1016/j.enbuild.2025.115924). *Energy and Buildings*, 344, 115924. DOI: 10.1016/j.enbuild.2025.115924.

3. Morten Brøgger; Kim Bjarne Wittchen. (2018). [Estimating the energy-saving potential in national building stocks – A methodology review](https://www.sciencedirect.com/science/article/abs/pii/S1364032117308857). *Renewable and Sustainable Energy Reviews*, 82, 1489–1496. DOI: 10.1016/j.rser.2017.05.239.

4. Giorgio Besagni; Marco Borgarello; Lidia Premoli Vilà; Behzad Najafi; Fabio Rinaldi. (2020). [MOIRAE – bottom-up MOdel to compute the energy consumption of the Italian REsidential sector: Model design, validation and evaluation of electrification pathways](https://www.researchgate.net/publication/344001872_MOIRAE_-_bottom-up_MOdel_to_compute_the_energy_consumption_of_the_Italian_REsidential_sector_Model_design_validation_and_evaluation_of_electrification_pathways). *Energy*, 211, 118674. DOI: 10.1016/j.energy.2020.118674.

5. L. Croci; S. Viani; Enrico Prataviera; Angelo Zarrella. (2021). [Applicativo per la valutazione dei consumi energetici degli edifici residenziali italiani](https://www.rse-web.it/rapporti/applicativo-per-la-valutazione-dei-consumi-energetici-degli-edifici-residenziali-italiani/). *Ricerca sul Sistema Energetico (RSE)*.

6. Enrico Prataviera; Jacopo Vivian; Francesca Gaudino; Angelo Zarrella. (2024). [Analysis of Energy Consumption Scenarios of the Italian Residential Building Stock](https://pro.unibz.it/library/bupress/publications/fulltext/9788860462022_21.pdf). *Building Simulation Applications BSA 2024*, 157–164.

7. Istituto nazionale di statistica. (2017). [Indagine sui consumi energetici delle famiglie: microdati ad uso pubblico, anno 2013](https://www.istat.it/microdati/indagine-sui-consumi-energetici-delle-famiglie-anno-2013-2/). *Istat*.

8. Giovanni Puglisi; Fabio Zanghirella; Paola Ungaro; Giuliano Cammarata. (2016). [A Methodology for the Generation of Energy Consumption Profiles in the Residential Sector](https://www.iieta.org/journals/ijht/paper/10.18280/ijht.340320). *International Journal of Heat and Technology*, 34, 491–497. DOI: 10.18280/ijht.340320.

9. Laura Carnieletto; Martina Ferrando; Lorenzo Teso; Kaiyu Sun; Wanni Zhang; Francesco Causone; Piercarlo Romagnoni; Angelo Zarrella; Tianzhen Hong. (2021). [Italian prototype building models for urban scale building performance simulation](https://escholarship.org/content/qt8q57r16d/qt8q57r16d_noSplash_a473a55b40e402f3c45f423bc4dbdf0c.pdf). *Building and Environment*, 192, 107590. DOI: 10.1016/j.buildenv.2021.107590.

10. Martina Ferrando; Francesco Causone; Alessia Banfi; Vincenzo Corrado; Ilaria Ballarini; Matteo Piro; Angelo Zarrella; Laura Carnieletto; Nicola Borgato; Gianpiero Evola; Maurizio Detommaso; Francesco Nicoletti; Andrea Vallati; Costanza Vittoria Fiorini. (2025). [Developing an Italian Library of Reference Buildings for Urban Building Energy Modeling (UBEM): Lessons Learnt from the URBEM Project](https://www.mdpi.com/1996-1073/18/22/6026). *Energies*, 18, 6026. DOI: 10.3390/en18226026.

11. Julia Sokol; Carlos Cerezo Davila; Christoph F. Reinhart. (2017). [Validation of a Bayesian-based method for defining residential archetypes in urban building energy models](https://www.sciencedirect.com/science/article/pii/S037877881631372X). *Energy and Buildings*, 134, 11–24. DOI: 10.1016/j.enbuild.2016.10.050.

12. Usman Ali; Mohammad Haris Shamsi; Cathal Hoare; Eleni Mangina; James O'Donnell. (2019). [A data-driven approach for multi-scale building archetypes development](https://researchrepository.ucd.ie/rest/bitstreams/42898/retrieve). *Energy and Buildings*, 202, 109364. DOI: 10.1016/j.enbuild.2019.109364.

13. Ian Richardson; Murray Thomson; David Infield; Conor Clifford. (2010). [Domestic electricity use: a high-resolution energy demand model](https://pureportal.strath.ac.uk/en/publications/domestic-electricity-use-a-high-resolution-energy-demand-model/). *Energy and Buildings*, 42, 1878–1887. DOI: 10.1016/j.enbuild.2010.05.023.

14. Joakim Widén; Ewa Wäckelgård. (2010). [A high-resolution stochastic model of domestic activity patterns and electricity demand](https://www.sciencedirect.com/science/article/pii/S0306261909004930). *Applied Energy*, 87, 1880–1892. DOI: 10.1016/j.apenergy.2009.11.006.

15. Giuseppina Buttitta; William J. N. Turner; Olivier Neu; Donal P. Finn. (2019). [Development of occupancy-integrated archetypes: Use of data mining clustering techniques to embed occupant behaviour profiles in archetypes](https://researchrepository.ucd.ie/server/api/core/bitstreams/d1e15168-1930-4c63-ade3-d9f66969c0ca/content). *Energy and Buildings*, 198, 84–99. DOI: 10.1016/j.enbuild.2019.05.056.

16. Giuseppina Buttitta; Donal P. Finn. (2020). [A high-temporal resolution residential building occupancy model to generate high-temporal resolution heating load profiles of occupancy-integrated archetypes](https://www.sciencedirect.com/science/article/abs/pii/S0378778819322170). *Energy and Buildings*, 206, 109577. DOI: 10.1016/j.enbuild.2019.109577.

17. Jianli Chen; Rajendra Adhikari; Eric Wilson; Joseph Robertson; Anthony Fontanini; Ben Polly; Opeoluwa Olawale. (2022). [Stochastic simulation of occupant-driven energy use in a bottom-up residential building stock model](https://www.sciencedirect.com/science/article/pii/S0306261922011540). *Applied Energy*, 325, 119890. DOI: 10.1016/j.apenergy.2022.119890.

18. Wooyoung Jung; Zhe Wang; Tianzhen Hong; Farrokh Jazizadeh. (2023). [Smart thermostat data-driven U.S. residential occupancy schedules and development of a U.S. residential occupancy schedule simulator](https://escholarship.org/content/qt4wd5w010/qt4wd5w010.pdf). *Building and Environment*, 243, 110628. DOI: 10.1016/j.buildenv.2023.110628.

19. Clare Hanmer; Michelle Shipworth; David Shipworth; Edwin Carter. (2019). [How household thermal routines shape UK home heating demand patterns](https://link.springer.com/article/10.1007/s12053-018-9632-x). *Energy Efficiency*, 12, 5–17. DOI: 10.1007/s12053-018-9632-x.

20. G. M. Huebner; M. McMichael; D. Shipworth; M. Shipworth; M. Durand-Daubin; A. Summerfield. (2013). [Heating patterns in English homes: Comparing results from a national survey against common model assumptions](https://discovery.ucl.ac.uk/id/eprint/1402869/). *Building and Environment*, 70, 298–305. DOI: 10.1016/j.buildenv.2013.08.028.

21. Milad Latifi; Claudia Zucca; Martijn C. Willemsen. (2026). [Beyond archetypes: Creating behavior-aware synthetic building populations through probabilistic multi-source data integration for multi-domain urban analysis](https://doi.org/10.1016/j.buildenv.2026.114872). *Building and Environment*, 302, 114872. DOI: 10.1016/j.buildenv.2026.114872.

22. Martín Mosteiro-Romero; Illias Hischier; Jimeno A. Fonseca; Arno Schlueter. (2020). [A novel population-based occupancy modeling approach for district-scale simulations compared to standard-based methods](https://research.tudelft.nl/en/publications/a-novel-population-based-occupancy-modeling-approach-for-district/). *Building and Environment*, 181, 107084. DOI: 10.1016/j.buildenv.2020.107084.

23. Enrico Prataviera; Jacopo Vivian; Giulia Lombardo; Angelo Zarrella. (2022). [Evaluation of the impact of input uncertainty on urban building energy simulations using uncertainty and sensitivity analysis](https://www.research.unipd.it/bitstream/11577/3417286/9/1-s2.0-S0306261922001568-main_compressed.pdf). *Applied Energy*, 311, 118691. DOI: 10.1016/j.apenergy.2022.118691.

24. F. G. H. F. Koene; B. B. Eslami-Mossallam. (2023). [Space heating demand profiles of districts considering temporal dispersion of thermostat settings in individual buildings](https://repository.tno.nl/SingleDoc?docId=55394). *Building and Environment*, 228, 109839. DOI: 10.1016/j.buildenv.2022.109839.

25. Roberto Rugani; Marco Picco; Giacomo Salvadori; Fabio Fantozzi; Marco Marengo. (2024). [A numerical analysis of occupancy profile databases impact on dynamic energy simulation of buildings](https://www.sciencedirect.com/science/article/pii/S0378778824002305). *Energy and Buildings*, 310, 114114. DOI: 10.1016/j.enbuild.2024.114114.

26. Enrico Prataviera; Pierdonato Romano; Laura Carnieletto; Francesco Pirotti; Jacopo Vivian; Angelo Zarrella. (2021). [EUReCA: An open-source urban building energy modelling tool for the efficient evaluation of cities energy demand](https://github.com/BETALAB-team/EUReCA). *Renewable Energy*, 173, 544–560. DOI: 10.1016/j.renene.2021.03.144.

27. Jacopo Vivian; Angelo Zarrella; Giuseppe Emmi; Michele De Carli. (2017). [An evaluation of the suitability of lumped-capacitance models in calculating energy needs and thermal behaviour of buildings](https://www.sciencedirect.com/science/article/pii/S0378778816314244). *Energy and Buildings*, 150, 447–465. DOI: 10.1016/j.enbuild.2017.06.021.

28. Iain Staffell; Dan Brett; Nigel Brandon; Adam Hawkes. (2012). [A review of domestic heat pumps](https://pubs.rsc.org/en/content/articlehtml/2012/ee/c2ee22653g). *Energy & Environmental Science*, 5, 9291–9306. DOI: 10.1039/C2EE22653G.

29. Ente Italiano di Normazione. (2019). [UNI/TS 11300-2:2019. Prestazioni energetiche degli edifici - Parte 2: Determinazione del fabbisogno di energia primaria e dei rendimenti per la climatizzazione invernale, per la produzione di acqua calda sanitaria, per la ventilazione e per l'illuminazione in edifici non residenziali](https://store.uni.com/uni-ts-11300-2-2019). *UNI*.

30. Ina De Jaeger; Annelies Vandermeulen; Bram van der Heijde; Lieve Helsen; Dirk Saelens. (2020). [Aggregating set-point temperature profiles for archetype-based: simulations of the space heat demand within residential districts](https://www.mech.kuleuven.be/en/tme/research/thermal_systems/publications/publications/view?fromnr=21&pubsonpage=20&pubtype=journal-article&sortby=scdate). *Journal of Building Performance Simulation*, 13, 285–300. DOI: 10.1080/19401493.2020.1727567.

31. Martina Ferrando; Sibilla Ferroni; Martina Pelle; Anita Tatti; Silvia Erba; Xing Shi; Francesco Causone. (2022). [UBEM’s archetypes improvement via data-driven occupant-related schedules randomly distributed and their impact assessment](https://re.public.polimi.it/bitstream/11311/1220975/2/Manuscript.pdf). *Sustainable Cities and Society*, 87, 104164. DOI: 10.1016/j.scs.2022.104164.

32. Sara Gilani; William O'Brien; H. Burak Gunay. (2018). [Simulating occupants' impact on building energy performance at different spatial scales](https://www.sciencedirect.com/science/article/abs/pii/S0360132318300581). *Building and Environment*, 132, 327–337. DOI: 10.1016/j.buildenv.2018.01.040.

33. Joakim Widén. (2025). [Computationally efficient upscaling of Markovian occupancy models for urban-level building simulations](https://www.tandfonline.com/doi/full/10.1080/19401493.2024.2445134). *Journal of Building Performance Simulation*, 18, 424–451. DOI: 10.1080/19401493.2024.2445134.

34. Divyanshu Sood; Ibrahim Alhindawi; Usman Ali; Donal Finn; James A. McGrath; Miriam A. Byrne; Anna Mavrogianni; James O'Donnell. (2025). [Quantifying uncertainty in multi-scale energy analysis of residential archetypes: A stochastic occupancy approach](https://pure.ul.ie/en/publications/quantifying-uncertainty-in-multi-scale-energy-analysis-of-residen/). *Building and Environment*, 279, 113026. DOI: 10.1016/j.buildenv.2025.113026.

35. Sanam Dabirian; Kartikay Sharma; Ursula Eicker. (2026). [How can stochastic occupant-based archetypes enhance urban building energy modeling for better decision-making?](https://www.frontiersin.org/journals/sustainable-cities/articles/10.3389/frsc.2026.1771845/full). *Frontiers in Sustainable Cities*, 8, 1771845. DOI: 10.3389/frsc.2026.1771845.

36. Carmen María Calama-González; Rocío Escandón; Rafael Suárez; Alicia Alonso; Ángel Luis León-Rodríguez. (2024). [Household energy vulnerability evaluation in southern Spain through parametric energy simulation models and socio-economic data](https://idus.us.es/server/api/core/bitstreams/d437d2eb-631e-4c9e-9d1f-5f19c3e4287b/content). *Sustainable Cities and Society*, 103, 105276. DOI: 10.1016/j.scs.2024.105276.

37. International Organization for Standardization. (2005). [ISO 15927-4:2005. Hygrothermal performance of buildings - Calculation and presentation of climatic data - Part 4: Hourly data for assessing the annual energy use for heating and cooling](https://www.iso.org/standard/41371.html). *ISO*.



