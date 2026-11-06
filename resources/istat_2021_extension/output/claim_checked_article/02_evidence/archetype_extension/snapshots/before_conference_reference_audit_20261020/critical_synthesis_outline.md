# Critical synthesis outline: implemented Italian user archetypes

Owner: literature worker; evidence date: 20 October 2026. Authoring packet, not polished article prose. Stable evidence claims: `AE-LIT-001`–`AE-LIT-033` in `proposed_literature_claims.md`; full records and access limits in `sources.json`. The former four-reference introduction should be replaced by a developed argument using the relevant sources below, rather than a long list appended to the same paragraphs. The 34 records include methods/data background; every record need not be cited.

## 1. What an archetype represents: physics, parameters, households or activities

Purpose: establish the model objects before discussing their advantages. “Archetype” is not a single interchangeable construct.

- First move: contrast Italian physical prototypes in Carnieletto et al. (2021), which specify fabric and systems with normative use assumptions, with URBEM's later harmonized reference-building scorecards. Their contribution is a traceable description of buildings under incomplete urban data. Claims AE-LIT-001–002. The new work can reuse such physical descriptions while changing household inputs; this does not create new physical prototypes.
- Second move: distinguish Sokol et al.'s probabilistic uncertain-parameter characterization from Khajedehi et al.'s demographic classes. Both use probabilities, but a calibrated distribution of an unknown U-value/setpoint is a different object from the probability that a household class inhabits a dwelling. Claims AE-LIT-003,017–018. Neither object is actual observed occupancy for a specific address.
- Third move: use Ali et al. (2019) to explain why representation depends on intended spatial scale. A national typology applied to an illustrative Padova-weather case is not evidence of a city sample. Claim AE-LIT-004. Connect the physical/user taxonomy to the article's case definition and explicit geographic inference boundary.

Suggested review table: columns **represented entity; source data; assignment mechanism; simulation input produced; evaluation evidence; transfer limit**. Include one row per construct, not a list of synonyms. This table can support a figure brief but requires no fabricated quantitative data.

## 2. Demography is not presence, heating control or appliance activity

Purpose: explain what information user archetypes add and what remains unobserved.

- First move: establish the four Italian demographic user groups from the full 2025 paper and its outcome definitions. Its heating response is declared on-time; its appliance electricity is calculated from survey answers. Claims AE-LIT-017,019. Demographic groups summarize household composition, not measured intraday trajectories or thermal preferences. Refer detailed group definitions and the recovered fitting/assignment assets to the context worker's audit.
- Second move: compare this information with time-use studies. Richardson et al. (2010) map active occupancy and daily activities to appliances; Widén and Wäckelgård (2010) generate activity sequences; Buttitta et al. (2019,2020) define occupancy-integrated archetypes and stochastic multi-day categories. Claims AE-LIT-005–008. Explain what diaries contribute—sequencing and duration relationships—which coarse energy surveys do not uniquely identify. Do not turn a daily hour total into a measured hourly schedule.
- Third move: use Jung et al. (2023) and Hanmer et al. (2019) as concrete counterpoints to an uncritical “measured data solves behaviour” claim. Sensor-derived occupancy status is not occupant count; controller IN states need not coincide with presence; a call-for-heat is not direct energy consumption. Claims AE-LIT-013–014. The relevant hierarchy is survey report → constructed control/gain schedule → physical load → delivered energy, with assumptions at every arrow.

Implementation bridge: document separately user class, people count, survey on-hours, high-setpoint schedule, setback rule, appliance energy and thermal gains. Do not assume one dimension supplies all the others. A household archetype must change a documented simulator input; attaching a group label to the original single household does not implement its energy consequences.

## 3. Linking users to dwellings: probabilistic assignment and joint structure

Purpose: make assignment a substantive modelling stage rather than a lookup hidden in preprocessing.

- First move: describe the 2025 Italian paper's probabilistic linkage as a response to uncertain dwelling-to-household classification. Section3.2/Algorithm1 estimates probabilities, not a certain individual class. Claim AE-LIT-018. Empirical association with dwelling properties does not justify a causal claim about why a household lives there.
- Second move: compare an empirical stock-sampling strategy with a synthetic linkage strategy. Chen et al. (2022) integrate stochastic end-use schedules into a conditional-stock framework; Latifi et al. (2026) link separate household/building/time-use/weather sources while acknowledging missing individual linkage labels. Claims AE-LIT-009,027. A draw from a fitted probability table is a synthetic assignment, with support and compatibility checks required. More input sources do not by themselves make the joint household–dwelling combinations correct.
- Third move: contrast register-based individual scheduling in Mosteiro-Romero et al. (2020) with national survey assignment. Claim AE-LIT-010. Registers give case-specific population structure but in a different mixed-use setting; national surveys give broader coverage but coarser geography. Both have scope limits. Do not import their numerical differences or model-quality claims into Italian residential cases.

Implementation bridge: first test all four recovered classes in matched illustrative dwellings, which exposes class-conditional model responses without asserting local prevalence. Any weighted mixture is a separate estimand requiring defensible probabilities/weights. Preserve conditional relationships and report out-of-support combinations. Compare class-specific simulation followed by aggregation with any pooled-input approximation; do not assert they are equivalent.

## 4. From user representation to thermal and electric outputs

Purpose: place the proposed implementation within actual simulation precedents, not just clustering literature.

- First move: make the direct comparisons substantive. Buttitta et al. (2019) already map occupancy-integrated archetypes to heating and internal-gain assumptions. Chen et al. (2022) already couples stochastic household/end-use events to a stock physics model. Claim AE-LIT-007–009. The proposed work is therefore not the first to implement heterogeneous users or stochastic schedules.
- Second move: establish the closest Italian antecedent. Prataviera et al. (2022) already use the same ISTAT three bands to construct random schedules and vary duration/start time in EUReCA. The CLIMA chapter remains the regional stock foundation; MODENA and Puglisi supply earlier stock/scenario/profile work. Claims AE-LIT-021,024–026,032. This narrows the contribution to a verified implementation and controlled evaluation of the separate published demographic representation within the conference-related model.
- Third move: compare profile sensitivity across weather. Rugani et al. (2024) already vary occupancy databases across three climates in EnergyPlus; Li and Dogan (2025) use modular survey-informed physical/system/load archetypes in a low-order model. Claim AE-LIT-031 and source SRC-LI2025. Varying weather is a controlled response test only when physical and user definitions stay fixed. A different city weather file does not create a locally representative stock.

Implementation bridge: specify the archetype-to-input mapping, its departure from the single-record baseline and its uncertainty before new runs. Hold physical assumptions, weather, controls and energy accounting matched when estimating the effect of representation. If operating duration and gains both change, a combined contrast alone cannot separate their effects. A small ablation of these channels can be more informative than more locations.

## 5. Aggregation, nonlinear heating and model-detail tradeoffs

Purpose: explain why annual totals, hourly loads and population mixtures need distinct tests, without predicting the result.

- First move: use De Jaeger et al. (2020) as the precise heating comparator: occupant setpoint profiles and the thermal response of the represented buildings complicate aggregation. Claim AE-LIT-011. “Simulate mean inputs” and “average simulated outputs” are different procedures to compare; equality is not supplied by an arithmetic average of heating hours.
- Second move: put evidence for detail alongside evidence for simplification. Ferrando et al. (2022) explicitly assess temporal/spatial aggregation; Mosteiro-Romero et al. (2020) distinguish annual and hourly model differences. Gilani et al. (2018) finds convergence for annual office-lighting simulations at larger aggregation. Claims AE-LIT-010,016,023. Avoid a universal threshold or generic assertion that aggregation either eliminates or preserves behaviour effects. End use, correlations, controls and endpoints differ between these studies.
- Third move: distinguish between-class heterogeneity from within-class stochasticity. Widén (2025) studies efficient aggregation of a common Markov process and explicitly leaves heterogeneity/building interactions for future work. Claim AE-LIT-015. Four deterministic class profiles implement between-class variation but not stochastic day-to-day behaviour; randomizing one respondent implements timing variability but not four demographic classes. Neither automatically estimates population uncertainty.

Implementation bridge: report class-specific annual useful heating and timing/peak endpoints; then define any mixture and aggregation level. Inspect seasonal restart and common-control artefacts before interpreting peaks. Include uncertainty sources beyond schedule placement if conclusions compare class effects with small numerical contrasts. Do not treat a deterministic class mean as a complete distribution.

## 6. Validation evidence, external validity and scientific positioning

Purpose: end the review with a checkable study contribution and validation boundary.

- First move: compare validation designs, not the word “validated.” Sokol et al. (2017) reports a training/test separation; Richardson et al. (2010) compares electricity statistics with recordings; Chen et al. (2022) distinguishes time-use checks from measured end-use evaluation. Claims AE-LIT-003,005,009. These address different outputs and do not validate Italian useful-heating predictions.
- Second move: distinguish the close Italian evaluation layers. Prataviera et al.'s Verona case has selected measured heating periods; its Milan application lacks equivalent observations. MOIRAE's abstract distinguishes national totals and household expenditure comparisons. The Italian monitoring campaign reports real household/appliance electricity but has access/completeness/geography limits. Claims AE-LIT-022,024 and SRC-MONITOR2020. TMY sensitivity cannot be compared directly to an unmatched historical meter period.
- Third move: use recent Sood et al. (2025) and Dabirian et al. (2026) as current implementation comparators, while respecting access depth and their Irish residential/Montreal office contexts. Claims AE-LIT-012,028. Do not import reported benefit percentages without their full calibration/holdout and end-use boundaries. Cross-context evidence motivates tests, not a universal superiority claim.

Permitted positioning: **A documented implementation of published Italian demographic user representations in a conference-related ita-domes/EUReCA configuration, with matched comparisons exposing how representation, schedule mapping and weather affect specified simulated endpoints.** This is conditional on the implementation actually being completed and audited. Its evidence can support sensitivity, differences, null effects and reproducibility. Improved predictive accuracy requires a suitable independent measured comparator. Claims AE-LIT-020,033.

Prohibited shortcuts: “first occupant-aware model”; “first survey-based Italian model”; “archetypes close the performance gap”; “probabilistic means representative”; “national survey validates city households”; or attributing a model difference exclusively to real behaviour. No unsuccessful search proves the absence of prior work.

## Coverage checks for the claim/prose authors

- Develop comparisons within each subsection, including at least one explicit limitation or conflicting implication, rather than adding citation clusters to a generic performance-gap statement.
- Cite original studies for their own methods/results; use the recovered 2025 paper's references as discovery leads rather than substitutes for opening the primary papers.
- Keep CLIMA modelling, demographic-archetype identification, and the newly implemented evaluation separate in the contribution statement.
- Use the full source-ledger limits; abstract-only sources support bounded methods statements here, not detailed error scores, generalizability or methodological endorsement.
- Every new numerical comparison or implementation result needs its own run-backed claim; literature evidence supplies no placeholder results.
