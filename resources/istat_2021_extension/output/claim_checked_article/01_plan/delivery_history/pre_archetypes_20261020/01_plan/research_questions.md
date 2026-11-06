# Selected research question

Decision date: 2026-10-20. Original scope was frozen before new simulation outputs were inspected and is preserved in research_questions_initial_20261020.md. This version clarifies the model construct after review 01; it does not change the predeclared annual-demand/peak endpoints or scenarios. See decisions D06–D07.

**Primary RQ:** Under a documented illustrative dwelling and a fixed 2013 Veneto survey response, how sensitive are EUReCA heating outputs to alternative within-block placements of a constructed two-level setpoint profile that preserve its encoded high-setpoint durations?

The reported operating-hour answers are 4/4/1 hours. The source maps them to 22°C high-setpoint durations with 19°C setback; heating remains available at setback. Equal encoded durations therefore do not constrain equal actual heater-on time. Temperatures and within-block placement are modelling assumptions, not observations.

**Secondary RQ1:** Does this conditional sensitivity differ across a small set of weather files selected for consistently measured climate contrast after the Padova pilot passes?

**Secondary RQ2:** Which output scales are sensitive: annual useful heating demand and hourly peak heating power? Aggregate similarity may coexist with timing/peak differences; this is a hypothesis to test, not an expected finding.

The experiment compares the current GRINS code's seeded contiguous within-block reconstruction with early and late admissible contiguous placements. Exact schedules, seed handling, thermostat mapping, block boundaries, season, model formulation, estimands, and checks must be specified by the methods owner. None is presumed to reproduce actual household timing. The chapter describes conversion of morning/afternoon/evening survey hours to hourly setpoints but does not establish empirical validity of a unique within-block placement.

The primary physical formulation should follow the chapter's 1C approach when feasible; any alternative is an explicit departure, not a conference reproduction. External code remains unchanged. The selected household record anchors input construction, not a representative Padova sample. Weather exposure supplies the geographic condition; it does not supply city household information.

The experiment can support a result about model response conditional on these assumptions. It cannot establish an improved schedule estimator, actual behaviour, household accuracy, causal effects, measured energy savings, a national stock result, or universal climate transferability.

If paired differences are negligible under the tested conditions, this still bounds the practical importance of the reconstruction assumption in those conditions. If differences are material, the result identifies an assumption requiring sensitivity reporting or better observations; it does not identify the correct reconstruction.

Empirical validation remains a separate workstream. Missing independent Padova measurements must remain visible in claims, prose, and validation matrix.

## Contribution-to-evidence map

| Candidate contribution | Required analysis / possible finding | Validation route | Planned visual |
|---|---|---|---|
| Audit of conference schedule inference boundary | Exact chapter and source-code mapping; unknown implementation equivalence | Page/source inspection; immutable code hashes | Conceptual evidence-to-schedule diagram |
| Quantified reconstruction sensitivity | Matched simulations retaining equal block hours; null or non-null differences | Schedule invariants, reproducibility, energy bookkeeping; no empirical accuracy inference | Schedule panel; paired annual-demand and peak table/plot |
| Conditional weather interaction | Same dwelling/equipment/household scenarios under selected weather files | Climate metrics; paired factors; limited site coverage | Climate descriptor panel and paired effects |
| Explicit validation readiness | Evidence and missing-data audit | Separate physics/benchmark/empirical/aggregate criteria | Validation matrix, primarily textual |

Novelty remains provisional against the closest literature. Prior survey-informed EUReCA uncertainty analysis exists; neither survey use nor hourly simulation is a novelty claim.
