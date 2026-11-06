# Decisions and rationale

## D01 — Separate artifacts and preserve provenance

New root is `output/claim_checked_article`. Existing research outputs and dirty external checkouts are read in place. The GRINS checkout's HEAD is a129009e177da391792b57c8af5055b18ce64374; its existing modifications/untracked files were observed and remain untouched. No publication or author contact is authorized.

## D02 — Evidence types and published baseline

The complete CLIMA chapter is the foundation: printed pp.1302–1311, published 2026 in CLIMA2025 proceedings. Its national final-energy comparisons are published results. Partial historical weather coverage and national aggregation preclude household/hourly/Padova accuracy claims. Local GRINS origin differs from the chapter's stated public repository; executable equivalence is unresolved.

## D03 — Choose schedule reconstruction sensitivity before outputs

The 2013–2021 blueprint proposes a much larger temporal transfer programme. Its common-domain mapping, frozen evaluation, appliance reconstruction, and downstream coupling require a separate substantial verification chain. Local discovery has instead established a direct, small test of the CLIMA hourly schedule mapping: reported block hours admit multiple hourly placements. A local EUReCA environment and Padova-coordinate EPW make this candidate feasible. The source routine uses seeded contiguous placements, not randomly scattered on-hours; alternatives must preserve the actual response constraints.

This scope retains the chapter's physics/survey connection and Padova-first requirement. It does not assume household information is outdated, claim a better reconstruction, or revive a correction narrative. Temporal survey transfer and retrofit ranking are deferred rather than presented as unfinished central contributions. The closest 2022 EUReCA study already addresses survey-informed operation uncertainty and measurements in Verona, so the contribution must remain the particular controlled disaggregation test, with novelty provisional.

## D04 — Climate expansion gate

Only expand after relevant Padova technical checks pass. Select named additional files using measured, consistently computed heating/cooling degree indicators, humidity, and solar conditions plus provenance. A file's name or geographic label alone is insufficient. Hold all physical and household assumptions fixed. No national inference from a few climates.

## D05 — Review and prose gates

Explicit requested model assignments are supported by spawn tools. Runtime-confirmed identities are unknown unless returned. Four serial full review/response/revision/register cycles precede prose. Each Astra reviewer starts fresh without prior verdicts/private reasoning; closure checks occur only after saved critique. Missing data may survive as explicit incomplete content; substantive errors may not be presented as settled.

## User steering — 2026-10-20

The user clarified that simulated results are not already available and that simulation belongs in the plan: select areas to fit the context, construct EUReCA inputs, then execute simulations. This confirms new simulation work; legacy local runs must not be treated as existing evidence for this article. Existing weather/code files are input candidates subject to audit. The reply does not establish that empirical measurements do not exist; the current inventory only establishes that a compatible independently documented meter dataset has not yet been identified.

## D06 — Independent review requires corrected execution, without changing the question

Review 01 reproduced a material state-update defect in the current EUReCA dispatcher using byte-identical archived outputs. The source hash and diagnostic are preserved in07_reviews/reviewer_01. This concerns the inspected current checkout; the exact conference executable is unknown, so no defect in the published national result is inferred. The original25annual runs remain quarantined. A local adapter performs candidate solves from the same beginning-of-hour state and commits only the accepted state once. The correction does not change scenario inputs, tune parameters to data, or introduce a new research question. Corrected runs receive new IDs and must pass transition and full-step state-balance checks before scientific use.

## D07 — Preserve the correct construct

The official survey asks reported mean winter-day operation hours. The current source maps those binary hours to a22°C high setpoint with19°C setback for the selected timer response. Thus the matched reconstructions preserve high-setpoint duration, not actual heater-on duration. Keep this extra modelling assumption explicit in the question, methods, captions and result interpretation. A strict equipment-availability experiment would require a separate design; it is not silently substituted here. Any partition of energy by setpoint is a labelled post-hoc diagnostic rather than a predeclared primary endpoint.

## D08 — Correct active weather forcing and report the remaining approximation

The second independent review identified that the EPW missing opaque-cloud code was used as physical cloud cover. The selected source files contain valid horizontal infrared data. The v4local correction derives effective radiative sky temperature from that IR, verifies units/time/field mapping, and replaces the native annual-mean air-minus-sky scalar before zone preprocessing. The annual-mean longwave approximation is deliberately retained and disclosed; this is not an empirical calibration or an unannounced hourly-longwave model change. All45earlier annual outputs remain quarantined.

Strict field QC also identified one genuinely negative Palermo wind speed. Raw data and the failed audit are preserved. The bounded fixed-ACH/no-ventilation experiment can proceed only with proof and a runtime guard that wind/cloud fields are not consumed after the correction. This is an explicit unused-variable exception, not a claim of universal EPW validity or a silently widened physical range. Native irradiance offset/time-label approximations remain documented limits on hourly fidelity.

## D09 — Replay into a fresh destination

The recorded v3command omitted a required site argument. Preserve the historical manifest, add a transparent correction, and provide a tested new-destination replay interface that records the site, campaign identity and executable. Completed run directories may not be overwritten. Replay verification cases are counted separately from the primary experiment.

## D10 — Climate class context does not alter the controlled test

Official-source contextual checks are in climate_classification_context.md. Reported municipal classes are E/E/D/B; legal/design degree-day values are distinct from the computed TMY HDD18 descriptors. These documents do not change site selection, fixed operating season or scenarios and are not used for present legal-compliance advice. Contextual evidence is kept outside the manuscript until individually registered/reviewed if used.
