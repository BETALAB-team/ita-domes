# Independent scientific review 04 — initial critique

Date: 20 October 2026. Reviewer: fresh `/root/review_04` context; required/requested model `gpt-6-astra`; backend model confirmation is not exposed to this reviewer and remains unknown.

Input: `06_claim_drafts/article_claims_v03.md`, SHA-256 `c96e3307de9b9587cc554c076f5d3527cc3ce8a6c29b12342777c07bc93c5eb2`. The hash was calculated directly. Companion: `claim_ledger_v03.json`. Scope: the full rubric and every central contribution/result, not merely closure of previous findings.

## Independence

This critique was written before opening any earlier review, response, issue register, author check/generator, private packet, decision-history file, or STATUS/agent-manifest narrative. References to those sources inside the supplied manuscript were not followed. The supplied technical method excerpts were used. The current validation matrix contained a brief earlier-review disposition; that incidental statement was not used as evidence and no earlier technical verdict/report was opened. The independent calculations below, not prior endorsements, support my recommendation.

## Recommendation

**Minor revision, with the bounded v5 scientific result admissible after the listed consistency fixes. No newly identified critical or major factual/methodological error is presented as settled in v03.** Complete the fourth author response/revision and a separate closure check before a prose-stage decision. This is not a submission-readiness or empirical-validation certificate.

The defensible deliverable is a transparent single-case computational sensitivity study extending one assumption in the conference workflow. Its potential journal contribution remains modest and explicitly provisional. It does not demonstrate a new schedule-generation method, an improved estimator, representative Italian/Padova results, or measured accuracy. A broader journal claim would need evidence absent here. The current manuscript mostly respects that boundary.

## Actual evidence inspection and independent calculations

- Read the human scientific scope, current research questions and full review rubric; inspected the 41-claim draft and machine ledger.
- Read the complete extracted target chapter, PDF pages 2–11 only. Inspected page images 7, 8 and 9 for Table 3 and both regional map figures. Natural-gas values, units, model/reference direction and the absence of regional measured comparators in those figures support RES-001/002. No neighbouring paper was used.
- Opened the primary [Prataviera et al. 2022 paper](https://www.research.unipd.it/bitstream/11577/3417286/9/1-s2.0-S0306261922001568-main_compressed.pdf), particularly p6, Table 1 p7 and measurement-boundary text. It indeed creates random profiles from the three bands and shifts availability by a discrete ±2 h parameter. LIT-002/NOV-001 correctly acknowledge the close precedent. This materially limits novelty.
- Opened the primary [MODENA BSA 2024 chapter](https://pro.unibz.it/library/bupress/publications/fulltext/9788860462022_21.pdf), methods pp158–160 and scenario definitions p160. Existing weather/envelope/heating-technology scenarios support LIT-003. Neither weather scenarios nor survey-informed EUReCA modelling are new here.
- Consulted [JRC TMY documentation](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/pvgis-typical-meteorological-year-tmy-generator_en) and the [EnergyPlus EPW dictionary](https://bigladdersoftware.com/epx/docs/24-2/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html). The retained phase and sky-aggregation caveats are scientifically material.
- Inspected original `C:/Works/GRINS/GRINS-Tool/classes/EndUse.py` methods around lines 593–656, original EUReCA `thermal_zone.py` pure 1C matrices around lines 1050–1110, current local single-commit, IR, surface-preprocessing and campaign scripts, run designs and network parameters. Source schedule extraction was executed independently; it exactly reproduces the seeded 8760-hour lower setpoint. Raw survey and compact-file selection were also checked.
- Wrote and executed **only** `07_reviews/reviewer_04/recheck.py`, using the existing EUReCA interpreter with `-B`. `independent_recheck.json` and `recheck_stdout.txt` contain the reproducible audit. An initial tab-delimiter assumption and Windows default-decoding error in my own helper were corrected before the final run; these were reviewer-helper errors, not study defects.
- Independently integrated all 20 v5 hourly files; reconstructed each peak and its previous/current setpoint and old air state; checked prior/current state continuity; reconstructed all air, surface and mass node equations using saved network parameters; independently recomputed all per-surface opaque SW/LW and glazed SW components and node forcing; compared node forcing with actual hourly solver exports. No earlier reviewer/author helper was imported.
- Recomputed sky scalars from the four raw EPW horizontal-IR columns; independently checked EPW-versus-JSON active-field row equality, hourly outdoor-temperature equality, HDD18 values, source/early/late band counts, repeated in-season daily profiles, output hashes against summaries, paired denominators, setback partition, warm-up contrasts, and separate replay byte identity.

## Rechecked central results

| Site | Seeded annual useful heat, kWh | Early / late annual change, % | Seeded hourly maximum, kW | Early / late maximum change, % |
|---|---:|---:|---:|---:|
| Padova | 9574.0182 | +0.894239 / +0.337144 | 10.049540 | +2.029262 / −2.103222 |
| Bolzano | 15851.2138 | −0.097977 / +0.026815 | 17.808621 | 0 / 0 |
| Roma | 6602.2288 | +2.033 (rounded) / +0.500 (rounded) | 8.909927 | +8.326 (rounded) / +3.381 (rounded) |
| Palermo | 5078.8372 | +3.868129 / +0.981468 | 8.292527 | +5.969112 / +1.524234 |

The figures in RES-004/005/007 agree at their reported precision. These are 12 primary matched runs; the additional repeat/warm-up runs and replay do not increase the number of physical cases.

RES-006 reproduces: Padova seeded has 2730 calendar hours at 19°C, 1323 with positive load above 1e-6 W, and 2375.617488 kWh there. It is correctly a post-hoc diagnostic, not a primary endpoint or observed operation.

RES-008 reproduces: all three Bolzano maxima occur at hour 6912, 16 October 00:00 in the imposed non-leap calendar, with 0→19°C lower-setpoint transition and old air 9.6413748°C. A shared seasonal recovery explains the equal maxima. Keeping this maximum and explaining it is preferable to retrospectively excluding it. No capacity-sizing inference is justified.

Padova maximum air temperature is 41.046114°C, supporting LIM-005. No positive heat was found at the 0°C off-season setting in these 20 files. This verifies the realised case, not universal equivalence between 0°C frost protection and disabled heating.

All 175,200 exported hours have exact adjacent old/current air and mass continuity. Independently evaluated residuals are below 3e-10 W, comfortably below 1e-6 W. My algebraic summation order gives a slightly different machine-level maximum from the manuscript's cited earlier checker; this is not a failed balance or meaningful physical discrepancy. Per-surface reconstruction error is about 1.4e-12 W; reconstructed-to-solver forcing differences are below 3e-12 W. The raw-IR annual mean air-minus-sky values include 12.775656 K for Padova and 17.620782 K for Bolzano. This verifies implementation of the declared approximation, not hourly radiative fidelity.

## Full-rubric red-team assessment

| Rubric / claims | Challenge and assessment |
|---|---|
| Conference foundation: LIT-001, GAP-002, RES-001/002, LIM-004 | Annual national final-energy checks cannot validate this hourly useful-sensible single-zone case. The manuscript correctly separates them and does not claim conference executable reproduction. Partial historical weather coverage in the chapter is material. |
| Closest work / extension: LIT-002/003/004, GAP-001, NOV-001 | The 2022 timing study is close enough to defeat broad first-method novelty. V03 acknowledges this. Isolating equal within-band durations across four weather exposures is an experimental distinction, not proof of a new scientific gap. LIT-004 remains partial-access and is not used to establish novelty. |
| Data and estimands: DAT-001–004, RQ-001/002, MET-001/002/004 | Mean survey on-hours become assumed high-setpoint durations. This mismatch is disclosed rather than erased. One unweighted respondent, an assumed envelope and four weather files provide no population inference. Seeded, earliest and latest profiles are not extrema or probability intervals. |
| Numerical provenance: RES-003–008, MET-005/010 | Independent direct-file calculations support the central values and same-site denominators. Duplicate execution is reproducibility evidence, not sample replication. The seasonal peak is exposed explicitly. |
| EUReCA implementation: MET-003/008/009/011 | Single-commit code is logically bounded; saved all-hour state and actual forcing checks pass independently. Surface calculation initializes each surface. Native transposition/time and annual-mean sky remain assumptions, not verified physics. General EUReCA/package certification does not follow. |
| Controlled weather comparison: MET-006, RES-005/007, INT-001 | Fixed building/setpoint/season supports conditional weather-exposure contrasts. Meteorological variables covary; no single-variable mechanism follows. A common climate-E seasonal mask at all sites limits local meaning, especially the Bolzano peak. |
| Validation: MET-007, GAP-002, DAT-004, LIM-002 | A matched independent benchmark and empirical observations remain missing, not implicitly passed. Numerical conservation does not establish accuracy. No acceptance statistic or fabricated empirical comparison is supplied. This boundary must remain prominent in prose. |
| Uncertainty: LIM-001/003/005/008, INT-001 | Annual changes as small as Bolzano's 0.027% can change under untested model assumptions. Rank/sign robustness, universal climate ordering, real savings and equipment sizing remain unsupported. V03 already states these limits. No confidence interval is warranted for this deterministic case. |
| Quarantine: LIM-006/007 | Earlier physical results are excluded from the argument. The current round did not independently re-investigate archived defects or execution counts; historical defect details await the permitted closure pass. They are not needed to establish the v5 numerical checks above. |

## Findings requiring revision

### R4-01 — Minor — Current permission/metadata fields still describe superseded work

**Claims:** MET-002/003/004, INT-001 and related machine-ledger fields.

**Evidence:** MET-003's permitted wording ends “weather forcing still requires correction,” although MET-009/011 and RES-004–008 describe the v5 correction and current conditional results. MET-002 still points to a v3 summary for a current invariant; MET-004's next action still directs reviewer 02; INT-001's population/geography field says Padova while its exact claim covers four exposures.

**Consequence:** A prose worker using the designated wording/metadata could incorrectly reintroduce a pending-correction statement, cite a quarantined run for a current invariant, or narrow an all-site claim to Padova. The underlying calculations are not affected.

**Checkable remedy:** In v04, make these fields explicitly current-v5 or explicitly historical. Replace the MET-003 pending-correction wording with the admitted v5 scope and retained approximation limits; cite v5 profile output for the invariant; remove obsolete reviewer assignments; align INT-001 geography with its four-coordinate claim. Mirror changes in the machine ledger and verify no scientific result gains broader wording.

## Unperformed checks and limits of this review

No fresh annual EUReCA rerun or new runtime synthetic perturbation/order suite was performed in this initial pass. I inspected the corresponding source logic and independently reconstructed actual exported all-hour forcing/state instead. Thus I do not independently re-certify every earlier transition/order-test assertion. A saved replay's byte identity was checked; its production was not observed.

I did not conduct a systematic literature search, obtain the unavailable full 2025 archetype PDF, re-inventory all possible local/third-party meters, independently benchmark against another solver, validate the envelope or recover original conference executable lineage. Official questionnaire semantics were assessed through the provided locator/audit and the independent 2022 primary description; this pass did not render the questionnaire PDF itself. The raw field values and source mapping were rechecked directly.

I did not retest all archived versions, certify every historical execution-count statement, or inspect prior findings for closure before saving this critique. Source-registry additions may be concurrent; a final unresolved-alias check should use the final frozen v04 registry. None of these limitations is replaced by the numerical pass.

After this initial critique is saved, prior issue materials may be opened for a separately labelled closure pass. Final v04 critical/major closure remains a distinct required check.
