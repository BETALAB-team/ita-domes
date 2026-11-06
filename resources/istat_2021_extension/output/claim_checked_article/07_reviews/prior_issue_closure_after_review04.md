# Prior-issue closure after independent review 04

20 October 2026. Reviewer `/root/review_04`, requested `gpt-6-astra`, runtime model confirmation unknown.

This is a separately authorized second pass after the initial full independent critique was saved. `review_04.md` remains frozen at SHA-256 `ce16dcca6b71bd1da3e5ce0c47ba44001628d66ed8535833da3d3c4fa549b687`. Input claim v03 remains `c96e3307de9b9587cc554c076f5d3527cc3ce8a6c29b12342777c07bc93c5eb2`.

Read the findings in reviews 01–03, responses 01–03, issue_register_03.json, closure_01_after_review02.md, closure_02_after_review03.md, v5_verification_supplement.md and the prior runtime/provenance/forcing diagnostic records. These were compared with the source and direct 20-case calculations independently completed **before** accessing those materials. No manuscript, old review or simulation was edited; no new annual run was executed.

## Dispositions

| Issue | Current closure recommendation | Evidence and boundary |
|---|---|---|
| **R1-01 critical: recursive dispatcher** | **Close for the bounded v5 single-commit path.** | Current source evaluates candidates from unchanged old state; my all-175,200-hour balances and adjacent-state continuity pass. Earlier reviewer02 runtime transition tests cover the actual stepper and compare against independently assembled matrices. I inspected those results after my independent pass, not as its foundation. This is not general EUReCA/HVAC certification. |
| **R1-02 major: high-setpoint hours versus operation** | **Closed; retain wording throughout prose/figures.** | Original EndUse mapping and my executed seeded source schedule match all 8760 setpoints; raw row 3523 has 4/4/1 and timer=1, compact filter has 372 eligible/minimum ID3523. All three daily high-setpoint profiles preserve their bands. Rechecked v5 setback heat proves actual heating runtime is not fixed. V03 explicitly labels the construct and post-hoc diagnostic. |
| R1-03 minor: inventory evidence class | Closed. | DAT-004 is a bounded derived estimate, not an observed measurement. No city-wide absence of measurements is asserted. |
| **R2-01 critical: cloud99/sky forcing** | **Close for the declared v5 IR-to-surface path.** | Independently derived raw-IR sky means and EPW/JSON active-field equality reproduce; source replaces the scalar before preprocessing and guards unused invalid fields. My per-surface actual-export and solver-node reconstruction passes at every site. The downstream surface gate that previously blocked this composite issue now passes. Annual-mean sky and unapplied solar offset remain limitations, not closed physical-accuracy questions. |
| R2-02 minor: incomplete historical command | Closed for interface/provenance. | Current runner takes SITE/campaign, records full argument vector/cwd and rejects existing output; historical omission is annotated. My direct replay-to-primary byte comparison passes. Prior saved refusal/replay tests support execution history, but I did not re-execute the interface or annual replay. |
| **R3-01 critical: roof inherits preceding-surface forcing** | **Close for exact v5 correction and bounded fixed-case result admission.** | Current source initializes each surface's own inputs and separates glazing. My independent per-surface SW/LW/window and solver-forcing reconstruction, full-hour balances, summary/hash/paired/peak/partition audit corroborate the exact corrected campaign. Earlier reviewer03 runtime roof-first, isolated-surface, zero-solar, zero-longwave and order tests were now inspected and agree with the corrected source design. The fresh-review04 condition has been substantively met; fourth-cycle bookkeeping still awaits v04 and final closure. |
| R3-02 minor: sky/time limitation absent | Closed in v03, subject to preservation. | MET-009, RQ-001/002, RES-004/005/007, INT-001 and LIM-008 attach the annual-mean/native-time qualifications. No robust rank/sign/physical peak inference is admitted. |
| R3-03 minor: closest ±2h precedent | Closed in v03, subject to preservation. | My initial review directly opened primary 2022 p6 and Table1; current LIT-002/GAP-001/NOV-001 explicitly distinguish prior timing uncertainty from the current fixed-duration contrast. Novelty remains unverified, not established by a failed search. |
| R3-04 minor: canonical current status | Substantive canonical-pointer remedy present; residual small field inconsistencies tracked by **R4-01**. | The current validation matrix leads with v5 and marks benchmark/meters missing. Frozen historical status stays historical. V03 MET-003 permitted wording and other metadata still need the R4-01 cleanup; do not treat this as a reopened physics defect. |

## Why critical/major closure is now justified

The three technical defects were separate: incorrect state advancement; invalid missing-cloud conversion; and stale surface-local preprocessing. A pass at one layer could not rescue another. V5 combines the one-commit state path, valid-IR scalar and explicit per-surface preprocessing. This round independently checked their **combined saved actual forcing and state trajectory**, alongside source mapping and result arithmetic, before learning prior verdicts. That supports admitting the current bounded deterministic contrasts. It does not retrospectively rehabilitate v1–v4.

The new v5 result also retains the inconvenient Bolzano seasonal-recovery maximum and the actual setback heating, avoiding a changed endpoint or an incorrect heater-runtime interpretation. No new critical/major issue was found in the independent review.

## Remaining dependencies and limits

- R4-01 minor wording/provenance/metadata cleanup must be reflected in v04 and its ledger, response and issue register. The inaccessible EPW24.1 URL should be replaced or explicitly qualified using the registered accessible24.2 documentation.
- Final closure must inspect the actual v04 hash and artifacts. This file **does not mark cycle04 complete**, release a prose author, or assert that absent artifacts exist.
- All earlier annual physical effects remain quarantined. I inspected historical explanations and diagnostics but did not rerun those defective paths or independently reconstruct the entire orchestration timeline.
- Missing matched building benchmark, independently documented meter series, full archetype text and quantified structural/timing/sky sensitivities remain missing. They are standing limitations rather than secretly completed validation.
- The supplementary registry now provides current aliases; the final frozen revision still needs a machine check that claim/source links and status enums remain consistent. Registry completeness is not scientific proof.
- My round04 runtime extent remains as recorded in the initial critique: independent source-profile execution and direct-file reconstruction, not a fresh annual solver run. Prior transition/order executions are specifically attributed to their original reviewer artifacts.

The next permitted step is the final v04 closure audit. Retain the reviewed scope: one unweighted response, one assumed dwelling, three setpoint placements, four coordinate TMYs, ideal useful sensible heat, and the documented seasonal/radiation/time conventions.
