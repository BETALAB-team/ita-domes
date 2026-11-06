# Closure check of review 01 after independent review 02

Date: 20 October 2026. Reviewer: the fresh-context review-02 agent, requested model `gpt-6-astra`; runtime-confirmed model metadata remains unknown.

This second pass was explicitly authorized only after the independent critique was saved. I then read `review_01.md`, `response_01.md` and `issue_register_01.json`, and checked their requested remedies against immutable v01 and the independent calculations already performed for review 02. No manuscript or issue-register edits were made. No annual campaign was rerun for this closure pass.

## Input integrity

| Artifact | SHA-256 |
|---|---|
| Reviewed v01 | `5e35bc02e77d6345a9c82be51a8a405d5d4ba3e8490f6c24d037b8488a04bab1` |
| Review 01 | `c7a8d509563fbef997f1615cb03dc0700a83800d619367b91207d944afaa6925` |
| Response 01 | `61a5c3fb372994fe8902ed2bb1e79ada8025ea9139d5be6f39697059fb567214` |
| Issue register 01 | `d842911844b06556f07b26f051af2e6093908e8f0b9323caf71ac3e784817314` |
| Independent review 02, unchanged | `3e258716d7516c05d8b78c306b358ff5651d204ed39110b831652877fa761fd4` |

## Closure disposition

| Earlier issue | Status | Meaning |
|---|---|---|
| R1-01: recursive dispatcher/state inconsistency | **Conditional closure** | The specific single-step state-update correction is independently verified within the declared sensible-only adapter scope. The wider admission of numerical findings remains blocked by the separate R2-01 weather-forcing defect. |
| R1-02: high-setpoint duration versus heating-operation duration | **Closed for the wording/estimand issue** | v01 and its machine ledger explicitly distinguish reported operating hours, encoded high-setpoint hours and unconstrained actual heating delivery. The new post-hoc arithmetic checks, but its v3 physical quantities remain withheld under R2-01. |
| R1-03: absence of measurements labelled observed measurement | **Closed** | DAT-004 uses `derived estimate` in the readable and machine versions, retains the bounded inventory scope, and makes no assertion that compatible observations cannot exist elsewhere. |

### R1-01 — Specific corrective action verified; final result gate remains open

Review 01 requested unchanged beginning-of-step states for all candidate modes, one commit of the accepted state, meaningful transition tests, complete hourly balances, preserved archives, and new paired/repeat/warm-up executions.

The v01 response and manuscript satisfy the archival and disclosure requirements: MET-003/MET-005/MET-008 describe the bounded local correction; LIM-006 quarantines original dispatcher-based effects; RES-004/RES-005/RES-007 and INT-001 remain candidates pending independent review. They do not claim that the conference executable shared the local defect.

Independent evidence already saved before reading review 01 confirms the technical remedy:

- Source inspection finds the free and constrained calls in `04_analysis/single_commit_1c.py` read one old air/mass state and commit one accepted state.
- `reviewer_02/runtime_checks.py` exercises the complete stepper for off-to-heat, heat-to-off, off-to-cool, cool-to-heat and equilibrium, and compares to an independently assembled linear system. Maximum output-vector difference is `4.55e-13`; exported old states match the initial states. The different prior-mode labels do not enter the selector's physical solve.
- `reviewer_02/independent_audit.py` recomputes all 175,200 v3 hourly node balances, verifies exact adjacent-hour state continuity and equal within-site forcing/parameters. Maximum residual is `2.0236257114447653e-10 W`, with no failures at `1e-6 W`.
- All saved annual/peak arithmetic, exact repeats, warm-up gates and inspected source/input/output hashes reproduce.

This supports closing the **specific recursive state-transition defect for the current ideal sensible adapter**. It does not establish validity of the complete weather-to-load pipeline, general EUReCA dispatch, latent/HVAC systems or empirical predictions.

Review 02's R2-01 independently identified untreated opaque-cloud sentinel values in every weather file, which produce invalid sky-radiation forcing upstream of the corrected network. That is a different defect. It must not be described as failure of the successful one-state-commit fix, nor hidden by marking the broader R1-01 gate fully complete. The original issue-register condition includes confirmation of candidate results; that wider condition is not met. Recommended register wording: “dispatcher correction verified within bounded scope; candidate physical results still blocked by R2-01.” Retain a linked open result gate until the forcing correction and new run evidence pass review.

### R1-02 — Corrected semantics and estimand

The central v01 blocks meet the requested distinction:

- GAP-001 separates reported mean winter-day operating hours from their encoding as 22°C periods.
- RQ-001 explicitly allows heating at the 19°C setback.
- MET-001 explains the source mapping; MET-002 says actual heating delivery duration is not preserved.
- RES-003 labels the 4/4/1 invariant as high-setpoint duration and repeats the setback caveat.
- RES-004/RES-005/RES-007 describe placement contrasts without treating equal encoded durations as measured or enforced equipment runtime.
- INT-001 limits interpretation to the constructed setpoint placements and one assumed case.

The same exact claims appear in `claim_ledger_v01.json`; the source-to-setpoint mapping was independently executed in review 02 and matched every annual primary-case setpoint. Original questionnaire pages 41–43 and `EndUse.py:593–625,643–656` substantiate the survey/control distinction.

RES-006 is labelled post-hoc, and its stored v3 partition independently reproduces 671 positive-load hours among 2730 setback hours, with 893.21783218 kWh. This confirms the **calculation over archived v3 output**, satisfying the arithmetic part of the earlier closure request. It does not overcome R2-01. The source code itself establishes that setback heating is allowed; closure of this semantic issue does not depend on retaining a now-compromised numerical illustration.

Future prose/figures must retain the distinction and label setpoint levels rather than equipment on/off. A future forcing-corrected partition may replace the example after review. Those continuing obligations do not reopen the resolved v01 wording issue.

### R1-03 — Evidence-class correction confirmed

DAT-004 reads `Evidence class: derived estimate` in v01. Its JSON `evidence_class` is also `derived estimate`. The claim describes the inspected local inventory, remains partially supported, and explicitly allows that suitable observations could exist elsewhere. A machine-ledger scan finds zero claims classified as `observed measurement`.

This closes the metadata error identified in review 01. It does not upgrade the underlying inventory into an exhaustive search or supply empirical validation.

## Gate consequence

The response to review 01 was substantive: it repaired the demonstrated state-update path, corrected the estimand wording and corrected the evidence class. Review 02 independently verified those changes and found a separate critical upstream defect. There is therefore no contradiction between bounded closure of the earlier corrective work and withholding the current physical outputs.

R2-01 remains open and blocks numerical scientific admission. R2-02's missing recorded site argument also remains to be addressed in future replay provenance. Neither this closure pass nor the original review 02 completes the four-cycle requirement or grants empirical validity.
