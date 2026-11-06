# Energy Studio 0.5 validation

## Scientific run

Completed independent run: `energy_studio_runs/20260930_120420_3a8b86`.

- 45 scenarios, 45 successful EnergyPlus runs, zero warnings.
- Milano / Multi-family / P2 is the single fitting reference.
- Independent schedules are frozen before simulation. Both sets of target predictions are frozen before target simulations.
- The saved reference fit and prediction checksums are unchanged after benchmarking.
- Known-aggregate test: heating CVRMSE 4.8963%, cooling 32.9703% (44 target cases).
- ISTAT-profile test: heating CVRMSE 11.3403%, cooling 37.6823%, appliance 31.2817% (44 target cases).
- Built-in schedules are synthetic. No measured-building validation is claimed.

## Automated checks

29 unit tests pass across the five Energy Studio test modules. The independent tests cover input isolation, equal-total timing pairs, demographic-only assignment, all 8,760 hours of EP schedule mapping, invalid input rejection, exclusion of the appliance identity from accuracy metrics, and prediction invariance to changes in target outcomes.

The saved demographic model reproduces all existing household archetype labels. Assignments for the independent cases use demographics only.

## Packaged application

`energy_studio/check_packaged_independent.py` passed the packaged GUI self-test and a reference-only run. The EXE launched exactly one EnergyPlus simulation and reproduced all 90 prediction rows from the full independent experiment, with rtol 1e-10 and atol 1e-7 kWh. Frozen input schedule hashes also match.

Acceptance receipt: `energy_studio_runs/packaged_v0_5/acceptance.json`.

The packaged PowerPoint refresh command completed with exit code 0. Its report revision is `energy_studio_runs/20260930_120420_3a8b86/reports/20260930_122617_8fc5cd`.

## Earlier workflow regression

The original shared-input benchmark `energy_studio_runs/20260929_225633_26db04` was reproduced with its saved reference and zero new EP simulations. All original predictions match at rtol 1e-12 and atol 1e-9 kWh. The legacy RC function was prohibited during this acceptance test.

Receipt: `energy_studio_runs/20260930_122600_97926a/reuse_acceptance.json`.

## Presentation checks

The 19-slide report contains 19 native editable charts and native tables. Structural integrity, chart data/workbooks, layout and import validation pass. Speaker notes are empty. Every slide was rendered and visually inspected; revised slides were rechecked. This verifies the rendered output and package structure, not execution in native PowerPoint.

The previous v0.4 EXE, reports and scientific run are preserved. No commit was made, and main was not modified.
