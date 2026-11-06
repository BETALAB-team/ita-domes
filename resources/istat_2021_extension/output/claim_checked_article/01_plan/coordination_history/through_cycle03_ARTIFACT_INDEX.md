# Research package index

Working date: 20 October 2026. This index is being assembled during the workflow; final status is recorded in [STATUS.md](STATUS.md). No missing empirical validation is implied by a completed review cycle.

## Scope and provenance

- [Research questions](01_plan/research_questions.md), [decisions](01_plan/decisions.md), [task DAG](01_plan/task_dag.md), and [readiness checklist](01_plan/readiness_checklist.md).
- [Conference-to-extension matrix](00_context/extension_matrix.md), [complete chapter evidence](00_context/conference_evidence.md), [source inventory](00_context/source_inventory.md), and [prior-work audit](00_context/prior_work_audit.md).
- [Official survey/selected-record verification](00_context/survey_schedule_verification.md).
- [Source ledger](02_evidence/source_ledger.md), [machine source records](02_evidence/sources.json), [conference source records](02_evidence/conference_sources.json), [bibliography](02_evidence/bibliography.bib), [search log](02_evidence/literature_search_log.md), and [closest-work comparison](02_evidence/nearest_work.md).
- [Local source/artifact ledger](02_evidence/local_source_ledger.md) and [machine local registry](02_evidence/local_sources.json) resolve code, raw input, run and verification aliases; producer-target hash checks distinguish summary CSV from JSON.
- [Actual agents and requested models](agent_manifest.md). Runtime-confirmed model metadata is not exposed.

## New simulations and validation

- [Padova-first simulation protocol](03_methods/simulation_protocol.md), [v2 addendum](03_methods/simulation_protocol_v2_addendum.md), [single-commit correction protocol](03_methods/correction_protocol_v3.md), [weather-forcing correction protocol](03_methods/weather_correction_protocol_v4.md), [climate selection](03_methods/climate_selection.md), and [official-source classification context](01_plan/climate_classification_context.md).
- [Validation protocol](03_methods/validation_protocol.md), [validation matrix](03_methods/validation_matrix.md), and [candidate Padova measurement access plan](03_methods/padova_measurement_access_plan.md).
- [Current candidate v4 simulation results](04_analysis/simulation_results.md), [v4 paired contrasts](04_analysis/paired_schedule_contrasts_v4.csv), and [post-hoc setpoint partition](04_analysis/posthoc_setpoint_partition_v4.md).
- [Local single-commit adapter](04_analysis/single_commit_1c.py), [transition checks](04_analysis/test_single_commit_1c.py), [current campaign script](04_analysis/run_climate_campaign_v4.py), and [independent balance checker](04_analysis/check_full_step_balances_v4.py). Exact frozen correction/forcing code and executed commands are linked from per-run manifests.
- [Methods execution report](03_methods/methods_execution_report.md) and [run inventory](05_simulations/run_inventory.csv). All 45 original v1/v2/v3 annual outputs remain quarantined. V4 has separate immutable input/configuration/source snapshots and is undergoing independent review. Its 20 campaign cases comprise 12 primary cases and 8 technical checks; one additional isolated replay checks the new-destination interface.

## Claim review and manuscript

The following later-stage links become deliverables only after their gates complete; consult STATUS.md while work is active.

| Cycle | Independent review | Author response | Immutable revision | Issue snapshot |
|---|---|---|---|---|
| 1 | [Review 01](07_reviews/review_01.md) | [Response 01](07_reviews/response_01.md) | [v01](06_claim_drafts/article_claims_v01.md) | [Register 01](07_reviews/issue_register_01.json) |
| 2 | [Review 02](07_reviews/review_02.md) | [Response 02](07_reviews/response_02.md) | [v02](06_claim_drafts/article_claims_v02.md) | [Register 02](07_reviews/issue_register_02.json) |
| 3 | [Review 03](07_reviews/review_03.md) | [Response 03](07_reviews/response_03.md) | [v03](06_claim_drafts/article_claims_v03.md) | [Register 03](07_reviews/issue_register_03.json) |
| 4 | [Review 04](07_reviews/review_04.md) | [Response 04](07_reviews/response_04.md) | [v04](06_claim_drafts/article_claims_v04.md) | [Register 04](07_reviews/issue_register_04.json) |

- [Initial claim draft v00](06_claim_drafts/article_claims_v00.md), [final reviewed claim snapshot](06_claim_drafts/article_claims_final.md), and [latest machine claim/evidence ledger](06_claim_drafts/claim_evidence_ledger.json).
- [Current issue register](07_reviews/issue_register.md), [final Astra closure](07_reviews/final_closure.md), and [workflow artifact/hash audit](01_plan/workflow_audit.json).
- [English prose first draft](08_manuscript/article_first_draft.md) and [claim-to-paragraph map](08_manuscript/claim_to_prose_map.md).
- [Figure briefs](09_figures/figure_briefs.md) and [figure/source/script/QA manifest](09_figures/figure_manifest.md).

## Scientific limits

The new experiment is conditional on one assumed dwelling, one selected unweighted Veneto survey response, a source-derived two-level temperature mapping, three constructed placements, and four coordinate TMY exposures. It does not estimate a representative city or national stock effect, identify actual household switching, demonstrate improved prediction, or establish empirical validation. Whole-building benchmarking and compatible independent observations remain separate dependencies. The 2026 CLIMA chapter's national comparisons remain published results, not newly reproduced results.
