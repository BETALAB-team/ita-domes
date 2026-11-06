# Compact 2021 H/E candidates

The predeclared rule selects **4 heating groups and 5 electricity groups**. This is a new compression method, not a retuning of the original Tukey significance threshold. It keeps the original 2021 demographic labels and the same 80 location × dwelling × archetype input cells.

For each K=1–15, exact dynamic programming partitions training cell means in increasing order to minimize count-weighted squared error. Five household folds estimate prediction loss with all means, ordering, partitions and predictions fitted to each training subset. Full-data fits provide final lookup tables only after candidate evaluation. The selected K is the smallest within one standard error of the minimum mean fold RMSE while retaining at least 95% of full-data between-cell mean variance. The SE is a five-fold descriptive uncertainty measure, not a survey confidence interval; this heuristic does not imply equal losses or an equivalence test.

|Response|Selected K|CV RMSE ± fold SE|Between-cell variance retained|Best CV K|
|---|---:|---:|---:|---:|
|Heating|4|4.79145 ± 0.02757 h/day|95.313%|15|
|Appliance electricity|5|720.6076 ± 3.6263 kWh/year|95.536%|10|

Heating's K=15 minimum lies at the upper search boundary; the prediction curve is nearly flat at larger K. No global optimum over all possible K is claimed. The selected compression deliberately trades some prediction performance for fewer groups and the declared fidelity requirement. In particular, a one-SE choice can have a positive paired loss difference from the best candidate; the paired contrasts are reported rather than described as no loss.

`candidate_curves.csv` gives every K and fold-RMSE SE, fidelity and selection flags. `fold_losses.csv`, `household_folds.csv` and `selected_paired_comparisons.csv` expose validation and paired losses. `uncompressed_cell_cv.csv` supplies the training-cell-mean benchmark. `all_K_cell_lookups.csv` retains every full-data candidate mapping. The selected cell lookups, group n/mean/SD and separately survey-weighted means, household assignments, and Appendix A long/wide tables are included.

K=7 heating / K=5 electricity sensitivities represent the original article's resolutions, with new 2021 DP memberships. K=9 / K=13 represent the prior 2021 run's resolutions. They are not the previous Tukey partitions; the actual previous Tukey partitions are compared only descriptively through full-data fidelity (99.104% H; 99.533% E), without leaking their full-data membership into cross-validation.

There are 80 final supported factor cells. Four heating segments absorb 76 cell boundaries and five electricity segments absorb 75; these are segment reductions, not a sequential significance-merge history. Selection operates on household counts to match the original unweighted grouping estimand. Official survey weights appear in separately labelled summaries, not in the optimization. The electricity outcome is modelled appliance kWh, not metered total energy. Heating is conditional reported hours, not cooling hours.

Validation is conditional on the already estimated demographic archetypes; it does not independently validate the whole demographic learning pipeline. Same-wave random household folds do not establish temporal or geographic transportability. Neither the 95% fidelity nor the fold SE quantifies explained household-level energy variance.
