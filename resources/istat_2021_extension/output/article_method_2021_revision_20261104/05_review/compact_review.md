# Independent check of compact 2021 heating and electricity groups

I independently implemented exact contiguous segmentation of ordered training cell means. My objective was direct household-count-weighted residual sum of squares for every possible interval and top-down dynamic programming. I did not import the candidate script. I verified all five fold assignments by shuffling sorted eligible household IDs with the declared seed, re-estimated means and ordering in each training fold and scored the held-out households. All 150 fold×K outcome RMSEs matched. No held-out cell lacked a training counterpart. I also verified the 2,400 full-data cell/K assignments and predicted levels, every selected group summary, all selected household assignments and the 160 H/E entries in the compact Appendix.

| Quantity | Heating | Modeled appliance electricity |
|---|---:|---:|
| Selected compact K | 4 | 5 |
| Minimum-error K in the tested 1–15 range | 15 | 10 |
| Best tested mean fold RMSE | 4.78517 h/day | 717.279 kWh/year |
| One-SE upper limit | 4.81277 | 720.832 |
| Compact mean fold RMSE | 4.79145 | 720.608 |
| Compact excess above best | 0.00628 h/day (0.13%) | 3.328 kWh/year (0.46%) |
| One-group mean fold RMSE | 5.05082 | 786.567 |
| Retained variation across the 80 cell means | 95.313% | 95.536% |
| Smallest K meeting only the one-SE criterion | 3 | 5 |
| Smallest K attaining only the 95% floor | 4 | 5 |

There is no conflict between the criteria at the selected K. For heating, K3 is within one SE but retains 91.50% of between-cell variation; the fidelity condition raises K to four. For electricity K4 retains 93.85%, and E5 is within the one-SE error band by only 0.225 kWh/year. These are pragmatic, sample-specific choices. “Within one SE” uses the marginal SE of the minimum-error candidate across its five fold RMSEs; the paired comparison still exhibits a nonzero predictive cost. The five folds share training data. These summaries do not establish equivalence, generalize to a different split with certainty, or identify an intrinsic number of physical behaviours. Heating achieves its minimum at the K15 upper search boundary, and should be called best *among tested K1–15*. The uncompressed 80-cell benchmark is separately reported.

With the same one-SE rule and alternative fidelity floors 90%, 95%, 97.5%, and 99%, the selected H counts become 3, 4, 6, 9, and E counts become 5, 5, 7, 10. The years' equal-count comparison fits 7H or 5E using 2021 cells and outcomes; H7 retains 98.65% and scores 4.78716 h/day. Neither 7H nor 5E implies unchanged historical class identity. The 95% criterion concerns variation **between cell means on the available sample**, not 95% of all household outcome variation or of predictive accuracy. Household-level CV is conditional on the previously fitted demographic labels. The full-data fidelity calculation is a design filter; completely independent confirmation would require a fresh evaluation sample.

The new groups are a practical segmentation of the same 80 cell inputs. The original-method 2021 Tukey partitions remain nine and thirteen; they are not the partitions that the compact DP subdivides or evaluates in its out-of-fold predictions. The reference 2013 outcomes remain seven and five. All original outputs were read-only. Dedicated verifier code and numeric results are beside this check.
