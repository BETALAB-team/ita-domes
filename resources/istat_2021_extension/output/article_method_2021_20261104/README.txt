ITALIAN USER ARCHETYPES AND ENERGY GROUPS — NEW 2021 RECONSTRUCTION
4 November 2026

Start with 06_report/Italian_user_archetypes_2021_reconstruction.pdf or its editable Word file.
The report's original article figure/table numbering is retained for corresponding outputs.

Main results
52,617 new 2021 demographic profiles; 46,712 records eligible for dwelling PFE.
63 empirical windows; 21 supported age/area windows in the primary polynomial fit.
A new 4-by-6 polynomial coefficient matrix and four conditional holdout scores.
9 new heating groups and 13 new appliance-electricity groups, estimated from 2021.
Appendix A covers 4 locations × 5 dwelling categories × 4 user types, with 160 H/E assignments.
All published 2013 values, original files and existing 2013 results remain unchanged.

Where the results are
01_inputs: audited source/model provenance, codebook mappings and input builder.
02_probabilities: primary labels, profile summaries, empirical/window-weighted probabilities,
  polynomial matrices and diagnostics, age/schooling sensitivity and household holdout.
03_energy: ANOVA, group means, lookup tables, merge histories and pairwise descriptions.
  appendix_A_assignment_2021.csv is the long 80-row lookup.
  appendix_A_assignment_2021_wide.csv is the paper-style 20-row lookup with 8 H/E columns.
04_figures: the new article-style plots as SVG, PNG and PDF, including H/E and Appendix A.
05_review: independent raw-record, probability, grouping, appendix and report checks.
06_report: Word/PDF, editable native equation and Appendix A table, source and export helpers.

Method definitions
All 2021 outputs use the same primary, freshly fitted four-type household labels.
The under-11 released age groups remain in the demographic profile; their blank schooling
is omitted from the education mean. Other unsupported active schooling profiles are excluded.
The primary merged no-credential/primary category uses 5 schooling years, with 0-year sensitivity.
The apartment categories follow the actual 2021 floor-position definitions.
Heating uses valid frequent-user daily time blocks referring to cold months of 2020.
Its 2021 grouped plot uses one box per H group because subjective insulation is not released.
Electricity is the audited 2021 ownership/use appliance model with fixed 2013 reference
parameters, split in its plot by declared cooling use. It is not a conversion from spending.
The final saved article inputs and Appendix A supply the location/dwelling/user cell structure.
The original-style Tukey-Kramer reduction uses full precision and deterministic tie handling;
terminal tests describe a partition selected on the same observations.
PESO-weighted summaries and a weighted probability matrix are separate from primary fits.

Probability matrix use
02_probabilities/polynomial_coefficient_matrix.csv stores full-precision coefficients.
Basis: [1, age/100, area/100, age*area/10000, age^2/10000, area^2/10000].
Primary empirical fitting centres: ages 20..140 step20 and areas 40,80,120 m².
Raw polynomial estimates are preserved with component-range and row-sum flags.
At fitting centres, their sums are 0.98044..1.03517. On the larger grid, 35 of 63 points
have component-range violations outside the primary area domain.
predict_probabilities_2021.py --age 60 --area 80 returns the raw estimates, flags,
and an explicitly labelled normalized vector when the raw components permit it.

Rebuilding
Use the Codex bundled Python recorded in the delivery manifest. The scientific dependencies
are pinned in requirements-2021.txt and are installed only under this task's .scientific folder.
Run run_analysis_2021.py from the repository resources/istat_data directory to rebuild the
isolated input, 2021 models and figures. It requires the native local ISTAT2021 release,
the audited revised_paper_outputs/appliance_audit/electricity_2021.csv and its provenance,
and the repository helper modules. It never runs a 2013 model or a legacy full runner.
For the report run 06_report/build_report.py, then 06_report/export_word_pdf.ps1
(native Microsoft Word), followed by 06_report/render_and_verify.py.

Package scope
The ZIP contains new result tables, coefficients, compact household assignments, plots,
reports, checks and scripts. Task-local installed packages and duplicate 364-column
household input/energy tables are left out. Their local paths and hashes are recorded.
Read-only source code copies are included for the recovered methods.
