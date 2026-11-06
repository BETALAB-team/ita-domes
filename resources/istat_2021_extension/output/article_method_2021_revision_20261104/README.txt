ITALIAN USER ARCHETYPES — REVISED 2021 DWELLING FIT AND COMPACT ENERGY GROUPS
4 November 2026

Open 06_report/Italian_user_archetypes_2021_reconstruction.pdf or the editable Word file.
These revised files supersede the previous 21-window probability model and the practical
use of its 9H/13E statistical grouping. The 2013 published results and previous run are intact.

Selected results
- Full 4×6 quadratic probability matrix on recovered historical 125-node geometry.
- Age centres 15..135 by5; area centres40,64,88,112,136; half-windows15years/40m².
- On identical held-out window targets, old21 training RMSE.040465 -> new125.022556.
- All probabilities sum to1 and stay in[0,1] throughout age15–135/area40–136.
- Practical 4 heating groups and5 electricity groups, using80 location×dwelling×user cells.
- These retain95.31% and95.54% of between-cell mean variance.
- Full new AppendixA:80 combinations and160 H/E assignments.

The primary outputs
02_probability_candidates/recommended_full_quadratic_matrix.csv
02_probability_candidates/recommended_apparent_fit_statistics.csv
02_probability_candidates/recommendation.json
02_probability_candidates/matched_vintage_area_curves.csv
02_probability_candidates/matched_vintage_observed_fitted_nodes.csv
02_probability_candidates/fair_geometry_validation_summary.csv
03_energy_candidates/H_selected_group_summary.csv
03_energy_candidates/E_selected_group_summary.csv
03_energy_candidates/H_selected_cell_lookup.csv
03_energy_candidates/E_selected_cell_lookup.csv
03_energy_candidates/appendix_A_compact_long.csv
03_energy_candidates/appendix_A_assignment_2021_wide_flat.csv
03_energy_candidates/compact_assignments_with_outcomes.csv
04_figures:article-style SVG,PNG,PDF figures and their exact node data.
05_review:independent probability,compact-group and report checks.
06_report:final Word/PDF, native editable equation and editable colour AppendixA.

Use
predict_probabilities_2021.py --age 60 --area 80
The helper evaluates the saved full-precision matrix without clipping/renormalization.
Its default application domain is age15–135years at2021/area40–136m².
All age/area curves are national and pooled over location/dwelling type.
Figure6follows the saved implementation:crosses are FITTED node probabilities,curves
are mean fits by window-centre construction era. Figure7compares ACTUAL empirical
proportions to those fits. Matched empirical/line diagnostic CSV uses the same node weights.
Era-centre windows overlap adjacent construction periods. Windows are not independent.

Energy selection
Exact contiguous dynamic-programming partition of sorted80cellmeans, weighted by counts.
Five training-only household folds compare K1..15. Select smallestK within1SE of minimum
CVRMSE that also retains at least95% of full-data between-cell mean variance.
95% is a declared compactness/fidelity choice; this is not a simulation-equivalence test.
Heating K15 is the search-boundary minimum. Electricity K5 passes the1SE rule narrowly.
The full K curves,paired losses,97.5/99% fidelity sensitivity,and7H/5E sensitivity are included.
This is a fresh practical partition of the80cells; it may split old Tukey memberships.
The prior9H/13E Tukey analysis remains saved in the previous output folder.
The electricity groups E1–E5 are the requested groups; cooling use is the plot's hue.

What stays fixed
All52,617 demographic labels and46,712 dwelling draws from the previous2021run.
2013publication numbers and original sources. Same2021cohort and energy response definitions.
Appliance kWh use fixed reference2013power/efficiency assumptions and2021ownership/use.
2021heating refers to coldmonths2020. No matching subjective-insulation rating is available.
2021apartments are floor-position categories. PESO weights are secondary diagnostics;
regional calibrated multiplier absent => modelling convention1.

Rebuilding in the existing repository
Use bundledPython and the pinned scientific packages in the prior run's.scientific folder.
Run02_probability_candidates/run_candidates.py and03_energy_candidates/compact_energy.py
from resources/istat_data; their specifications and source paths are saved alongside them.
Use build_revision_figures.py then06_report/build_report.py, export_word_pdf.ps1 and
render_and_verify.py for final publication artifacts. Native Microsoft Word exports the PDF.
The bundle expects the existing repository input paths; it is not a standalone raw-data release.
01_reference copies the fixed labels/dwelling samples needed to inspect or port the models.
Do not execute historical full runners against original2013output locations.

Archive scope
Includes revised code,coefficients,all candidate/validation tables,compact assignments,
figures,reports,source-reference notes,and independent checks. Excludes QApageimages,
temporary rasterizer folders and installed libraries. PACKAGE_MANIFEST.json lists SHA256s.
