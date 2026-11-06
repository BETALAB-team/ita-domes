# Corrected 2013 and 2021 manuscripts

Run from this folder:

```powershell
.\.venv\Scripts\python.exe run_revised_papers.py
```

This regenerates the two scientific analyses, vector figures, tables and PDFs:

- `output/pdf/italian_user_archetypes_corrected_2013.pdf`
- `output/pdf/italian_user_archetypes_corrected_2021.pdf`

The legacy workflow, configurations and outputs are preserved. All new scientific
results are under `revised_paper_outputs/`. The editable manuscript generator is
`build_revised_papers.py`; its narrative and tables draw numbers from the results.
Each manuscript includes Figures 1-10 and A.1, a sensitivity figure B.1, Tables 1-8,
the two algorithms, equations, conclusions, references and a correction register.
Every figure is also exported independently as SVG, PDF and PNG.

## Scientific scope

These are revised research drafts, not approved journal corrigenda. The original
question sequence and figure types are retained, but unsupported conclusions and
invalid statistical formulations are not copied. In particular:

- Ages remain interval proxies. Education uses a documented common attainment
  proxy, not the original faulty age-class denominator or a claimed UNDP index.
- The weighted PCA and four-cluster k-means are fitted only to 2013 training
  households. The 2021 demographic transformation and centers remain frozen.
- The dwelling model uses jointly valid multinomial probabilities and genuine
  held-out household scores. Original independent polynomial F/R-squared metrics
  are replaced with appropriate coefficient and predictive-score tables.
- Energy groups are constructed on training data, selected on tuning data and
  evaluated once on test data. They are not forced to become pairwise significant.
- 2021 cooling ownership and use are observed. Insulation/window improvements in
  the last five years enter the heating regression and stratify Figure 8;
  Table 9 and Figure 11 summarize equipment, improvements and cooling hours.
  The improvement indicator is distinct from the 2013 subjective insulation
  rating. The 2021 appliance-kWh outcome is now modeled from ownership and use.
  The component sum reproduces all 20,000 saved 2013 totals to numerical
  precision. Appendix B documents coarser-input assumptions and alternatives.
- Figure 10 is a regional heating-hours choropleth. A municipal climate-zone map
  cannot be reconstructed from the released household geography. Historical
  schedule benchmarks are illustrative, not current compliance assessments.
- Model-based robust tests are not full survey-design-adjusted inference. No
  measured building-energy or simulation-performance validation is claimed.

The configuration `revised_paper_config.json` controls input-config location,
output paths, seeds, split proportions, proxy assumptions, probability penalties
and support thresholds. The source releases are resolved through
`multiyear_workflow_config.json`. Only the documented 2013 and 2021 source schemas
are supported by this scientific revision; a new survey needs a verified codebook
crosswalk, not just renamed columns.

## Rebuild only the manuscripts

To update only the 2021 equipment analysis and PDF against the existing frozen
reference fit, run `.\.venv\Scripts\python.exe revise_2021_equipment.py`.
This checks that the reused demographic and probability inputs are unchanged,
reruns heating groups and tests, and renders the revised 2021 manuscript.
Cooling hours retain the frequency-1-to-3 routing and refer to summer 2020.
Blank insulation selections are interpreted using the parent investment
question; an unanswered parent stays unknown.

The appliance reconstruction scripts read the model and workbook from `main`
using `git show`; they do not check out, modify or commit to that branch.
`audit_appliance_model.py` verifies the 2013 component boundary and
`model_electricity_2021.py` creates the 2021 reference-parameter estimate and
sensitivity scenarios. Run these with the bundled PDF Python before a focused
2021 rebuild. The full paired runner executes them automatically. Heating
Figure 8 also displays within-group yes/no-improvement HC3 contrasts, adjusted
with Holm over the fixed groups; unknown responses are excluded.

```powershell
.\.venv\Scripts\python.exe run_revised_papers.py --manuscripts-only
```

Do not use this option after changing scientific settings: rerun the full command.

## Tests and PDF inspection

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`verify_revised_papers.py` checks the PDF structure and renders every page with
Poppler, creating contact sheets for human visual inspection. Run it with a Python
environment containing `reportlab`, `pypdf` and `Pillow`. This machine's bundled
PDF environment is discovered automatically by the main runner; set
`ISTAT_PDF_PYTHON` to override it. Rendering also requires Poppler. A PDF must be
visually checked again after changing its content or layout.

Input/output hashes are recorded in `revised_paper_outputs/manifest.json` and the
individual `manuscript_checks.json` files. Household-level intermediate outputs
stay local; verify ISTAT research-file conditions before redistributing records.

## Dependencies

The installed project environment provides numpy, pandas, scipy, statsmodels and
matplotlib. The PDF renderer uses the bundled reportlab, pypdf and Pillow packages
and Windows Times New Roman fonts. On another machine, install
`requirements-revised-papers.txt` and configure equivalent fonts/rendering paths.
