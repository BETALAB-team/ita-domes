# ISTAT household archetypes and energy groups

This folder collects the 2013–2021 analysis, its research and simulation materials, and the academic print comparison. The source checkout was `viv89/GRINS-Tool` on `data_analysis`, at `a129009` plus the local analysis work. The integration is based on `BETALAB-team/ita-domes` `UserArchetypes` at `1b5972b`.

## Results

- [2013 print comparison](output/pdf/Italian_archetypes_2013_print_comparison.pdf) and [2021 print comparison](output/pdf/Italian_archetypes_2021_print_comparison.pdf) are matching six-page A4 editions. Page 3 contains each probability matrix equation. Page 6 contains the group assignment matrices.
- [Revised 2021 analysis](output/article_method_2021_revision_20261104/README.txt) uses the recovered 125-window geometry, the 4 × 6 quadratic probability matrix and the compact four heating / five appliance electricity groups. Its [Word and PDF reports](output/article_method_2021_revision_20261104/06_report), [figures](output/article_method_2021_revision_20261104/04_figures), [probability results](output/article_method_2021_revision_20261104/02_probability_candidates) and [energy results](output/article_method_2021_revision_20261104/03_energy_candidates) are included.
- [Published 2013 benchmarks](output/temporal_extension_2013_2021_20261104/00_baseline/published_benchmarks.csv), [published Appendix A](output/claim_checked_article/00_context/archetype_extension/published_appendix_A_group_lookup.csv), and the [2021 Appendix A](output/article_method_2021_revision_20261104/03_energy_candidates/appendix_A_assignment_2021_wide_flat.csv) are available as tables.
- [Initial reconstruction](output/article_method_2021_20261104/README.txt), [temporal comparison](output/temporal_extension_2013_2021_20261104/STATUS.md), and [corrected manuscript workflow](REVISED_PAPERS.md) document the earlier analyses.
- [Manuscript and evidence](output/claim_checked_article/ARTIFACT_INDEX.md) include literature, methods, EUReCA simulation code and summaries, figures, reviews and compiled drafts. [Energy Studio](energy_studio/README.md) includes the simulation application source and its validation notes.

## Use the revised probability matrix

From this directory, run the dependency-free predictor on the saved full-precision coefficients.

```console
python output/article_method_2021_revision_20261104/predict_probabilities_2021.py --age 60 --area 80
```

Age is in years at the 2021 survey and area is in square metres. The checked domain is 15–135 years and 40–136 m².

## Rebuild with local inputs

Run analysis commands from this directory so their relative paths resolve here. [WORKFLOW.md](WORKFLOW.md) covers the general workflows. The two 2021 analysis READMEs specify the order and the additional inputs for their respective runs. Dependency files, source code and existing tests are included.

The 2013 baseline input filenames used by the scripts can be sourced locally from the existing upstream `../istat_data/UserClusters/istat_data/` tree. The raw 2021 release belongs at the configured path under `istat/Microdati21/`. The baseline config also expects the 2013 modeled energy table. Place authorized copies locally before running data-dependent stages. The published-figure print builder also requires a local copy of the source article, the compact household outcome file and the fonts specified in its source.

The transfer includes project code, prepared reports, aggregate results, and selected simulation inputs, summaries and verification records. Raw survey releases, household-level exports, bulk simulation run files, bundled runtimes, and duplicate archives were excluded. Original delivery manifests and reports retain their historical provenance, so they may refer to omitted local inputs, original paths, or full run directories. [TRANSFER_MANIFEST.tsv](TRANSFER_MANIFEST.tsv) records the files copied into this checkout.
