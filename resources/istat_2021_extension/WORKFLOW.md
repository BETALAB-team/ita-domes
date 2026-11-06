# Reproducible ISTAT archetype and energy workflow

This workflow regenerates the four household archetypes, the windowed
dwelling-to-archetype probabilities, and the paper-aligned energy ANOVA/Tukey
analysis directly from the raw ISTAT and GRINS energy inputs.

## Run everything

Use Python 3.10 or newer. Install the scientific dependencies once:

```powershell
python -m pip install -r requirements-workflow.txt
```

Then regenerate and verify every result with one command:

```powershell
python run_workflow.py
```

## Run the 2013 and 2021 ISTAT waves

The multi-year entry point reads the raw tab-delimited 2013 release and the raw
caret-delimited 2021 release under `istat/`, harmonizes only the fields used by
the analysis, and runs the comparable archetype and PFE stages for both years:

```powershell
python run_multiyear_workflow.py
```

Outputs are isolated under `multiyear_outputs/2013/` and
`multiyear_outputs/2021/`. Each year has its own harmonized input, validation
report, SHA-256 manifest, baseline/PFE tables, and SVG/PNG figures. The combined
archetype table is `multiyear_outputs/archetype_summary_2013_2021.csv`.

The 2013 run also executes the energy ANOVA/Tukey stage and regenerates all ten
paper-replica figures. Seven figures are directly comparable in 2021: the
workflow flowchart, PCA variance, clustering elbow, regional centroids,
population shares, dwelling probability profiles, and probability fit check.
The three energy-specific figures are intentionally limited to 2013 because
the 2021 microdata provide energy expenditure rather than the joined GRINS kWh
response and do not contain the 2013 thermal-insulation rating.

The year-specific category ranges are taken from the supplied ISTAT legends.
The 2021 release contains member-level age and education fields for five people;
households coded as six or more members are therefore excluded with the explicit
reason `member_count_exceeds_available_columns` instead of fabricating a sixth
member.

Paths, random seeds, the PFE support threshold, and the ANOVA alpha are all in
`workflow_config.json`. Relative paths are resolved from the config file, so
the command does not depend on the current working directory.

### Different source column names

`column_mappings` accepts source names mapped to the canonical names used by
the workflow. The source files are never renamed or overwritten:

```json
"column_mappings": {
  "raw_istat": {
    "household_key": "id",
    "household_members": "q_1_1_sq1",
    "building_type": "q_2_1",
    "building_age_band": "q_2_4_ric",
    "floor_area_band": "q_2_7_class",
    "region_code": "reg"
  },
  "energy_results": {
    "household_key": "id",
    "annual_appliance_kwh": "appliances_el_kWh"
  }
}
```

The direction is always `source column -> canonical column`. Unmapped fields
are expected to already use their canonical names. The loader rejects missing
mapped columns, duplicate targets, unknown targets, and files containing both
a canonical column and its configured alias. Mappings rename fields only;
category recoding, unit conversion, or person-to-household reshaping still
requires a preprocessing adapter.

To check existing outputs without regenerating them:

```powershell
python run_workflow.py --verify-only
```

The automated regression checks can also be run independently:

```powershell
python -m unittest discover -s tests -v
```

## Output contract

- `baseline_outputs/`: one household-archetype table, archetype statistics,
  preprocessing exclusions, model metadata, and paper-tolerance validation.
- `pfe_outputs/`: the 63 age/area windows, sample and archetype counts, four
  probabilities, unsupported-window audit, model metadata, and validation.
- `energy_outputs/`: the ID-joined analysis dataset, exclusions, two 14-term
  Type-II ANOVA tables, deterministic Tukey mappings and merge history,
  composite and separate-factor summaries, model metadata, and validation.
- `workflow_outputs/`: the cross-stage validation report and SHA-256 manifest.
  It also contains `istat_archetype_energy_report.docx`, the report validation
  record, and the regenerated SVG/PNG figure sources under `report_figures/`.

The Word report is generated automatically on every full workflow run. It
contains captioned methods/results tables, the complete 63-window probability
appendix, and ten paper-matched figures: the study flow, two-panel PCA variance,
clustering elbow, regional centroid network, population-share bars, four-panel
dwelling probability curves, probability fit check, paired energy boxplots,
standards comparison, and the appendix assignment matrix. Every plot is stored
as SVG and embedded natively in Word with a matching high-resolution PNG
fallback. Interpretation is intentionally omitted.

The regional centroid replica requires the raw census region field with the
canonical name `reg`. If a newer census uses a different name, map that source
field to `reg` in `column_mappings.raw_istat`.

Every CSV has an explicit checked schema. Household joins use `id` with
one-to-one validation; positional joins and legacy intermediate files are not
used. The old exploratory scripts remain untouched for provenance, but they
are not part of the executable workflow.

## Methodology and paper comparison

Daily heating use is `q_3_11A + q_3_11B + q_3_11C`. Annual electricity is
`results_grins.csv:appliances_el_kWh`. Building age and area use the same
seeded class expansion as the probability stage. Rows with insulation code 0
are explicitly recorded in `energy_exclusions.csv` instead of being silently
coerced.

The ANOVA uses the seven main effects and seven interactions in paper Table 5.
The Tukey-Kramer procedure repeatedly merges the least-separated non-significant
pair, recalculating the pooled error after every merge, and stops only when all
remaining pairs are significant.

Paper agreement is reported separately from computational validity. With the
restored deterministic archetypes, 25 of 28 ANOVA significance decisions match
Table 5; the regenerated Tukey solution contains eight heating and four
electricity groups. The article reports seven heating groups and is internally
inconsistent for electricity: the text says six, while Table 6 lists E1-E5.
Those differences remain visible in the validation report and are never
silently forced.
