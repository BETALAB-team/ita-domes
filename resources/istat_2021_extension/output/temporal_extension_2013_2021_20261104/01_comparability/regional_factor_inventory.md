# Regional factor inventory

Use official coef_red/PESO under the user-authorized fallback. No qualified custom regional correction table was identified. These coefficients expand households; they do not calibrate physical energy.

## Official household expansion weights

- **path**: ['C:\\Works\\GRINS\\GRINS-Tool\\resources\\istat_data\\istat\\ECH_2013_IT\\MICRODATI\\ECH_Microdati_Anno_2013.txt', 'C:\\Works\\GRINS\\GRINS-Tool\\resources\\istat_data\\istat\\Microdati21\\500Tj00000M4IEcIAN-1_00250525-1\\MICRODATI\\ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt', 'C:/Works/GRINS/GRINS-Tool/funcs/auxiliary_functions.py:85']
- **schema**: nprog,coef_red (2013);ID_RECORD,PESO (2021)
- **source_year**: 2013/2021
- **purpose**: Population household summaries
- **equation**: mean=sum(w*y)/sum(w); share=sum(w*I)/sum(w); regional means restrict sums to region
- **applicability**: Use supplied final positive household weights in extension. Published baseline remains unweighted.
- **audit**: Both releases unique IDs and all positive weights, see raw_release_checks.json. Official tracciato2013 row385;2021 row8. Not physical calibration.

## Compositional standardization

- **path**: No pre-existing qualified factor table located
- **schema**: Proposed target cell proportions p_ref(c),observed p_t(c)
- **source_year**: New extension only
- **purpose**: Hold macroregion/demographic composition fixed
- **equation**: standardized mean=sum_c p_ref(c)*mean_t(c); optional weight multiplier p_ref(c)/p_t(c)
- **applicability**: Only explicit supported cells with documented target/overlap. Not an official ISTAT supplied coefficient.
- **audit**: Separate from official weights; no new baseline fit or inferred existing factor.

## Archived regional centroid summaries

- **path**: ['C:\\Works\\GRINS\\GRINS-Tool\\resources\\istat_data\\replicability_outputs\\regional_centroids_audit.csv', 'C:\\Works\\GRINS\\GRINS-Tool\\resources\\istat_data\\revised_paper_outputs\\2013\\regional_centroids.csv', 'C:\\Works\\GRINS\\GRINS-Tool\\resources\\istat_data\\revised_paper_outputs\\2021\\regional_centroids.csv']
- **schema**: audit:reg,archetype,pc1_scaled,pc2_scaled,education_scaled; revised:region,archetype,n,centroid columns
- **source_year**: 2013/2021 derived outputs
- **purpose**: Regional cluster-location diagnostics
- **equation**: Regional centroid=within-cluster feature mean; alignment to reference by matching
- **applicability**: Descriptive only; not a row weight, prevalence ratio or energy calibration multiplier
- **audit**: Paperp6Table2 regional MANOVA and p7 exclusion of weights; centroid similarity does not establish prevalence equivalence.

## Physical energy calibration multipliers

- **path**: No qualified region-by-carrier/end-use/year multiplier table found in bounded search
- **schema**: Would require region,year,carrier,end_use,benchmark/model ratio,source,units
- **source_year**: Not established
- **purpose**: Align model energy to independent compatible benchmark
- **equation**: E_adjusted=m_region,year,enduse*E_model
- **applicability**: Not used. Cannot substitute household weights or centroid coordinates.
- **audit**: Model appliance coefficients/reference efficiencies and unit conversions exist, but are not verified regional energy calibration tables.

Search bounds: repository resources/funcs filename queries; Python/config content queries; targeted legacy functions,source paper,old archetype audits and regional output schemas. Giant manifests,binaries,virtualenvs and unrelated simulation-run snapshots were excluded from broad grep. No claim of exhaustive historical-object or opaque-archive inspection.


**Critical coverage restriction (fresh raw audit):**2021education is structurally blank for all released members aged0–10. A strict all-member complete-schooling profile excludes every household containing an under11child. This is selection,not population prevalence;regional weights cannot repair zero inclusion. See structural_schooling_coverage.json and02_analysis/coverage_under15_2021.csv.
