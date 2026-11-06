# Fixed archived assignment transfer

## Evidence status

The published estimator's original fitted centroid object, exact min/max, random age-class realization and software/default record are unavailable. Rounded Table3 means cannot recover it. Published equations use exact thirds, whereas the archived saved generation coordinate uses0.33 and0.66. Do not label an archive-derived transfer as exact reproduction of the published estimator.

An operational frozen archive transfer is recoverable without fitting: use the saved normal_l,normal_g,normal_s columns of lowercase clusters.csv; compute their means within the unchanged saved segment labels. Those are fixed-label arithmetic summaries, not a k-means optimization. Replay nearest-center assignment on these saved coordinates yields zero disagreement for all17,855 archived labels. archive_transfer_parameters.json stores the numerical centers, min/max and residual checks. No archived membership is changed by this diagnostic or by any new-year assignment.

## Transformation and prediction

For a household with harmonized minimum age a,maximum age b,member count n and education index e:

1. L=0.25a+0.75b−10n; G=0.33a+0.66b+10n.
2. zL=(L+40.5)/114.5; zG=(G−30.790000000000003)/91.05; zS=e.
3. Assign nearest fixed center using unweighted squared Euclidean distance across zL,zG,zS. Resolve exact ties by lowest semantic archetype ID. No refitting, normalization to new-year min/max, regional recentering or forcing shares.

Centers in feature order zL,zG,zS:

| Archetype | zL | zG | zS |
|---|---:|---:|---:|
|1|0.8604156366383043|0.6835784416884528|0.49237465163054994|
|2|0.4554191312035611|0.5549410411362908|0.7446408872049364|
|3|0.6681971915403716|0.5349778725328681|0.5158518129474011|
|4|0.7651687045927483|0.6145189974865297|0.7694965218432915|

The archive uses a sign reversal relative to paper PC1; a consistent sign reversal of observations and centers preserves Euclidean distances. The0.33/0.66 approximation is independently material and remains explicit. Transforming only observations with exact thirds while using these archived centers is inconsistent.

Do not clip out-of-range coordinates for primary assignment: retain frozen geometry and flag ranges. Record nearest and second-nearest distance/margin and missing-feature exclusions. Outside2013 support is extrapolation, not observed evidence of a new archetype. Age and education harmonization across surveys remains a modeling assumption independent of the replay success. The saved2013 age realization must remain unchanged. New-year coarsened ages require an explicit fixed rule/sensitivity; education categories must be crosswalked with ambiguous consolidations documented. The legacy education recoding has an in-place replacement collision (code3 effectively12years), and must not be described as a validated normative education index.

## Comparison contract

Use the printed paper tables as unchanged2013 numerical reference. For record-level comparisons or within-archive diagnostics, use the archived labels and clearly label derived2013 statistics as supplementary archive calculations, never overwrite printed means or shares. New2021 labels are archive-calibrated out-of-sample assignments, not ground truth. Weighted2021 results and any harmonized2013 sensitivity have distinct estimands; no reinterpretation of the paper as weighted is permitted. Report differences as conditional on the harmonization and assignment assumptions.
