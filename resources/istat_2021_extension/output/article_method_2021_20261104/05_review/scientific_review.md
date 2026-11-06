# Independent scientific review of the 2021 article-method reconstruction

**Assessment.** This package contains a newly fitted 2021 demographic partition, an empirical dwelling-probability grid, a new polynomial matrix and two independently estimated energy-group partitions. The 2013 reference belongs to the original article. The new cluster and energy numbers can be compared across waves as labelled study outputs. Cross-wave code identity is not implied by matching names.

## Source, cohort and archetype checks

I read the native release, its classification tables for members, education, location, dwelling type, dwelling construction and area, the survey's questionnaire and the article's method materials, the original study's implementation and the three new analysis stages. The 54,069 native identifiers are unique. All 54,069 agree with the input and household assignment files. I independently joined and checked final survey weight, location, dwelling type, heating response and existing modeled appliance kWh. The published baseline is read-only. Fourteen protected source files matched their recorded hashes.

The household roster supports 1–5 members. The 549 top-coded records lack sufficiently detailed full rosters. Among the remaining records, schooling in both age bands spanning 0–10 is empirically blank. The new computation keeps the children in household size and age and uses only observed member schooling in the education mean. Specifically the instrument's F.10 asks about age ≥10 and relationship codes 1–5; age 10 cannot be isolated in the released 6–10 band. Thus it is correct to report that all released under-11 schooling is blank and that 7,737 under-11 households are **retained**. Avoid the broader assertion that no 10-year-olds were asked schooling by the instrument.

Of the 54,069 households, 52,617 (92.484% of full-wave household weight) enter the demographic fit. Another 903 size-eligible profiles fail the schooling requirements. I reconstructed the seeded age values, capped education index and normalized coordinates from raw fields. Nearest assignment to the new K4 centres, the minimum-distance one-to-one alignment to the frozen historical coordinate means, and the three PCA variance fractions agree. A1–A4 counts are 12,735, 17,296, 11,451 and 11,135. The new unweighted percentages are 24.20, 32.87, 21.76 and 21.16. The printed unweighted 2013 references remain 40.1, 26.9, 22.9 and 10.1. Geometry, schooling and household coverage have changed. Switching the lowest merged schooling category from the assumed five years to zero yields 15.93, 28.01, 26.39 and 29.67 percent, with ARI 0.471 versus the primary partition. The fitted point estimates alone cannot identify a national change in archetype prevalence.

## Dwelling probabilities and coefficients

The native building classifications support the 2021 interval mappings. Source code 9 is unknown year of construction. Only 46,712 eligible household records have both a known construction band and an eligible floor-area band. Each PFE dwelling's two draws and its demographic membership agree on independent reconstruction. The 150-year and 400-square-metre upper endpoints and 10-square-metre lower endpoint are explicit finite assumptions. The PFE stage pools all confirmed 2021 dwelling types and macroregions when conditioning on age and area.

Independent window counts and probabilities agree for all 63 nonempty grid centres and all four archetypes. The grid windows collectively capture 12,334 distinct household draws. Fifty-seven windows contain at least five sampled households. Twenty-one supported windows, at ages from 20 to 140 years and areas 40, 80 and 120 m², supply the primary regression. Their observed conditional shares sum to one. I independently ran statsmodels backwards OLS against these raw-count ratios using a six-term basis with pinned intercept. Each of the 24 entries below and each corresponding survey-weighted sensitivity coefficient agrees with the output.

| Row | Constant | Age/100 | Area/100 | Age×area/10,000 | Age²/10,000 | Area²/10,000 |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 0.0649592 | 0.5497158 | 0 | 0 | −0.2697043 | 0 |
| A2 | 0.3951931 | −0.5505812 | 0.1030504 | 0 | 0.3084934 | 0 |
| A3 | 0.4571235 | −0.0889840 | −0.2134687 | 0 | 0 | 0 |
| A4 | 0.1527772 | 0 | 0 | 0 | 0 | 0.0791919 |

The source article presents a six-term matrix, while its exploratory historical script contains narrower candidate formulas. The new estimator starts each row with all six candidates, uses a 0.05 elimination threshold and fits equal-weighted windows. This is an explicit adaptation of the paper's functional form. Separate OLS fits do not constrain probability sums. Across the 21 fitted-domain centres components are within [0,1], but row sums range from 0.98044 to 1.03517. All 63 sums differ from one beyond numerical tolerance, and 35 of the 42 extrapolation windows have at least one component outside [0,1]. Use empirical ratios for valid local proportions and disclose the limitations of using the raw polynomial as a sampling distribution. The four held-out window RMSEs are 0.07195, 0.08931, 0.09830 and 0.05112. The holdout validates the probability stage conditional on a full-wave demographic fit. I separately recomputed the split and these four scores. This does not validate the whole fitted chain on an independent survey.

## Heating, modeled electricity and Appendix A

The heating outcome is summed 0–8-hour blocks among respondents with frequency 1 or 2 during the cold months of 2020. The second outcome is an annual appliance model joining 2021 ownership/use to fixed reference assumptions. This audit verified the join and component-total source. It did not independently recalibrate the appliance parameters or compare them to a national meter dataset. The 2021 five-way dwelling code uses the apartment's floor, unlike the 2013 classification by the number of apartments in the building. The published insulation rating lacks a 2021 counterpart.

There are 80 starting cells per outcome, defined by four macroregions × five 2021 dwelling categories × four newly fitted archetypes. The smallest heating and electricity cells contain 217 and 228 households. I independently replayed each of the 71 H and 67 E merges from native observations, recomputing the pooled within-group error, unequal-size Tukey–Kramer q and the minimum-q candidate each time. Every merger and final rank-ordered cell assignment agrees with the saved history. The historic exploratory script reads the maximum from a rounded statsmodels summary. Full-precision smallest-q selection is deterministic but can resolve near or rounded ties differently. “Original-style iterative Tukey–Kramer” accurately describes this implementation.

| Measure | Heating | Modeled appliance electricity |
|---|---:|---:|
| Eligible households | 48,690 | 52,617 |
| Groups obtained from 80 starting cells | 9 | 13 |
| Group-mean range | 6.17–11.11 h/day | 1,621–2,843 kWh/year |
| Overall unweighted / survey-weighted mean | 8.551 / 8.506 h/day | 2,068.66 / 1,936.46 kWh/year |
| Largest final adjusted pairwise p-value | 0.001993 | 0.014253 |

I independently checked every group count, unweighted mean, SD and PESO-weighted mean. Both terminal sets have all pairwise adjusted p<0.05 under the within-wave procedure. Since the partitions were selected on the same observations, these p-values describe the stopping condition. The historical seven H/five E benchmark means in published Table 6 and the new nine H/thirteen E means are ordered within their respective waves. A shared suffix does not guarantee a shared population or an energy transition.

The 80-row long Appendix A and 20-row wide appendix each encode **160** assignments; every value matches the H/E lookup for its location, dwelling label and archetype. The actual 2021 heating plot shows one box for each H group and no fictitious insulation categories. The electricity plot stratifies each E group by actual 2021 cooling-use classification from native frequency and inventory. There are 29,948 absent/unused and 22,669 used records. I compared stratum counts for each E group and inspected the figure. For the final visual the metric tick labels E1–E13 should be short enough to avoid touching; category 3's display should include its semi-basement component as well as ground and raised floors.

On the age/area-supported ANOVA domains I independently recomputed Type-II sums of squares, degrees of freedom and F statistics with a separately specified formula. Heating n=43,569 and R²=0.110821, and electricity n=46,712 and R²=0.266429. Both designs have rank 48. The 2021 available-factor ANOVA omits the unavailable insulation main effect and its interactions. Its p-values are conditional model results rather than design-based estimates. More electricity groups in the 2021 sample does not by itself demonstrate additional physical modes of energy use.

## Reproducibility and disposition

The three check scripts beside this file use the isolated scientific runtime and do not run a historical estimator. The JSON result files record the independently reproduced counts and scores. The K4 verification reconstructed normalized features, PCA eigenspectrum, centroid replay and optimal alignment from the observations; it did not independently repeat every random k-means initialization or the four sensitivity refits. Energy grouping, its terminal comparisons and ANOVA were independently reproduced. No new subgroup counts were mandated. The scientific outputs support a report comparing the new wave directly with the printed historical results under the scope above.

