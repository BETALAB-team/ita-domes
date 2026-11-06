# Frozen 2013 published baseline

The authoritative numerical baseline is Khajedehi, Vivian and Zarrella (2025), *Identification of Italian user archetypes for building energy simulations based on a national survey*, Energy & Buildings 344,115924, DOI10.1016/j.enbuild.2025.115924. The full15-page primary text was read. Tables4 and6 were additionally checked against rendered pages8 and10. Published values are transcribed in published_benchmarks.csv/json and must never be replaced with recalculated values.

The four clusters have household shares40.1,26.9,22.9,10.1% and person shares29.9,40.2,21.3,8.6% (p7Fig5). Table3 means and SD are included as separate benchmark rows. The paper explicitly treats each surveyed family equally, excluding national expansion weights (p7§4.2). These are unweighted sample reference results; weighted estimates are separate extension estimands.

The lowercase clusters.csv is a supplementary archived assignment, with17,855 unique household IDs and segment0–3 mapped to archetype1–4. It agrees with39/40 one-decimal Table3/Fig5 cells; A4 maximum-age mean remains72.2 in the paper versus archive72.0607, rounding72.1. Archival correspondence does not establish exact identity to published remote UserArchetypes branch. The17,855-record domain is frozen;2,145 of20,000 released2013 records have no archived label. Join by ID, never row position.

Do not substitute Clustered.csv (five groups), baseline_outputs/household_archetypes.csv (later reconstruction differs318 labels), revised_paper_outputs/2013/households.csv (different19,445-record weighted method), or a new fit. No2013 memberships, PCA or cluster model were refit here.

## Published energy results and definitions

P5§3.3 defines heating as declared daily heating-system on-hours, not measured thermal energy or household presence. Appliance electricity is calculated using a survey-based model, not metered whole-house electricity. P10Table6 reports H1–H7 means(SD)4.67(3.54),5.25(3.77),6.91(4.43),7.53(4.41),8.53(4.00),9.14(4.17),9.67(4.28)h/day; E1–E5 means(SD)1801(1502),2023(1631),2320(1502),2646(1844),2895(1853)kWh/year. AppendixA p14 provides80 combinations, transcribed in the source packet's published_appendix_A_group_lookup.csv. These groups combine archetype, dwelling and macro-location; they are not one energy value per demographic archetype.

Printed ambiguities remain unresolved: six electricity groups in prose/conclusion versus five in Table6/Fig8/AppendixA; duplicated H6/H7 examples; p9 prose statistics differ from Table5; A4 age-squared coefficient0.152 in p7Eq3 versus0.02 in p8Table4. Both coefficient versions are stored separately, including the distinct printed precision. Do not normalize these polynomials into a simplex and call the result the printed model. Eq3 row order is intercept,age/100,area/100,age*area/10000,age²/10000,area²/10000. Table4 fit statistics are also transcribed.

## Regional findings and correction factors

P3§2 describes ISTAT deterministic/probabilistic response correction and expansion/calibration weights intended to align demographic and regional distributions; it does not provide a numeric regional correction-factor table. P4§3.1 explains regional MANOVA. P6§4.1 reports region centroids approximately align, especially for larger regional samples, and lower regional CHI than pooled CHI. Table2 prints Wilks lambda0.8105,F0.2226,p1; Roy greatest root0.1687,F0.5329,p0.9357. The authors interpret these as no significant regional effect on demographic cluster centroids. This does not establish no regional differences in energy or archetype shares. P9Table5 reports significant location effects for heating and electricity and combined dwelling/location groups. Paper-based inference supports common demographic archetypes with location-dependent behavior, not region-specific demographic recentering.

The paper does not publish operational regional centroid correction factors. Any2021 regional behavioral ratio, population calibration, poststratification or centroid shift must be labelled a newly estimated extension and defined explicitly. Official expansion weights are a separate concept from geometric centroid correction.

## Integrity

baseline_freeze.json holds SHA256 and byte size for the PDF, archived assignments/features, original source code, relevant input and candidate replacement artifacts. The same manifest includes after-audit hashes; every listed file remained unchanged. verify_integrity.py can recheck without modifying sources. freeze_baseline.py is a read-only source audit whose sole writes are in this00_baseline directory. It does not import or run project pipelines.
