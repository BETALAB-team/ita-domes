# Cross-wave comparability — 2026-11-04

Published 2013 results stay fixed. This is a conditional temporal extension; reconstructed and revised model outputs must never replace the published reference.

|Concept|2013|2021|Restriction|
|---|---|---|---|
|Identifiers|nprog|ID_RECORD|Wave-local; repeated cross-section, never panel linkage.|
|Weight|coef_red|PESO|Official final household expansion weights. Separate weighted extension from published unweighted 2013 reference.|
|Region|reg 1–20|REGIO 10,20,30,41,42,50…200|10→1,20→2,30→3,41/42→4,50→5,…200→20; 41 Bolzano,42 Trento.|
|Members|q_1_1_sq1; 12 age/education slots|ncomp_mfr; 5 slots,6=6+|Common domain size1–5; never treat padded slots as observed; no claim for all households.|
|Age|cletad_1…12:0–14,15–24,…65–74,75+|eta_mfr1…5:0–5,6–10,11–14,15–19,20–29,…70–79,80+|Non-nested intervals. Declare midpoint/draw and open-age cap assumptions; fixed 2013 definition and sensitivity.|
|Education|titstu_1…12; more detailed categories|istr_mfr1…5:1 none/primary,3 lower secondary,4 upper secondary,6 tertiary|Common coarse groups only; combined none/primary and2–3/4–5yr diplomas lose detail. Do not change published baseline to match new proxy.|
|Heating hours|q_3_10 frequency;q_3_11A/B/C|v3_7 frequency;v3_8a/b/c|2013 typical winter day;2021 cold months2020. Common frequency1/2; sum three0–8h blocks only when all present; hours not annual kWh/runtime observation.|
|Cooling hours|q_5_7;q_5_8A/B/C|v5_6;v5_7a/b/c|2013 typical summer;2021 hot months2020. Preserve routing,presence,frequency eligibility and missingness; not all-household unconditional means.|
|Dwelling|q_2_1,q_2_4_ric,q_2_7_class|v1_1,v1_2_new,v1_7_mfr|Type,construction and area bands differ; collapse only to semantic intersections; construction unknown not a valid recent year.|
|Insulation|q_3_17 subjective rating|No equivalent rating|v3_13/v3_14gh_new are recent5yr improvement indicators, not insulation quality. No pooled rating trend or causal savings.|
|Energy|results_grins.csv appliances_el_kWh model|2020 expenditures and ownership/use; separate kWh model possible|Euros not kWh; neither modeled appliance kWh nor heating hours is metered total energy. Price/pandemic/weather/end-use differences prevent attribution.|
|Modes/target|CATI,resident families main dwelling|CAWI/CATI,resident families main dwelling|Mode and nonresponse may change; no full design variance from final weights alone.|
|Later wave|2013|2021 local release|2024 genuine survey exists;9Oct2025 release;research microdata page lists2013/2021 only. No2024 local energy microdata identified.|

Evidence: raw_release_checks.json; official tracciato files; extracted questionnaires (section3.10/3.11 versus3.7/3.8;5.7/5.8 versus5.6/5.7); codebook classification var6,10,16,31; legenda workbook extracts; istat_harmonizer.py; audit_replicability.py; revised_analysis.py. Exact baseline assignment and transformation audit is owned by the frozen-baseline packet.


**Critical coverage restriction (fresh raw audit):**2021education is structurally blank for all released members aged0–10. A strict all-member complete-schooling profile excludes every household containing an under11child. This is selection,not population prevalence;regional weights cannot repair zero inclusion. See structural_schooling_coverage.json and02_analysis/coverage_under15_2021.csv.
