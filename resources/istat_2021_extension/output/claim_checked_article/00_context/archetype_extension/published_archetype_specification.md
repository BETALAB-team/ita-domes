# Published archetype specification and implementable assignment route

Date: 20 October 2026. Owner: `context_evidence`. Requested model: `gpt-6.1-sol`; confirmed backend model metadata is unknown. This is an evidence/specification packet, not manuscript prose or a new simulation result. All existing research and the previous delivery are preserved.

## Decision and source access

Use the archived **lowercase `clusters.csv`** as the primary article-consistent four-group membership input. Join its `identi` to official survey `nprog` or compact survey `id`, and map saved `segment`0/1/2/3 to archetype1/2/3/4. Do not fit new clusters to stand in for the published archetypes. The ready local mapping is `archived_four_archetype_assignments.csv`; the explicit schema/provenance is `assignment_mapping_schema.json`.

This choice implements the published four demographic groups through an archived assignment that has strong aggregate correspondence with the paper. **Exact membership identity with the published UserArchetypes branch is not established.** Matching rounded summaries is not proof of identical per-household labels or validation. Preserve this qualification in the study.

**ARCH2025-FULL:** Khajedehi, M.H.; Vivian, J.; Zarrella, A. (2025), *Identification of Italian user archetypes for building energy simulations based on a national survey*, Energy and Buildings344,115924. DOI10.1016/j.enbuild.2025.115924. Available online27May2025. Full local publisher PDF: `C:/Users/khajmoh18975/Downloads/1-s2.0-S0378778825006541-main.pdf`; SHA256 `8995443d5c1b92313730c75c99512f94122aac4f8e0a79b4fe00bca5b269aae2`. All15pages were read, and all15 rendered page images inspected. The paper states CC BY4.0 onp1. The institutional repository identifies a publisher version and Creative Commons access. The duplicate Downloads PDF has a different byte hash; no identical-byte claim is made.

The paper's p14 data-availability statement links `https://github.com/BETALAB-team/ita-domes/tree/UserArchetypes`. Web access produced cache misses/inaccessibility; institutional PDF open returned403; Git remote inspection failed through the locally configured127.0.0.1:9proxy. These access failures do not establish absence of the branch. The local full PDF supplies the complete primary content. Local GRINS origin is `viv89/GRINS-Tool`; exact identity to that published branch remains unverified. Metadata, access attempts and file/Git hashes are in `primary_sources_and_local_inventory.json`.

## Demographic definitions and assignment

| Item | Primary locator | Verified specification and boundary |
|---|---|---|
| Four inputs | pp3–4,§3.1/Table1 | Minimum age, maximum age, household-member count, and household-average education index. Building/equipment fields are excluded from demographic clustering. Individual ages are released in classes; derived ages are not exact observed ages. |
| Education | p4,Table1 | Described as mean individual years-studied/potential-years-by-age, bounded0–1. The paper cites UNDP; exact release recoding, missing handling and reproducible preprocessing parameters are not completely specified in the article. Do not certify the local legacy index as a validated UNDP measure. |
| Components | pp5–6,§4.1/Eqs1–2 | PC1=−0.25age_min−0.75age_max+10n_members. PC2=(1/3)age_min+(2/3)age_max+10n_members. Education retained separately. |
| Clustering | pp3–4,§3.1; p6 | Min–max scale PC1,PC2,education to0–1; Euclidean k-means; WCSS/CHI/elbow discussion; K=4. Article does not supply exact fitted centroid coordinates, fitted min/max, random age realization or complete software/default/seed record. |
| Household membership | p7,§4.2 | Nearest centroid defines membership. Verbal examples are interpretations, **not** age/education thresholds or deterministic household rules. No new household can be exactly assigned from the prose alone. |
| Weighting | p7,§4.2 final paragraph | Analysis explicitly excludes the national institute's weighting factor and treats surveyed families equally. This qualifies the earlier p3 description of released expansion weights. Use unweighted published-reference summaries; label any new weighted estimand separately. |
| Geography | pp6–7,§4.1–4.2 | Authors report regional centroid similarity. This is not proof of city representativeness, causal invariance or empirical energy accuracy. |

| Published archetype | Source-located interpretation (p7) | Table3 means(SD): minimum age;maximum age;member count;education |
|---|---|---|
|1|Few older members,mid-level education;retired couple or single elderly person are examples|70.2(11.2);78.3(5.0);1.8(0.7);0.5(0.1)|
|2|More members,typically a younger and a middle-age member;parents/children or young cohabitants are examples|21.2(7.6);57.2(8.6);3.7(0.9);0.7(0.1)|
|3|Few middle-age members,medium education;working couple or parent with working-age child are examples|46.0(12.8);63.0(8.0);2.3(0.8);0.5(0.1)|
|4|Few middle/older-age members,higher education;similar to3 with higher education/closer retirement is an example|55.6(17.1);72.2(9.0);2.1(0.8);0.8(0.1)|

These means and SDs are **not ranges**. The article does not report hard demographic cutoffs or complete per-cluster min/max ranges; do not invent them. Figure5 reports household shares40.1/26.9/22.9/10.1% and person shares29.9/40.2/21.3/8.6%, unweighted.

## Direct archived membership audit

Lowercase `clusters.csv` contains17,855 unique `identi` values and four numeric labels. All17,855 join both the20,000-row official raw release and the compact input one-to-one;2,145 release records lie outside this archived assignment domain and must remain unassigned/excluded unless a separately specified method is adopted.

| Saved segment | Semantic archetype | Archived n | Household share | Person share |
|---|---:|---:|---:|---:|
|0|1|7,163|40.1176%|29.8937%|
|1|2|4,801|26.8888%|40.2103%|
|2|3|4,080|22.8507%|21.2538%|
|3|4|1,811|10.1428%|8.6422%|

Direct arithmetic reproduces39of40 one-decimal Table3/Figure5 cells. The exception is archetype4maximum-age mean: archive72.0607→72.1, paper72.2. Preserve it; do not force agreement. Saved features contain a random age-class realization, so rerandomizing ages is not the same input. `local_assignment_audit.json` records all40 checks, source summaries, confusion matrices and join counts.

The file is tracked in local Git; its last logged change is `e7094f4`,3October2024, “clustering, probablity regression”. At current HEAD `a129009e177da391792b57c8af5055b18ce64374`, the working file and Git blob have identical text after CRLF→LF normalization. Their byte hashes differ because of line endings: worktree `0df1bcc11c5b89f4a485affb6e0d661c417ef13b7879d10b8fa8550d29cded90`; Git blob SHA256 `5fe10886fc70792cb0a2212fc8249283654e4ace9f172d061f8428b036efadd0`. This establishes local archival provenance, not equality to the remote published branch.

Candidate files must remain distinct:

- `Clustered.csv` has **five** labels, with counts6,953/4,172/2,702/1,451/2,577. It is unsuitable as the published four-group mapping.
- `01_PCA_Exploratory.py` fits four k-means groups. It uses saved `family_trees.csv`, min–max features, PC1 sign reversal and0.66/0.33 approximations in its second component. Article equations use exact thirds. Saved membership avoids silently recomputing under changed formulas/defaults.
- Current `03_PCA_Clustering.py` fits20groups. `00_Clustering_Exploratory.py` uses another five-group exploratory input set. Neither is the four-group membership authority.
- `baseline_outputs/household_archetypes.csv` is a **later deterministic reconstruction**, produced by `archetype_pipeline.py` with frozen seed/tolerance rules and legacy recoding. Its labels differ from lowercase archived labels(+1) for **318of17,855 households**, agreeing on17,537. Do not call it exact published assignment. It can be a separately labelled reconstruction sensitivity.
- `revised_paper_outputs/2013/households.csv` retains19,445 records under a different weighted/training-based specification. It is a later revised representation, not the original four-group mapping. No row mixing is permitted.

## Behaviour, dwelling linkage and internal gains

The paper distinguishes demographic archetypes from later energy-use groups. Onp5,§3.3 defines outcomes as **declared daily heating-system on-hours** and **annual appliance electricity calculated by a survey-based appliance model**. Neither is metered thermal demand. Heating on-hours are not household presence, exact burner runtime or hourly heat demand.

Pages5,7–8 describe probabilistic dwelling linkage from construction age/area with initial location/type/insulation variables. Final independent polynomial equations are not a unique demographic assignment rule and are not guaranteed a valid probability simplex without checking. The printed Eq3 cluster4age² coefficient0.152 disagrees with Table4's0.02. The direct ID-based archived route avoids synthetic dwelling assignment and this unresolved coefficient ambiguity.

Pages9–11/Table6/Fig8 and p14AppendixA define heating groups H1–H7 and electricity groups E1–E5 according to demographic group, dwelling type and macro-location. `published_appendix_A_group_lookup.csv` is a visual transcription of all80 combinations, with exact printed labels. It is a published lookup, not a new validated predictor. Heating means are4.67,5.25,6.91,7.53,8.53,9.14,9.67h/day; appliance means are1801,2023,2320,2646,2895kWh/year for listed groups. These are group summaries, **not** one heating schedule/appliance demand per demographic archetype independent of dwelling context.

Printed ambiguities remain visible: text/conclusions say six electricity groups although Table6/Fig8/AppendixA use E1–E5; H6/H7 Table6 examples are duplicated despite different means; p9 narrative statistics conflict with Table5 in places. Do not silently “repair” them or use Table6 means as measured or validated energy demand. AppendixA labels can be catalogued without adopting every numerical interpretation.

Pages10–12,§5.1 compares appliance electricity and standard assumptions for a120m²example. Page12's discussion speculates that a working household may have lower daytime occupancy; it does **not** supply a measured archetype-specific hourly presence curve. The article provides no complete archetype-specific8760 occupancy, internal-gain, ventilation or temperature schedule. Section6,p13 explicitly leaves the impact on individual-building simulations to future study, despite broader accuracy rhetoric in§5. The present extension can implement/test operational consequences, but cannot claim published evidence already validates that impact.

## EUReCA implementation route

1. Freeze this archived mapping and source hashes. Join exact survey records by ID; keep the17,855-record domain and original four classes. Select a declared common survey-response universe before comparing groups.
2. Use each selected record's actual three heating-hour responses and controls to construct schedules under the documented model mapping. Preserve block-hour, routing, temperature, season, clock and random-placement assumptions. Stratifying actual response-linked inputs by archetype implements the classes without inventing occupancy curves.
3. For household count/gains, distinguish released coarsened member count, modeled appliance kWh, assumed heat-to-zone fractions and assumed temporal shapes. The current `EndUse.py` uses one standard weekday/weekend occupancy shape scaled by member count and a standard appliance shape scaled by annual estimated components. This is a code assumption, not a measured demographic-archetype curve.
4. Hold physical dwelling and weather fixed for controlled archetype-input comparisons; retain the corrected v5 surface/weather/state path and its limitations. If varying heating only, hold gains common and name the estimand accordingly. If varying gains/member count too, use paired/ablation contrasts to separate the changes. No particular gain/occupancy conversion is justified solely by the demographic label.
5. Retain within-archetype response variability or systematically selected actual records; do not present one convenient household as the group's full distribution. Treat the existing single-record study as a technical pilot within archetype1, not implementation of all four classes.
6. Use the paper's unweighted reference for fidelity; a separate weighted analysis may use officially calibrated `coef_red` with its own population/domain statement. Official methodological note p7 describes expansion/calibration weights. Such weighting is a new estimand, not reproduction of published percentages.

Minimum evidence gate: four archived groups materially enter the simulated input design; selected records/weights/controls and exclusions are traceable; input differences are explicit; every run uses known corrected code and common physical assumptions; resulting claims remain conditional simulations without empirical validation. No annual runs or clustering refit were executed by this evidence worker.
