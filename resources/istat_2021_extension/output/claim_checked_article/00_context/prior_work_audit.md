# Prior-work audit

Owner:context_evidence.20 October2026. This audit uses existing outputs as leads and checks a bounded sample directly. It does not endorse previous scientific narratives or rerun complete pipelines.

## What was rechecked

`prior_number_recheck.json` independently recomputes weighted summaries from saved household tables using standard-libraryCSV arithmetic. SHA256 of2013/2021households.csv,2013/2021results.json,and appliance_audit/electricity_2021.csv match the existing revised manifest. The raw2013/2021 release paths exist;their hashes are recorded in `source_inventory.json`.

| Quantity | Saved/recomputed result | Meaning/provenance |
|---|---:|---|
|2013 demographic eligible records|19,445|`revised_paper_outputs/2013/households.csv` row count;from revised_analysis.py::features_from_codes common1–5member,age/education domain.|
|2021 demographic eligible records|52,632|Same specification;not54,069all-release or53,520earlier common-domain sample.|
|2013/2021 weighted retained population|25,266,108 /23,608,115.215|Sum of retained released weights;analytic-domain totals,not independently verified complete population counts.|
|2013/2021 Kish effective n|6,925.619 /23,054.242|Weight concentration diagnostic;not full survey-design-adjusted effective sample sizes.|
|2013/2021 heating-hour means|7.87839(n18,105) /8.50764(n48,705)h/day|Conditional declared-use summaries;`heating=heating_hours.where(heating_comparable)`. Different periods/domains and routing need codebook verification;not thermal energy or measured energy.|
|2013/2021 eligible appliance means|2,175.0767 /1,936.7522kWh/household/year|Modelled estimates in revised-household tables;not meter data. Population and component boundaries must accompany comparisons.|
|2021 cooling ownership|24,073records;48.5734%weighted|Within52,632eligible records;simple sample fraction would answer a different weighted estimand.|
|2021 cooling-use mean|5.96701h/day;n17,493|Conditional equipment/frequency routing,not average for all households.|
|2021 insulation/window improvement|9.96311%weighted among52,200known records|Recent improvement response;not the2013subjective insulation-quality question.|
|2021 outside reference transformation range|3,189records|Diagnostic flag,not automatic model invalidity or household transitions.|

Selected narrative values in `literature_revised_draft.md`(52,632eligible;48.6%ownership;5.97cooling hours;10.0%improvements) match these stored summaries. This checks table-to-prose correspondence only;questionnaire definitions/sampling variance and causal interpretation remain separate scientific questions.

The full-release appliance report states mean1,949.1860kWh,lower1,527.1184andupper2,705.6711,n54,069. It differs from1,936.7522because the latter uses the52,632eligible domain. This is a denominator distinction,not by itself a contradiction. `model_electricity_2021.py` lines85–115 reads saved components and computes electricity from fixed2013parameters plus missing-detail assumptions;the saved report's2013component error is1.81899e−12kWh for20,000rows. This is an algebraic replication check,not empirical validation. The underlying model functions were not rerun in this bounded audit.

## Conflicting generations and why they must remain separate

1. `WORKFLOW.md`/legacy outputs use17,855eligible2013 and53,520eligible2021 for archetypes;an earlier common-domain audit uses19,688/53,520;revised outputs use19,445/52,632. Algorithms,education proxies,member restrictions and weighting changed. Do not splice numbers from these specifications.
2. Legacy replica produced8heating/4electricityTukeygroups and25/28published ANOVA decisions;revised split-sample outputs produce7/5groups in2013and8/8in2021. Significance-driven merging and tuning-selected group construction are distinct procedures. None defines a conference model validation result.
3. The older audit describes no directly comparable2021modelled appliance outcome;later appliance code adds a reference-parameter reconstruction. That new outcome is still an estimate with coarser inputs. It does not supply observed2021kWh or an updated appliance-efficiency database.
4. Revised code fits weighted PCA/k-means only on2013training households,freezes the demographic transform for2021,but probability/energy models are separately fit within waves. A frozen demographic representation is not a completed frozen2013versus refitted2021prediction experiment. The prose drafts explicitly mark transfer,decomposition,common-information coarsening and regional retrofit runs pending.
5. The published2013household-archetype paper is a distinct source from the CLIMA stock-model chapter. Its methods/results cannot be inferred from repository correction narratives. Its independent primary-source verification belongs to the literature worker.

## Reuse decision

Reuse raw releases,codebooks and demonstrably relevant physical/schedule functions with frozen hashes. Reuse stored summaries only with their exact population,weights,units and estimated/observed labels. Preserve all historical artifacts. Do not carry the legacy correction narrative into this article,or claim completed temporal transfer,regional savings,Padova validation or climate accuracy from existing tables. The selected direct schedule-reconstruction experiment avoids making the much larger temporal inference chain a prerequisite;cross-wave work remains a documented candidate extension.

No scientific tests were executed in this worker's audit beyond read-only table arithmetic and hash checks. A prior successful test/exit or matching manifest proves neither scientific validity nor independent validation.
