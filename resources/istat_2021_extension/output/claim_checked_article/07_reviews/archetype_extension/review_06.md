**Independent scientific review 06 — initial critique**

Date: 20 October 2026. Reviewer: new independent review06 agent. Requested model in dispatch: gpt-6-astra; backend identity was not independently exposed to this reviewer. This review follows the full rubric, rather than checking only previously reported issues.

**Recommendation: admit the v06 scientific claims to the bounded prose-development gate. No new critical, major, or minor scientific correction is required by the evidence inspected in this review.** This is a recommendation about the claim-stage evidence and the explicitly limited implementation experiment, not journal acceptance, empirical validation, confirmation of absolute novelty, or closure of the earlier review chain. The separate procedural/previous-issue closure pass has not been performed.

The study now contains an actual implementation of the recovered demographic groups: their training-set membership changes the probabilities assigned to four real survey-derived operating profiles. It also contains a substantive literature framework anchored in the original CLIMA chapter. It does not merely rename the original single-household sensitivity pilot. Its demonstrated contribution remains narrow: fidelity to detailed-input outputs from the same model for a small held-out cohort in one assumed physical dwelling.

**Frozen input and independence**

All paths below are relative to `C:/Works/GRINS/GRINS-Tool/resources/istat_data/output/claim_checked_article`, unless stated otherwise.

| Input | Independently calculated SHA-256 |
|---|---|
| `06_claim_drafts/article_claims_v06.md` — 106 claims | `0075415fec3e985bbd55a6cfc363fd94a7d73b9421ddc9699ff82a4ae79f5c7b` |
| `06_claim_drafts/claim_ledger_v06.json` | `32a76ba4e2e4ed5c1114a1e2a2be4d269579782d5fe48777e12d4f951018fdc8` |
| `06_claim_drafts/claim_evidence_ledger_v06.json` | `91227a4d60a5d6bbcf736ac0c7070f75b10cd7ead8e1eaf0a2ee160d70f46b66` |
| `02_evidence/archetype_extension/local_sources.json` | `703de24ecf972037ec5fa492f949f9af32810cbc7b709c63b0b06d9fcb537f94` |

I inspected all 106 exact claims and their scopes, assumptions, counterevidence and permitted wording, together with the machine ledgers. The exact claims and permitted-wording fields agree across the Markdown and both ledgers. I used the registry for locating sources and binding identities, not as proof of scientific correctness.

Before saving this critique I did not open previous reviewer reports or helpers, author responses, issue registers, pre-run audit reports/helpers, private author checks/builders/reasoning, final acceptance/closure artifacts, or manuscript prose. The required current validation matrix and claim text contain incidental references to prior reviews; those statements were not used as evidence. I did not solicit other agents' opinions. The dispatch's later registry-hash update conveyed source identity only.

I wrote and ran only the reviewer-owned helpers `reviewer_06_recalculate.py` and `reviewer_06_extra_checks.py`, saving `reviewer_06_calculations.json` and `reviewer_06_extra_checks.json` beside this report. These import numerical libraries, not the study's analyzer, profile bridge or thermal engine. They read existing outputs and reconstruct calculations independently. No annual simulation, clustering fit, environment modification or external-checkout edit was performed.

**Source correspondence, original CLIMA scope and literature**

I read the complete extracted original CLIMA chapter, PDF pages 2–11 / printed pages 1302–1311, including its full bibliography, and the complete 15-page 2025 archetype paper. I additionally viewed the archetype paper's rendered page 7, checking Table 3 and Figure 5, and CLIMA's rendered page 8, checking the regional GWh maps. Primary local paths are `00_context/conference_complete_text.txt`, `00_context/archetype_extension/primary_complete_text.txt`, and their page images. The source specification and assignment schema correctly distinguish four demographic classes from the paper's later heating/electricity groups and probabilistic dwelling link.

For LIT-001/003/004/021–023/034/039, DAT-005/009–012, GAP-002/004, RES-001/002 and INT-002, the inspected primary material supports the bounded statements. CLIMA already used household responses to infer heating operation and gains, included physical archetypes and final-energy conversions, and compared national carrier totals. Its Table 3 gives the stated gas values, while its maps have no measured regional comparator. The extension does not reproduce that national stock pipeline: its useful sensible heating, fixed geometry and gains, PVGIS coordinate TMYs, common seasonal calendar and study-local solver path are material departures. They are disclosed in the claims and methods rather than hidden behind the conference citation.

The archetype paper explicitly excludes survey expansion weights in its reported group summaries, distinguishes declared heating hours from model-derived appliance electricity, and reserves individual-building simulation consequences for future work. The v06 claim language does not adopt the paper's broad accuracy rhetoric as completed validation. It also avoids using demographic means as assignment thresholds or manufacturing measured hourly occupancy from the group labels.

I inspected the 48-record literature catalog, claim-to-source access statements, and all 17 rows of `conference_reference_crosswalk.md` against the original bibliography. Direct external primary-source checks included:

- [Prataviera et al. 2022](https://www.research.unipd.it/bitstream/11577/3417286/9/1-s2.0-S0306261922001568-main_compressed.pdf), especially §3.1.1, p6 and the measurement boundary: three survey bands, randomized profiles and the discrete ±2 h startup perturbation already exist. Verona and Milan do not share the same validation evidence.
- [Buttitta et al. 2019 author manuscript](https://researchrepository.ucd.ie/server/api/core/bitstreams/d1e15168-1930-4c63-ade3-d9f66969c0ca/content), particularly §§3.1.1–3.1.2 and the comparative results: occupancy-conditioned heating and gain inputs are established precedents.
- [MODENA BSA2024](https://pro.unibz.it/library/bupress/publications/fulltext/9788860462022_21.pdf), §§2.2–3.2, and the [official RSE2021 report description](https://www.rse-web.it/rapporti/applicativo-per-la-valutazione-dei-consumi-energetici-degli-edifici-residenziali-italiani/): Italian survey information, sociodemographics and hourly modelling precede this extension.
- Publisher/author primary abstracts or indexed sections for [Sokol et al.](https://www.sciencedirect.com/science/article/pii/S037877881631372X), [Chen et al.](https://www.sciencedirect.com/science/article/pii/S0306261922011540), and [De Jaeger et al.](https://www.tandfonline.com/doi/abs/10.1080/19401493.2020.1727567): probabilistic parameter fitting with a separate test set, stochastic ResStock schedules, and setpoint aggregation are not novel in general.
- [Hanmer et al.](https://link.springer.com/article/10.1007/s12053-018-9632-x), controller-analysis and limitations sections: programmed states and calls for heat are distinct from actual presence and energy-meter observations.
- [Li and Dogan's primary proceedings record](https://publications.ibpsa.org/conference/paper/?id=bs2025_1336), [Dabirian et al.'s primary article](https://www.frontiersin.org/journals/sustainable-cities/articles/10.3389/frsc.2026.1771845/full), and [Puglisi et al.'s primary article](https://www.iieta.org/journals/ijht/paper/10.18280/ijht.340320), as additional survey/archetype/scheduling comparators.
- The [official ISO15927-4 abstract](https://www.iso.org/standard/41371.html) supports the limited reference-year-versus-extreme-year statement. The primary author-institution indexed Calama-González methods material supports the stock/comfort/socioeconomic distinction; it does not validate the present model.

These comparisons support NOV-002's specific implementation contribution and its explicit rejection of generic first-method claims. They do not establish absolute priority. Full text was not independently obtained or read for every catalog entry. Some direct publisher fetches failed; accessible primary indexed excerpts were used only for the corresponding bounded statements. In particular, this review does not newly certify the full RSE report, full standards, the full Brøgger/Vivian2017 papers or Latifi's complete methods. The catalog's 48-source count must not become a claim of 48 complete independent full-text appraisals.

**Independent assignment, response and selection calculations**

The direct inputs were the workspace's lowercase `clusters.csv`, compact `istat_microdata_2013.csv`, official `istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt`, and the frozen v2 files in `05_simulations/archetype_campaign_v2/pre_run`.

I recovered 17,855 unique assigned IDs, all present in each 20,000-row survey file. The assignment artifact agrees with archived `segment + 1`. Recalculated group counts are 7,163/4,801/4,080/1,811. Rounded demographic means, standard deviations and household/person percentages reproduce 39 of 40 printed Table 3/Figure 5 cells; the retained exception is group 4's mean maximum age, 72.06074 → 72.1 rather than printed 72.2. Thus DAT-006 and LIM-009 have the correct strength: article-consistent archived membership, not proven exact published-branch identity.

Independent raw-file eligibility reconstructs exactly the same 13,624 IDs as the frozen compact-input selection, with group counts 5,430/3,693/2,991/1,510. All observed band/control values agree. The documented raw-blank-to-compact-zero counts are reproduced: 3,364 for each timer/valve field, 10,248 for the retrospective control-investment field, and 8,789 for each stove/fireplace field. None changes the implemented affirmative (`==1`) branch predicates. These checks establish implementation equivalence for this rule, not that a routed blank is an observed negative answer.

A fresh PCG64(20261020) selection reproduces all 32 test IDs. Independently fitting the four 24-hour branch centroids and nearest real training profiles reproduces IDs 2941, 981, 5303 and 9574 for 22/11, 22/19, 24/13 and 24/21°C. Test IDs are excluded from this fit. Training counts reproduce all conditional probabilities exactly; evaluation weights agree within 9.8e-17. The counts support DAT-013's percentages.

I independently reconstructed the source-ID-seeded contiguous segments using the inspected `EndUse.py` rules and the fixed season, without importing the bridge. All 36 full 8,760-value profiles and their array hashes match. The four shared components are real inputs; the centroids are not simulated. All 47 runtime hash targets and every target in the frozen selection manifest matched. The same shared components with different group probabilities implement all four demographic groups while leaving physical/gain assumptions common.

The experiment has more conditional probability parameters, only two test observations per group×branch stratum, and one input-selected component per branch. It consequently omits within-branch group differences and much within-group variation. MET-013–015, LIM-010/011 and INT-003 disclose these points. The training mixture weights and evaluation-stratum weights are appropriately distinct. I found no outcome fitting, changed holdout membership or averaged-input simulation in the inspected executed path.

**Independent physical and numerical checks**

I inspected the actual `profile_bridge.py`, both selection scripts, `run_mixture_campaign.py`, `analyze_mixtures.py`, and the local `single_commit_1c.py`, `surface_forcing_1c.py`, `ir_sky_forcing.py`. The external `EndUse.py` and EUReCA `thermal_zone.py` were read in place. The network's implicit air/mass balances and surface equation were reconstructed from raw hourly states and per-case parameters, rather than by calling the existing checking scripts.

All 152 new annual files have 8,760 finite correctly indexed rows, totalling 1,331,520. Maximum absolute independently reconstructed residuals are 5.935e-11 W (air), 1.617e-10 W (surface), and 2.831e-10 W (mass). Old-state continuity errors are zero. Every exported heating-setpoint array matches its frozen input, active heating attains that setpoint, and cooling load is zero. Every hourly file's hash matches its run summary.

Per-surface shortwave/longwave/window terms reconstruct within 1.365e-12 W, including each surface's own areas and coefficients. A separate BSpline evaluation reproduces the glazing coefficients within 3.34e-16. Node forcing exported by the preprocessor matches the solver inputs exactly. Horizontal-IR reconstruction gives the stored annual mean air-minus-effective-sky values exactly: 12.775656/17.620782/13.551708/14.073727 K for Padova/Bolzano/Roma/Palermo. All four EPWs retain 8,760 missing-cloud sentinels; Palermo alone has one negative wind entry. The inspected path replaces the cloud-dependent sky scalar and guards those unused fields.

Each technical repeat is numerically identical to its reference. The largest absolute annual warm-up relative difference is 2.15e-8 (Palermo); all peak differences are zero. These values are well within the frozen limits.

This establishes numerical consistency of the declared sensible-only model. I did not independently reproduce pvlib solar geometry/transposition, perform a timestep refinement study, rerun surface-order perturbation fixtures, or benchmark the zone against another engine or measurements. Annual-mean sky exchange, unapplied irradiation offset, the same climate-E season at every site, daily repetition, assumed control temperatures and the ideal unlimited heater remain physical assumptions. LIM-008/010/011 and the endpoint qualifications correctly prevent numerical residuals from being presented as physical validation.

**Central results independently recalculated**

The following RMSEs are reconstructed from raw hourly loads, independently re-estimated training probabilities, and original eligible-stratum counts. They agree with RES-009/010.

| TMY | Annual useful heat, pooled → conditional (kWh) | Individual annual maximum, pooled → conditional (kW) |
|---|---:|---:|
| Padova | 1828.20305 → 1787.94010 | 4.715544 → 4.601489 |
| Bolzano | 2633.85896 → 2565.52670 | 2.365811 → 2.357069 |
| Roma | 1437.67638 → 1411.15791 | 3.551397 → 3.474229 |
| Palermo | 1215.81481 → 1196.74295 | 3.238879 → 3.194494 |

All 40 overall/subgroup endpoint/site rows were independently reconstructed. The largest absolute discrepancy against the reported RMSE/paired-MSE fields is 6.52e-9, attributable to numerical summation. The paired annual MSE differences are −145596.59455, −355285.78841, −75546.73032 and −46011.94709 kWh²; the corresponding peak MSE differences are −1.062655, −0.041289, −0.542149 and −0.285542 kW².

Bolzano group 2 peak RMSE worsens from 1.965770 to 2.008767 kW and group 4 from 2.929172 to 2.941238 kW, confirming RES-011. The Padova detailed annual range is 5538.60434–13145.25903 kWh, conditional group predictions span 7736.54542–7953.89312 kWh, and the pooled prediction is 7834.85665 kWh, confirming RES-012. Overall improvement therefore does not imply uniformly better individual or subgroup prediction.

The raw Bolzano annual maxima occur on restart-day hours 6912–6935 for 27/32 test respondents and 2/4 components; 18/32 and 2/4 respectively peak exactly at 6912. The other five test respondents and two components peak in January. There are no ties. I calculated these counts without reading the existing diagnostic helper. RES-013 has the right denominators and correctly leaves the original annual maximum and loss endpoints intact. It establishes calendar coincidence; a counterfactual season-change experiment would be needed to quantify a causal restart effect.

For the retained original v5 scope, I integrated the saved raw annual files at all four sites. Padova seeded/early/late reproduce 9574.01817/9659.63278/9606.29644 kWh and 10.049540/10.253472/9.838176 kW. The claimed site-relative annual/peak contrasts follow. The Padova setback diagnostic reproduces 2,730 calendar hours, 1,323 positive-load hours and 2375.61749 kWh, and maximum air temperature is 41.04611°C. Bolzano's original three profiles all peak at hour 6912 and 17.808621 kW. These support RES-003–008 and LIM-005 without using a previous reviewer verdict as evidence.

**Findings and limitations of this recommendation**

There are no new actionable scientific findings to assign critical/major/minor severity. The existing limitations are material and must survive prose editing: exact paper-branch identity unresolved; archetypes implemented only through operational branch probabilities; one physical case; small stratified holdout; additional conditional parameters; fixed assumed temperatures and gains; strong information compression; unquantified timing/longwave sensitivity; restart-dominated Bolzano maxima; and absent independent empirical or external-engine validation. These are already explicit rather than errors presented as settled.

The full claim set was considered, but the checks are not all equally deep. Primary empirical literature was inspected at the access depths described above, not exhaustively replicated. I did not rederive every historical pre-v5 defect or re-open their diagnostic reports; their exclusion is conservative, and no quarantined annual output enters the new calculation. The exact historical review-approval statements in RQ-001/002 and related fields are procedural claims for the later closure pass, not independently established by this scientific review. I also did not prove the absence of a compatible meter series outside the bounded local inventory.

The defensible result is that conditional branch frequencies yield modestly smaller overall same-model losses in this frozen experiment, with documented subgroup exceptions and substantial unresolved variation. It is not evidence that demographic archetypes generally improve real heating predictions, that the model reproduces a particular household, or that four TMY exposures are independent population replications. Within those limits, the evidence is concrete, reproducible and sufficient to support the requested implemented-archetype extension and its claim-stage literature argument.
