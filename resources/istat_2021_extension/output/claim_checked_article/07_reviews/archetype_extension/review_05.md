# Independent scientific review 05 — frozen archetype claim manuscript v05

Initial critique saved 20 October 2026, before access to earlier reviews, author responses, issue registers, pre-run audit helpers, author arithmetic checks, final acceptance records or prose.

Reviewed manuscript: `06_claim_drafts/article_claims_v05.md`, SHA256 `2e5889e3948a72dab10af6867b4fad193f7bfd40efecab8248c3fb3aad9773fa`; 105 claim blocks. Machine ledger: `06_claim_drafts/claim_ledger_v05.json`, SHA256 `7b2f31d57d8bbadcb17a07b1249136033caa39f5381c0fb8dd985076798003b0`. The assigned role is a fresh independent scientific reviewer, requested model `gpt-6-astra`; runtime model metadata are not exposed to this reviewer. Procedural independence is not empirical validation.

## Recommendation

**Minor revision. No critical or major scientific error was found in the bounded claims inspected.** This recommendation applies to the actual narrow experiment: archived article-consistent demographic labels condition frequencies over four shared source-rule operating components, and predictions are assessed against outputs of the same model for 32 held-out respondents. It does not approve national prediction, observed accuracy, general algorithm superiority, exact reproduction of the published membership, or an independently benchmarked building model.

The implementation is substantive: demographic labels enter the conditional weights rather than merely labelling one earlier respondent. The conditional and pooled representations share the four components, but have different probability parameter counts. The draft correctly retains that limitation, adverse subgroup results, the failed initial compression design, and the separation between source assumptions and observed heating hours. The literature has become a useful, qualified comparison of methods rather than a citation count. Both CLIMA continuity and earlier occupant-aware precedents are present. Resolve the two minor findings below in the next immutable claim revision and response. The requested further fresh review and final closure still apply before updated prose.

## Independent checks and results

I wrote `reviewer_05/check.py` and `reviewer_05/supplement.py` without importing the authors' analysis code or previous reviewers' checkers. Both ran with the existing read-only EUReCA Python environment using `-B`; no annual simulation or external checkout modification was performed. Outputs are `audit.json`, `supplement.json`, `metrics.csv`, `endpoints.csv`, `run_checks.csv`, `primary_peak_context.csv`, and `old_pilot_endpoints.csv` in that directory. `audit.json` records exact SHA256 values of raw files consumed by the main check, including all 152 new annual files, profile inputs and network parameters.

### Source, assignment and input bridge

- Read the complete target CLIMA chapter text, PDF pages 2–11/printed 1302–1311, excluding adjacent papers. Visually inspected the regional end-use map page. The chapter itself supports the stock/software lineage, assumed heating schedules, 1C description, partial historical weather coverage and national final-energy comparison. It does not validate the present dwelling or hourly outputs. The natural-gas numbers in RES-001 match the chapter; its two quoted deviations follow directly from the displayed numerator/denominator.
- Read the original archetype paper's methods and definitions, demographic results, dwelling-linkage interpretation, conclusion and data availability through `primary_complete_text.txt`; visually inspected original page 7, Table 3/Figure 5. DAT-009–012 preserve the published values and appropriately treat them as summaries rather than assignment rules. The published conclusion explicitly leaves individual-building simulation impacts for future work. Its accuracy rhetoric is not imported as evidence.
- Directly joined the archived assignment CSV to compact inputs. All 17,855 assigned records join; recomputed the 13,624 eligible records and 13,592 training records. Independently read the official raw 2013 release and reconstructed eligibility; all 13,624 eligible IDs have matching hours/main-system values and the same `==1` control/fireplace predicates as compact inputs. This is a raw-value equivalence check, not endorsement of missing-as-no semantics.
- Inspected actual `C:/Works/GRINS/GRINS-Tool/classes/EndUse.py` source. Independently reconstructed its three separately source-ID-seeded contiguous segments, night rotation, four assumed temperature branches and 105–288 day seasonal exclusion. All 36 complete frozen profiles match exactly and their binary-array hashes agree. All corresponding new solver setpoint columns match their frozen profile.
- Independently recomputed training branch counts and conditional probabilities, group and empirical-domain evaluation weights, and nearest-centroid branch components with the specified tolerance and smallest-ID ties. Selected component IDs agree and none is in the 32 test IDs. Preserved v1 and v2 test IDs agree; all five v1 prototype roles share one full-profile hash. I did not rerun clustering or certify exact remote published-branch membership.

### New raw numerical results

Reintegrated all 152 annual hourly outputs: 1,331,520 rows, of which 144 annual files are scientific cases and eight are repeat/warm-up checks. Reconstructed all eight overall and 32 subgroup MSE/RMSE comparisons from raw component/reference endpoints and independently calculated weights. All 40 rows agree with `weighted_metrics.csv` within numerical precision.

| Weather | Annual RMSE pooled → conditional (kWh) | Peak RMSE pooled → conditional (kW) |
|---|---:|---:|
| Padova | 1828.203053 → 1787.940102 | 4.715544 → 4.601489 |
| Bolzano | 2633.858963 → 2565.526700 | 2.365811 → 2.357069 |
| Roma | 1437.676381 → 1411.157909 | 3.551397 → 3.474229 |
| Palermo | 1215.814805 → 1196.742952 | 3.238879 → 3.194494 |

The RES-009/010 paired MSE differences reproduce. The only adverse subgroup RMSE comparisons among the 32 subgroup rows are precisely the two disclosed in RES-011: Bolzano peak group 2, 1.965770 → 2.008767 kW; group 4, 2.929172 → 2.941238 kW. Overall improvement therefore cannot be described as universal subgroup improvement. The prediction is a probability-weighted mean of separately computed individual maxima; I did not substitute the maximum of a mean curve.

The component-selection code reads inputs rather than annual outcomes; the recorded v1 collapse and explicit v2 pre-output amendment are scientifically coherent. The current evidence supports the frozen comparison, not an independently established unconditional error rate for other splits or homes. Two tests per stratum, one split, one physical dwelling, a four-component compression and extra conditional probabilities remain substantive limits even though the arithmetic is correct.

### Physics and original pilot

Independently reconstructed air, surface and mass residuals for every new hour from saved network constants, states, gains and HVAC radiant/convective partitions, guided by the external solver's equations. Maximum absolute residual is **2.837623469531536e-10 W**. Old-air and old-mass continuity is exact. Saved solver forcing matches exported surface forcing. Independently recalculated each wall/roof's opaque shortwave, longwave and window contribution and its distribution to surface/mass nodes at all four sites; all agree within 1e-8 W. This checks the actual exported algebra, not just a summary's `pass`.

Read `run_mixture_campaign.py`, `profile_bridge.py`, both selection scripts, `analyze_mixtures.py`, `single_commit_1c.py`, and `surface_forcing_1c.py`. The new runner keeps the same 80 m² geometry, envelope, infiltration, electric gains and seasonal convention across profiles/sites. It passes the frozen lower setpoints to the local one-commit 1C stepper. Group labels do not secretly alter gains or geometry.

Reintegrated all 20 old v5 pilot annual files directly, independently of historical review helpers. The seeded/early/late annual and maximum values supporting RES-004/005/007 reproduce, including the equal Bolzano 17.808620809 kW maxima at hour 6912. New archetype quantities were never sourced from these old runs. I did not re-adjudicate the historical defect/quarantine findings before saving this initial critique; those historical review claims are not new physical evidence.

The declared annual-mean sky approximation, native solar-time convention, fixed seasonal discontinuity and lack of benchmark/meters are important. Balance closure cannot validate them. The manuscript does not present current agreement with the RC equations as a matched independent model benchmark or observation.

### Literature and novelty appraisal

Primary sources directly reopened online, in addition to the two local primary papers:

- [Buttitta et al. 2019, author manuscript](https://researchrepository.ucd.ie/server/api/core/bitstreams/d1e15168-1930-4c63-ade3-d9f66969c0ca/content), especially PDF page 16, §§3.1.1–3.1.2. This confirms an earlier occupancy-to-heating/internal-gain implementation and constrains generic integration novelty.
- [Prataviera et al. 2022](https://www.research.unipd.it/bitstream/11577/3417286/9/1-s2.0-S0306261922001568-main_compressed.pdf), PDF pages 6 and 11, operating-time/start-up uncertainty and Verona measurement boundary. This confirms the ±2 h start-up precedent and the distinct measured district-heating context.
- [Prataviera et al. 2024, MODENA](https://pro.unibz.it/library/bupress/publications/fulltext/9788860462022_21.pdf), printed pages 158–160, survey/engine/accounting/scenario methods. This substantiates the conference lineage and prior scenario work.
- [Li and Dogan 2025](https://publications.ibpsa.org/proceedings/bs/2025/papers/bs2025_1336.pdf), title/abstract and modular classification description. This independently corroborates a recent survey-based modular 5R1C implementation already listed in the nearest-work matrix. No accuracy magnitude is adopted here.

The conference crosswalk covers all 17 references and distinguishes methods lineage from policy/standards background. Source access depths are expressly unequal. NOV-002 is a defensible application/implementation contribution, not a demonstrated methodological first. The current experiment is narrower than full stock implementation, occupancy modelling, or the archetype paper's dwelling-allocation method, and LIM-011 states that clearly. Before prose conversion, the synthesis should retain substantive comparisons with the closest studies, not merely their references. Li/Dogan is currently a contextual ledger entry without its own manuscript claim; any substantive sentence added about it needs the normal claim gate.

## Findings

### R5-01 — Minor — clearly distinguish superseded validation rows from the operative v2 design

**Claims:** MET-013–016, LIM-010; their linked methods documentation.

**Evidence:** `03_methods/archetype_extension/validation_matrix.md`, first table, “Prototypes” row still specifies four conditional and one pooled medoid and a four-versus-one capacity difference. Its compression row emphasizes signed/absolute differences. The later v2 supplement and frozen `mixture_protocol_v2.md` correctly specify four shared branch components, 12 versus 3 probability parameters and primary squared deviation/RMSE.

**Consequence:** The two stages are recoverable, so this is not a wrong executed experiment or invalid result. However, the first table is not explicitly labelled historical/superseded and can misstate the active validation design when used alone by the prose/figure author. MET-014's counterevidence also says “four methods share components”; there are two representations sharing four components.

**Checkable remedy:** Preserve frozen historical files. Clearly mark the first matrix's superseded rows as v1-only, or provide one current v2 matrix with historical references. Correct “four methods” to “two representations”. Verify that current documentation consistently specifies four shared components, 12/3 free probabilities, squared loss/RMSE and the expected-individual-maximum estimand.

### R5-02 — Minor — make the seasonal-restart qualification specific to the new peak result

**Claims:** RES-010, RES-011, INT-003; linked RES-008 and LIM-008.

**Evidence:** Own `reviewer_05/primary_peak_context.csv`, extracted directly from the 144 scientific v2 annual files. Under Bolzano exposure, **27 of the 32 held-out respondents** and **2 of the 4 components** attain their annual maximum on the prescribed restart day, model hours 6912–6935 (16 October). Of those, 18 respondents and both components peak exactly at hour 6912. The other five test maxima are in January. The strong quantified RES-008 seasonal explanation currently concerns the old single-respondent v5 timing pilot; RES-010 provides only the generic statement that maxima “can occur” at onset.

**Consequence:** The reported metric remains correct and the endpoint was predeclared; no maximum should be removed. But the new Bolzano primary result is substantially tied to the imposed seasonal recovery. Readers should not mistake the old pilot's three equal maxima for a complete characterization of the new heterogeneous cohort, or miss how much of the new endpoint lies on the restart day.

**Checkable remedy:** Add a narrowly labelled **post-hoc v2 peak-context diagnostic**, verified by the author from raw hourly files, adjacent to the new peak interpretation. Preserve all annual maxima and all primary metrics. State the exact 32-test/four-component denominators and avoid causal attribution, local regulatory realism or equipment-sizing interpretation. A short count sentence/table is sufficient; no new annual simulation or alternate endpoint is required.

## Limits of this review

I did not reread every page of all 48 external sources, independently authenticate the inaccessible published assignment branch, reproduce the original clustering, inspect every raw questionnaire page, obtain independent measurements, rerun a formal benchmark, rerun any annual simulation, or quantify sensitivity to envelope/sky/solar phase/seasonal assumptions. Not every historical review-specific diagnostic has been independently reconstructed in this initial pass. My raw force check starts from exported plane irradiance/AOI and network constants; it does not independently certify native solar transposition or physical material parameter truth. These are explicit review limits, not claims that missing empirical validation has been completed.

No author-check output, previous reviewer verdict/helper, pre-run audit verdict/helper, or final prose was used to establish this initial recommendation. Incidental historical-review references inside the manuscript and canonical methods were treated only as citations. Any prior-issue closure must be a separately labelled subsequent pass after this critique has been saved and its hash reported.
