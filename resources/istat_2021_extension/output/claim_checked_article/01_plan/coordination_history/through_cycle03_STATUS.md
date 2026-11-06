# Latest checkpoint — v5 independently checked; 2 full cycles complete

V5 new surface preprocessing and20annual campaign cases plus1isolated replay completed. Reviewer03 independently reconstructed forcing from rawEPWs/live surfaces, checked order invariance and all175,200annual states/arithmetic:07_reviews/v5_verification_supplement.md. Specific R3-01conditionally closed for exactv5implementation/campaign; R2sky-to-envelope gate satisfied within bounded model. Fresh fullreview04still required. Current article-run count87=66quarantined earlier executions+20v5campaign+1v5replay; reviewer01diagnostic annual reproductions5separate.

Author is preparing v03/response03/register03 using independently reintegratedv5data and supplement; these are not yet complete, so fullcyclecount remains2. Formalreview03SHAdda6e7288e34af7368f458cc40390297efa7b3da0ab097cabb1cb606e9e77e34 is saved and unchanged. Originalreview03discovered roof bug before v5fix; supplements are separate. PriorR2closure is07_reviews/closure_02_after_review03.md. V03must address R3-02nativeannualmeansky/solarphase limits,R3-03closest2022±2hstartup precedent,R3-04canonicalpointers and statusenumQA. Bolzano all3annualpeaks occur16October00:00atseasonrestart; author verifiedrawCSV and will add boundedposthoclimitation, leavingpredeclaredendpointunchanged.

Next: waitfrozenauthorv03hash andcompletecycle03artifacts; update localsourcealiasregistry forv03; launch NEWfreshgpt-6-astrareviewer04onv03withprimaryevidence andnoearlierverdicts/privateauthorreasoning. Thenauthorv04/response04/register04, finalAstra closure. Onlyafteratleast4completecycles+gate may NEWgpt-6.1-solproseauthorandseparateNEWgpt-6.1-solvisualizationagent writearticle/figures. No prose/figuresyet. Needfinalindexcurrentv5paths,sourceledgeraliases,workflow/hash/link/fidelityaudit. Canonicalmethodsdocsarebeingreconciled;frozenrunsourcesunchanged.

Below is the prior detailed checkpoint/history; superseded run-status statements there do not override this latest checkpoint.
# Current status — 20 October 2026

## Objective and scope
Complete the user's research workflow through at least four genuine serial claim-draft/review/revision cycles, final closure, a separate English prose first draft and a separate programmatic visualization handoff. Full request:00_context/user_request.md. User followup confirms simulations are new planned/executed work; choose areas to fit context and construct inputs. No publication/contact authorization. No pending user question is required for current work.

Selected question: sensitivity of one assumed dwelling's EUReCA heating outputs to within-block placement of a constructed22/19°Cprofile derived from one2013Veneto response (reported4/4/1hours). These are high-setpoint durations, not heater runtime. Four coordinate TMY exposures:Padova,Bolzano,Roma,Palermo. Identical building/season/other inputs; no city/national representativeness or accuracy claim. Temporal2013–2021blueprint deferred with evidence-based rationale.

## Exact completed review count: 2

- v00 SHA:f2df38afd47db701eba7edf7304a7bd7e9da158d65e7bebd3acace32ecfc0089.
- Cycle01 complete:review_01.md,response_01.md,v01,issue_register_01.json. v01 SHA:5e35bc02e77d6345a9c82be51a8a405d5d4ba3e8490f6c24d037b8488a04bab1.
- Cycle02 complete:review_02.md,response_02.md,v02,issue_register_02.json. Current immutable v02 SHA:91373b60fe5f574057f2e2fdfc597cf9e373c090ae4d2aba4757682b183c0045 (38claims). Numeric physical results withheld.
- Review03 is running fresh on v02. No cycle03 response/revision/register exists yet. No prose or figures have been authored. Prose gate closed.

## Scientific execution and active blocker

66article annual executions are preserved:5v1pilot+20v2+20v3+20v4+1v4isolated replay. Reviewer01 also performed5separate diagnostic annual reproductions. NO current physical effects are accepted. All pre-v4outputs(45)are quarantined; v4promotion is now also blocked by a new independently reproduced opaque-surface preprocessing issue. Methods worker will reconcile its quarantine records.

1. R1-01 current EUReCA recursive dispatcher advanced/overwrote states at mode transitions. Local single-commit adapter implemented; all v3/v4node balances passed. Independent review02 conditionally closed this bounded correction. R1-02 high-setpoint/runtime wording and R1-03 inventory evidence class closed in separate closure_01_after_review02.md.
2. R2-01 opaque-cloud missing99was used as physical cloud cover, causing wrong sky forcing. V4uses valid horizontalIRto derive effective sky temperature and replace native annual-mean air-minus-sky scalar; source/units/time/QC/guard checks pass. R2-02 command missingSITEargument addressed by new-campaign CLI, byte-identical isolated replay and existing-destination refusal. Independent issue closure awaits review03second pass after its critique.
3. CURRENT OPEN issue discovered by review03and methods extraQA: thermal_zone.py~802–842 assigns A_op/h_r/TRV and plane irradiance only underhasattr(surface,'window'), then uses them for all opaque ExtWall/Roof surfaces. Unglazed roof reuses previous wall values (80m²roof versus19.2m²wall), causing wrong solar/LWforcing and surface-order dependence. Candidatev4Padova actual sky response−160.55Wdiffers from expected−289.99W. Direct diagnostics:07_reviews/reviewer_03/runtime_forcing_checks.json and05_simulations/native_longwave_preprocessing_v4/native_forcing_difference_checks.json. Formal review03and conclusive order-invariance proof pending. Methods authorized only study-local preprocessing correction, independent per-surface solar/LWchecks, then new Padova-first runIDs. External checkouts remain untouched.

Remaining external gaps:independent whole-building benchmark and empirical validation not performed. Candidate ANACI annual STDMCfields exist but provenance/units/meter boundary/calibration holdout are unverified. Native annualmeanlongwave and irradiance/time approximations remain explicit limits. Palermo has one unused negativewind source record; rawretained, runtimeguards forbid wind/cloud consumption in active boundedpath. No generic all-variableweather-validity claim.

## Active real agents / ownership

- /root orchestrator:01_plan,STATUS,manifest,index.
- /root/claim_author requestedgpt-6-sol:06_claim_drafts,responses/registers; v02frozen; awaiting formalreview03and correctedrun evidence forv03.
- /root/review_03 requestedgpt-6-astra:fresh independent fullrubric/currentprimaryevidence; ownsreview_03andreviewer_03helpers. Earlier verdicts/privateauthorreasoning withheld until savedcritique.
- /root/simulation_methods requestedgpt-6.1-sol:03_methods/04_analysis/05_simulations; currently investigating opaqueforcing correction. No furtherannualrun authorized before diagnosis/pre-run correctionprotocol/checks.
- Completed context/literature agents requestedgpt-6.1-sol available for followup. Local source registry currently covers v01aliases only; update it for later aliases before prose gate, preserving initialregistry snapshot.
- Prose author must be a NEW separategpt-6.1-solagent afterminimum4cycles+closure. Visualization must be another NEWgpt-6.1-solagent; no generativeimages anywhere. Modelselection accepted by tools; runtimebackendconfirmation unknown for allagents.

## Next actions

1. Receive review03formalcritique; preserve its hash. Then authorize separate prior-issueclosurepass. Author revisesv02→v03withresponse03andregister03; include only independently checked correctedrun evidence or explicit pendingresults.
2. Complete local opaquepreprocessing fix, per-surface analytical/order-invariance checks and new Padova-first campaign if feasible. Preserve all earlier runs and exactcounts. Do not promote balance-only success into validforcing/empiricalaccuracy.
3. Fresh reviewer04onfrozenv03withallprimaryevidence/fullredteamrubric. Priorreviewswithheld untilcritique saved. Authorv04/response04/register04, thenAstra finalclosureonv04. If materialerrorsremain, further cycles rather than falsepass.
4. Update sourcealiases/ledgers and readiness/gate. Then NEWproseagent writes08_manuscript/article_first_draft.md,claim_to_prose_map.md,09_figures/figure_briefs.md; no newunsupportedclaims. NEWvisualizationagent makesreproducibleSVG/PNGfigures withQAandmanifest; nofabricateddata. Proseauthorintegratesreadyfigures/keepsblockedplaceholders.
5. Finalfidelity,workflow/hash/linkaudit,cleanSTATUSandmanifest,openprosedraft,deliverARTIFACT_INDEXplusrequestedlinks. Finalreportmustdistinguishcompletedreviews,acceptedruns,quarantinedattempts,andmissingempiricalvalidation.

Supportingrecords:agent_manifest.md;01_plan/decisions.md;task_dag.md;review_rubric.md;author_packet.md;workflow_audit.json. Run01_plan/audit_workflow.py to verify structuralcyclecount,reviewedhashes,stableIDsandlatestledgerhash. ARTIFACT_INDEX.md is stillworkinprogress; updatecurrentrun/figurepaths beforedelivery.

