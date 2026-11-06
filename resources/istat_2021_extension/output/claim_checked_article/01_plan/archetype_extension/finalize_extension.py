"""Finalize the current artifact handoff after root's prose/figure fidelity read."""
from pathlib import Path
import hashlib
import json
import re
import runpy

ROOT=Path(__file__).resolve().parents[2]
def read(p): return p.read_text(encoding='utf-8-sig')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

expected={
 '08_manuscript/article_first_draft.md':'79d39383485e4310254a2fa74764fd93dd63e252bda37379d187f9e13642614c',
 '08_manuscript/claim_to_prose_map.md':'71ab5816c3cb2beaee24499a1fa8666788afb4cff17c6561547b27800331c000',
 '08_manuscript/fidelity_check.md':'15d5c2e2265c50e5521bda806c7dd2bd13b15d437b9b42a280ba0d8a27d78ca9',
 '06_claim_drafts/article_claims_final.md':'9aa90444db044ab1480872fbf496db171f359e39e10cf1207f76a4c586757df9',
 '09_figures/archetype_extension/figure_briefs.md':'9eaad3a1892bf650600857986b350b1abb234c598f26bff9627c7e5b269bf993',
 '09_figures/archetype_extension/FIG-04_components_probabilities.png':'ee36af5984145817d556e54d279d5158e35c647025ae07c6efdf5e51d917daf7',
 '09_figures/archetype_extension/FIG-05_fidelity_metrics.png':'5c8071b66b63bf6340060c3731645f81bb5c040d587ec730eb7f4cb786e6e924',
 '09_figures/figure_briefs.md':'3a3f9f3bd803c75e97e3369524d7e7dcee4792550c15a00cfb3d6f412c42bd4d'}
assert all(sha(ROOT/p)==h for p,h in expected.items())
article=read(ROOT/'08_manuscript/article_first_draft.md')
mapping=read(ROOT/'08_manuscript/claim_to_prose_map.md')
paragraphs=re.findall(r'<!-- (P\d+|CAPT?\d+) -->',article)
assert len(paragraphs)==54 and len(set(paragraphs))==54
assert all(x in mapping for x in paragraphs)
assert len(re.findall(r'^\d+\. ',article.split('## References',1)[1],re.M))==37
assert len(re.findall(r'!\[[^\]]*\]\([A-Z]:/',article))==2
assert 'FIG-06 BLOCKED' in article and '[RESULT PENDING:' in article
assert not (ROOT/'09_figures/archetype_extension/FIG-06_empirical_comparison.png').exists()

(ROOT/'STATUS.md').write_text('''# Revised first-manuscript delivery complete — 20 October 2026

The requested archetype implementation and literature expansion are delivered. The English article now centers the actual four-group operational mixture, includes two native tables and two integrated programmatic figures, and has a developed 1,527-word critical literature section with 37 source-linked references. All 17 references from the original CLIMA chapter were crosswalked; relevant stock, survey, engine, prototype and evaluation sources form the argument's backbone. The broader evidence pack contains 48 records with explicit access depths, not 48 claimed full-text critical readings.

Accepted scientific authority: 106-claim immutable v07 and its exact canonical copy, SHA2569aa90444db044ab1480872fbf496db171f359e39e10cf1207f76a4c586757df9. Six actual serial review-response-revision-register cycles are complete, including two for this extension. Round06 required no scientific correction; v07 is procedural only. Fresh initial critiques preceded release of earlier verdicts/private checks. Separate final closure closes both new issues and retains ten historical bounded dispositions; all12final entries are closed. Review closure is not empirical validation. The new sealed gate preceded separate prose and visualization work.

New simulations completed: 152annual executions (144scientific and eight repeat/warm-up checks), Padova first then Bolzano/Roma/Palermo, 1,331,520hourly rows. The four archived demographic groups condition training probabilities over four real source-control components, with three pooled versus twelve conditional free probabilities. Training uses13,592records;32held-out respondents are compared against same-model detailed-input references for one assumed80m²dwelling under four coordinate TMYs and fixed physical inputs/gains/season. The v1 singleton collapse was input-only, with no v1 annual run. The original87article executions remain distinct, including66quarantined runs and one separate v5replay; total article executions239. The original reviewer's five diagnostic annual reproductions remain outside that count.

The prose reports lower overall conditional RMSE for both endpoints at all four exposures, adverse Bolzano peak results for groups2/4, substantial remaining individual spread and separate post-hoc restart-day counts. These are model-reference deviations, not measured predictive errors or general demographic superiority. Peak predictions average individual annual maxima, not coincident urban peaks or sizing targets. Published descriptive group means are not assignment thresholds or measured occupancy/gain schedules.

Final source registry:116resolved claim aliases,121records,1404files and2000reported hash/identity checks, zero missing/mismatched/unresolved. V05–v07 registry histories, all immutable drafts/reviews/registers, original data/runs and the complete earlier delivery remain preserved. Root's current structural audit checks source aliases, schema, exact final/index binding, six actual cycles, local links, original figure-file preservation and saved delivery identities. It is distinct from scientific review and validation.

Delivered figure package: FIG04actual component profiles/probabilities and FIG05overall/subgroup RMSE, editable SVGs,400dpiPNGs, exact plotting data, code, source hashes and actual publication-width visual QA. TAB01/02transcriptions pass. FIG06empirical comparison remains blocked, with no fabricated data or image. Root read the full prose/map, reviewed source/claim correspondence and viewed both final publication previews including the corrected high/setback label. The final prose author sealed paragraph/caption coverage and integrated hashes.

Requested roles: gpt-6-sol claim author; fresh gpt-6-astra reviewers; gpt-6.1-sol evidence/methods/simulation workers, separate prose and visualization owners. Tools accepted these model selections; independently confirmed backend identity/version remains unknown. All worker handoffs are complete; no continuing work is claimed after delivery. No user question is pending. No generative images, external contact, submission, publication, external checkout edit or environment installation occurred.

The scientific manuscript remains explicitly incomplete for predictive/transfer claims: matched independent whole-building benchmarking, compatible monitored-building measurements, broader uncertainty/sensitivity, exact remote published assignment identity and conference executable identity remain unresolved. The index links the manuscript, claims, all six reviews/responses, evidence, protocols, figures/code and final audits.
''',encoding='utf-8')

p=ROOT/'agent_manifest.md'
s=read(p)
s+='''
## Final archetype-extension delivery

All extension worker handoffs are complete. The separate prose owner delivered the1,527-word literature section,37source-linked references,54mapped paragraph/caption IDs, two verified native tables and integrated FIG04/05. The separate visualization owner delivered SVGs/400dpiPNGs/code/data/numerical and actual visual QA; its terminology-only high/setback label update preserves data/briefs/captions. Root read the complete prose/map and inspected both publication previews, including the corrected final FIG04. Final articleSHA79d39383485e4310254a2fa74764fd93dd63e252bda37379d187f9e13642614c. All six actual cycles and current106-claim authority are preserved; round06's revision was procedural only. No worker task remains represented as running. Requested role routing is unchanged and backend confirmation remains unknown. Final current records are01_plan/archetype_extension/delivery_audit.json,delivery_manifest.json and final_fidelity_check.md. Original delivery/figures and immutable scientific evidence remain preserved. Missing benchmark/meters/uncertainty/provenance remain scientific gaps, not uncompleted hidden worker tasks.
'''
p.write_text(s,encoding='utf-8')

p=ROOT/'01_plan/archetype_extension/task_dag.md'
s=read(p).replace('Active after sealed new gate','Complete: expanded article/map/briefs, final figure integration and prose fidelity sealed').replace('Active: exact new briefs frozen before separate visualization dispatch','Complete: separate figures/code/data/QA, native table checks, integration and root fidelity')
p.write_text(s,encoding='utf-8')

p=ROOT/'ARTIFACT_INDEX.md'
s=read(p)
start=s.index('Working date:');end=s.index('\n## Current extension',start)
s=s[:start]+'''Delivery date: 20 October 2026. The user-requested archetype/literature extension is complete as an explicitly incomplete scientific first manuscript. Six actual serial claim-review/revision cycles, final independent closure and exact v07 source binding preceded separate prose and visualization work. Start with the [revised English manuscript](08_manuscript/article_first_draft.md), [accepted 106-claim checker](06_claim_drafts/article_claims_final.md), [status](STATUS.md) and [actual agents/models](agent_manifest.md). Review completion does not establish empirical validation.
'''+s[end:]
start=s.index('## Prose and figures');end=s.index('\n## Preserved evidence',start)
s=s[:start]+'''## Prose and figures

- [Revised English article](08_manuscript/article_first_draft.md), [complete claim-to-prose map](08_manuscript/claim_to_prose_map.md), [prose fidelity seal](08_manuscript/fidelity_check.md), and [reference-to-verified-source map](08_manuscript/reference_source_map.txt).
- [Complete new figure/table briefs](09_figures/archetype_extension/figure_briefs.md) and [figure/data/code/source/QA manifest](09_figures/archetype_extension/figure_manifest.md). TAB01/02 are verified native Markdown; FIG06 remains blocked without invented data.
- [FIG04 actual component inputs and probabilities: PNG](09_figures/archetype_extension/FIG-04_components_probabilities.png), [SVG](09_figures/archetype_extension/FIG-04_components_probabilities.svg), [profile data](09_figures/archetype_extension/FIG-04_profile_plotting_data.csv), and [probability/count data](09_figures/archetype_extension/FIG-04_probability_plotting_data.csv).
- [FIG05 overall and Bolzano subgroup RMSE: PNG](09_figures/archetype_extension/FIG-05_fidelity_metrics.png), [SVG](09_figures/archetype_extension/FIG-05_fidelity_metrics.svg), and [exact selected metric rows](09_figures/archetype_extension/FIG-05_plotting_data.csv).
- [Editable new renderer](09_figures/archetype_extension/render_figures.py), [figure numerical/provenance QA](09_figures/archetype_extension/figure_provenance_qa.json), and [native-table QA](09_figures/archetype_extension/table_transcription_qa.json).
- [Root final fidelity read](01_plan/archetype_extension/final_fidelity_check.md), [current structural delivery audit](01_plan/archetype_extension/delivery_audit.json), and [final delivery identities](01_plan/archetype_extension/delivery_manifest.json).

The literature section develops1,527words with37source-linked references selected by relevance from48records. All17original CLIMA references are audited in the crosswalk; relevant sources are integrated into the critical argument. The [sealed handoff](01_plan/archetype_extension/prose_and_figure_handoff.md) and gate preserve the drafting/visualization sequence. Original timing-only materials remain preserved in the [earlier delivery](01_plan/delivery_history/pre_archetypes_20261020/snapshot_manifest.json).

Original scientific figures retain their original scope and unchanged brief binding: [FIG01 endpoints PNG](09_figures/FIG-01_paired_contrasts.png), [SVG](09_figures/FIG-01_paired_contrasts.svg), [FIG02 setback diagnostic PNG](09_figures/FIG-02_setback_diagnostic.png), [SVG](09_figures/FIG-02_setback_diagnostic.svg), [old briefs](09_figures/figure_briefs.md), [old manifest](09_figures/figure_manifest.md), and [original editable script](09_figures/render_figures.py). They remain secondary one-response evidence.
'''+s[end:]
p.write_text(s,encoding='utf-8')

(ROOT/'01_plan/archetype_extension/final_fidelity_check.md').write_text('''# Orchestrator final fidelity and delivery check

20 October2026. This is a final artifact/prose/graphics check, not a new independent scientific review or empirical validation.

Root read the full revised article, complete paragraph/visual map, prose fidelity record, frozen briefs, figure manifest, final independent critique/closure, accepted claims and final source binding. The article has actual four-group operational mixtures, the source-backed demographic/cohort tables, substantial original-CLIMA-based critical literature, all eight overall RMSE/MSE comparisons, adverse Bolzano group2/4 results, residual spread and separate post-hoc restart denominators. The one-response v5pilot is secondary. Counts, equations, units, training-versus-evaluation weights, three-versus-twelve probability parameters, source assumptions and major inference limits agree with accepted claims. No new substantive scientific claim was identified. All54paragraph/caption IDs are mapped, with explicit disposition of accepted claims not repeated in prose.

Root viewed FIG04/05 at actual1000px publication width and re-viewed final FIG04 after its high/setback label correction. No clipping, collision or scientific mismatch was observed. Comparable absolute setpoint axes,0–1probability scale, exact near-equal RMSE labels, separate endpoint units, adverse groups2/4, n32/n8, parameter counts and post-hoc27/32test plus2/4component context are legible. The worker additionally viewed full PNGs and recorded plotting arithmetic/source hashes. Two SVG/400dpiPNG pairs, code and exact plotting tables are delivered; no annual model was rerun for final plotting. Native tables have independently checked text identities/cells. FIG06 has an informative blocked placeholder and no image or fabricated observation.

Root's two editorial requests clarify actual primary-source access depth and state the implemented bridge directly. A citation-link refinement points the already verified DeJaeger2020record to its canonical DOI. Primary publisher spot-checks confirmed its unusual printed colon title and the existing Vivian2017PII/DOI identity; these checks introduced no new scientific claim. Other primary literature checks/access limits are recorded by the evidence and fresh reviewers. The final text does not claim48complete independent full-text readings or a systematic review.

Six actual serial cycles and final independent closure preceded the new sealed gate and separate prose/visualization handoffs. Round06required no scientific correction; v07updates elapsed procedural labels only. Current accepted claims/index are byte-identical to frozenv07. All12issues have bounded closed dispositions. The source registry binds106claims/116aliases with no reported missing/mismatched/unresolved items; root checked final registry/ledger identities and version binding. The worker's2000aggregate checks are attributed rather than claimed as a new root rerun. Original figure files and saved earlier-delivery identities pass preservation checks. External runtime/source immutability is supported by the worker audits and independent review calculations; no external environment was changed.

Independent benchmark, compatible meters, wider physical/assignment/split sensitivity, exact remote assignment identity and exact conference executable identity remain unresolved. Review closure, numerical balances and same-model fidelity do not supply empirical accuracy, demographic causality, population transfer, sizing or coincident district peaks. Final source/schema/link/hash checks are delivery_audit.json; exact deliverable identities are delivery_manifest.json. All worker tasks are complete; no background continuation, publication, submission, contact or generative-image use is claimed.
''',encoding='utf-8')

def manifest():
    paths=['ARTIFACT_INDEX.md','STATUS.md','agent_manifest.md',
       '01_plan/archetype_extension/task_dag.md','01_plan/archetype_extension/prose_gate.json',
       '01_plan/archetype_extension/final_fidelity_check.md','01_plan/archetype_extension/workflow_audit.json',
       '01_plan/archetype_extension/delivery_audit.json',
       '06_claim_drafts/article_claims_final.md','06_claim_drafts/final_acceptance.md',
       '06_claim_drafts/claim_ledger_v07.json','06_claim_drafts/claim_evidence_ledger.json',
       '07_reviews/archetype_extension/final_closure.md','07_reviews/archetype_extension/issue_register_final.json',
       '02_evidence/archetype_extension/local_sources.json','02_evidence/archetype_extension/registry_audit_v07.json',
       '02_evidence/archetype_extension/conference_reference_crosswalk.md',
       '08_manuscript/article_first_draft.md','08_manuscript/claim_to_prose_map.md','08_manuscript/fidelity_check.md',
       '08_manuscript/reference_source_map.txt']
    paths.extend(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'09_figures/archetype_extension').iterdir() if p.is_file())
    return {'date':'2026-10-20','delivery':'Requested revised first-manuscript workflow complete; empirical/benchmark/uncertainty evidence remains incomplete.',
       'claim_version':'v07','claims':106,'full_cycles':6,'extension_full_cycles':2,'last_revision_procedural_only':True,
       'new_annual_executions':152,'new_scientific_annual_executions':144,'new_technical_annual_executions':8,
       'prior_article_annual_executions':87,'total_article_annual_executions':239,'quarantined_prior_executions':66,
       'reviewer_diagnostic_annual_executions_excluded_from_article_total':5,
       'critical_literature_section_whitespace_words':1527,'manuscript_references':37,'literature_source_records':48,
       'original_conference_references_crosswalked':17,'paragraph_and_caption_ids':54,
       'ready_figures':['FIG-04','FIG-05'],'native_tables':['TAB-01','TAB-02'],'blocked_figures':['FIG-06'],
       'empirical_validation':False,'matched_independent_whole_building_benchmark':False,
       'requested_model_routing':{'claim_author':'gpt-6-sol','fresh_reviewers':'gpt-6-astra','specialists_prose_visualization':'gpt-6.1-sol'},
       'confirmed_backend_model_identity':'unknown','hashes':{p:sha(ROOT/p) for p in paths}}

dest=ROOT/'01_plan/archetype_extension/delivery_manifest.json'
dest.write_text(json.dumps(manifest(),indent=2),encoding='utf-8')
runpy.run_path(str(ROOT/'01_plan/archetype_extension/audit_extension.py'),run_name='__main__')
runpy.run_path(str(ROOT/'01_plan/archetype_extension/audit_delivery_extension.py'),run_name='__main__')
audit=json.loads(read(ROOT/'01_plan/archetype_extension/delivery_audit.json'))
for k in ('enum_errors','unresolved_source_aliases','missing_required_files','broken_delivery_links',
          'unknown_claim_ids_in_prose_map','prior_delivery_preservation_errors'):
    assert not audit[k], (k,audit[k])
assert all(audit[k] for k in ('current_index_ledger_hash_matches','ledger_draft_hash_matches','final_snapshot_matches_latest','extension_final_closure_exists'))
assert audit['full_cycles']==6
dest.write_text(json.dumps(manifest(),indent=2),encoding='utf-8')
print('Revised manuscript delivery sealed; source/schema/link/identity and preservation checks pass.')
