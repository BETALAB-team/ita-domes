"""Seal coordination and delivery records after prose/figure fidelity checks."""
from pathlib import Path
import hashlib
import json
import re
import runpy

ROOT=Path(__file__).resolve().parents[1]
def read(p): return p.read_text(encoding='utf-8-sig')

(ROOT/'STATUS.md').write_text('''# Delivery status — 20 October 2026

The requested claim-review-to-first-manuscript workflow is complete within the documented evidence limits. Four distinct serial claim-draft/independent-review/response/revision/register cycles and final Astra closure preceded separate prose and visualization agents. The accepted claim snapshot has 41 claims and is byte-identical to reviewed v04 (SHA-256 8a641ab68bd80fb96aad3016f3deea37ba0dbd4ab50ce9d984d8b59cfe50da7e). All ten review issues have bounded final dispositions. Review closure does not establish empirical validation or submission readiness.

New simulations were designed and executed: Padova first, then Bolzano, Roma and Palermo coordinate TMY exposures with unchanged assumed dwelling, one unweighted Veneto 2013 response, common seasonal mask and three constructed high-setpoint placements. Current v5 contains 20 annual cases:12 primary comparisons, four deterministic repeats and four longer-warm-up checks. One separate seeded replay matches exactly. All 66 earlier executions remain quarantined and preserved. Total:87 article annual executions. Five reviewer diagnostic annual reproductions are separate. External checkouts and environments were read in place and not modified.

The admitted results concern useful sensible heating and maximum hourly load for one assumed 80 m² dwelling. Encoded 4/4/1 hours at 22°C with 19°C setback do not fix heater runtime. Post-hoc setback heating and Bolzano's shared season-start maximum are labelled. No city/national representation, empirical accuracy, savings, equipment sizing, new schedule-generation method or causal weather-variable attribution is established. Native annual-mean sky exchange, solar-time approximation, fixed season, uncharacterized physical inputs and missing uncertainty ensemble remain limitations.

An independent matched whole-building benchmark and compatible independently documented meter validation remain unperformed. Candidate Padova annual attributes require provenance, units, period, boundary and independence checks before use. The manuscript and FIG-03 retain explicit informative dependencies; no empirical curve or error statistic was invented. No further work is represented as running after this handoff.

Delivered: English article with two integrated scientific figures; paragraph/caption-to-claim map; complete figure briefs; editable plotting code/data, SVGs, 400 dpi PNGs and visual QA; all four reviews/responses/revisions/registers; final closure; conference/extension/source inventories; simulation and validation protocols. ARTIFACT_INDEX.md links every deliverable. The final source registry audited205/205expected hashes with zero missing/unresolved aliases. Final delivery audit reports no missing required files, broken delivery links, enum errors or unresolved aliases.

Actual model-selection arguments: gpt-6-sol claim author; fresh gpt-6-astra reviewers in each round; gpt-6.1-sol evidence/literature/methods workers, separate prose author and separate visualization worker. Selection was supported and accepted by tools; independently confirmed backend identity/version remains unknown. See agent_manifest.md. No generative-image tools, publication, submission or external contact occurred.

Root inspected manuscript text, claim correspondence and both rendered figures, including the corrected publication-width preview. Prose/figure fidelity and administrative audits are separate from scientific validation. All historical snapshots and transparent corrections remain preserved. No user question is pending.
''',encoding='utf-8')

p=ROOT/'agent_manifest.md'
s=read(p).replace('| Active |','| Delivery complete |')
s=s.replace('| Active after four complete cycles and final closure |','| Complete; two figures integrated and final fidelity sealed |')
s=s.replace('| Active: two ready figures; empirical figure blocked |','| Complete; two rendered figures with QA; empirical figure remains blocked |')
s+='\n## Final delivery seal\n\nThe separate prose author integrated both ready figures and sealed article/map/fidelity identities. The visualization owner corrected the Bolzano annotation to all three profiles, re-rendered and checked it. Root viewed both full PNGs and the corrected 900 px preview. All required artifact/link/schema checks pass; remaining benchmark, measurement and uncertainty gaps remain explicit. No worker task is claimed to continue after delivery.\n'
p.write_text(s,encoding='utf-8')

p=ROOT/'ARTIFACT_INDEX.md'
s=read(p).replace('Working date: 20 October 2026. This index is being assembled during the workflow; final status is recorded in [STATUS.md](STATUS.md). No missing empirical validation is implied by a completed review cycle.',
 'Delivery date:20 October 2026. Four complete claim-review-revision cycles and final closure preceded the English first draft and separate scientific figures. [STATUS.md](STATUS.md) records scope and remaining evidence gaps. The independent benchmark and empirical validation remain missing.')
s=s.replace('Current candidate v5 simulation results','Reviewed conditional v5 simulation results')
s=s.replace('Technical review03 passed; fresh full-claim review04 is underway.','Technical and full-claim review04/final closure passed within the bounded scope; no empirical validation follows.')
s=s.replace('The following later-stage links become deliverables only after their gates complete; consult STATUS.md while work is active.',
 'All four serial cycles and final closure are complete. Frozen stage labels are explained by the dated final acceptance note; missing empirical content remains explicit.')
s=s.replace('- [Figure briefs](09_figures/figure_briefs.md) and [figure/source/script/QA manifest](09_figures/figure_manifest.md).',
 '- [Figure briefs](09_figures/figure_briefs.md) and [figure/source/script/QA manifest](09_figures/figure_manifest.md).\n- [FIG-01 paired endpoint contrasts, PNG](09_figures/FIG-01_paired_contrasts.png), [SVG](09_figures/FIG-01_paired_contrasts.svg), [FIG-02 setback diagnostic, PNG](09_figures/FIG-02_setback_diagnostic.png), and [SVG](09_figures/FIG-02_setback_diagnostic.svg).\n- [Editable figure script](09_figures/render_figures.py), [figure numerical/provenance QA](09_figures/figure_provenance_qa.json), [prose fidelity check](08_manuscript/fidelity_check.md), and [root final fidelity record](01_plan/final_fidelity_check.md).\n- [Final delivery link/schema audit](01_plan/delivery_audit.json) and [delivery identities](01_plan/delivery_manifest.json). FIG-03 remains blocked without fabricated data or image.')
p.write_text(s,encoding='utf-8')

p=ROOT/'01_plan/task_dag.md'
s=read(p).replace('Separate prose author active.','Prose, figures and final fidelity delivered.')
statuses={'D1':'Complete; separate prose agent and final integration',
          'D2':'Complete; separate visualization worker, SVG/400dpiPNG/code/data/QA; empirical figure blocked',
          'D3':'Complete; final fidelity, links, statuses and identities checked'}
out=[]
for line in s.splitlines():
    if line.startswith('| '):
        cells=line.split('|')
        for k,v in statuses.items():
            if cells[1].strip().startswith(k+' '):
                cells[-2]=' '+v+' '
                line='|'.join(cells)
                break
    out.append(line)
p.write_text('\n'.join(out)+'\n',encoding='utf-8')
p=ROOT/'01_plan/readiness_checklist.md'
s=read(p).replace('- [ ] Prose, paragraph map, complete figure briefs and separate visualization outputs delivered and checked together.',
 '- [x] Prose, paragraph map, complete figure briefs and separate visualization outputs delivered and checked together; FIG-03 remains explicitly blocked.')
p.write_text(s,encoding='utf-8')

article=read(ROOT/'08_manuscript/article_first_draft.md')
mapping=read(ROOT/'08_manuscript/claim_to_prose_map.md')
ids=re.findall(r'<!-- (P\d+|CAP\d+) -->',article)
assert len(ids)==26 and len(set(ids))==26 and all(i in mapping for i in ids)
assert 'FIG-03 BLOCKED' in article
assert not (ROOT/'09_figures/FIG-03_empirical_comparison.png').exists()
assert (ROOT/'06_claim_drafts/article_claims_final.md').read_bytes()==(ROOT/'06_claim_drafts/article_claims_v04.md').read_bytes()

(ROOT/'01_plan/final_fidelity_check.md').write_text('''# Orchestrator final fidelity and delivery check

20 October 2026. This is an artifact/fidelity check, not empirical validation or a new independent scientific review.

Root read the full prose manuscript, paragraph map, accepted claim blocks, review04 and final closure. Padova energy/peak values, four-site signed same-site percentages, setback hours/heat, Bolzano season-start context and model/validation limitations agree with the admitted claims. The372eligible-record count is explicitly in DAT-001 provenance/counterevidence. All23substantive paragraphs and three captions have unique mapped IDs. No new substantive claim requiring a further claim cycle was identified.

Root viewed both full rendered PNG figures and the final900pxFIG-01 preview. Signed values, two separate endpoint units, zero Bolzano contrasts, deterministic alternatives, same-site denominator and post-hoc setback labels agree with the claims and prose. The annotation now explicitly covers all three Bolzano profiles. Labels, marker shapes, zero reference, spacing and scope notes are legible without clipping. The separate worker records raw-file arithmetic and rendering QA in figure_manifest.md/figure_provenance_qa.json. Root did not rerun annual simulations during final fidelity checking.

The prose author integrated both absolute PNG paths and kept the stable brief references/captions. SVG, editable code, exact plotting data and source hashes are delivered. FIG-03 remains blocked with an informative dependency list; no measured curve, error metric or proxy ground truth was fabricated. Missing independent benchmark, meter validation, full physical characterization and uncertainty sensitivities remain explicit.

Four complete serial cycles have distinct draft hashes and saved reviews/responses/revisions/registers. Final Astra closure precedes prose_gate.json and the separate prose/visualization handoffs. Final accepted claims remain byte-identical to v04. Historical outputs and response-hash correction are preserved transparently. Requested model routing is recorded; unexposed backend identity remains unknown. No generative-image tool, external contact, publication or submission occurred.

Final structural audit is delivery_audit.json; exact delivery identities are delivery_manifest.json. Successful structural and fidelity checks do not establish measured accuracy or broader novelty.
''',encoding='utf-8')

paths=['ARTIFACT_INDEX.md','STATUS.md','agent_manifest.md','01_plan/final_fidelity_check.md',
       '01_plan/prose_gate.json','06_claim_drafts/article_claims_final.md',
       '07_reviews/final_closure.md','07_reviews/issue_register_final.json',
       '08_manuscript/article_first_draft.md','08_manuscript/claim_to_prose_map.md','08_manuscript/fidelity_check.md',
       '09_figures/figure_briefs.md','09_figures/figure_manifest.md','09_figures/render_figures.py',
       '09_figures/figure_provenance_qa.json','09_figures/FIG-01_paired_contrasts.png','09_figures/FIG-01_paired_contrasts.svg',
       '09_figures/FIG-01_plotting_data.csv','09_figures/FIG-02_setback_diagnostic.png','09_figures/FIG-02_setback_diagnostic.svg',
       '09_figures/FIG-02_plotting_data.csv','02_evidence/local_sources.json','02_evidence/local_source_ledger.md']
manifest={'date':'2026-10-20','delivery':'Incomplete scientific first manuscript; requested workflow completed.',
          'full_cycles':4,'claims':41,'current_campaign_annuals':20,'separate_replay_annuals':1,
          'quarantined_annuals':66,'total_article_annual_executions':87,
          'empirical_validation':False,'independent_whole_building_benchmark':False,
          'ready_figures':['FIG-01','FIG-02'],'blocked_figures':['FIG-03'],
          'hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
(ROOT/'01_plan/delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
runpy.run_path(str(ROOT/'01_plan/audit_workflow.py'),run_name='__main__')
runpy.run_path(str(ROOT/'01_plan/audit_delivery.py'),run_name='__main__')
audit=json.loads(read(ROOT/'01_plan/delivery_audit.json'))
assert all(not audit[k] for k in ('enum_errors','unresolved_source_aliases','missing_required_files','broken_delivery_links'))
assert audit['final_snapshot_matches_latest']
print('Final delivery sealed; no missing required artifacts or broken delivery links.')
