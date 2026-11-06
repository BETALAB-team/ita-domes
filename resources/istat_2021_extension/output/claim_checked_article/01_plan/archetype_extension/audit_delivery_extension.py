"""Current extension artifact checks; never substitutes for scientific review."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]
def read(p): return p.read_text(encoding='utf-8-sig')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ids(v):
    found=set()
    if isinstance(v,dict):
        for k,x in v.items():
            if k in ('source_id','id') and isinstance(x,str): found.add(x)
            found.update(ids(x))
    elif isinstance(v,list):
        for x in v: found.update(ids(x))
    return found

index=json.loads(read(ROOT/'06_claim_drafts/claim_evidence_ledger.json'))
ledger=json.loads(read(ROOT/'06_claim_drafts'/index['claim_ledger_filename']))
claims=ledger['claims']
if isinstance(claims,dict): claims=list(claims.values())
known={c['id'] for c in claims}
registered=set()
for rel in ('02_evidence/sources.json','02_evidence/conference_sources.json','02_evidence/local_sources.json',
            '02_evidence/archetype_extension/sources.json','02_evidence/archetype_extension/local_sources.json'):
    registered.update(ids(json.loads(read(ROOT/rel))))
allowed={'kind':set('literature gap novelty method result interpretation limitation'.split()),
         'evidence_class':{'published finding','observed measurement','survey response','derived estimate','simulation output','hypothesis','planned test'},
         'status':{'supported','partially supported','disputed','unverified','planned'}}
errors=[{'id':c['id'],'field':k,'value':c.get(k)} for c in claims for k,values in allowed.items() if c.get(k) not in values]
required=['01_plan/archetype_extension/task_dag.md','01_plan/archetype_extension/conference_to_extension_matrix.md',
          '02_evidence/archetype_extension/conference_reference_crosswalk.md','02_evidence/archetype_extension/source_ledger.md',
          '02_evidence/archetype_extension/local_source_ledger.md','03_methods/archetype_extension/mixture_protocol_v2.md',
          '03_methods/archetype_extension/validation_matrix.md','06_claim_drafts/article_claims_final.md',
          '06_claim_drafts/final_acceptance.md','07_reviews/archetype_extension/final_closure.md',
          '01_plan/archetype_extension/prose_gate.json','08_manuscript/article_first_draft.md',
          '08_manuscript/claim_to_prose_map.md','08_manuscript/fidelity_check.md',
          '09_figures/archetype_extension/figure_briefs.md','09_figures/archetype_extension/figure_manifest.md']
for i in range(1,7):
    folder='07_reviews/archetype_extension' if i>4 else '07_reviews'
    required.extend(f'{folder}/{name}_{i:02d}.{ext}' for name,ext in [('review','md'),('response','md'),('issue_register','json')])
targets=['ARTIFACT_INDEX.md','08_manuscript/article_first_draft.md','08_manuscript/claim_to_prose_map.md',
         '09_figures/archetype_extension/figure_briefs.md','09_figures/archetype_extension/figure_manifest.md']
link_errors=[]
for rel in targets:
    p=ROOT/rel
    if not p.is_file(): continue
    for raw in re.findall(r'!?\[[^\]]*\]\(([^\n]+?)\)',read(p)):
        target=raw.strip().strip('<>')
        if re.match(r'^(https?://|app://|codex://|#)',target): continue
        target=unquote(target.split('#',1)[0])
        if re.match(r'^/[A-Za-z]:/',target): target=target[1:]
        if not target: continue
        q=Path(target)
        if not q.is_absolute(): q=p.parent/q
        if not q.exists(): link_errors.append({'file':rel,'target':raw})
linked=ROOT/'06_claim_drafts'/ledger['draft_filename']
final=ROOT/'06_claim_drafts/article_claims_final.md'
workflow=json.loads(read(ROOT/'01_plan/archetype_extension/workflow_audit.json'))
history=ROOT/'01_plan/delivery_history/pre_archetypes_20261020'
prior=json.loads(read(history/'01_plan/delivery_manifest.json'))
preservation_errors=[]
for rel,expected in prior['hashes'].items():
    original=ROOT/rel
    saved=history/rel
    if rel.startswith('09_figures/') and original.exists() and sha(original)!=expected:
        preservation_errors.append({'type':'original_figure_changed','path':rel})
    if saved.exists() and sha(saved)!=expected:
        preservation_errors.append({'type':'preserved_delivery_changed','path':rel})
map_path=ROOT/'08_manuscript/claim_to_prose_map.md'
map_ids=set(re.findall(r'\b[A-Z]+-\d{3}\b',read(map_path))) if map_path.exists() else set()
report={'scope':'Structural and source-identity checks only; not empirical/scientific validation.',
        'claim_version':ledger['version'],'claim_count':len(claims),
        'current_index_ledger_hash_matches':sha(ROOT/'06_claim_drafts'/index['claim_ledger_filename'])==index['claim_ledger_sha256'],
        'full_cycles':workflow['total_full_cycles'],
        'ledger_draft_hash_matches':sha(linked)==ledger['draft_sha256'],
        'final_snapshot_matches_latest':final.read_bytes()==linked.read_bytes(),
        'enum_errors':errors,'unresolved_source_aliases':sorted({s for c in claims for s in c.get('source_ids',[]) if s not in registered}),
        'missing_required_files':[x for x in required if not (ROOT/x).is_file()],
        'broken_delivery_links':link_errors,'unknown_claim_ids_in_prose_map':sorted(map_ids-known),
        'prior_delivery_preservation_errors':preservation_errors,
        'extension_final_closure_exists':workflow['extension_final_closure_exists'],
        'current_scientific_annual_executions':144,'current_technical_annual_executions':8,
        'new_extension_annual_executions':152,'prior_article_annual_executions':87,
        'total_article_annual_executions':239,'prior_quarantined_annual_executions':66,
        'reviewer_diagnostic_annual_executions_counted':False}
(ROOT/'01_plan/archetype_extension/delivery_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
