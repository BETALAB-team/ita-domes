"""Structural delivery checks, distinct from scientific review and validation."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
def read(p): return p.read_text(encoding='utf-8-sig')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def find_ids(value):
    found=set()
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('source_id','id') and isinstance(v,str): found.add(v)
            found.update(find_ids(v))
    elif isinstance(value,list):
        for v in value: found.update(find_ids(v))
    return found

ledger=json.loads(read(ROOT/'06_claim_drafts/claim_evidence_ledger.json'))
claims=ledger['claims']
if isinstance(claims,dict): claims=list(claims.values())
allowed={'kind':set('literature gap novelty method result interpretation limitation'.split()),
         'evidence_class':{'published finding','observed measurement','survey response','derived estimate','simulation output','hypothesis','planned test'},
         'status':{'supported','partially supported','disputed','unverified','planned'}}
enum_errors=[{'id':c['id'],'field':k,'value':c.get(k)} for c in claims for k,values in allowed.items() if c.get(k) not in values]
registered=set()
for name in ('02_evidence/sources.json','02_evidence/conference_sources.json','02_evidence/local_sources.json'):
    p=ROOT/name
    if p.exists(): registered.update(find_ids(json.loads(read(p))))
unresolved=sorted({s for c in claims for s in c.get('source_ids',[]) if s not in registered})
required=['01_plan/task_dag.md','00_context/extension_matrix.md','02_evidence/source_ledger.md',
          '02_evidence/local_source_ledger.md','03_methods/simulation_protocol.md','03_methods/climate_selection.md',
          '03_methods/validation_protocol.md','03_methods/validation_matrix.md','06_claim_drafts/article_claims_final.md',
          '07_reviews/final_closure.md','08_manuscript/article_first_draft.md','08_manuscript/claim_to_prose_map.md',
          '09_figures/figure_briefs.md','09_figures/figure_manifest.md']
for i in range(1,5):
    required.extend([f'07_reviews/review_{i:02d}.md',f'07_reviews/response_{i:02d}.md',f'07_reviews/issue_register_{i:02d}.json'])
link_errors=[]
for rel in ['ARTIFACT_INDEX.md','08_manuscript/article_first_draft.md','08_manuscript/claim_to_prose_map.md','09_figures/figure_briefs.md','09_figures/figure_manifest.md']:
    p=ROOT/rel
    if not p.exists(): continue
    for raw in re.findall(r'!?\[[^\]]*\]\(([^\n]+?)\)',read(p)):
        target=raw.strip().strip('<>')
        if re.match(r'^(https?://|app://|codex://|#)',target): continue
        target=unquote(target.split('#',1)[0])
        if not target: continue
        # Markdown paths may use /C:/... in this Windows-hosted task.
        if re.match(r'^/[A-Za-z]:/',target): target=target[1:]
        q=Path(target)
        if not q.is_absolute(): q=p.parent/q
        if not q.exists(): link_errors.append({'file':rel,'target':raw})
linked=ROOT/'06_claim_drafts'/ledger['draft_filename']
final=ROOT/'06_claim_drafts/article_claims_final.md'
report={'scope':'Structural checks only; no claim of empirical or scientific validation.',
        'claim_version':ledger.get('version'),'claim_count':len(claims),
        'ledger_draft_hash_matches':sha(linked)==ledger['draft_sha256'],
        'final_snapshot_matches_latest':final.exists() and final.read_bytes()==linked.read_bytes(),
        'enum_errors':enum_errors,'unresolved_source_aliases':unresolved,
        'missing_required_files':[x for x in required if not (ROOT/x).is_file()],
        'broken_delivery_links':link_errors}
(ROOT/'01_plan/delivery_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
