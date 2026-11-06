"""Track actual extension cycles and source/schema fidelity, not scientific truth."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
def read(p): return p.read_text(encoding='utf-8-sig')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def source_ids(v):
    found=set()
    if isinstance(v,dict):
        for k,x in v.items():
            if k in ('source_id','id') and isinstance(x,str): found.add(x)
            found.update(source_ids(x))
    elif isinstance(v,list):
        for x in v: found.update(source_ids(x))
    return found

drafts=[]
for n in (5,6,7):
    p=ROOT/f'06_claim_drafts/article_claims_v{n:02d}.md'
    if p.exists():
        ids=re.findall(r'^###\s+([A-Z]+-\d+)\b',read(p),re.M)
        drafts.append({'version':f'v{n:02d}','path':str(p.relative_to(ROOT)),'sha256':sha(p),'claims':len(ids),'duplicate_ids':sorted({x for x in ids if ids.count(x)>1})})

cycles=[]
for review,input_v,output_v in ((5,5,6),(6,6,7)):
    paths={'input':ROOT/f'06_claim_drafts/article_claims_v{input_v:02d}.md',
           'review':ROOT/f'07_reviews/archetype_extension/review_{review:02d}.md',
           'response':ROOT/f'07_reviews/archetype_extension/response_{review:02d}.md',
           'revision':ROOT/f'06_claim_drafts/article_claims_v{output_v:02d}.md',
           'register':ROOT/f'07_reviews/archetype_extension/issue_register_{review:02d}.json'}
    present={k:p.is_file() and p.stat().st_size>0 for k,p in paths.items()}
    cycles.append({'round':review,'input_version':input_v,'revision_version':output_v,
                   'files_present':present,'complete':all(present.values()),
                   'review_input_hash_recorded':present['input'] and present['review'] and sha(paths['input']) in read(paths['review']).lower(),
                   'hashes':{k:sha(p) for k,p in paths.items() if p.is_file()}})

registered=set()
for rel in ('02_evidence/sources.json','02_evidence/conference_sources.json','02_evidence/local_sources.json',
            '02_evidence/archetype_extension/sources.json','02_evidence/archetype_extension/local_sources.json'):
    p=ROOT/rel
    if p.is_file(): registered.update(source_ids(json.loads(read(p))))
ledger_check={}
for n in (7,6,5):
    p=ROOT/f'06_claim_drafts/claim_ledger_v{n:02d}.json'
    if not p.is_file(): continue
    ledger=json.loads(read(p)); claims=ledger['claims']
    if isinstance(claims,dict): claims=list(claims.values())
    allowed={'kind':{'literature','gap','novelty','method','result','interpretation','limitation'},
             'evidence_class':{'published finding','observed measurement','survey response','derived estimate','simulation output','hypothesis','planned test'},
             'status':{'supported','partially supported','disputed','unverified','planned'}}
    draft=ROOT/f'06_claim_drafts/article_claims_v{n:02d}.md'
    ledger_check={'version':n,'claims':len(claims),'draft_hash_matches':draft.exists() and sha(draft)==ledger.get('draft_sha256'),
                  'enum_errors':[{'claim':c['id'],'field':k,'value':c.get(k)} for c in claims for k,values in allowed.items() if c.get(k) not in values],
                  'unresolved_source_ids':sorted({x for c in claims for x in c.get('source_ids',[]) if x not in registered}),
                  'prior_v04_ids_retained':set(re.findall(r'^###\s+([A-Z]+-\d+)\b',read(ROOT/'06_claim_drafts/article_claims_v04.md'),re.M)).issubset({c['id'] for c in claims})}
    break
completed=sum(c['complete'] and c['review_input_hash_recorded'] for c in cycles)
report={'scope':'Structural/provenance audit only; not scientific or empirical validation.',
        'prior_full_cycles':4,'extension_full_cycles':completed,'total_full_cycles':4+completed,
        'pre_run_audit_counted_as_cycle':False,'drafts':drafts,'cycles':cycles,'latest_ledger':ledger_check,
        'extension_final_closure_exists':(ROOT/'07_reviews/archetype_extension/final_closure.md').is_file()}
dest=ROOT/'01_plan/archetype_extension/workflow_audit.json'
dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
