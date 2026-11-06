from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parents[2]
read=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256((root/p).read_bytes()).hexdigest()
a=read('06_claim_drafts/claim_ledger_v03.json')
b=read('06_claim_drafts/claim_ledger_v04.json')
current=read('06_claim_drafts/claim_evidence_ledger.json')
text=(root/'06_claim_drafts/article_claims_v04.md').read_text(encoding='utf-8')
ids=set()
for file in ['02_evidence/sources.json','02_evidence/local_sources.json','02_evidence/conference_sources.json']:
 for s in read(file)['sources']:ids.add(s.get('id',s.get('source_id')))
old={c['id']:c for c in a['claims']}
out={'hashes':{p:sha(p) for p in ['06_claim_drafts/article_claims_v04.md','06_claim_drafts/claim_ledger_v04.json','06_claim_drafts/claim_evidence_ledger.json','07_reviews/response_04.md','07_reviews/issue_register_04.json','07_reviews/review_04.md','07_reviews/prior_issue_closure_after_review04.md']},
'count':len(b['claims']),'same_ids':set(old)=={c['id'] for c in b['claims']},'exact_claims_match_markdown':all(c['exact_claim'] in text for c in b['claims']),'enums':{k:sorted(set(c[k] for c in b['claims'])) for k in ['kind','evidence_class','status']},
'unresolved_aliases':sorted(set(x for c in b['claims'] for x in c['source_ids'])-ids),
'changed_fields':{c['id']:[k for k in c if c[k]!=old[c['id']].get(k)] for c in b['claims'] if c!=old[c['id']]},
'changed_exact_claims':{c['id']:{'before':old[c['id']]['exact_claim'],'after':c['exact_claim']} for c in b['claims'] if c['exact_claim']!=old[c['id']]['exact_claim']},
'current_index_correct':current['draft_sha256']==sha('06_claim_drafts/article_claims_v04.md') and current['claim_ledger_sha256']==sha('06_claim_drafts/claim_ledger_v04.json'),
'result_numbers_same':{c['id']:re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',c['exact_claim'])==re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',old[c['id']]['exact_claim']) for c in b['claims'] if c['id'].startswith('RES-')}}
out['cycles']=[]
for i in range(1,5):
 paths=[f'06_claim_drafts/article_claims_v{i-1:02}.md',f'07_reviews/review_{i:02}.md',f'07_reviews/response_{i:02}.md',f'06_claim_drafts/article_claims_v{i:02}.md',f'07_reviews/issue_register_{i:02}.json']
 out['cycles'].append({'round':i,'all_files_exist':all((root/p).exists() for p in paths),'draft_before':sha(paths[0]),'draft_after':sha(paths[3]),'different_revision':sha(paths[0])!=sha(paths[3])})
(Path(__file__).parent/'final_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(out,indent=2,ensure_ascii=True))
