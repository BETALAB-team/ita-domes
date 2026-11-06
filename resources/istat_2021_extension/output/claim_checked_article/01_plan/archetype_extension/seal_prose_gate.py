"""Seal the extension prose dispatch only after actual reviews and final closure."""
from pathlib import Path
import hashlib
import json
import runpy

ROOT=Path(__file__).resolve().parents[2]
def read(p): return p.read_text(encoding='utf-8-sig')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

runpy.run_path(str(ROOT/'01_plan/archetype_extension/audit_extension.py'),run_name='__main__')
audit=json.loads(read(ROOT/'01_plan/archetype_extension/workflow_audit.json'))
assert audit['extension_full_cycles']==2 and audit['total_full_cycles']==6
assert all(c['review_input_hash_recorded'] for c in audit['cycles'])
assert audit['extension_final_closure_exists']
assert audit['latest_ledger']['version']==7
assert audit['latest_ledger']['draft_hash_matches']
assert not audit['latest_ledger']['enum_errors']
assert not audit['latest_ledger']['unresolved_source_ids']
final=ROOT/'06_claim_drafts/article_claims_final.md'
v07=ROOT/'06_claim_drafts/article_claims_v07.md'
assert final.read_bytes()==v07.read_bytes()
ledger=json.loads(read(ROOT/'06_claim_drafts/claim_evidence_ledger.json'))
assert ledger['version']=='v07' and ledger['draft_sha256']==sha(v07)
issues=json.loads(read(ROOT/'07_reviews/archetype_extension/issue_register_final.json'))
assert not any(e.get('still_open',False) for e in issues['entries'])
registry=json.loads(read(ROOT/'02_evidence/archetype_extension/registry_audit_v07.json'))
assert all(registry['counts'][k]==0 for k in ('hash_mismatch_count','missing_file_count','unresolved_alias_count'))
assert registry['claim_ledger']['sha256']==sha(ROOT/'06_claim_drafts/claim_ledger_v07.json')
assert registry['claim_ledger']['draft_sha256']==sha(v07)
assert registry['registry_sha256']==sha(ROOT/'02_evidence/archetype_extension/local_sources.json')
assert registry['ledger_sha256']==sha(ROOT/'02_evidence/archetype_extension/local_source_ledger.md')
paths=['06_claim_drafts/article_claims_final.md','06_claim_drafts/final_acceptance.md',
       '06_claim_drafts/claim_ledger_v07.json','06_claim_drafts/claim_evidence_ledger.json',
       '07_reviews/archetype_extension/review_06.md','07_reviews/archetype_extension/response_06.md',
       '07_reviews/archetype_extension/issue_register_06.json','07_reviews/archetype_extension/final_closure.md',
       '07_reviews/archetype_extension/issue_register_final.json',
       '02_evidence/archetype_extension/local_sources.json','02_evidence/archetype_extension/registry_audit_v07.json',
       '01_plan/archetype_extension/prose_and_figure_handoff.md']
record={'date':'2026-10-20','decision_owner':'/root orchestrator following independent final closure',
        'decision':'Open the separate prose and visualization workflow for a truthful incomplete scientific first manuscript.',
        'prior_full_cycles':4,'extension_full_cycles':2,'total_full_cycles':6,
        'claim_count':audit['latest_ledger']['claims'],'claim_version':'v07',
        'closure_read_and_assessed_by_root':True,
        'scope':'Four archived demographic groups condition mixtures of four real heating-control profiles; bounded within-model held-out comparison in one assumed dwelling and four TMYs.',
        'remaining_scientific_gaps':['Matched independent whole-building benchmark','Compatible independently documented meter validation',
                                   'Broader physical-input/assignment/split uncertainty','Quantified sky/solar-time/season sensitivity',
                                   'Exact remote published assignment identity','Population and city representativeness'],
        'hashes':{p:sha(ROOT/p) for p in paths},
        'prohibited':['Unsupported new scientific claims','Generative images','Publication','Submission','External contact']}
(ROOT/'01_plan/archetype_extension/prose_gate.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Extension prose gate sealed after six actual full cycles and final independent closure.')
