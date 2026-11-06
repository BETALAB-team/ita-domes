"""Record the orchestrator's evidence-based prose gate after final closure."""
from pathlib import Path
import hashlib
import json
import runpy

ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'01_plan/audit_workflow.py'),run_name='__main__')
runpy.run_path(str(ROOT/'01_plan/audit_delivery.py'),run_name='__main__')
workflow=json.loads((ROOT/'01_plan/workflow_audit.json').read_text(encoding='utf-8-sig'))
delivery=json.loads((ROOT/'01_plan/delivery_audit.json').read_text(encoding='utf-8-sig'))
register=json.loads((ROOT/'07_reviews/issue_register_final.json').read_text(encoding='utf-8-sig'))
assert workflow['structurally_complete_cycles']==4
assert all(c['reviewed_draft_hash_recorded'] for c in workflow['cycles'])
assert delivery['final_snapshot_matches_latest'] and delivery['ledger_draft_hash_matches']
assert not delivery['enum_errors'] and not delivery['unresolved_source_aliases']
assert not any(e['still_open'] for e in register['entries'])
paths=['06_claim_drafts/article_claims_final.md','06_claim_drafts/final_acceptance.md',
       '06_claim_drafts/claim_ledger_v04.json','07_reviews/final_closure.md',
       '07_reviews/issue_register_final.json','02_evidence/local_sources.json',
       '02_evidence/local_registry_audit_v04.md','01_plan/canonical_status_closure_update.json']
record={'date':'2026-10-20','decision':'Open for a truthful incomplete first English prose manuscript.',
        'decision_owner':'/root orchestrator, following saved independent Astra final closure',
        'complete_cycles':4,'claim_count':41,
        'scope':'Bounded v5 deterministic model contrasts; one assumed dwelling and one respondent, three placements, four coordinate TMYs.',
        'remaining_scientific_gaps':['Independent matched whole-building benchmark','Compatible independent meter validation','Physical-input characterization and uncertainty ensemble','Quantified sky/solar-time/season sensitivity','Established broader novelty or population inference'],
        'administrative_dependencies':'Final registry and current canonical status refresh completed before dispatch.',
        'hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        'not_authorized':['Submission','Publication','External contact','New unsupported scientific claims','Generative images']}
(ROOT/'01_plan/prose_gate.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Prose gate sealed with four full cycles and final independent closure.')
