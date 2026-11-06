"""Audit persisted workflow artifacts; existence/hashes do not certify scientific truth."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    drafts = []
    prior_ids = set()
    for p in sorted((ROOT / '06_claim_drafts').glob('article_claims_v[0-9][0-9].md')):
        ids = re.findall(r'^###\s+([A-Z]+-\d+)\b', p.read_text(encoding='utf-8-sig'), re.M)
        drafts.append({'file': str(p.relative_to(ROOT)), 'sha256': sha(p),
                       'claim_count': len(ids), 'duplicate_ids': sorted({x for x in ids if ids.count(x) > 1}),
                       'ids_absent_since_prior_version': sorted(prior_ids - set(ids))})
        prior_ids = set(ids)
    cycles = []
    for n in range(1, 5):
        # Canonical filenames are specified in author/reviewer packets.
        paths = {'review': ROOT / '07_reviews' / f'review_{n:02d}.md',
                 'response': ROOT / '07_reviews' / f'response_{n:02d}.md',
                 'revised_draft': ROOT / '06_claim_drafts' / f'article_claims_v{n:02d}.md',
                 'register_snapshot': ROOT / '07_reviews' / f'issue_register_{n:02d}.json'}
        exists = {k: p.is_file() and p.stat().st_size > 0 for k, p in paths.items()}
        reviewed = ROOT / '06_claim_drafts' / f'article_claims_v{n-1:02d}.md'
        reviewed_hash_recorded = (sha(reviewed) in paths['review'].read_text(encoding='utf-8-sig').lower()
                                  if reviewed.is_file() and paths['review'].is_file() else False)
        cycles.append({'cycle': n, 'artifacts_present': exists,
                       'artifact_complete': all(exists.values()),
                       'reviewed_draft_hash_recorded': reviewed_hash_recorded,
                       'hashes': {k: sha(p) for k, p in paths.items() if p.is_file()}})
    latest_ledger = ROOT / '06_claim_drafts' / 'claim_evidence_ledger.json'
    ledger_check = {'exists': latest_ledger.is_file()}
    if latest_ledger.is_file():
        ledger = json.loads(latest_ledger.read_text(encoding='utf-8-sig'))
        linked_draft = latest_ledger.parent / ledger.get('draft_filename', '')
        ledger_check.update({'draft_filename': ledger.get('draft_filename'),
                             'draft_hash_matches': linked_draft.is_file() and sha(linked_draft) == ledger.get('draft_sha256'),
                             'claim_count': len(ledger.get('claims', {}))})
    report = {'artifact_root': str(ROOT), 'scope': 'Structural/provenance audit only; does not evaluate scientific adequacy.',
              'drafts': drafts, 'cycles': cycles,
              'latest_claim_ledger_check': ledger_check,
              'structurally_complete_cycles': sum(x['artifact_complete'] for x in cycles),
              'final_claims_exists': (ROOT/'06_claim_drafts'/'article_claims_final.md').is_file(),
              'closure_exists': (ROOT/'07_reviews'/'final_closure.md').is_file(),
              'prose_exists': (ROOT/'08_manuscript'/'article_first_draft.md').is_file()}
    dest = ROOT / '01_plan' / 'workflow_audit.json'
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'drafts': len(drafts), 'structurally_complete_cycles': report['structurally_complete_cycles'], 'report': str(dest)}))

if __name__ == '__main__':
    main()
