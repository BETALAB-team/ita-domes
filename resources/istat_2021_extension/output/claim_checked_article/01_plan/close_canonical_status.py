"""Post-review status annotations only; no frozen run/source/data edits."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
closure=ROOT/'07_reviews/final_closure.md'
assert closure.is_file()
files=['03_methods/climate_selection.md','03_methods/feasibility_audit.md',
       '03_methods/methods_execution_report.md','03_methods/simulation_protocol.md',
       '03_methods/validation_protocol.md','03_methods/validation_matrix.md',
       '04_analysis/simulation_results.md','05_simulations/SCIENTIFIC_USE_HOLD.md']
history=ROOT/'01_plan/coordination_history/pre_final_closure'
records=[]
replacements={
 'v5 CANDIDATE':'v5 — reviewed conditional model illustration',
 'fresh full-claim review 04 remains pending':'fresh full-claim review 04 and final v04 closure passed for the bounded conditional claims (07_reviews/final_closure.md)',
 'These are candidate results pending independent scientific review; no empirical accuracy is claimed.':'These conditional results have passed the bounded claim-review gate; no empirical accuracy is claimed.',
 'The fresh full-claim review 04 and manuscript gate remain separate pending tasks.':'The fresh full-claim review 04 and final v04 closure passed; four complete cycles permit an explicitly incomplete prose first draft.',
 'V5 is a candidate conditional illustration pending independent scientific review, not an empirically validated results paper.':'V5 is a reviewed conditional illustration. Independent benchmark and empirical validation remain missing.',
 'Independent scientific review of v5 is pending.':'Independent scientific review and final v04 closure are complete for the bounded v5 claims; this does not establish empirical validation.',
 'Status: CANDIDATE.':'Status: reviewed conditional model illustration.',
 'Use only current v5 data for candidate result claims.':'Use only current v5 data for the reviewed conditional result claims.',
}
for rel in files:
    p=ROOT/rel
    before=p.read_bytes()
    archived=history/rel
    archived.parent.mkdir(parents=True,exist_ok=True)
    if not archived.exists(): shutil.copy2(p,archived)
    s=before.decode('utf-8-sig')
    for a,b in replacements.items(): s=s.replace(a,b)
    s+='\nPost-review status annotation by the orchestrator: the saved final closure supersedes procedural pending-review labels in historical sections. Frozen protocols and run snapshots remain unchanged.\n'
    p.write_text(s,encoding='utf-8')
    records.append({'path':rel,'before_sha256':hashlib.sha256(before).hexdigest(),
                    'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                    'change_scope':'Current procedural status only; historical run snapshots, numerical tables and code unchanged.'})
(ROOT/'01_plan/canonical_status_closure_update.json').write_text(json.dumps({
    'closure_sha256':hashlib.sha256(closure.read_bytes()).hexdigest(),'files':records},indent=2),encoding='utf-8')
print('Updated current status annotations in '+str(len(records))+' files; previous versions preserved.')
