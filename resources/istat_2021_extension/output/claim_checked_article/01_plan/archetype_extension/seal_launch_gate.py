"""Seal root's annual-launch decision against the exact audited pre-run state."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
expected={
 '03_methods/archetype_extension/mixture_protocol_v2.md':'0abd5e8952bd9021d4bc8c8ef6a0649b1c14ffc4c467e5b1da7c13dc018633ba',
 '05_simulations/archetype_campaign_v2/pre_run/selection_manifest.json':'a34a21e694b1c0117e915567c0eab56f83bfa3aa03b586ab6443006e0b8daed4',
 '05_simulations/archetype_campaign_v2/pre_run/runtime_source_manifest.json':'bf1eb3abaf62373885533464e954a79088b7fbe4b9b98bc8a1fd8f41e4d23340',
 '04_analysis/archetype_extension/run_mixture_campaign.py':'2c435e83aa14b073078ccd27ea9585079a8e3ff1225dbf53d2f5c76db3413a08',
 '04_analysis/archetype_extension/analyze_mixtures.py':'07fb6f014aa8ffe84e3cb72669b07fa1012827b6db05eeac9168bc821f7285fc'}
for rel,want in expected.items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==want,rel
audit=ROOT/'07_reviews/archetype_extension/pre_run_audit/v2_pre_run_audit.md'
assert audit.is_file()
readiness=json.loads((ROOT/'05_simulations/archetype_campaign_v2/pre_run/readiness.json').read_text(encoding='utf-8-sig'))
assert readiness['pass'] and not readiness['annual_execution']
assert readiness['total_planned']==152
for p in (ROOT/'05_simulations/archetype_campaign_v2').glob('*/*_hourly.csv'):
    raise RuntimeError('Annual output exists before launch seal: '+str(p))
decision={'date':'2026-10-20','owner':'/root orchestrator','launch_approved':True,
 'authorization_basis':'Human request to implement user archetypes and run the research workflow; saved independent exact-state pre-run audit and root source/hash checks.',
 'scope':'Padova38 annual cases first; expand to Bolzano/Roma/Palermo only after its saved numerical, full-state and surface-forcing gates pass.',
 'planned_counts':{'detailed_test_respondents':32,'shared_component_profiles':4,'scientific_annuals':144,'technical_annuals':8,'total_annuals':152},
 'conditions':['No code/input/protocol/split/selection/threshold changes after this seal without declared amendment and appropriate recheck.',
 'No outcome-based resampling, prototype replacement or tuning.',
 'Preserve all raw outputs, failed checks and exact commands; stop expansion if a required gate fails.',
 'Aggregate only with exact frozen code/manifests and independently recompute endpoints/mixtures/losses.',
 'New results remain candidate until fresh full scientific claim review and revision/closure.'],
 'frozen_sha256':expected,
 'audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),
 'readiness_sha256':hashlib.sha256((ROOT/'05_simulations/archetype_campaign_v2/pre_run/readiness.json').read_bytes()).hexdigest(),
 'v1_status':'Preserved input-only prototype collapse; no v1 annual execution.'}
dest=ROOT/'01_plan/archetype_extension/launch_gate.json'
assert not dest.exists(),'Do not overwrite launch seal'
dest.write_text(json.dumps(decision,indent=2),encoding='utf-8')
print('Root launch gate sealed for exact audited v2; Padova first.')
