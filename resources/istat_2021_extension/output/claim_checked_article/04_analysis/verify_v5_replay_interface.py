from pathlib import Path
import json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'05_simulations/climate_campaign_v5/Padova'
def fingerprints():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.iterdir() if p.is_file()}
before=fingerprints()
command=[sys.executable,'-B',str(ROOT/'04_analysis/run_climate_campaign_v5.py'),'Padova','--campaign','climate_campaign_v5']
refusal=subprocess.run(command,capture_output=True,text=True)
after=fingerprints()
record={'test':'existing completed destination is rejected without modifying any file','command_argv':command,'return_code':refusal.returncode,'stderr':refusal.stderr,'all_before_after_file_hashes_equal':before==after,'file_count':len(before),'pass':refusal.returncode!=0 and 'Refusing to overwrite' in refusal.stderr and before==after}
base=json.loads((RUN/'summary.json').read_text())[0]
replay=ROOT/'05_simulations/climate_campaign_v5_replay/Padova'
replaycase=json.loads((replay/'summary.json').read_text())[0]
record['separate_genuine_annual_replay']={'campaign':'climate_campaign_v5_replay','case':'seeded','hours':8760,'not_an_additional_scientific_sample':True,'byte_identical_hourly_output':base['hourly_sha256']==replaycase['hourly_sha256'],'independent_full_step_gate_pass':json.loads((replay/'full_step_balance_checks.json').read_text())['pass']}
record['pass']=record['pass'] and all(record['separate_genuine_annual_replay'][k] for k in ['byte_identical_hourly_output','independent_full_step_gate_pass'])
(ROOT/'05_simulations/v5_replay_interface_checks.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
assert record['pass']
