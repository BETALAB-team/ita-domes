"""Independent completed-site profile/chronology gate; no model execution."""
from pathlib import Path
import csv,json,hashlib,sys,time,argparse
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('SITE');p.add_argument('--campaign',required=True);a=p.parse_args();run=ROOT/'05_simulations'/a.campaign/a.SITE;pre=ROOT/'05_simulations/archetype_campaign_v2/pre_run'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
design=json.loads((run/'design.json').read_text());result=[]
for case,_ in design['cases']:
    pid=design['case_to_profile_id'][case]
    frozen=list(csv.DictReader((pre/f'{pid}.csv').open(newline='')))
    copied=list(csv.DictReader((run/f'{case}_input_profile.csv').open(newline='')))
    output=list(csv.DictReader((run/f'{case}_hourly.csv').open(newline='')))
    good=len(frozen)==len(copied)==len(output)==8760 and all(int(f['hour'])==int(c['hour'])==int(o['hour'])==i and float(f['lower_C'])==float(c['lower_C'])==float(o['heating_setpoint_C']) for i,(f,c,o) in enumerate(zip(frozen,copied,output)))
    result.append({'case':case,'profile_id':pid,'profile_csv_sha256':sha(pre/f'{pid}.csv'),'hourly_sha256':sha(run/f'{case}_hourly.csv'),'pass':good})
checks={filename:json.loads((run/filename).read_text())['pass'] for filename in ['checks.json','full_step_balance_checks.json','independent_surface_forcing_checks.json']}
profiles={'cases':result,'pass':all(r['pass'] for r in result),'checker_sha256':sha(__file__)}
(run/'profile_preservation_checks.json').write_text(json.dumps(profiles,indent=2),encoding='utf-8')
completion=json.loads((run/'completion_manifest.json').read_text())
checks['profile_preservation_checks.json']=profiles['pass'];checks['completion_manifest.json']=completion['status']=='complete' and completion['checks_pass']
gate={'site':a.SITE,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'campaign':a.campaign,'annual_executions':len(design['cases']),'scientific_executions':36,'technical_executions':2,'checks':checks,'pass':all(checks.values()),'expansion_permitted_if_Padova':a.SITE=='Padova' and all(checks.values()),'launch_gate_sha256':sha(ROOT/'01_plan/archetype_extension/launch_gate.json'),'selection_manifest_sha256':sha(pre/'selection_manifest.json'),'frozen_manifest_sha256':sha(run/'frozen_manifest.json'),'checker_sha256':sha(__file__),'diagnostic_note':'Historical failure.txt, if present, is a preserved source-drift preflight diagnostic; not an annual execution.'}
(run/'site_gate.json').write_text(json.dumps(gate,indent=2),encoding='utf-8');print(json.dumps(gate,indent=2));assert gate['pass']
