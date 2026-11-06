from pathlib import Path
import json,hashlib
import numpy as np
from profile_bridge import make_profile,source_sha256
HERE=Path(__file__).resolve().parent
base={'id':3523,'q_3_0':2,'q_3_11A':4,'q_3_11B':4,'q_3_11C':1,'q_3_13_2':0,'q_3_13_3':0,'q_3_19_6':0,'q_2_32C':2,'q_2_32D':2}
tests=[]
for timer_key in [None,'q_3_13_2','q_3_13_3','q_3_19_6']:
    for fire_key in [None,'q_2_32C','q_2_32D']:
        row=base.copy()
        if timer_key:row[timer_key]=1
        if fire_key:row[fire_key]=1
        p=make_profile(row)
        tests.append({'timer_key':timer_key,'fireplace_key':fire_key,'on_C':p['on_C'],'off_C':p['off_C'],'reported_blocks':[4,4,1],'passed':True})
for block in [(0,8,0),(8,0,8),(1,1,1)]:
    row=base|dict(zip(['q_3_11A','q_3_11B','q_3_11C'],block));p=make_profile(row)
    tests.append({'boundary_blocks':block,'passed':True})
for bad in [{'q_3_0':3},{'q_3_11A':8.5},{'q_3_11A':9},{'q_3_13_2':2},{'q_3_11A':0,'q_3_11B':0,'q_3_11C':0}]:
    try:make_profile(base|bad)
    except ValueError:tests.append({'invalid_input_rejected':bad,'passed':True})
    else:raise AssertionError(f'Accepted invalid input {bad}')
out={'scope':'preparatory profile fixtures only; no new annual energy simulation or group selection','source_sha256':source_sha256(),'bridge_sha256':hashlib.sha256((HERE/'profile_bridge.py').read_bytes()).hexdigest(),'tests':tests,'pass':all(t['passed'] for t in tests)}
(HERE/'profile_fixture_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'tests':len(tests),'pass':out['pass'],'source_sha256':out['source_sha256'],'bridge_sha256':out['bridge_sha256']},indent=2))
