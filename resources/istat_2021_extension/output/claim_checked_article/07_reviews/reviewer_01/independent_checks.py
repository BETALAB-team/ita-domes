from pathlib import Path
import csv,json,hashlib,math,ast,random
import numpy as np
root=Path.cwd(); run=root/'05_simulations/padova_pilot_v1'; out=root/'07_reviews/reviewer_01'
summary=json.loads((run/'summary.json').read_text()); results=[]
for s in summary:
 p=run/(s['case']+'_hourly.csv'); rows=list(csv.DictReader(p.open())); q=[float(r['sensible_load_W']) for r in rows]
 prof=s['daily_on_profile']; setbacks=[i for i,r in enumerate(rows) if float(r['heating_setpoint_C'])==19 and q[i]>1e-6]
 results.append({'case':s['case'],'rows':len(rows),'finite':all(math.isfinite(float(x)) for r in rows for x in r.values()),'heating_kWh':sum(max(x,0) for x in q)/1000,'peak_kW':max(q)/1000,'block_counts':[sum(prof[5:13]),sum(prof[13:21]),sum(prof[21:]+prof[:5])],'positive_heat_at_19C_hours':len(setbacks),'heat_at_19C_kWh':sum(q[i] for i in setbacks)/1000,'hash_matches':hashlib.sha256(p.read_bytes()).hexdigest()==s['hourly_sha256']})
for r in results:r['delta_percent']={k:100*(r[k]/results[0][k]-1) for k in ['heating_kWh','peak_kW']}
base=Path(r'C:\Works\GRINS\GRINS-Tool\resources\istat_data')
compact=list(csv.DictReader((base/'istat_microdata_2013.csv').open(encoding='utf-8-sig'),delimiter=';'))
elig=[r for r in compact if r['reg']=='5' and all(0<float(r[k])<8 for k in ['q_3_11A','q_3_11B','q_3_11C'])]
rawpath=base/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt'
with rawpath.open() as f:
 raw=next(r for r in csv.DictReader(f,delimiter='\t') if r['nprog'].strip()=='3523')
obj={'cases':results,'selection':{'eligible':len(elig),'minimum_id':min(int(r['id']) for r in elig),'raw_fields':{k:raw[k].strip() for k in ['nprog','reg','q_3_11A','q_3_11B','q_3_11C','q_3_13_2']}},'gas_deviation_percent':[100*(18090/18073-1),100*(17115/15151-1)]}
(out/'independent_checks.json').write_text(json.dumps(obj,indent=2));print(json.dumps(obj,indent=2))


