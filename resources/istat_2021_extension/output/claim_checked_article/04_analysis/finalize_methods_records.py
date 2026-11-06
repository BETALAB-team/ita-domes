"""Finalize post-run records; never modify immutable campaign snapshots."""
from pathlib import Path
import importlib.metadata as md
import json,csv,math
ROOT=Path(__file__).resolve().parents[1]
packages=sorted([{'name':d.metadata.get('Name'),'version':d.version} for d in md.distributions()],key=lambda d:(d['name'] or '').lower())
(ROOT/'05_simulations/environment_packages.json').write_text(json.dumps({'inventory_method':'importlib.metadata.distributions; pip is absent in this read-only existing environment','packages':packages},indent=2))
(ROOT/'05_simulations/environment_freeze.txt').write_text('\n'.join(f"{d['name']}=={d['version']}" for d in packages)+'\n')
replacements={'assumed80m²dwelling':'assumed 80 m² dwelling','actual2013Veneto':'actual 2013 Veneto','4/4/1daily blockhours':'4/4/1 daily block hours','20new annual':'20 new annual','12primary':'12 primary','4coordinatecases×3schedules':'4 coordinate cases × 3 schedules','4exact-repeat':'4 exact-repeat','4longer-warm-up':'4 longer-warm-up','adds5annual':'adds 5 annual','25totalnew annual':'25 total new annual','all20outputs':'all 20 outputs','below1e-9kWh':'below 1e-9 kWh','had8760finite':'had 8760 finite','analytical1Cnetwork':'analytical 1C network','maximumerror9.71e−13':'maximum error 9.71e−13','3327.272727W':'3327.272727 W','absoluteerror2.28e−12W':'absolute error 2.28e−12 W','largest<6e−12W':'largest <6e−12 W','TMYscenarios2005–2023':'TMY scenarios for 2005–2023','elevationPadova25m, Bolzano272m, Roma72m, Palermo29m':'elevations Padova 25 m, Bolzano 272 m, Roma 72 m, Palermo 29 m','local-v1weather':'local v1 weather','official-v2Padova':'official v2 Padova','execute04_analysis':'execute 04_analysis','networkaccess':'network access','EUReCA.venvPython-B':'EUReCA .venv Python with -B','finally04_analysis':'finally 04_analysis','weatherchecksums':'weather checksums','under05_simulations':'under 05_simulations','runID':'run ID','runtime was14.715s':'runtime was 14.715 s'}
p=ROOT/'04_analysis/simulation_results.md';text=p.read_text(encoding='utf-8')
for a,b in replacements.items():text=text.replace(a,b)
p.write_text(text,encoding='utf-8')
inventory=[]
for group,sites in [('padova_pilot_v1',[None]),('climate_campaign_v2',['Padova','Bolzano','Roma','Palermo'])]:
    for site in sites:
        run=ROOT/'05_simulations'/group
        if site:run=run/site
        for s in json.loads((run/'summary.json').read_text()):
            inventory.append({'campaign':group,'site':site or 'Padova_local_weather','case':s['case'],'purpose':'primary_contrast' if s['case'] in ['seeded','early','late'] else 'technical_check','warmup_hours':s['warmup_h'],'hourly_path':str(run/f"{s['case']}_hourly.csv"),'hourly_sha256':s['hourly_sha256'],'annual_sensible_heating_kWh':s['annual_sensible_heating_kWh'],'peak_sensible_heating_kW':s['peak_sensible_heating_kW']})
assert len(inventory)==25
with (ROOT/'05_simulations/run_inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=inventory[0].keys());w.writeheader();w.writerows(inventory)
# Explicitly post-hoc, descriptive partition requested after primary results.
diagnostic=[];tolerance_W=1e-6
for r in inventory:
    if r['campaign']!='climate_campaign_v2' or r['purpose']!='primary_contrast':continue
    with open(r['hourly_path'],newline='') as f:hourly=list(csv.DictReader(f))
    for sp in [0.,19.,22.]:
        subset=[x for x in hourly if float(x['heating_setpoint_C'])==sp]
        q=[float(x['sensible_load_W']) for x in subset]
        diagnostic.append({'site':r['site'],'schedule':r['case'],'setpoint_C':sp,'calendar_hours_at_setpoint':len(q),'positive_heating_hours_gt_1e_minus6W':sum(v>tolerance_W for v in q),'positive_sensible_heating_kWh':math.fsum(max(v,0.) for v in q)/1000,'share_annual_sensible_heating_percent':100*math.fsum(max(v,0.) for v in q)/1000/r['annual_sensible_heating_kWh'],'positive_load_tolerance_W':tolerance_W})
with (ROOT/'04_analysis/posthoc_setpoint_partition_v2.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=diagnostic[0].keys());w.writeheader();w.writerows(diagnostic)
note=['# Post-hoc descriptive setpoint partition','','Requested after the primary simulations and endpoints were complete; no new annual runs or primary tests were added. Positive heating hours use sensible load >1e-6 W. Energy sums include max(load,0) at the one-hour timestep. Columns distinguish calendar hours assigned each lower setpoint, hours with positive modeled heating, and their energy. Every schedule has the same number of hours at 19/22/0°C; the positive-load hours and energy need not match.','','The source mapping preserves 4/4/1 hours assigned the high setpoint in the three reported blocks. It does not enforce 4/4/1 modeled heater-runtime hours. The selected timer-control branch imposes 19°C setback outside its high-setpoint hours, so the ideal system may continue supplying heat at setback. Outside the source-defined season the lower setpoint is 0°C. The reported survey on-hours therefore become a temperature-schedule assumption in this adapter, not a constraint guaranteeing equal realized heating runtime. This is a material interpretation limit.','','| Site | Schedule | Heating at 19°C kWh | Positive hours at 19°C | Heating at 22°C kWh | Positive hours at 22°C | Heating at 0°C kWh |','|---|---|---:|---:|---:|---:|---:|']
for site in ['Padova','Bolzano','Roma','Palermo']:
    for mode in ['seeded','early','late']:
        d={r['setpoint_C']:r for r in diagnostic if r['site']==site and r['schedule']==mode}
        note.append(f"|{site}|{mode}|{d[19.]['positive_sensible_heating_kWh']:.3f}|{d[19.]['positive_heating_hours_gt_1e_minus6W']}|{d[22.]['positive_sensible_heating_kWh']:.3f}|{d[22.]['positive_heating_hours_gt_1e_minus6W']}|{d[0.]['positive_sensible_heating_kWh']:.3f}|")
(ROOT/'04_analysis/posthoc_setpoint_partition_v2.md').write_text('\n'.join(note),encoding='utf-8')
print('Finalized package inventory, results readability, exact 25-run inventory, and post-hoc setpoint diagnostic')
