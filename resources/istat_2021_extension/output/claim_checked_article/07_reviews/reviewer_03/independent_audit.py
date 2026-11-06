"""Reviewer 03: independent standard-library recomputation, no prior review imports."""
import csv,json,hashlib,math,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(newline='',encoding='utf-8-sig')))
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
result={'draft_sha256':sha(ROOT/'06_claim_drafts/article_claims_v02.md'),'sites':{},'runs':{}}
for site in ['Padova','Bolzano','Roma','Palermo']:
    p=ROOT/f'05_simulations/climate_inputs_v2/{site}.epw'
    raw=list(csv.reader(p.open()))[8:]
    data=[[float(x) for x in r[:5]+r[6:24]] for r in []] # no coercion of flag string
    air=[float(r[6]) for r in raw];ir=[float(r[12]) for r in raw]
    sky=[(x/5.6697e-8)**.25-273.15 for x in ir]
    daily=[sum(air[i:i+24])/24 for i in range(0,8760,24)]
    result['sites'][site]={'epw_sha256':sha(p),'rows':len(raw),'opaque99_count':sum(float(r[23])==99 for r in raw),
      'negative_wind_records':[{'index':i,'value':float(r[21]),'label':r[:5]} for i,r in enumerate(raw) if float(r[21])<0],
      'mean_air_minus_sky_K':sum(a-s for a,s in zip(air,sky))/8760,
      'sky_C_range':[min(sky),max(sky)],'HDD18':sum(max(18-t,0) for t in daily),
      'CDD26':sum(max(t-26,0) for t in daily),'GHI_kWh_m2':sum(float(r[13]) for r in raw)/1000,
      'RH_mean':sum(float(r[8]) for r in raw)/8760}
    for version in [3,4]:
        run=ROOT/f'05_simulations/climate_campaign_v{version}/{site}'
        if not (run/'completion_manifest.json').exists():continue
        summ={x['case']:x for x in load(run/'summary.json')}
        manifest=load(run/'completion_manifest.json')
        cases={}
        saved={}
        for case in ['seeded','early','late','repeat_seeded','warmup_seeded']:
            f=run/f'{case}_hourly.csv'
            rr=[{k:float(v) for k,v in r.items()} for r in rows(f)]
            saved[case]=rr
            par=load(run/f'{case}_network_parameters.json')
            maxima=[0.,0.,0.];continuity=0.
            for i,r in enumerate(rr):
                ta,ts,tm,to=[r[k] for k in ['air_temperature_C','surface_C','mass_C','outdoor_temperature_C']]
                q=r['sensible_load_W']
                residuals=[
                    par['_air_thermal_capacity']*(ta-r['old_air_C'])/3600-par['Htr_is']*(ts-ta)-r['infiltration_H_W_K']*(to-ta)-r['phi_air_W']-r['sigma_convective']*q,
                    par['Htr_is']*(ta-ts)+par['Htr_w']*(to-ts)+par['Htr_ms']*(tm-ts)+r['phi_surface_W']+r['sigma_radiant']*q,
                    par['Cm']*(tm-r['old_mass_C'])/3600-par['Htr_em']*(to-tm)-par['Htr_ms']*(ts-tm)-r['phi_mass_W']]
                maxima=[max(a,abs(b)) for a,b in zip(maxima,residuals)]
                if i:continuity=max(continuity,abs(r['old_air_C']-rr[i-1]['air_temperature_C']),abs(r['old_mass_C']-rr[i-1]['mass_C']))
            annual=sum(max(r['sensible_load_W'],0) for r in rr)/1000
            peak=max(rr,key=lambda r:r['sensible_load_W'])
            h=int(peak['hour'])
            prof=summ[case]['daily_on_profile']
            cases[case]={'sha256':sha(f),'hash_matches_summary':sha(f)==summ[case]['hourly_sha256'],
              'n':len(rr),'residual_abs_max_W':maxima,'continuity_abs_max_K':continuity,
              'finite':all(math.isfinite(v) for r in rr for v in r.values()),'hour_sequence':all(r['hour']==i for i,r in enumerate(rr)),
              'annual_kWh':annual,'peak_kW':peak['sensible_load_W']/1000,
              'annual_summary_abs_error_kWh':abs(annual-summ[case]['annual_sensible_heating_kWh']),
              'peak_hour':h,'peak_datetime':str(datetime.datetime(2023,1,1)+datetime.timedelta(hours=h)),
              'peak_row':peak,'previous_row':rr[h-1] if h else None,
              'temperature_range_C':[min(r['air_temperature_C'] for r in rr),max(r['air_temperature_C'] for r in rr)],
              'cooling_kWh':sum(max(-r['sensible_load_W'],0) for r in rr)/1000,
              'heating_setpoint_max_error_K':max([abs(r['air_temperature_C']-r['heating_setpoint_C']) for r in rr if r['sensible_load_W']>1e-6],default=0),
              'block_counts':[sum(prof[5:13]),sum(prof[13:21]),sum(prof[21:]+prof[:5])],
              'setback_positive_hours':sum(r['heating_setpoint_C']==19 and r['sensible_load_W']>1e-6 for r in rr),
              'setback_heating_kWh':sum(max(r['sensible_load_W'],0) for r in rr if r['heating_setpoint_C']==19)/1000}
        repeat=max(abs(a[k]-b[k]) for a,b in zip(saved['seeded'],saved['repeat_seeded']) for k in a)
        pairs={case:{'annual_percent':100*(cases[case]['annual_kWh']/cases['seeded']['annual_kWh']-1),
                     'peak_percent':100*(cases[case]['peak_kW']/cases['seeded']['peak_kW']-1)} for case in ['early','late','warmup_seeded']}
        result['runs'][f'v{version}/{site}']={'manifest_sha256':sha(run/'completion_manifest.json'),
          'script_sha256':sha(run/'adapter_snapshot.py'),'script_hash_matches_manifest':sha(run/'adapter_snapshot.py')==manifest['script_sha256'],
          'cases':cases,'repeat_max_difference':repeat,'pairs':pairs,
          'package_hash_mismatches':[p for p,h in manifest['package_source_sha256'].items() if sha(p)!=h]}
rawp=ROOT.parents[1]/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt'
compactp=ROOT.parents[1]/'istat_microdata_2013.csv'
raw=[{k:v.strip() for k,v in r.items()} for r in csv.DictReader(rawp.open(encoding='utf-8-sig'),delimiter='\t')]
compact=list(csv.DictReader(compactp.open(encoding='utf-8-sig'),delimiter=';'))
fields=['reg','q_3_11A','q_3_11B','q_3_11C','q_3_13_2','q_2_32C','q_2_32D']
selected=next(r for r in raw if r['nprog']=='3523')
selectedc=next(r for r in compact if r['id']=='3523')
eligible=[r for r in compact if r['reg']=='5' and all(0<float(r[k])<8 for k in ['q_3_11A','q_3_11B','q_3_11C'])]
result['survey']={'raw_sha256':sha(rawp),'compact_sha256':sha(compactp),'raw':{k:selected[k] for k in fields},
  'compact':{k:selectedc[k] for k in fields},'eligible_count':len(eligible),'min_id':min(int(r['id']) for r in eligible)}
result['published_gas_deviation_percent']=[100*(18090/18073-1),100*(17115/15151-1)]
(OUT/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'draft_sha256':result['draft_sha256'],'survey':result['survey'],'sites':result['sites'],
  'v4':{k:{'pairs':v['pairs'],'max_residual_W':max(max(c['residual_abs_max_W']) for c in v['cases'].values()),'peak_hours':{c:x['peak_datetime'] for c,x in v['cases'].items()}} for k,v in result['runs'].items() if k.startswith('v4')}},indent=2))
