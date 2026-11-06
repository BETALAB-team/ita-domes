"""Audit every weather field consumed by this bounded EUReCA pathway."""
from pathlib import Path
from datetime import datetime
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'05_simulations/climate_inputs_v2'
OUT=ROOT/'05_simulations/weather_forcing_v4';OUT.mkdir(parents=True,exist_ok=True)
SIGMA=5.6697e-8  # EnergyPlus primary documentation convention.
# zero-based EPW column, unit, missing sentinel, declared validity range
FIELDS={'dry_bulb':(6,'degC',99.9,-70,70),'dew_point':(7,'degC',99.9,-70,70),
'relative_humidity':(8,'percent',999,0,100),'pressure':(9,'Pa',999999,31000,120000),
'horizontal_IR':(12,'Wh/m2 per preceding hour',9999,0,1000),
'GHI':(13,'Wh/m2 per preceding hour',9999,-.01,1500),'DNI':(14,'Wh/m2 per preceding hour',9999,-.01,1500),'DHI':(15,'Wh/m2 per preceding hour',9999,-.01,1500),
'wind_direction':(20,'degree',999,0,360),'wind_speed':(21,'m/s',999,0,40),
'opaque_sky_cover':(23,'tenths',99,0,10)}
result={}
for site in ['Padova','Bolzano','Roma','Palermo']:
    epw=INPUT/f'{site}.epw';obj=json.loads((INPUT/f'{site}.json').read_text());raw=pd.read_csv(epw,skiprows=8,header=None)
    js=pd.DataFrame(obj['outputs']['tmy_hourly']);n=len(raw);assert n==len(js)==8760
    qc={}
    for name,(col,unit,sentinel,lo,hi) in FIELDS.items():
        a=raw[col].to_numpy(float)
        qc[name]={'epw_zero_based_column':col,'unit':unit,'missing_sentinel':sentinel,'sentinel_count':int((a>=sentinel).sum()),'min':float(a.min()),'max':float(a.max()),'finite':bool(np.isfinite(a).all()),'range_failure_count':int(((a<lo)|(a>hi)).sum()),'validity_range':[lo,hi]}
    # Wind is loaded but has no causal role in this fixed-ACH, no-natural-
    # ventilation, no-PV sensible-only pathway. Preserve and report its invalid
    # source record; do not reclassify it as valid or change the physical limit.
    failures={k:x for k,x in qc.items() if k not in ['opaque_sky_cover','wind_speed'] and not(x['finite'] and x['sentinel_count']==0 and x['range_failure_count']==0)}
    if failures:
        (OUT/f'{site}_QC_failure.json').write_text(json.dumps(failures,indent=2))
        raise ValueError(f'{site} field QC failed: {failures}')
    assert qc['opaque_sky_cover']['sentinel_count']==8760
    # Compare corresponding exported records, not inferred UTC instants.
    mapping={name:float(np.max(np.abs(raw[col].to_numpy(float)-js[jname].to_numpy(float)))) for name,col,jname in [('dry_bulb',6,'T2m'),('RH',8,'RH'),('pressure',9,'SP'),('IR',12,'IR(h)'),('GHI',13,'G(h)'),('DNI',14,'Gb(n)'),('DHI',15,'Gd(h)'),('wind_direction',20,'WD10m'),('wind_speed',21,'WS10m')]}
    assert all(x<=1e-9 for name,x in mapping.items() if name!='wind_speed')
    assert mapping['wind_speed']<=.050000001 # EPW export rounding to one decimal.
    time_matches=[]
    for i,time in enumerate(js['time(UTC)']):
        dt=datetime.strptime(time,'%Y%m%d:%H%M')
        time_matches.append((int(raw.iloc[i,0]),int(raw.iloc[i,1]),int(raw.iloc[i,2]),int(raw.iloc[i,3])-1)==(dt.year,dt.month,dt.day,dt.hour))
    assert all(time_matches)
    ir=raw[12].to_numpy(float);air=raw[6].to_numpy(float)
    sky_K=np.power(ir/SIGMA,.25);sky_C=sky_K-273.15
    delta=air-sky_C
    recon=SIGMA*sky_K**4
    assert np.isfinite(sky_C).all() and np.max(np.abs(recon-ir))<1e-9
    assert -100<sky_C.min() and sky_C.max()<70 and 0<float(delta.mean())<50
    pd.DataFrame({'hour':range(n),'air_C':air,'horizontal_IR_Wh_m2':ir,'interval_hours':np.ones(n),'horizontal_IR_rate_W_m2':ir,'effective_sky_C':sky_C,'air_minus_sky_K':delta,'IR_reconstruction_error_W_m2':recon-ir}).to_csv(OUT/f'{site}_derived_sky.csv',index=False)
    header=epw.read_text().splitlines()[:8]
    result[site]={'fields':qc,'epw_json_recordwise_max_abs_difference':mapping,'all_date_hour_end_minus1_match_json_labels':all(time_matches),'EPW_timezone_hours':float(header[0].split(',')[8]),'EPW_comments2':header[6],'JSON_irradiance_time_offset_hours':obj['inputs']['location']['irradiance_time_offset'],
      'effective_sky_C_range':[float(sky_C.min()),float(sky_C.max())],'IR_derived_mean_air_minus_sky_K':float(delta.mean()),'IR_reconstruction_max_abs_error_W_m2':float(np.max(np.abs(recon-ir))),
      'dew_point_above_air_count':int((raw[7].to_numpy(float)>air+.05).sum()),'active_forcing_QC_pass':True,'sky_derivation_pass':True,
      'loaded_unused_wind_speed_valid':qc['wind_speed']['range_failure_count']==0 and qc['wind_speed']['sentinel_count']==0,
      'wind_scope':'Raw wind retained, not imputed. Source reads wind only for natural ventilation, disabled and guarded in this campaign; no wind-quality or ventilation inference follows.',
      'epw_sha256':hashlib.sha256(epw.read_bytes()).hexdigest(),'json_sha256':hashlib.sha256((INPUT/f'{site}.json').read_bytes()).hexdigest(),
      'time_interpretation':'Record i EPW hour-ending H matches JSON label H-1 exactly in date, IR and other fields. Native pvlib labels it H-1 in EPW fixed UTC+1; no extra shift/interpolation. IR and air paired from same exported record. Solar offset is reported but not applied by native EUReCA; retained timing approximation, not a claim of exact solar phase.'}
audit={'sigma_W_m2_K4':SIGMA,'EPW_interval_hours':1,'method':'IR rate=(IR Wh/m2)/1hour; effective sky K=(IRrate/sigma)^.25; native annual scalar is arithmetic mean(airC-skyC)','no_cloud_imputation':True,'sites':result,'gate_scope':'active forcing, mapping and sky derivation; not a statement that every exported weather variable is valid','pass':all(x['active_forcing_QC_pass'] and x['sky_derivation_pass'] for x in result.values())}
(OUT/'weather_forcing_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({s:{k:v for k,v in d.items() if k in ['IR_derived_mean_air_minus_sky_K','effective_sky_C_range','epw_json_recordwise_max_abs_difference','all_date_hour_end_minus1_match_json_labels']} for s,d in result.items()},indent=2))
