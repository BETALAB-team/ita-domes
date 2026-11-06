from pathlib import Path
import hashlib,json,ast,random
import numpy as np
import pandas as pd
from scipy.interpolate import splev
ROOT=Path(__file__).resolve().parents[2]
out={'draft_sha256':hashlib.sha256((ROOT/'06_claim_drafts/article_claims_v03.md').read_bytes()).hexdigest(),'sites':{}}
for site in ['Padova','Bolzano','Roma','Palermo']:
 d=ROOT/'05_simulations/climate_campaign_v5'/site
 raw=pd.read_csv(ROOT/f'05_simulations/climate_inputs_v2/{site}.epw',skiprows=8,header=None)
 delta=np.mean(raw[6]-((raw[12]/5.6697e-8)**.25-273.15))
 f=pd.read_csv(d/'surface_forcing_hourly.csv')
 p=json.loads((d/'surface_forcing_parameters.json').read_text())
 solar=np.zeros(8760); errs=[]
 for name,s in p['surfaces'].items():
  col=lambda x:f[name+'::'+x].to_numpy()
  sw=s['shading']*col('irradiance_global_W_m2')*s['absorptance']*s['Rse_m2_K_W']*s['U_net_W_m2_K']*s['opaque_area_m2']
  lw=-s['sky_view_factor']*s['Rse_m2_K_W']*s['U_net_W_m2_K']*s['opaque_area_m2']*s['h_r_W_m2_K']*delta
  win=np.zeros(8760)
  if s['glazed_area_m2']>0:
   spline=s['shgc_spline']; tck=(spline['knots'],spline['coefficients'],spline['degree'])
   win=s['glazed_area_m2']*s['window_shading_external']*s['window_shading_internal']*(1-s['frame_factor'])*(s['shading']*col('irradiance_direct_W_m2')*splev(col('AOI_degree'),tck)+(col('irradiance_global_W_m2')-col('irradiance_direct_W_m2'))*splev(70,tck))
  for a,b in [(sw,col('opaque_shortwave_W')),(lw,col('opaque_longwave_W')),(win,col('window_shortwave_W'))]:errs.append(float(np.max(np.abs(a-b))))
  solar+=sw+lw+win
 forcing=[f.internal_convective_W.to_numpy(),p['node_surface_weight']*(solar+f.internal_radiative_W),p['node_mass_weight']*(solar+f.internal_radiative_W)]
 results={}
 for case in ['seeded','early','late','repeat_seeded','warmup_seeded']:
  x=pd.read_csv(d/f'{case}_hourly.csv'); n=json.loads((d/f'{case}_network_parameters.json').read_text())
  q=x.sensible_load_W;ta=x.air_temperature_C;ts=x.surface_C;tm=x.mass_C;te=x.outdoor_temperature_C
  rair=x.phi_air_W+x.sigma_convective*q+n['Htr_is']*(ts-ta)+x.infiltration_H_W_K*(te-ta)-n['_air_thermal_capacity']*(ta-x.old_air_C)/3600
  rs=x.phi_surface_W+x.sigma_radiant*q+n['Htr_is']*(ta-ts)+n['Htr_w']*(te-ts)+n['Htr_ms']*(tm-ts)
  rm=x.phi_mass_W+n['Htr_ms']*(ts-tm)+n['Htr_em']*(te-tm)-n['Cm']*(tm-x.old_mass_C)/3600
  peak=int(q.idxmax());mask=x.heating_setpoint_C==19
  results[case]={'energy_kWh':float(q.clip(lower=0).sum()/1000),'peak_kW':float(q.max()/1000),'peak_hour':peak,'peak_setpoint':float(x.heating_setpoint_C.iloc[peak]),'peak_previous_setpoint':float(x.heating_setpoint_C.iloc[peak-1]),'peak_old_air':float(x.old_air_C.iloc[peak]),'max_node_residual_W':max(float(abs(r).max()) for r in [rair,rs,rm]),'continuity_max_K':max(float(np.max(np.abs(x.old_air_C.to_numpy()[1:]-ta.to_numpy()[:-1]))),float(np.max(np.abs(x.old_mass_C.to_numpy()[1:]-tm.to_numpy()[:-1])))),'forcing_error_W':max(float(np.max(abs(x[c]-v))) for c,v in zip(['phi_air_W','phi_surface_W','phi_mass_W'],forcing)),'weather_error_K':float(abs(te-raw[6]).max()),'setback_hours':int(mask.sum()),'setback_heating_hours':int((mask&(q>1e-6)).sum()),'setback_kWh':float(q[mask].clip(lower=0).sum()/1000),'maximum_air_C':float(ta.max()),'offseason_positive_heat_kWh':float(q[x.heating_setpoint_C==0].clip(lower=0).sum()/1000),'sha256':hashlib.sha256((d/f'{case}_hourly.csv').read_bytes()).hexdigest()}
  if case!='seeded':
   results[case]['energy_change_pct']=100*(results[case]['energy_kWh']/results['seeded']['energy_kWh']-1)
   results[case]['peak_change_pct']=100*(results[case]['peak_kW']/results['seeded']['peak_kW']-1)
 out['sites'][site]={'surface_error_W':max(errs),'sky_delta_K':float(delta),'HDD18':float(np.maximum(18-raw[6].to_numpy().reshape(365,24).mean(axis=1),0).sum()),'cases':results}
out['replay_identical']=(ROOT/'05_simulations/climate_campaign_v5/Padova/seeded_hourly.csv').read_bytes()==(ROOT/'05_simulations/climate_campaign_v5_replay/Padova/seeded_hourly.csv').read_bytes()
tree=ast.parse(Path('C:/Works/GRINS/GRINS-Tool/classes/EndUse.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='User')
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['create_prof','create_random_prof','substitute_istat_heatingTSP'] or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='periodi_risc' for t in n.targets)]
env={'np':np,'random':random}
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'original_EndUse','exec'),env)
design=json.loads((ROOT/'05_simulations/climate_campaign_v5/Padova/design.json').read_text())
u=env['User']();u.substitute_istat_heatingTSP(design['response'])
x=pd.read_csv(ROOT/'05_simulations/climate_campaign_v5/Padova/seeded_hourly.csv')
out['source_schedule_max_error']=float(np.max(abs(u.heatingTSP-x.heating_setpoint_C)))
out['source_profile']=u.create_prof(4,4,1,3523).tolist()
rawsurvey=pd.read_csv(ROOT.parents[1]/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt',sep='\t',dtype=str)
out['raw_survey_selected']=rawsurvey.loc[rawsurvey.nprog.str.strip()=='3523',['nprog','reg','q_3_11A','q_3_11B','q_3_11C','q_3_13_2','q_2_32C','q_2_32D']].fillna('BLANK').to_dict('records')
compact=pd.read_csv(ROOT.parents[1]/'istat_microdata_2013.csv',sep=';')
eligible=compact[(compact.reg==5)&compact[['q_3_11A','q_3_11B','q_3_11C']].gt(0).all(axis=1)&compact[['q_3_11A','q_3_11B','q_3_11C']].lt(8).all(axis=1)]
out['selection']={'eligible':len(eligible),'minimum_id':int(eligible.id.min())}
out['profile_checks']={}
for case in ['seeded','early','late']:
 v=pd.read_csv(ROOT/f'05_simulations/climate_campaign_v5/Padova/{case}_hourly.csv').heating_setpoint_C.to_numpy()
 daily=(v[:24]==22).astype(int)
 out['profile_checks'][case]={'bands':[int(daily[5:13].sum()),int(daily[13:21].sum()),int(daily[21:].sum()+daily[:5].sum())],'daily_high_hours':np.where(daily)[0].tolist(),'inseason_daily_repetition':bool(all(np.array_equal(row,v[:24]) for row in v.reshape(365,24) if row.max()>0))}
out['weather_mapping']={}
out['summary_comparison']={}
for site in out['sites']:
 raw=pd.read_csv(ROOT/f'05_simulations/climate_inputs_v2/{site}.epw',skiprows=8,header=None)
 j=pd.DataFrame(json.loads((ROOT/f'05_simulations/climate_inputs_v2/{site}.json').read_text())['outputs']['tmy_hourly'])
 out['weather_mapping'][site]={name:float(np.max(abs(raw[col]-j[name]))) for name,col in [('T2m',6),('RH',8),('IR(h)',12),('G(h)',13),('Gb(n)',14),('Gd(h)',15)]}
 summary=json.loads((ROOT/f'05_simulations/climate_campaign_v5/{site}/summary.json').read_text())
 out['summary_comparison'][site]=[{'case':s['case'],'energy_error_kWh':out['sites'][site]['cases'][s['case']]['energy_kWh']-s['annual_sensible_heating_kWh'],'peak_error_kW':out['sites'][site]['cases'][s['case']]['peak_kW']-s['peak_sensible_heating_kW'],'hash_match':out['sites'][site]['cases'][s['case']]['sha256']==s['hourly_sha256']} for s in summary]
ledger=json.loads((ROOT/'06_claim_drafts/claim_ledger_v03.json').read_text(encoding='utf-8'))
text=(ROOT/'06_claim_drafts/article_claims_v03.md').read_text(encoding='utf-8')
out['ledger']={'count':len(ledger['claims']),'exact_claims_all_in_text':all(c['exact_claim'] in text for c in ledger['claims']),'hash_match':ledger['draft_sha256']==out['draft_sha256']}
(Path(__file__).parent/'independent_recheck.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
