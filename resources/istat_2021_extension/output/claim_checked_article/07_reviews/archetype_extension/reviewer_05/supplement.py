from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
from scipy.interpolate import splev
R=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent;D=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raw=pd.read_csv(D/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt',sep='\t',low_memory=False)
raw=raw.apply(lambda c:pd.to_numeric(c.astype(str).str.strip(),errors='coerce'))
arc=pd.read_csv(R/'00_context/archetype_extension/archived_four_archetype_assignments.csv')
s=pd.read_csv(D/'istat_microdata_2013.csv',sep=';')
r=arc.merge(raw,left_on='household_id',right_on='nprog',validate='1:1')
hs=['q_3_11A','q_3_11B','q_3_11C']
ok=r.q_3_0.isin([1,2]) & r[hs].apply(lambda x:x.between(0,8)&(x%1==0)).all(axis=1)&r[hs].sum(axis=1).gt(0)
r=r[ok];assert len(r)==13624
c=s.set_index('id').loc[r.household_id].reset_index()
for f in hs+['q_3_0']:
    assert np.array_equal(r[f],c[f])
for f in ['q_3_13_2','q_3_13_3','q_3_19_6','q_2_32C','q_2_32D']:
    assert np.array_equal(r[f].eq(1),c[f].eq(1))
roles=pd.read_csv(R/'05_simulations/archetype_campaign_v2/pre_run/role_manifest.csv')
v1=pd.read_csv(R/'05_simulations/archetype_campaign_v1/pre_run/role_manifest.csv')
assert v1[v1.role!='test'].profile_sha256.nunique()==1
assert set(v1[v1.role=='test'].household_id)==set(roles[roles.role=='test'].household_id)
old=[];surfaces=[];peak=[]
for site in ['Padova','Bolzano','Roma','Palermo']:
    run=R/'05_simulations/archetype_campaign_v2'/site
    pars=json.loads((run/'surface_forcing_parameters.json').read_text())
    h=pd.read_csv(run/'surface_forcing_hourly.csv')
    reconstructed=np.zeros(8760)
    for name,p in pars['surfaces'].items():
        g=h[name+'::irradiance_global_W_m2'];direct=h[name+'::irradiance_direct_W_m2']
        sw=p['shading']*g*p['absorptance']*p['Rse_m2_K_W']*p['U_net_W_m2_K']*p['opaque_area_m2']
        lw=-p['sky_view_factor']*p['Rse_m2_K_W']*p['U_net_W_m2_K']*p['opaque_area_m2']*p['h_r_W_m2_K']*p['mean_air_minus_sky_K']
        glass=np.zeros(8760)
        if p['glazed_area_m2']:
            sp=p['shgc_spline'];tck=(sp['knots'],sp['coefficients'],sp['degree'])
            gd=splev(h[name+'::AOI_degree'],tck);gf=splev(70,tck)
            fact=p['window_shading_external']*p['window_shading_internal']*(1-p['frame_factor'])*p['glazed_area_m2']
            glass=fact*(p['shading']*direct*gd+(g-direct)*gf)
        assert np.max(abs(sw-h[name+'::opaque_shortwave_W']))<1e-8
        assert np.max(abs(lw-h[name+'::opaque_longwave_W']))<1e-8
        assert np.max(abs(glass-h[name+'::window_shortwave_W']))<1e-8
        reconstructed+=sw+lw+glass
    assert np.max(abs(reconstructed-h.total_surface_gain_W))<1e-8
    for node,key in [('surface','node_surface_weight'),('mass','node_mass_weight')]:
        assert np.max(abs((h.internal_radiative_W+reconstructed)*pars[key]-h['phi_'+node+'_W']))<1e-8
    surfaces.append({'site':site,'surface_algebra_pass':True})
    for _,rr in roles.iterrows():
        x=pd.read_csv(run/f'{rr.profile_id}_hourly.csv')
        idx=int(x.sensible_load_W.idxmax())
        peak.append({'site':site,'role':rr.role,'household_id':int(rr.household_id),'hour':idx,'old_air_C':x.old_air_C[idx],'setpoint_C':x.heating_setpoint_C[idx]})
    oldrun=R/'05_simulations/climate_campaign_v5'/site
    for name in ['seeded','early','late','repeat_seeded','warmup_seeded']:
        p=oldrun/f'{name}_hourly.csv'
        if not p.exists():continue
        x=pd.read_csv(p);q=x.sensible_load_W.clip(lower=0)
        old.append({'site':site,'case':name,'kWh':float(q.sum()/1000),'kW':float(q.max()/1000),'peak_hour':int(q.idxmax()),'sha256':sha(p)})
pd.DataFrame(old).to_csv(O/'old_pilot_endpoints.csv',index=False)
pd.DataFrame(peak).to_csv(O/'primary_peak_context.csv',index=False)
(O/'supplement.json').write_text(json.dumps({'raw_eligibility_and_predicates_pass':True,'v1_five_roles_one_profile':True,'unchanged_test_ids':True,'surface_checks':surfaces,'raw_sha256':sha(D/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt'),'old_pilot_files_checked':len(old)},indent=2))
print(json.dumps({'raw_n':len(r),'old_files':len(old),'surface_checks':surfaces,'Bolzano_oct16_maxima':len([v for v in peak if v['site']=='Bolzano' and 6912<=v['hour']<6936]),'Bolzano_test_oct16_maxima':len([v for v in peak if v['site']=='Bolzano' and v['role']=='test' and 6912<=v['hour']<6936])}))
