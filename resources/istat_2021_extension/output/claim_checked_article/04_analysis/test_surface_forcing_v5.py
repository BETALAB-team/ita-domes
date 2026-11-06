"""Independent numerical fixtures for separate surface physics and ordering."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,hashlib
import numpy as np
from scipy.interpolate import splrep
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,r'C:\Works\EUReCA\EUReCA')
from eureca_building.config import load_config
load_config(str(ROOT/'05_simulations/climate_campaign_v4/Padova/eureca_config.json'))
from surface_forcing_1c import preprocess_1c_surface_loads
n=8760
constant=lambda x:np.full(n,x,dtype=float)
construction=SimpleNamespace(ext_absorptance=.6,_R_se=.04,_u_value_net=.8)
window=SimpleNamespace(_shading_coef_ext=.9,_shading_coef_int=.8,_frame_factor=.1,solar_heat_gain_coef_profile=splrep([0.,30.,70.,90.],[.6,.6,.6,.6],k=1))
wall=SimpleNamespace(name='wall',surface_type='ExtWall',_opaque_area=20.,_glazed_area=4.,_construction=construction,_sky_view_factor=.5,_azimuth_round=0.,_height_round=90.,get_surface_external_radiative_coefficient=lambda:4.5,window=window)
roof=SimpleNamespace(name='roof_no_window',surface_type='Roof',_opaque_area=80.,_glazed_area=0.,_construction=construction,_sky_view_factor=1.,_azimuth_round=0.,_height_round=0.,get_surface_external_radiative_coefficient=lambda:4.5)
weather=SimpleNamespace(general_data={'average_dt_air_sky':12.},hourly_data={'out_air_db_temperature':constant(5)},hourly_data_irradiances={0.:{90.:{'global':constant(150),'direct':constant(60),'AOI':constant(30)},0.:{'global':constant(300),'direct':constant(100),'AOI':constant(40)}}})
def zone(surfaces):return SimpleNamespace(_surface_list=surfaces,Am=40.,Atot=100.,Htr_w=10.,extract_convective_radiative_latent_electric_load=lambda:{'convective [W]':constant(10),'radiative [W]':constant(20)})
full=zone([wall,roof]);frame,p=preprocess_1c_surface_loads(full,weather)
# Scalar equations written independently of the exported helper components.
expected={'wall':{'opaque_shortwave_W':150*.6*.04*.8*20,'opaque_longwave_W':-.5*.04*.8*20*4.5*12,'window_shortwave_W':150*.9*.8*.9*.6*4},'roof_no_window':{'opaque_shortwave_W':300*.6*.04*.8*80,'opaque_longwave_W':-1*.04*.8*80*4.5*12,'window_shortwave_W':0.}}
errors={name:{component:float(np.max(np.abs(frame[f'{name}::{component}']-target))) for component,target in values.items()} for name,values in expected.items()}
standalone={}
for s in [wall,roof]:
    f,_=preprocess_1c_surface_loads(zone([s]),weather)
    standalone[s.name]=max(float(np.max(np.abs(f[c]-frame[c]))) for c in f if c.startswith(s.name+'::'))
order=[]
for surfaces in [[roof,wall],[wall,roof][1:]+[wall,roof][:1]]:
    z=zone(surfaces);preprocess_1c_surface_loads(z,weather)
    order.append(max(float(np.max(np.abs(getattr(z,k)-getattr(full,k)))) for k in ['phi_ia','phi_st','phi_m']))
total=sum(v for d in expected.values() for v in d.values()); ws=1-.4-10/910; wm=.4
node_error=[float(np.max(np.abs(full.phi_ia-10))),float(np.max(np.abs(full.phi_st-ws*(20+total)))),float(np.max(np.abs(full.phi_m-wm*(20+total))))]
out={'purpose':'pre-run independent per-surface shortwave/longwave/window fixtures and ordering; no annual thermal simulation','expected_components_W':expected,'maximum_component_error_W':errors,'standalone_component_error_W':standalone,'reverse_rotate_max_node_forcing_error_W':order,'independent_node_distribution_error_W':node_error,'roof_has_no_window_attribute':not hasattr(roof,'window'),'source_sha256':hashlib.sha256((ROOT/'04_analysis/surface_forcing_1c.py').read_bytes()).hexdigest()}
out['pass']=max([v for e in errors.values() for v in e.values()]+list(standalone.values())+order+node_error)<1e-8 and out['roof_has_no_window_attribute']
(ROOT/'05_simulations/surface_forcing_v5_fixture_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2));assert out['pass']
