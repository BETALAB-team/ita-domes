"""Independent all-hour component reconstruction and solver-forcing agreement."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.interpolate import splev
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('SITE');parser.add_argument('--campaign',required=True);args=parser.parse_args()
RUN=ROOT/'05_simulations'/args.campaign/args.SITE
assert RUN.resolve().is_relative_to((ROOT/'05_simulations').resolve())
df=pd.read_csv(RUN/'surface_forcing_hourly.csv');p=json.loads((RUN/'surface_forcing_parameters.json').read_text());total=np.zeros(len(df));surface_checks={}
for name,s in p['surfaces'].items():
    get=lambda key:df[f'{name}::{key}'].to_numpy()
    g=get('irradiance_global_W_m2');b=get('irradiance_direct_W_m2');aoi=get('AOI_degree')
    opaque=g*s['shading']*s['absorptance']*s['Rse_m2_K_W']*s['U_net_W_m2_K']*s['opaque_area_m2']
    longwave=np.full(len(df),-s['sky_view_factor']*s['Rse_m2_K_W']*s['U_net_W_m2_K']*s['opaque_area_m2']*s['h_r_W_m2_K']*s['mean_air_minus_sky_K'])
    glazed=np.zeros(len(df));curve_errors=[]
    if s['glazed_area_m2']>0:
        spline=s['shgc_spline'];curve=(spline['knots'],spline['coefficients'],spline['degree'])
        gd=splev(aoi,curve);gf=float(splev(70.,curve))
        curve_errors=[float(np.max(np.abs(gd-get('SHGC_direct')))),float(np.max(np.abs(gf-get('SHGC_diffuse'))))]
        scale=s['window_shading_external']*s['window_shading_internal']*(1-s['frame_factor'])*s['glazed_area_m2']
        glazed=(s['shading']*b*gd+(g-b)*gf)*scale
    errors=[float(np.max(np.abs(opaque-get('opaque_shortwave_W')))),float(np.max(np.abs(longwave-get('opaque_longwave_W')))),float(np.max(np.abs(glazed-get('window_shortwave_W'))))]
    surface_checks[name]={'type':s['type'],'opaque_glazed_area_m2':[s['opaque_area_m2'],s['glazed_area_m2']],'max_SW_LW_window_reconstruction_error_W':errors,'SHGC_curve_errors':curve_errors,'pass':max(errors+curve_errors)<1e-8}
    total+=opaque+longwave+glazed
nodes=[df['internal_convective_W'].to_numpy(),(total+df['internal_radiative_W'].to_numpy())*p['node_surface_weight'],(total+df['internal_radiative_W'].to_numpy())*p['node_mass_weight']]
node_errors=[float(np.max(np.abs(a-df[k].to_numpy()))) for a,k in zip(nodes,['phi_air_W','phi_surface_W','phi_mass_W'])]
case_checks=[]
design=json.loads((RUN/'design.json').read_text())
for name,warmup in design['cases']:
    annual=pd.read_csv(RUN/f'{name}_hourly.csv')
    errors=[float(np.max(np.abs(a-annual[k].to_numpy()))) for a,k in zip(nodes,['phi_air_W','phi_surface_W','phi_mass_W'])]
    order=json.loads((RUN/f'{name}_surface_order_checks.json').read_text())
    case_checks.append({'case':name,'solver_forcing_max_air_surface_mass_error_W':errors,'surface_order_checks':order,'pass':max(errors)<1e-8 and order['pass']})
out={'site':args.SITE,'rows':len(df),'surfaces':surface_checks,'node_reconstruction_max_error_W':node_errors,'case_checks':case_checks,'pass':len(df)==8760 and all(x['pass'] for x in surface_checks.values()) and max(node_errors)<1e-8 and all(x['pass'] for x in case_checks)}
(RUN/'independent_surface_forcing_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2));assert out['pass']
