"""Reviewer 03 v5 checks from raw EPW + live surfaces; does not import author checker."""
from pathlib import Path
import hashlib,importlib.util,json
HERE=Path(__file__).resolve().parent
# Reuse only our frozen-geometry fixture and imports; no v4 diagnostic executes.
exec((HERE/'runtime_forcing_check.py').read_text().split('results={}')[0])
from scipy.interpolate import BSpline
fixpath=ROOT/'05_simulations/climate_campaign_v5/Padova/surface_forcing_1c.py'
spec=importlib.util.spec_from_file_location('reviewed_v5_fix',fixpath)
fix=importlib.util.module_from_spec(spec);spec.loader.exec_module(fix)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'fix_sha256':sha(fixpath),'sites':{}}
for site in ['Padova','Bolzano','Roma','Palermo']:
    epw=ROOT/f'05_simulations/climate_inputs_v2/{site}.epw'
    raw=pd.read_csv(epw,skiprows=8,header=None)
    weather=WeatherFile(str(epw),year=2023,time_steps=1,azimuth_subdivisions=8,height_subdivisions=4)
    delta=float(np.mean(raw[6].to_numpy()-(np.power(raw[12].to_numpy()/5.6697e-8,.25)-273.15)))
    weather.general_data['average_dt_air_sky']=delta
    weather.hourly_data=GuardUnusedInvalidFields(weather.hourly_data)
    z=zone(np.full(8760,19.))
    frame,parameters=fix.preprocess_1c_surface_loads(z,weather)
    original=list(z._surface_list);expected=np.zeros(8760);surface_results={}
    for surface in original:
        if surface.surface_type not in ['ExtWall','Roof']:continue
        s=surface;c=s._construction
        rad=weather.hourly_data_irradiances[s._azimuth_round][s._height_round]
        incident=np.asarray(rad['global']);beam=np.asarray(rad['direct'])
        fshade=getattr(s,'shading_coefficient',1.)
        sw=incident*(s._opaque_area*c._u_value_net*c._R_se*c.ext_absorptance*fshade)
        lw=np.full(8760,-delta*(s._opaque_area*c._u_value_net*c._R_se*s._sky_view_factor*s.get_surface_external_radiative_coefficient()))
        gl=np.zeros(8760)
        if s._glazed_area>0:
            w=s.window
            # Independent spline API and grouping from author splev implementation.
            spline=BSpline(*w.solar_heat_gain_coef_profile)
            gl=(beam*fshade*spline(rad['AOI'])+(incident-beam)*spline(70))*(s._glazed_area*(1-w._frame_factor)*w._shading_coef_ext*w._shading_coef_int)
        component=sw+lw+gl;expected+=component
        errors=[float(np.max(abs(a-frame[f'{s.name}::{key}'].to_numpy()))) for a,key in [(sw,'opaque_shortwave_W'),(lw,'opaque_longwave_W'),(gl,'window_shortwave_W')]]
        # Same surface alone must recover its own combined-run contribution.
        z._surface_list=[surface]
        alone,_=fix.preprocess_1c_surface_loads(z,weather)
        oneerr=float(np.max(abs(alone.total_surface_gain_W.to_numpy()-component)))
        surface_results[s.name]={'errors_SW_LW_window_W':errors,'isolated_surface_error_W':oneerr,'opaque_area_m2':s._opaque_area,'longwave_W':float(lw[0])}
    z._surface_list=original
    fix.preprocess_1c_surface_loads(z,weather)
    n1=190.08*np.ones(8760);n2=(expected+47.52)*(1-z.Am/z.Atot-z.Htr_w/(9.1*z.Atot));n3=(expected+47.52)*z.Am/z.Atot
    errors=[float(np.max(abs(a-b))) for a,b in zip([n1,n2,n3],[z.phi_ia,z.phi_st,z.phi_m])]
    order_errors={}
    roof=next(s for s in original if s.surface_type=='Roof')
    for label,order in [('reverse',original[::-1]),('roof_first',[roof]+[s for s in original if s is not roof]),('rotate',original[3:]+original[:3])]:
        z._surface_list=order
        fix.preprocess_1c_surface_loads(z,weather)
        order_errors[label]=max(float(np.max(abs(a-b))) for a,b in zip([n1,n2,n3],[z.phi_ia,z.phi_st,z.phi_m]))
    z._surface_list=original
    fix.preprocess_1c_surface_loads(z,weather)
    base=z.phi_m.copy()/(z.Am/z.Atot)
    weather.general_data['average_dt_air_sky']=0.
    fix.preprocess_1c_surface_loads(z,weather)
    actual_lw=base-z.phi_m/(z.Am/z.Atot)
    declared_lw=sum(v['longwave_W'] for v in surface_results.values())
    lw_error=float(np.max(abs(actual_lw-declared_lw)))
    weather.general_data['average_dt_air_sky']=delta
    for az,v in weather.hourly_data_irradiances.items():
        for ht,d in v.items():d['direct']=np.zeros(8760);d['global']=np.zeros(8760)
    fix.preprocess_1c_surface_loads(z,weather)
    zero_error=float(np.max(abs(z.phi_m/(z.Am/z.Atot)-(47.52+declared_lw))))
    run=ROOT/f'05_simulations/climate_campaign_v5/{site}'
    casechecks={}
    if (run/'completion_manifest.json').exists():
        design=json.loads((run/'design.json').read_text())
        for name,warmup in design['cases']:
            df=pd.read_csv(run/f'{name}_hourly.csv')
            forcing_errors=[float(np.max(abs(a-df[key].to_numpy()))) for a,key in zip([n1,n2,n3],['phi_air_W','phi_surface_W','phi_mass_W'])]
            v4par=json.loads((ROOT/f'05_simulations/climate_campaign_v4/{site}/{name}_network_parameters.json').read_text())
            v5par=json.loads((run/f'{name}_network_parameters.json').read_text())
            casechecks[name]={'actual_solver_forcing_max_error_W':forcing_errors,'network_parameters_unchanged':v4par==v5par,'hourly_sha256':sha(run/f'{name}_hourly.csv')}
    result['sites'][site]={'epw_sha256':sha(epw),'surface_checks':surface_results,'node_errors_W':errors,'order_errors_W':order_errors,
       'declared_longwave_W':declared_lw,'actual_longwave_perturbation_error_W':lw_error,'zero_solar_error_W':zero_error,
       'completed_campaign_cases_checked':casechecks}
(HERE/'v5_forcing_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
