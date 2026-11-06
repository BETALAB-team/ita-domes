"""Reviewer runtime audit: actual load preprocessing, zero-solar perturbation, guard."""
from pathlib import Path
import sys,os,ast,json,copy,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
os.chdir(OUT)
sys.path.insert(0,r'C:\Works\EUReCA\EUReCA')
sys.path.insert(0,str(ROOT/'04_analysis'))
from eureca_building.config import load_config
load_config(str(ROOT/'05_simulations/climate_campaign_v4/Padova/eureca_config.json'))
from eureca_building.weather import WeatherFile
from eureca_building.construction import Construction
from eureca_building.window import SimpleWindow
from eureca_building.surface import Surface,SurfaceInternalMass
from eureca_building.internal_load import ElectricLoad
from eureca_building.thermal_zone import ThermalZone
from eureca_building.schedule import Schedule
from eureca_building.setpoints import SetpointDualBand
from eureca_building.ventilation import Infiltration,MechanicalVentilation
from eureca_building.air_handling_unit import AirHandlingUnit
from ir_sky_forcing import GuardUnusedInvalidFields
from single_commit_1c import solve_1c_single_commit,select_solution
# Geometry/setup kept exactly as frozen primary run; checker formula is separate.
tree=ast.parse((ROOT/'05_simulations/climate_campaign_v4/Padova/adapter_snapshot.py').read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
zonefn=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='zone')
exec(compile(ast.fix_missing_locations(ast.Module(body=[zonefn],type_ignores=[])),'frozen-zone-setup','exec'))
results={}
for site in ['Padova','Bolzano','Roma','Palermo']:
    path=ROOT/f'05_simulations/climate_inputs_v2/{site}.epw'
    weather=WeatherFile(str(path),year=2023,time_steps=1,azimuth_subdivisions=8,height_subdivisions=4)
    r=pd.read_csv(path,skiprows=8,header=None)
    old=weather.general_data['average_dt_air_sky']
    delta=float(np.mean(r[6].to_numpy()-(np.power(r[12].to_numpy()/5.6697e-8,.25)-273.15)))
    weather.general_data['average_dt_air_sky']=delta
    weather.hourly_data=GuardUnusedInvalidFields(weather.hourly_data)
    saved=pd.read_csv(ROOT/f'05_simulations/climate_campaign_v4/{site}/seeded_hourly.csv')
    z=zone(saved.heating_setpoint_C.to_numpy())
    surface_records=[{'name':s.name,'type':s.surface_type,'opaque_area_m2':getattr(s,'_opaque_area',None),'has_window':hasattr(s,'window')} for s in z._surface_list]
    original_solar=copy.deepcopy(weather.hourly_data_irradiances)
    mismatch=max(np.max(abs(getattr(z,attr)-saved[col].to_numpy())) for attr,col in [('phi_ia','phi_air_W'),('phi_st','phi_surface_W'),('phi_m','phi_mass_W')])
    # Recover aggregate radiative+solar forcing from the mass-node split.
    base=z.phi_m.copy()/(z.Am/z.Atot)
    weather.general_data['average_dt_air_sky']=0.
    z.calculate_zone_loads_ISO13790(weather)
    zero_delta=z.phi_m.copy()/(z.Am/z.Atot)
    coeff=sum(s._sky_view_factor*s._construction._R_se*s._construction._u_value_net*s._opaque_area*s.get_surface_external_radiative_coefficient() for s in z._surface_list if s.surface_type in ['Roof','ExtWall'])
    expected=-coeff*delta
    maxerr=float(np.max(abs((base-zero_delta)-expected)))
    # Zero both direct and global irradiation, rerun genuine preprocessing.
    weather.general_data['average_dt_air_sky']=delta
    for az,v in weather.hourly_data_irradiances.items():
        for ht,d in v.items():
            d['direct']=np.zeros(8760);d['global']=np.zeros(8760)
    z.calculate_zone_loads_ISO13790(weather)
    phi=z.phi_m/(z.Am/z.Atot)
    # ElectricLoad radiant contribution is 240*.99*.2, same declared inputs.
    zeroerr=float(np.max(abs(phi-(240*.99*.2+expected))))
    # A permutation of the same physical surfaces must not alter forcing.
    z._surface_list[0],z._surface_list[3]=z._surface_list[3],z._surface_list[0]
    z.calculate_zone_loads_ISO13790(weather)
    reordered_zero=z.phi_m/(z.Am/z.Atot)
    zero_order=float(np.max(abs(reordered_zero-phi)))
    weather.hourly_data_irradiances=original_solar
    z.calculate_zone_loads_ISO13790(weather)
    reordered_solar=z.phi_m/(z.Am/z.Atot)
    full_order=float(np.max(abs(reordered_solar-base)))
    roof=next(s for s in z._surface_list if s.surface_type=='Roof')
    z._surface_list.remove(roof);z._surface_list.insert(0,roof)
    try:
        z.calculate_zone_loads_ISO13790(weather)
        roof_first_error=None
    except Exception as e:
        roof_first_error=type(e).__name__+': '+str(e)
    guards={}
    for key in ['wind_speed','opaque_sky_coverage']:
        try:weather.hourly_data[key];guards[key]=False
        except RuntimeError:guards[key]=True
    trials=[]
    for oldT,lower,upper in [(10,19,50),(25,19,50),(20,19,50),(20,22,50),(60,19,50)]:
        z.Ta0=float(oldT);z.Tm0=[float(oldT)]*2
        chosen,mode,sigma=select_solution(z,lower,upper,[0.,24.],15.,15.,[190.08,0.,0.])
        trials.append({'old_C':oldT,'lower':lower,'upper':upper,'mode':mode,'q_W':float(chosen[0]),'old_state_unchanged':z.Ta0==oldT and z.Tm0==[oldT,oldT]})
    results[site]={'old_invalid_mean_delta_K':old,'new_mean_delta_K':delta,
      'saved_node_forcing_max_abs_error_W':float(mismatch),'longwave_coefficient_W_K':float(coeff),
      'longwave_term_W':float(expected),'actual_preprocessing_delta_max_error_W':maxerr,
      'zero_solar_recovered_forcing_max_error_W':zeroerr,'runtime_guards':guards,
      'surfaces':surface_records,'same_surface_set_reordered_zero_solar_max_difference_W':zero_order,
      'same_surface_set_reordered_full_solar_max_difference_W':full_order,'roof_first_error':roof_first_error,
      'actual_native_longwave_term_W':float(np.mean(base-zero_delta)),
      'direct_transition_trials':trials,'native_index_first':str(weather._epw_hourly_data.index[0])}
(OUT/'runtime_forcing_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
