"""Independent preprocessing-only test: no annual thermal simulation is run."""
from pathlib import Path
import ast,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_simulations/native_longwave_preprocessing_v4';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,r'C:\Works\EUReCA\EUReCA')
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
from ir_sky_forcing import apply_ir_sky_forcing,verify_native_longwave
source=ROOT/'04_analysis/run_climate_campaign_v4.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
zone_function=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='zone')
result={}
for site in ['Padova','Bolzano','Roma','Palermo']:
    directory=OUT/site;directory.mkdir(exist_ok=True)
    epw=ROOT/f'05_simulations/climate_inputs_v2/{site}.epw'
    weather=WeatherFile(str(epw),year=2023,time_steps=1,azimuth_subdivisions=8,height_subdivisions=4)
    apply_ir_sky_forcing(weather,epw,directory,site,ROOT/'05_simulations/weather_forcing_v4/weather_forcing_audit.json')
    namespace=globals().copy();namespace['weather']=weather
    exec(compile(ast.fix_missing_locations(ast.Module(body=[zone_function],type_ignores=[])),str(source),'exec'),namespace)
    lower=pd.read_csv(ROOT/f'05_simulations/climate_campaign_v4/{site}/seeded_hourly.csv')['heating_setpoint_C'].to_numpy()
    z=namespace['zone'](lower)
    corrected=[z.phi_ia.copy(),z.phi_st.copy(),z.phi_m.copy()]
    verify_native_longwave(z,weather,directory,'preprocess')
    longwave=json.loads((directory/'preprocess_longwave_check.json').read_text())
    loss=-longwave['opaque_longwave_term_W']
    weather.general_data['average_dt_air_sky']=0.
    z.calculate_zone_loads_ISO13790(weather)
    no_sky=[z.phi_ia,z.phi_st,z.phi_m]
    weights=[0.,1-z.Am/z.Atot-z.Htr_w/(9.1*z.Atot),z.Am/z.Atot]
    errors=[float(np.max(np.abs((a-b)-(-loss*w)))) for a,b,w in zip(corrected,no_sky,weights)]
    result[site]={'no_annual_thermal_run':True,'comparison':'native preprocessed node forcing with corrected scalar minus native zero-sky reference','opaque_longwave_term_W':-loss,'node_distribution_weights':list(map(float,weights)),'maximum_air_surface_mass_difference_error_W':errors,'pass':max(errors)<1e-8}
    print(site,result[site],flush=True)
(OUT/'native_forcing_difference_checks.json').write_text(json.dumps({'sites':result,'pass':all(v['pass'] for v in result.values())},indent=2))
assert all(v['pass'] for v in result.values())
