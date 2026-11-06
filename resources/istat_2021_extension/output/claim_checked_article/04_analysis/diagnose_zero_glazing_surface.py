"""Read-only-source, preprocessing-only proof of windowless-roof forcing issue."""
from pathlib import Path
import ast,inspect,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_simulations/zero_glazing_surface_diagnostic';OUT.mkdir(parents=True,exist_ok=True)
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
epw=ROOT/'05_simulations/climate_inputs_v2/Padova.epw'
weather=WeatherFile(str(epw),year=2023,time_steps=1,azimuth_subdivisions=8,height_subdivisions=4)
apply_ir_sky_forcing(weather,epw,OUT,'Padova',ROOT/'05_simulations/weather_forcing_v4/weather_forcing_audit.json')
source=ROOT/'04_analysis/run_climate_campaign_v4.py'
tree=ast.parse(source.read_text(encoding='utf-8')); main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');func=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='zone')
ns=globals().copy();exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(source),'exec'),ns)
lower=pd.read_csv(ROOT/'05_simulations/climate_campaign_v4/Padova/seeded_hourly.csv')['heating_setpoint_C'].to_numpy()
src_lines,start=inspect.getsourcelines(ThermalZone.calculate_zone_loads_ISO13790)
target=start+next(i for i,line in enumerate(src_lines) if 'phi_sol_op = F_sh_urban_shading * TRV' in line)
trace=[]
def observer(frame,event,arg):
    if event=='line' and frame.f_code==ThermalZone.calculate_zone_loads_ISO13790.__code__ and frame.f_lineno==target:
        d=frame.f_locals;s=d.get('surface')
        if s is not None and s.surface_type=='Roof':
            correct=weather.hourly_data_irradiances[float(s._azimuth_round)][float(s._height_round)]['global']
            trace.append({'source_line':target,'surface':'Roof','has_window_attribute':hasattr(s,'window'),'actual_roof_opaque_area_m2':s._opaque_area,'local_A_op_used_m2':float(d['A_op']),'local_TRV_equals_roof_irradiance':bool(np.array_equal(d['TRV'],correct))})
    return observer
sys.settrace(observer)
old=ns['zone'](lower)
sys.settrace(None)
class AttachZeroAreaWindow(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if any(k.arg=='surface_type' and isinstance(k.value,ast.Name) and k.value.id=='typ' for k in node.keywords):
            node.keywords.append(ast.keyword(arg='window',value=ast.Name(id='win',ctx=ast.Load())))
        return node
func=AttachZeroAreaWindow().visit(func)
ns2=globals().copy();exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(source),'exec'),ns2)
fixed=ns2['zone'](lower)
verify_native_longwave(fixed,weather,OUT,'fixed')
actual=[fixed.phi_ia.copy(),fixed.phi_st.copy(),fixed.phi_m.copy()]
loss=-json.loads((OUT/'fixed_longwave_check.json').read_text())['opaque_longwave_term_W']
weather.general_data['average_dt_air_sky']=0.;fixed.calculate_zone_loads_ISO13790(weather)
weights=[0.,1-fixed.Am/fixed.Atot-fixed.Htr_w/(9.1*fixed.Atot),fixed.Am/fixed.Atot]
errors=[float(np.max(np.abs((a-b)+loss*w))) for a,b,w in zip(actual,[fixed.phi_ia,fixed.phi_st,fixed.phi_m],weights)]
same={k:float(getattr(old,k))==float(getattr(fixed,k)) for k in ['Htr_is','Htr_w','Htr_ms','Htr_em','Cm','_air_thermal_capacity','Atot','Am']}
roof=next(s for s in fixed._surface_list if s.surface_type=='Roof')
result={'no_annual_simulation':True,'original_roof_trace':trace,'local_object_workaround':'Attach the same window descriptor to zero-glazed roof/floor; glazed area remains zero. This makes the native branch initialize its own area and irradiance. No external source edit.','fixed_roof_glazed_area_m2':roof._glazed_area,'unchanged_network_parameters':same,'fixed_preprocessing_longwave_difference_max_air_surface_mass_error_W':errors,'workaround_pass':all(same.values()) and roof._glazed_area==0 and max(errors)<1e-8}
(OUT/'diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
