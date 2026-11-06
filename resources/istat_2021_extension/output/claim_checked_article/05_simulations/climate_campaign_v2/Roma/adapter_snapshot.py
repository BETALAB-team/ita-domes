"""Frozen illustrative EUReCA 1C schedule sensitivity pilot, no final-energy claim."""
from pathlib import Path
import ast, hashlib, json, sys, time, random, traceback, os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SITE=sys.argv[1]
assert SITE in ['Padova','Bolzano','Roma','Palermo']
RUN=ROOT/f'05_simulations/climate_campaign_v2/{SITE}'
EURECA=Path(r'C:\Works\EUReCA\EUReCA')
GRINS=Path(r'C:\Works\GRINS\GRINS-Tool')
WX=ROOT/f'05_simulations/climate_inputs_v2/{SITE}.epw'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj): Path(p).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def source_user_class():
    tree=ast.parse((GRINS/'classes/EndUse.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='User')
    methods={'create_random_prof','create_prof','substitute_istat_heatingTSP'}
    keep=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='periodi_risc' for t in n.targets)]
    cls.body=keep
    env={'np':np,'random':random}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(GRINS/'classes/EndUse.py'),'exec'),env)
    return env['User']

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    os.chdir(RUN)
    (RUN/'adapter_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for pname in ['simulation_protocol.md','simulation_protocol_v2_addendum.md']:
        (RUN/pname).write_bytes((ROOT/'03_methods'/pname).read_bytes())
    audit=json.loads((ROOT/'04_analysis/input_audit.json').read_text())
    audit['sha256']={k:v for k,v in audit['sha256'].items() if not k.endswith('.epw')}
    audit['sha256'][str(WX)]=sha(WX)
    audit['sha256'][str(WX.with_suffix('.json'))]=sha(WX.with_suffix('.json'))
    response=audit['selected_survey_response']|{'Fascia_clima':'E'}
    cfg=json.loads((EURECA/'eureca_building/config.json').read_text())
    cfg['model']['Output path']=str(RUN/'engine_output')
    cfg['simulation settings'].update({'Building energy model':'1C','time steps per hour':'1','simulation reference year':'2023','start date':'01-01 00:00','final date':'12-31 23:00'})
    cfg['solar radiation settings']['do solar shading calculation']='False'
    write(RUN/'eureca_config.json',cfg)
    design={'site':SITE,'protocol':'03_methods/simulation_protocol_v2_addendum.md','response':response,'building':{'width_m':10,'depth_m':8,'height_m':3,'floor_m2':80,'volume_m3':240,'U_wall':1.0,'U_roof':0.8,'U_floor':0.8,'U_window':2.8,'g_window':0.65,'wwr':0.2,'frame_factor':0.15,'weight_class':'Heavy','internal_mass_area_m2':[400,160],'gain_W':240,'infiltration_ACH':0.3},'cases':[['seeded',168],['early',168],['late',168],['repeat_seeded',168],['warmup_seeded',336]],'cooling_upper_C':50,'calendar':'2023 nonleap fixed UTC+1 no DST','energy_boundary':'ideal sensible heating thermal demand, no system/final energy'}
    write(RUN/'design.json',design)
    frozen={'script_sha256':sha(__file__),'config_sha256':sha(RUN/'eureca_config.json'),'design_sha256':sha(RUN/'design.json'),'protocol_sha256':sha(RUN/'simulation_protocol.md'),'protocol_v2_sha256':sha(RUN/'simulation_protocol_v2_addendum.md'),'package_source_sha256':{str(p):sha(p) for p in sorted((EURECA/'eureca_building').glob('*.py'))},'inputs':audit['sha256'],'command':f'{sys.executable} -B {__file__}','python':sys.version,'executable':sys.executable,'status':'frozen before engine execution'}
    write(RUN/'frozen_manifest.json',frozen)
    sys.path.insert(0,str(EURECA))
    from eureca_building.config import load_config
    load_config(str(RUN/'eureca_config.json'))
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
    import eureca_building.thermal_zone as tz_module
    assert Path(tz_module.__file__).resolve().is_relative_to(EURECA)
    print('PINNED_IMPORT',tz_module.__file__,flush=True)
    weather=WeatherFile(str(WX),year=2023,time_steps=1,azimuth_subdivisions=8,height_subdivisions=4)
    print('WEATHER_INDEX',str(weather._epw_hourly_data.index[0]),str(weather._epw_hourly_data.index[-1]),flush=True)
    raw=pd.read_csv(WX,skiprows=8,header=None)
    assert len(raw)==8760 and len(raw[[1,2,3]].drop_duplicates())==8760
    for c,lo,hi in [(6,-70,70),(8,0,100),(13,-0.1,1500),(14,-0.1,1500),(15,-0.1,1500)]:
        assert np.isfinite(raw[c].to_numpy()).all() and raw[c].between(lo,hi).all(),f'weather QC col{c}'
    SourceUser=source_user_class()
    def heating(mode):
        u=SourceUser()
        if mode in ['early','late']:
            def block(n,seed):
                n=int(n); start=0 if mode=='early' else 8-n
                return np.r_[np.zeros(start),np.ones(n),np.zeros(8-start-n)]
            u.create_random_prof=block
        u.substitute_istat_heatingTSP(response)
        prof=u.create_prof(response['q_3_11A'],response['q_3_11B'],response['q_3_11C'],response['id'])
        night=np.r_[prof[21:],prof[:5]]
        assert [sum(prof[5:13]),sum(prof[13:21]),sum(night)]==[4,4,1]
        assert len(u.heatingTSP)==8760
        return u.heatingTSP,prof
    def zone(lower):
        cs={typ:Construction.from_U_value(typ,u,weight_class='Heavy',construction_type=typ) for typ,u in [('ExtWall',1.),('Roof',.8),('GroundFloor',.8),('IntWall',1.)]}
        win=SimpleWindow(name='assumed_window',u_value=2.8,solar_heat_gain_coef=.65,frame_factor=.15,shading_coef_int=.99,shading_coef_ext=.99)
        surfaces=[]
        for az,area in [(0.,30.),(90.,24.),(180.,30.),(-90.,24.)]:
            surfaces.append(Surface.from_area_azimuth_height(f'wall{az}',area=area,height=90.,azimuth=az,wwr=.2,surface_type='ExtWall',construction=cs['ExtWall'],window=win))
        for typ,height in [('Roof',0.),('GroundFloor',180.)]:
            surfaces.append(Surface.from_area_azimuth_height(typ,area=80.,height=height,azimuth=0.,wwr=0.,surface_type=typ,construction=cs[typ]))
        surfaces.extend([SurfaceInternalMass('internal_walls',area=400.,surface_type='IntWall',construction=cs['IntWall']),SurfaceInternalMass('internal_ceiling',area=160.,surface_type='IntCeiling',construction=cs['IntWall'])])
        z=ThermalZone('illustrative80m2',surface_list=surfaces,net_floor_area=80.,volume=240.,number_of_units=1)
        z._ISO13790_params()
        def schedule(name,typ,a):return Schedule(name,typ,np.array(a,dtype=float))
        def constant(name,typ,x):return Schedule.from_constant_value(name=name,schedule_type=typ,value=float(x))
        z.add_internal_load(ElectricLoad(name='constant_internal_gain',unit='W',nominal_value=240.,schedule=constant('gain','dimensionless',1),fraction_radiant=.2,fraction_convective=.8,fraction_to_zone=.99))
        z.extract_convective_radiative_latent_electric_load()
        z.calculate_zone_loads_ISO13790(weather)
        z.add_temperature_setpoint(SetpointDualBand('temperature','temperature',schedule_lower=schedule('lower','temperature',lower),schedule_upper=constant('upper','temperature',50)))
        z.add_humidity_setpoint(SetpointDualBand('humidity','relative_humidity',schedule_lower=constant('rh_lower','dimensionless',0),schedule_upper=constant('rh_upper','dimensionless',1)))
        z.add_infiltration(Infiltration(name='infiltration',unit='Vol/h',nominal_value=.3,schedule=constant('infiltration_schedule','dimensionless',1)))
        z.calc_infiltration(weather)
        vent=MechanicalVentilation(name='zero_mechanical_ventilation',unit='Vol/h',nominal_value=0.,schedule=constant('vent','dimensionless',0))
        AirHandlingUnit('zero_AHU',vent,constant('supply_temperature','temperature',23),constant('supply_humidity','dimensionless',.007),constant('ahu_availability','availability',0),True,0.,0.,.99,weather,z)
        z.reset_init_values(T=15.,X=.007)
        return z
    summaries=[]
    for name,warmup in design['cases']:
        started=time.perf_counter()
        mode=name if name in ['early','late'] else 'seeded'
        lower,prof=heating(mode)
        z=zone(lower)
        rows=[]
        for h in range(-warmup,8760):
            z.solve_timestep(h,weather,flag=z.flag,model='1C')
            if h>=0: rows.append((h,z.sensible_zone_load,z.zone_air_temperature,lower[h],weather.hourly_data['out_air_db_temperature'][h]))
        out=pd.DataFrame(rows,columns=['hour','sensible_load_W','air_temperature_C','heating_setpoint_C','outdoor_temperature_C'])
        out.to_csv(RUN/f'{name}_hourly.csv',index=False)
        q=out.sensible_load_W.to_numpy()
        active=q>1e-6
        summary={'case':name,'warmup_h':warmup,'annual_sensible_heating_kWh':float(q.clip(0).sum()/1000),'peak_sensible_heating_kW':float(q.clip(0).max()/1000),'annual_sensible_cooling_kWh':float(-q.clip(max=0).sum()/1000),'minimum_air_C':float(out.air_temperature_C.min()),'maximum_air_C':float(out.air_temperature_C.max()),'active_heating_max_setpoint_error_K':float(np.abs(out.air_temperature_C.to_numpy()[active]-lower[active]).max(initial=0)),'daily_on_profile':prof.tolist(),'runtime_seconds':time.perf_counter()-started,'rows':len(out),'finite':bool(np.isfinite(out.to_numpy()).all()),'hourly_sha256':sha(RUN/f'{name}_hourly.csv')}
        summaries.append(summary)
        write(RUN/'summary.json',summaries)
        print('RUN_COMPLETE',json.dumps(summary),flush=True)
    pd.DataFrame([{k:v for k,v in x.items() if k!='daily_on_profile'} for x in summaries]).to_csv(RUN/'summary.csv',index=False)
    base=pd.read_csv(RUN/'seeded_hourly.csv'); repeat=pd.read_csv(RUN/'repeat_seeded_hourly.csv')
    checks={'block_totals_exact':True,'all8760_finite':all(s['rows']==8760 and s['finite'] for s in summaries),'zero_cooling':all(s['annual_sensible_cooling_kWh']<1e-9 for s in summaries),'temperature_load_screens':all(-10<s['minimum_air_C'] and s['maximum_air_C']<50 and s['peak_sensible_heating_kW']<100 for s in summaries),'setpoint_adherence':all(s['active_heating_max_setpoint_error_K']<=.1 for s in summaries),'repeat_max_difference':float(np.max(np.abs(base.to_numpy()-repeat.to_numpy())))}
    for key,limit in [('annual_sensible_heating_kWh',.005),('peak_sensible_heating_kW',.01)]:
        rel=abs(summaries[-1][key]/summaries[0][key]-1)
        checks['warmup_'+key]={'relative_difference':rel,'limit':limit,'pass':rel<=limit}
    checks['pass']=all(checks[k] for k in ['block_totals_exact','all8760_finite','zero_cooling','temperature_load_screens','setpoint_adherence']) and checks['repeat_max_difference']<=1e-9 and all(v['pass'] for v in checks.values() if isinstance(v,dict))
    write(RUN/'checks.json',checks)
    print('CHECKS',json.dumps(checks),flush=True)
    frozen.update({'status':'complete','imported_zone_module':tz_module.__file__,'checks_pass':checks['pass'],'summary_sha256':sha(RUN/'summary.csv'),'elapsed_run_seconds':sum(s['runtime_seconds'] for s in summaries)})
    write(RUN/'completion_manifest.json',frozen)
if __name__=='__main__':
    try: main()
    except Exception:
        RUN.mkdir(parents=True,exist_ok=True)
        (RUN/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
