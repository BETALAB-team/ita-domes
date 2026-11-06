"""Generate a simple single-zone epJSON, run the installed engine, read SQLite."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import numpy as np
import pandas as pd
from .physics import geometry,schedules


def make_model(cfg,context,profile,weather):
    s=cfg['simulation'];g=geometry(s,context);L,W,H=g['length'],g['width'],s['height_m'];m={}
    def put(kind,name,**values):m.setdefault(kind,{})[name]=values
    put('Version','Version',version_identifier='25.1')
    put('Building','Reference dwelling',north_axis=s['north_axis_deg'],terrain='Suburbs',solar_distribution='FullExterior',maximum_number_of_warmup_days=50,minimum_number_of_warmup_days=6)
    put('SimulationControl','Control',do_zone_sizing_calculation='No',do_system_sizing_calculation='No',do_plant_sizing_calculation='No',run_simulation_for_sizing_periods='No',run_simulation_for_weather_file_run_periods='Yes')
    put('Timestep','Step',number_of_timesteps_per_hour=s['timestep'])
    put('RunPeriod','Annual',begin_month=1,begin_day_of_month=1,end_month=12,end_day_of_month=31,day_of_week_for_start_day='Monday',use_weather_file_daylight_saving_period='No',use_weather_file_holidays_and_special_days='No')
    put('GlobalGeometryRules','Rules',starting_vertex_position='UpperLeftCorner',vertex_entry_direction='Counterclockwise',coordinate_system='Relative')
    put('Site:Location','Site',latitude=weather['latitude'],longitude=weather['longitude'],time_zone=weather['timezone'],elevation=weather['elevation'])
    import calendar
    put('Site:GroundTemperature:BuildingSurface','Ground',**{f'{calendar.month_name[i].lower()}_ground_temperature':s['ground_temperature_c'] for i in range(1,13)})
    put('Zone','Dwelling',direction_of_relative_north=0,x_origin=0,y_origin=0,z_origin=0,multiplier=1,ceiling_height=H,volume=s['area_m2']*H,floor_area=s['area_m2'])
    for name,u in [('Wall',s['wall_u']),('Roof',s['roof_u']),('Floor',s['floor_u'])]:
        # Nominal films .17 m²K/W; EnergyPlus calculates actual convection.
        put('Material',name,roughness='MediumRough',thickness=.15,conductivity=.15/(1/u-.17),density=1000,specific_heat=1000,thermal_absorptance=.9,solar_absorptance=.6,visible_absorptance=.6)
        put('Construction',name,outside_layer=name)
    put('WindowMaterial:SimpleGlazingSystem','Glass',u_factor=s['window_u'],solar_heat_gain_coefficient=s['window_shgc'],visible_transmittance=.7)
    put('Construction','Window',outside_layer='Glass')
    coords={'South':[(0,0,H),(0,0,0),(L,0,0),(L,0,H)],'East':[(L,0,H),(L,0,0),(L,W,0),(L,W,H)],
            'North':[(L,W,H),(L,W,0),(0,W,0),(0,W,H)],'West':[(0,W,H),(0,W,0),(0,0,0),(0,0,H)],
            'Roof':[(0,W,H),(0,0,H),(L,0,H),(L,W,H)],'Floor':[(0,0,0),(0,W,0),(L,W,0),(L,0,0)]}
    for face,vertices in coords.items():
        boundary=context['roof'] if face=='Roof' else context['floor'] if face=='Floor' else 'Outdoors' if face in context['exposed_walls'] else 'Adiabatic'
        kind=face if face in ['Roof','Floor'] else 'Wall'
        put('BuildingSurface:Detailed',face,surface_type=kind,construction_name=kind,zone_name='Dwelling',outside_boundary_condition=boundary,sun_exposure='SunExposed' if boundary=='Outdoors' else 'NoSun',wind_exposure='WindExposed' if boundary=='Outdoors' else 'NoWind',number_of_vertices=4,vertices=[dict(zip(['vertex_x_coordinate','vertex_y_coordinate','vertex_z_coordinate'],v)) for v in vertices])
        if face in context['exposed_walls'] and s['window_wall_ratio']>0:
            center=np.mean(vertices,axis=0);points=center+(np.array(vertices)-center)*np.sqrt(s['window_wall_ratio'])
            fields={f'vertex_{i}_{axis}_coordinate':float(v) for i,pt in enumerate(points,1) for axis,v in zip('xyz',pt)}
            put('FenestrationSurface:Detailed',face+' Window',surface_type='Window',construction_name='Window',building_surface_name=face,number_of_vertices=4,**fields)
    put('ScheduleTypeLimits','Any',numeric_type='Continuous')
    put('Schedule:Constant','Always',schedule_type_limits_name='Any',hourly_value=1)
    put('Schedule:Constant','Cooling available',schedule_type_limits_name='Any',hourly_value=int(s['cooling_enabled']))
    put('Schedule:Constant','ControlType',schedule_type_limits_name='Any',hourly_value=4)
    put('Schedule:Constant','Activity',schedule_type_limits_name='Any',hourly_value=s['people_sensible_w'])
    sch=profile.get('annual_schedules') or schedules(s,profile)
    def daily(name,values):
        if len(values)==8760:
            # Explicit non-leap year, Jan 1 Monday. Preserve every supplied hour.
            from datetime import date,timedelta
            day_names=[];unique={}
            for day in np.asarray(values).reshape(365,24):
                key=tuple(float(v) for v in day)
                if key not in unique:
                    dn=f'{name} Day {len(unique)+1}';unique[key]=dn
                    put('Schedule:Day:Hourly',dn,schedule_type_limits_name='Any',**{f'hour_{i+1}':v for i,v in enumerate(key)})
                day_names.append(unique[key])
            weeks=[]
            for first in range(0,365,7):
                last=min(first+6,364);wn=f'{name} Week {first//7+1}'
                fields={f'{d}_schedule_day_name':day_names[min(first+i,364)] for i,d in enumerate(['monday','tuesday','wednesday','thursday','friday','saturday','sunday'])}
                fields.update({f'{d}_schedule_day_name':day_names[first] for d in ['holiday','summerdesignday','winterdesignday','customday1','customday2']})
                put('Schedule:Week:Daily',wn,**fields)
                start,end=[date(2001,1,1)+timedelta(days=d) for d in [first,last]]
                weeks.append({'schedule_week_name':wn,'start_month':start.month,'start_day':start.day,'end_month':end.month,'end_day':end.day})
            put('Schedule:Year',name,schedule_type_limits_name='Any',schedule_weeks=weeks)
            return
        put('Schedule:Day:Hourly',name+' Day',schedule_type_limits_name='Any',**{f'hour_{i+1}':float(x) for i,x in enumerate(values)})
        fields={f'{x}_schedule_day_name':name+' Day' for x in ['sunday','monday','tuesday','wednesday','thursday','friday','saturday','holiday','summerdesignday','winterdesignday','customday1','customday2']}
        put('Schedule:Week:Daily',name+' Week',**fields)
        put('Schedule:Year',name,schedule_type_limits_name='Any',schedule_weeks=[{'schedule_week_name':name+' Week','start_month':1,'start_day':1,'end_month':12,'end_day':31}])
    for name,key in [('Heating','heat'),('Cooling','cool'),('Occupancy','occupancy'),('Appliances','appliance_w')]:daily(name,sch[key])
    put('People','Household',zone_or_zonelist_or_space_or_spacelist_name='Dwelling',number_of_people_schedule_name='Occupancy',number_of_people_calculation_method='People',number_of_people=profile['members'],fraction_radiant=.3,sensible_heat_fraction=1,activity_level_schedule_name='Activity')
    put('ElectricEquipment','Appliances',zone_or_zonelist_or_space_or_spacelist_name='Dwelling',schedule_name='Appliances',design_level_calculation_method='EquipmentLevel',design_level=1,fraction_latent=0,fraction_radiant=.3*s['appliance_heat_fraction'],fraction_lost=1-s['appliance_heat_fraction'])
    put('ZoneInfiltration:DesignFlowRate','Infiltration',zone_or_zonelist_or_space_or_spacelist_name='Dwelling',schedule_name='Always',design_flow_rate_calculation_method='AirChanges/Hour',air_changes_per_hour=s['infiltration_ach'],constant_term_coefficient=1,temperature_term_coefficient=0,velocity_term_coefficient=0,velocity_squared_term_coefficient=0)
    put('ThermostatSetpoint:DualSetpoint','Setpoints',heating_setpoint_temperature_schedule_name='Heating',cooling_setpoint_temperature_schedule_name='Cooling')
    put('ZoneControl:Thermostat','Thermostat',zone_or_zonelist_name='Dwelling',control_type_schedule_name='ControlType',control_1_object_type='ThermostatSetpoint:DualSetpoint',control_1_name='Setpoints')
    put('ZoneHVAC:IdealLoadsAirSystem','Ideal',availability_schedule_name='Always',zone_supply_air_node_name='Supply',zone_exhaust_air_node_name='Exhaust',maximum_heating_supply_air_temperature=50,minimum_cooling_supply_air_temperature=13,heating_limit='NoLimit',cooling_limit='NoLimit',dehumidification_control_type='None',humidification_control_type='None',outdoor_air_economizer_type='NoEconomizer',heat_recovery_type='None')
    m['ZoneHVAC:IdealLoadsAirSystem']['Ideal']['cooling_availability_schedule_name']='Cooling available'
    put('ZoneHVAC:EquipmentList','Equipment',load_distribution_scheme='SequentialLoad',equipment=[{'zone_equipment_object_type':'ZoneHVAC:IdealLoadsAirSystem','zone_equipment_name':'Ideal','zone_equipment_cooling_sequence':1,'zone_equipment_heating_or_no_load_sequence':1}])
    put('ZoneHVAC:EquipmentConnections','Connections',zone_name='Dwelling',zone_conditioning_equipment_list_name='Equipment',zone_air_inlet_node_or_nodelist_name='Supply',zone_air_exhaust_node_or_nodelist_name='Exhaust',zone_air_node_name='Zone Air',zone_return_air_node_or_nodelist_name='Return')
    put('Output:SQLite','SQLite',option_type='SimpleAndTabular')
    for name in ['Zone Ideal Loads Zone Sensible Heating Energy','Zone Ideal Loads Zone Sensible Cooling Energy','Zone Electric Equipment Electricity Energy','Zone Mean Air Temperature']:
        put('Output:Variable',name,key_value='*',variable_name=name,reporting_frequency='Hourly')
    return m


def run_model(model,epw,exe,folder,timeout,cancel):
    folder.mkdir(parents=True,exist_ok=True);model_path=folder/'model.epJSON';model_path.write_text(json.dumps(model,indent=2))
    cmd=[str(exe),'-w',str(Path(epw).resolve()),'-d',str(folder.resolve()),str(model_path.resolve())]
    start=time.perf_counter()
    with (folder/'console.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=folder,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        while process.poll() is None:
            if cancel.wait(.2) or time.perf_counter()-start>timeout:
                process.kill();process.wait();raise RuntimeError('EnergyPlus cancelled or exceeded configured timeout.')
    errors=(folder/'eplusout.err').read_text(errors='replace') if (folder/'eplusout.err').exists() else ''
    if process.returncode or '** Severe **' in errors or '**  Fatal  **' in errors:
        raise RuntimeError(f'EnergyPlus failed; inspect {folder}/eplusout.err\n{errors[-2500:]}')
    if 'EnergyPlus Completed Successfully' not in errors:raise RuntimeError(f'EnergyPlus success not confirmed: {folder}')
    con=sqlite3.connect(folder/'eplusout.sql')
    query='''SELECT t.TimeIndex,t.Month,d.Name,r.Value FROM ReportData r
      JOIN ReportDataDictionary d ON r.ReportDataDictionaryIndex=d.ReportDataDictionaryIndex
      JOIN Time t ON r.TimeIndex=t.TimeIndex WHERE COALESCE(t.WarmupFlag,0)=0 AND d.ReportingFrequency='Hourly' '''
    raw=pd.read_sql_query(query,con);con.close()
    table=raw.pivot(index=['TimeIndex','Month'],columns='Name',values='Value').reset_index()
    rename={'Zone Ideal Loads Zone Sensible Heating Energy':'heating_kwh','Zone Ideal Loads Zone Sensible Cooling Energy':'cooling_kwh','Zone Electric Equipment Electricity Energy':'appliance_kwh','Zone Mean Air Temperature':'temperature_c','Month':'month'}
    table=table.rename(columns=rename)
    for key in ['heating_kwh','cooling_kwh','appliance_kwh']:
        if key not in table:raise RuntimeError(f'Missing EnergyPlus output {key}')
        table[key]/=3.6e6
    if len(table)!=8760:raise RuntimeError(f'Expected 8760 simulated hours, got {len(table)}')
    table.to_csv(folder/'hourly.csv',index=False)
    return table,time.perf_counter()-start,errors.count('** Warning **')
