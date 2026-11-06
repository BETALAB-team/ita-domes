"""Transparent hourly 1R1C approximation, not a substitute for a validated model."""
import calendar
import math
import numpy as np
import pandas as pd


def read_weather(path):
    with open(path,encoding='utf-8-sig',errors='replace') as f: header=f.readline().strip().split(',')
    if header[0]!='LOCATION':raise ValueError(f'Not an EPW: {path}')
    data=pd.read_csv(path,skiprows=8,header=None)
    # Use a common non-leap calendar, irrespective of source-year blocks in TMY.
    data=data[~((data[1]==2)&(data[2]==29))].copy()
    if len(data)!=8760:raise ValueError(f'Expected 8760 non-leap hourly EPW records, got {len(data)}.')
    dry=data[6].to_numpy(float);dni=data[14].to_numpy(float);dhi=data[15].to_numpy(float)
    if not np.isfinite(dry).all() or np.any(abs(dry)>80) or np.any((dni<0)|(dni>=9999)|(dhi<0)|(dhi>=9999)):
        raise ValueError('Missing/out-of-range EPW temperature or direct/diffuse radiation.')
    return {'city':header[1], 'latitude':float(header[6]), 'longitude':float(header[7]),
            'timezone':float(header[8]),'elevation':float(header[9]),'dry':dry,'dni':dni,'dhi':dhi,
            'month':data[1].to_numpy(int),'hour':data[3].to_numpy(int)-1}


def geometry(s,context):
    length=math.sqrt(s['area_m2']*s['aspect_ratio']);width=s['area_m2']/length;h=s['height_m']
    walls={'South':length*h,'North':length*h,'East':width*h,'West':width*h}
    exposed=sum(walls[w] for w in context['exposed_walls']);windows=exposed*s['window_wall_ratio']
    hout=(exposed-windows)*s['wall_u']+windows*s['window_u']
    if context['roof']=='Outdoors':hout+=s['area_m2']*s['roof_u']
    hout+=1.2*1006*s['area_m2']*h*s['infiltration_ach']/3600
    ground=s['area_m2']*s['floor_u'] if context['floor']=='Ground' else 0.
    return {'length':length,'width':width,'walls':walls,'window_area':windows,'h_out':hout,'h_ground':ground}


def daily_operation(hours,kind):
    if not np.isfinite(hours) or not 0<=hours<=24:raise ValueError('Operating hours must be in [0,24].')
    priority=([18,19,20,21,6,7,8,17,22,9,16,10,15,11,14,12,13,23,5,4,3,2,1,0] if kind=='heat'
              else [14,15,16,17,18,19,20,21,13,12,22,11,23,10,9,8,7,6,5,4,3,2,1,0])
    values=np.zeros(24);left=hours
    for hour in priority:
        values[hour]=min(1.,left);left=max(0.,left-1)
    return values


def schedules(s,profile):
    heat=daily_operation(float(profile['heating_hours']),'heat')
    cool=daily_operation(float(profile['cooling_hours']),'cool')
    occ=np.asarray(profile.get('occupancy_profile',s['occupancy_profile']),float)
    app=np.asarray(profile.get('appliance_profile',s['appliance_profile']),float)
    if len(occ)!=24 or len(app)!=24 or app.sum()<=0 or np.any(occ<0) or np.any(occ>1):raise ValueError('Invalid per-archetype hourly profile.')
    watts=float(profile['appliance_kwh'])*1000/365*app/app.sum()
    return {'heat':s['heat_setback_c']+heat*(s['heat_setpoint_c']-s['heat_setback_c']),
            'cool':s['cool_setback_c']+cool*(s['cool_setpoint_c']-s['cool_setback_c']) if s['cooling_enabled'] else np.full(24,60.),
            'occupancy':occ,'appliance_w':watts,
            'gains_w':occ*profile['members']*s['people_sensible_w']+watts*s['appliance_heat_fraction']}


def solar_gain(weather,s,context):
    """Approximate isotropic sky transposition to vertical exposed windows."""
    g=geometry(s,context);day=np.repeat(np.arange(1,366),24);hour=np.tile(np.arange(24)+.5,365)
    lat=np.deg2rad(weather['latitude']);b=2*np.pi*(day-81)/364
    eq=9.87*np.sin(2*b)-7.53*np.cos(b)-1.5*np.sin(b)
    solar_hour=hour+(4*(weather['longitude']-15*weather['timezone'])+eq)/60
    ha=np.deg2rad(15*(solar_hour-12));dec=np.deg2rad(23.45)*np.sin(2*np.pi*(284+day)/365)
    up=np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(ha)
    south=np.sin(lat)*np.cos(dec)*np.cos(ha)-np.cos(lat)*np.sin(dec)
    west=np.cos(dec)*np.sin(ha)
    horizontal=weather['dni']*np.maximum(up,0)+weather['dhi'];gain=np.zeros(8760)
    azimuth={'South':0,'West':90,'North':180,'East':270}
    for wall in context['exposed_walls']:
        az=np.deg2rad(azimuth[wall]+s['north_axis_deg'])
        cosine=np.maximum(south*np.cos(az)+west*np.sin(az),0)*(up>0)
        irradiance=weather['dni']*cosine+.5*weather['dhi']+.1*horizontal
        gain+=irradiance*g['walls'][wall]*s['window_wall_ratio']*s['window_shgc']
    return gain


def simulate_rc(weather,s,context,profile):
    g=geometry(s,context);sch=schedules(s,profile);h=g['h_out']+g['h_ground']
    if h<=0:raise ValueError('No heat exchange in simplified model.')
    capacity=s['rc_capacity_kj_m2k']*1000*s['area_m2'];decay=np.exp(-h*3600/capacity)
    gains=np.tile(sch['gains_w'],365)+solar_gain(weather,s,context)
    heat=np.zeros(8760);cool=np.zeros(8760);temperature=np.zeros(8760);t=20.
    # Repeat the first week for spin-up; do not count it as annual demand.
    indices=list(range(168))*4+list(range(8760))
    for step,i in enumerate(indices):
        equilibrium=(g['h_out']*weather['dry'][i]+g['h_ground']*s['ground_temperature_c']+gains[i])/h
        free=equilibrium+(t-equilibrium)*decay
        lo,hi=sch['heat'][i%24],sch['cool'][i%24] if s['cooling_enabled'] else np.inf
        target=min(max(free,lo),hi);power=(target-free)*h/(1-decay);t=target
        if step>=672:heat[i]=max(power,0)/1000;cool[i]=max(-power,0)/1000;temperature[i]=t
    return pd.DataFrame({'month':weather['month'],'heating_kwh':heat,'cooling_kwh':cool,
                         'appliance_kwh':np.tile(sch['appliance_w'],365)/1000,'temperature_c':temperature})
