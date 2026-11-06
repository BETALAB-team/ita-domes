"""Non-ISTAT schedule inputs and demographic-only assignment.

The built-in profiles are declared synthetic assumptions, not observations or
standard profiles. No survey data or EnergyPlus result enters their generation.
"""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCHEDULE_COLUMNS = ('occupancy', 'heat', 'cool', 'appliance_w')


def default_profiles():
    """Editable experimental cases. Hours refer to hour-beginning 0..23."""
    away = [.95]*7 + [.7] + [.15]*9 + [.7] + [.95]*6
    home = [.95]*8 + [.8]*10 + [.95]*6
    weekend = [.95]*8 + [.7]*10 + [.95]*6
    cases = [
        ('P1', 'Older pair', 2, 72, 77, 5/18, home, list(range(7,22)), list(range(12,22)), 100, 130),
        ('P2', 'Family', 4, 8, 46, 12/18, away, [6,7,17,18,19,20,21,22], list(range(15,23)), 160, 330),
        ('P3', 'Working pair', 2, 40, 48, 12/18, away, [6,7,18,19,20,21], list(range(18,23)), 90, 160),
        ('P4', 'Single adult', 1, 32, 32, 1., away, [6,19,20,21], list(range(18,23)), 65, 110),
    ]
    profiles=[]
    for pid,label,n,lo,hi,edu,occ,heat,cool,base,extra in cases:
        profiles.append({'id':pid,'label':label,'members':n,
            'demographics':{'agemin':lo,'agemax':hi,'nmembers':n,'education':edu},
            'weekday':{'occupancy':occ,'heating_on_hours':heat,'cooling_on_hours':cool,
                       'appliance_w':[base+(extra if h in [7,12,18,19,20,21] else 0) for h in range(24)]},
            'weekend':{'occupancy':weekend,'heating_on_hours':sorted(set(heat+[9,10,11,12])),
                       'cooling_on_hours':sorted(set(cool+[13,14])),
                       'appliance_w':[base+(extra if h in [8,9,12,13,18,19,20,21] else 0) for h in range(24)]}})
    shifted=copy.deepcopy(profiles[1]);shifted.update(id='P2_shift',label='Family, timing shifted',paired_with='P2')
    for kind in ['weekday','weekend']:
        day=shifted[kind]
        for key in ['occupancy','appliance_w']:day[key]=np.roll(day[key],8).tolist()
        for key in ['heating_on_hours','cooling_on_hours']:day[key]=sorted((h+8)%24 for h in day[key])
    profiles.append(shifted)
    return {'schema_version':1,'evidence_kind':'synthetic',
            'source':'Predeclared Energy Studio synthetic stress-test cases; no ISTAT inputs.',
            'calendar':'365 days, 1 January Monday, local standard time; no DST or holidays',
            'profiles':profiles}


def load_profiles(cfg):
    filename=cfg['benchmark']['profiles_file']
    doc=json.loads(Path(filename).read_text(encoding='utf-8-sig')) if filename else default_profiles()
    if doc.get('schema_version')!=1:raise ValueError('Independent profiles require schema_version 1.')
    if doc.get('evidence_kind') not in ['synthetic','external_schedule','measured_schedule']:
        raise ValueError('Declare profile evidence_kind: synthetic, external_schedule or measured_schedule.')
    if not str(doc.get('source','')).strip():raise ValueError('Independent schedules need a source description.')
    items=doc.get('profiles',[])
    ids=[p['id'] for p in items]
    if not items or len(set(ids))!=len(ids):raise ValueError('Independent profile IDs must be nonempty and unique.')
    root=Path(filename).resolve().parent if filename else Path.cwd()
    built={}
    for p in items:
        if not p['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in p['id']):
            raise ValueError('Use letters, digits, underscore or hyphen in profile IDs.')
        built[p['id']]=build_profile(p,cfg,root)
    if cfg['benchmark']['anchor_profile'] not in built:raise ValueError('Independent reference profile does not exist.')
    return doc,built


def build_profile(item,cfg,root=None):
    s=cfg['simulation'];n=float(item['members'])
    if not np.isfinite(n) or n<=0:raise ValueError('Independent member count must be positive.')
    if item.get('hourly_csv'):
        path=Path(item['hourly_csv']);path=path if path.is_absolute() else Path(root or '.')/path
        frame=pd.read_csv(path)
        if 'hour' not in frame or not np.array_equal(frame.hour.to_numpy(),np.arange(1,8761)):
            raise ValueError('Hourly CSV must have ordered hour 1..8760 without gaps or duplicates.')
        arrays={k:frame[k].to_numpy(float) for k in SCHEDULE_COLUMNS}
    else:
        days={}
        for kind in ['weekday','weekend']:
            d=item[kind];days[kind]={}
            for key in ['occupancy','appliance_w']:
                arr=np.asarray(d[key],float)
                if arr.shape!=(24,):raise ValueError(f'{item["id"]}: {kind} {key} needs 24 values.')
                days[kind][key]=arr
            for key,prefix in [('heat','heating'),('cool','cooling')]:
                hours=d[prefix+'_on_hours']
                if len(set(hours))!=len(hours) or any(type(h) is not int or not 0<=h<24 for h in hours):
                    raise ValueError('On-hours must be distinct integers 0..23.')
                days[kind][key]=np.array([s[key+'_setpoint_c'] if h in hours else s[key+'_setback_c'] for h in range(24)])
        arrays={k:np.concatenate([days['weekend' if day%7>=5 else 'weekday'][k] for day in range(365)]) for k in SCHEDULE_COLUMNS}
    validate_arrays(arrays)
    if s['heat_setpoint_c']==s['heat_setback_c'] or s['cool_setpoint_c']==s['cool_setback_c']:
        raise ValueError('Independent aggregate-hour conversion requires different active and setback temperatures.')
    for key in ['heat','cool']:
        lo,hi=sorted([s[key+'_setpoint_c'],s[key+'_setback_c']])
        if arrays[key].min()<lo or arrays[key].max()>hi:
            raise ValueError(f'{key} temperatures must be within configured setpoint/setback bounds.')
    h=24*float(np.mean((arrays['heat']-s['heat_setback_c'])/(s['heat_setpoint_c']-s['heat_setback_c'])))
    c=24*float(np.mean((arrays['cool']-s['cool_setback_c'])/(s['cool_setpoint_c']-s['cool_setback_c'])))
    e=float(arrays['appliance_w'].sum()/1000)
    profile={'members':n,'heating_hours':h,'cooling_hours':c,'appliance_kwh':e,
             'occupancy_profile':arrays['occupancy'].reshape(365,24).mean(axis=0).tolist(),
             'presence_index':float(arrays['occupancy'].mean()),
             'annual_schedules':{k:v.tolist() for k,v in arrays.items()},
             'profile_source':'independent schedule specification'}
    for index,value in [('members',n),('heating',h),('cooling',c),('appliance',e)]:
        profile.update({index+'_index':value,index+'_index_n':None,
                        index+'_index_coverage':None,index+'_index_source':'independent known aggregate'})
    return profile


def validate_arrays(arrays):
    for key in SCHEDULE_COLUMNS:
        v=np.asarray(arrays[key],float)
        if v.shape!=(8760,) or not np.isfinite(v).all():raise ValueError(f'{key} needs 8760 finite values.')
    if np.any((np.asarray(arrays['occupancy'])<0)|(np.asarray(arrays['occupancy'])>1)):raise ValueError('Occupancy must be in [0,1].')
    if np.any(np.asarray(arrays['appliance_w'])<0):raise ValueError('Appliance watts cannot be negative.')
    if np.any(np.asarray(arrays['heat'])>=np.asarray(arrays['cool'])):raise ValueError('Every heating setpoint must be below cooling.')


def classify_demographics(records,model):
    """Same frozen transformation as revised_analysis, without heavy imports."""
    f=pd.DataFrame(records)
    names=['agemin','agemax','nmembers','education'] if 'pca' in model else model['features']
    if any(k not in f for k in names) or not np.isfinite(f[names].to_numpy(float)).all():
        raise ValueError('Complete finite demographic inputs are required for independent assignment.')
    if np.any(f.agemin>f.agemax) or np.any((f.education<0)|(f.education>1)):
        raise ValueError('Invalid demographic ages or education proxy (0..1).')
    if 'pca' in model:
        pc=model['pca'];z=(f[['agemin','agemax','nmembers']].to_numpy(float)-pc['mean'])/pc['scale']
        pcs=z@np.asarray(pc['loadings']).T
        x=(np.column_stack([pcs[:,:2],f.education])-model['minimum'])/model['span']
        outside=((x<0)|(x>1)).any(axis=1)
    else:
        x=((f[names].to_numpy(float)-model['mean'])/model['scale'])@np.asarray(model['rotation'])
        outside=np.zeros(len(x),dtype=bool)
    labels=((x[:,None]-np.asarray(model['centers'])[None])**2).sum(2).argmin(1)+1
    return labels,outside
