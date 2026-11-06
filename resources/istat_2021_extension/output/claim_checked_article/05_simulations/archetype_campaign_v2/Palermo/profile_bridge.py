"""Input-only bridge from actual survey rows to the reviewed EUReCA path.

Archetype identity conditions row selection; it never supplies an invented
temperature, occupancy or gain value. The hourly map remains a source-rule
assumption, not an observed schedule.
"""
from pathlib import Path
import ast,hashlib,random
import numpy as np
GRINS=Path(r'C:\Works\GRINS\GRINS-Tool')
FIELDS=['id','q_3_0','q_3_11A','q_3_11B','q_3_11C','q_3_13_2','q_3_13_3','q_3_19_6','q_2_32C','q_2_32D']
def source_sha256():return hashlib.sha256((GRINS/'classes/EndUse.py').read_bytes()).hexdigest()
def source_class():
    p=GRINS/'classes/EndUse.py';tree=ast.parse(p.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='User')
    methods={'create_random_prof','create_prof','substitute_istat_heatingTSP'}
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='periodi_risc' for t in n.targets)]
    namespace={'np':np,'random':random}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(p),'exec'),namespace)
    return namespace['User']
SOURCE_CLASS=source_class()
def validated_row(row):
    values={k:row[k] for k in FIELDS}
    for k,v in values.items():
        if not np.isfinite(v) or float(v)!=int(v):raise ValueError(f'Noninteger/nonfinite {k}')
        values[k]=int(v)
    if values['q_3_0'] not in [1,2]:raise ValueError('Outside central/individual main-heating domain')
    hours=[values[k] for k in ['q_3_11A','q_3_11B','q_3_11C']]
    if any(n<0 or n>8 for n in hours) or sum(hours)<=0:raise ValueError('Invalid/nonpositive reported block hours')
    for k in ['q_3_13_2','q_3_13_3','q_3_19_6']:
        if values[k] not in [0,1]:raise ValueError(f'Unsupported compact control code {k}')
    for k in ['q_2_32C','q_2_32D']:
        if values[k] not in [0,1,2]:raise ValueError(f'Unsupported compact fireplace code {k}')
    values['Fascia_clima']='E'
    return values
def make_profile(row):
    values=validated_row(row);u=SOURCE_CLASS()
    u.substitute_istat_heatingTSP(values)
    lower=np.asarray(u.heatingTSP,dtype=float)
    daily=u.create_prof(values['q_3_11A'],values['q_3_11B'],values['q_3_11C'],values['id'])
    expected=[values['q_3_11A'],values['q_3_11B'],values['q_3_11C']]
    observed=[int(sum(daily[5:13])),int(sum(daily[13:21])),int(sum(np.r_[daily[21:],daily[:5]]))]
    assert expected==observed and len(lower)==8760
    control_source=any(values[k]==1 for k in ['q_3_13_2','q_3_13_3','q_3_19_6'])
    fireplace=any(values[k]==1 for k in ['q_2_32C','q_2_32D'])
    on=22.+2*fireplace;off=11.+8*control_source+2*fireplace
    assert np.array_equal(lower[:24],off+(on-off)*daily)
    assert np.all(lower[105*24:288*24]==0)
    return {'response':values,'lower_C':lower,'daily_high_setpoint':daily,'on_C':on,'off_C':off,'control_branch':f'{on:g}/{off:g}','profile_sha256':hashlib.sha256(lower.astype('<f8').tobytes()).hexdigest()}
