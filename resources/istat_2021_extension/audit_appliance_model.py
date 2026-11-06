"""Read-only execution audit of the appliance model committed on main.

Only dependency-free helper definitions are loaded to avoid importing unrelated
GIS/HVAC packages. Appliance functions and formulas are executed unmodified.
"""
import ast
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def git_blob(path):
    return subprocess.check_output(['git','-C',str(REPO),'show',f'main:{path}'])


def load_model():
    scope = {'pd':pd,'np':np,'DataFrame':pd.DataFrame}
    auxiliary = ast.parse(git_blob('funcs/auxiliary_functions.py').decode('utf-8-sig'))
    helper = ast.Module(body=[node for node in auxiliary.body if isinstance(node,ast.FunctionDef)
                            and node.name in ['weightCoeff','decodificRegion','fascia_climatica']],type_ignores=[])
    exec(compile(helper,'main:funcs/auxiliary_functions.py','exec'),scope)
    code = git_blob('classes/Appliances.py').decode('utf-8-sig')
    # Two legacy metadata-label replacements use sets instead of mappings.
    # Repair only these display fields for modern pandas; energy formulas stay intact.
    code = code.replace(".replace({'none',''})", ".replace({'none':''})")
    code = code.replace(".replace({0,''})", ".replace({0:''})")
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
    exec(compile(tree,'main:classes/Appliances.py','exec'),scope)
    return scope


def components(frame):
    frame = frame.assign(sample=frame['id'], ComponentiFamiglia=frame['q_1_1_sq1'])
    scope = load_model()
    db = pd.ExcelFile(io.BytesIO(git_blob('resources/appliances_data/Database_elettrodomestici_2013.xlsx')))
    out = pd.DataFrame(index=frame.index)
    for name,fn,sheet,attribute in [
        ('lights','loadLights','illuminazioneMOIRAE','lights_demand'),
        ('small','loadLittleAppliances','piccoli_elettrodomestici_MOIRAE','little_appliances_demand'),
        ('fridge_moirae','loadRefrigerators_moirae','frigoriferi','refrigerators_demand'),
        ('big_moirae','loadAppliances_moirae','grandi_elettrodomestici','big_appliances_demand'),
        ('big_ce','loadAppliances_CE','grandi_elettrodomestici','big_appliances_demand'),
        ('screens','loadScreens','schermi_MOIRAE','screen_demand'),
        ('standby','loadStandby','Standby','standby_demand')]:
        print(name,flush=True)
        table = pd.read_excel(db,sheet_name=sheet,index_col=None if name=='standby' else 0)
        value = scope[fn](frame.copy(),table)
        if isinstance(value,tuple): value=value[0]
        out[name]=getattr(value,attribute)
    return out


def main():
    raw=pd.read_csv(ROOT/'istat_microdata_2013.csv',sep=';').set_index('id',drop=False)
    folder=ROOT/'revised_paper_outputs/appliance_audit';folder.mkdir(exist_ok=True)
    modeled=components(raw)
    if 'fridge_ce' not in modeled:
        cols=['q_8_2','q_8_5','q_8_6','q_8_7','q_8_3_ric','q_8_8_ric']
        unique=raw[cols].drop_duplicates().reset_index(drop=True)
        unique['id']=unique.index;unique['reg']=1
        scope=load_model()
        print('Unique CE fridge configurations',len(unique),flush=True)
        value=scope['loadRefrigerators_CE'](unique.copy(),None)[0].refrigerators_demand
        unique['fridge_ce']=value
        modeled['fridge_ce']=raw.reset_index(drop=True).merge(unique[cols+['fridge_ce']],on=cols,validate='many_to_one').set_index('id').fridge_ce
    reference=pd.read_csv(ROOT/'results_grins.csv').set_index('id').appliances_el_kWh
    joined=modeled.join(reference).dropna()
    coef,_,_,_=np.linalg.lstsq(joined[modeled.columns].to_numpy(float),joined.appliances_el_kWh,rcond=None)
    residual=float(np.linalg.norm(joined[modeled.columns].to_numpy(float)@coef-joined.appliances_el_kWh))
    print(json.dumps({'component_coefficients':dict(zip(modeled.columns,coef)),
                      'residual_norm':residual,'n':len(joined)},indent=2))
    modeled.to_csv(folder/'components_2013.csv')


if __name__=='__main__':main()
