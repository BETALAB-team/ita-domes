"""Transition and limiting-case checks before corrected annual campaigns."""
from pathlib import Path
from types import SimpleNamespace,MethodType
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,r'C:\Works\EUReCA\EUReCA')
from eureca_building.config import load_config
load_config(str(ROOT/'05_simulations/climate_campaign_v2/Padova/eureca_config.json'))
from eureca_building.thermal_zone import ThermalZone
from single_commit_1c import select_solution
def network(ta=20.,tm=20.):
    z=SimpleNamespace(Htr_is=300.,Htr_w=50.,Htr_ms=1000.,Htr_em=250.,Cm=2e7,_air_thermal_capacity=3e5,Ta0=ta,Tm0=[tm,tm],heating_sigma={'1C':[.2,.8]},cooling_sigma={'1C':[.2,.8]})
    z.sensible_balance_1C=MethodType(ThermalZone.sensible_balance_1C,z)
    return z
def residual(z,oldta,oldtm,a,ext,gains,sigma):
    q,ta,ts,tm=a;rad,conv=sigma
    return [q*conv+gains[0]+z.Htr_is*(ts-ta)+30*(ext-ta)-z._air_thermal_capacity*(ta-oldta)/3600,
    q*rad+gains[1]+z.Htr_is*(ta-ts)+z.Htr_w*(ext-ts)+z.Htr_ms*(tm-ts),
    gains[2]+z.Htr_ms*(ts-tm)+z.Htr_em*(ext-tm)-z.Cm*(tm-oldtm)/3600]
cases=[]
# Explicit previous-mode labels cannot affect the physical one-step selection.
for name,oldta,oldtm,ext,lower,upper,gains,previous,expected in [
 ('off_to_heating',15.,15.,0.,22.,50.,[0.,0.,0.],'off','heating on'),
 ('heating_to_off',22.,22.,25.,19.,50.,[0.,0.,0.],'heating on','off'),
 ('steady_off',20.,20.,20.,19.,50.,[0.,0.,0.],'off','off'),
 ('continuing_heating',20.,18.,0.,20.,50.,[100.,80.,20.],'heating on','heating on'),
 ('off_to_cooling',30.,30.,35.,19.,26.,[0.,0.,0.],'off','cooling on')]:
    z=network(oldta,oldtm);z.flag=previous
    a,mode,sigma=select_solution(z,lower,upper,[0.,30.],ext,ext,gains)
    unchanged=z.Ta0==oldta and z.Tm0[0]==oldtm
    errors=residual(z,oldta,oldtm,a,ext,gains,sigma)
    ok=mode==expected and unchanged and max(abs(x) for x in errors)<1e-6
    cases.append({'case':name,'expected_mode':expected,'selected_mode':mode,'old_state_unchanged_during_trials':unchanged,'solution_W_C_C_C':a.tolist(),'node_residual_W':list(map(float,errors)),'pass':bool(ok)})
out={'purpose':'pre-run local selector transition verification; no annual results used','cases':cases,'pass':all(x['pass'] for x in cases),'adapter_sha256':hashlib.sha256((ROOT/'04_analysis/single_commit_1c.py').read_bytes()).hexdigest()}
(ROOT/'05_simulations/single_commit_transition_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
assert out['pass']
