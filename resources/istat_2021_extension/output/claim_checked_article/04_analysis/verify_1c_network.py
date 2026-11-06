"""Analytical limiting cases for actual imported EUReCA 1C network solver."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,r'C:\Works\EUReCA\EUReCA')
from eureca_building.config import load_config
load_config(str(ROOT/'05_simulations/padova_pilot_v1/eureca_config.json'))
from eureca_building.thermal_zone import ThermalZone
import eureca_building.thermal_zone as module
def network():return SimpleNamespace(Htr_is=300.,Htr_w=50.,Htr_ms=1000.,Htr_em=250.,Cm=2e7,_air_thermal_capacity=3e5,Ta0=20.,Tm0=[20.,20.])
z=network()
zero=ThermalZone.sensible_balance_1C(z,'Tset',[0.,30.],20.,20.,[0.,0.,0.],sigma=[0.,1.],T_set=20.)
z=network()
for i in range(5000):
    steady=ThermalZone.sensible_balance_1C(z,'Tset',[0.,30.],0.,0.,[0.,0.,0.],sigma=[0.,1.],T_set=20.)
    z.Ta0=float(steady[1]);z.Tm0=[float(steady[3])]*2
K=50.+1/(1/1000.+1/250.)
analytical_W=(30.+1/(1/300.+1/K))*20.
z=network(); z.Ta0=18.;z.Tm0=[17.,17.]
load,ta,ts,tm=ThermalZone.sensible_balance_1C(z,'Tset',[12.,30.],3.,16.,[100.,80.,20.],sigma=[.2,.8],T_set=20.)
residuals=[load*.8+100+300*(ts-ta)+30*(3-ta)+12*(16-ta)-3e5*(ta-18)/3600,
load*.2+80+300*(ta-ts)+50*(3-ts)+1000*(tm-ts),
20+1000*(ts-tm)+250*(3-tm)-2e7*(tm-17)/3600]
result={'module':module.__file__,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
'description':'hand-specified 1C conductance/capacitance network, convective steady heat and zero-gradient equilibrium; independent node residual arithmetic',
'zero_gradient_result_W_C_C_C':zero.tolist(),'zero_gradient_max_error':float(np.max(np.abs(zero-[0,20,20,20]))),
'steady_solver_W':float(steady[0]),'steady_analytical_W':analytical_W,'steady_error_W':float(abs(steady[0]-analytical_W)),
'transient_air_surface_mass_residual_W':[float(x) for x in residuals],
'pass':bool(np.max(np.abs(zero-[0,20,20,20]))<1e-8 and abs(steady[0]-analytical_W)<1e-6 and max(abs(x) for x in residuals)<1e-8),
'scope':'implementation/physics verification of network algebra only; no benchmark certification, envelope parameter verification or empirical accuracy'}
(ROOT/'05_simulations/padova_pilot_v1/analytical_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
