"""Reviewer runtime checks. Writes only in reviewer_02; never launches author campaigns."""
from pathlib import Path
import sys, ast, json, csv
from types import SimpleNamespace, MethodType
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,r"C:\Works\EUReCA\EUReCA")
sys.path.insert(0,str(ROOT/"04_analysis"))
from eureca_building.config import load_config
load_config(str(ROOT/"05_simulations/climate_campaign_v3/Padova/eureca_config.json"))
from eureca_building.thermal_zone import ThermalZone
from eureca_building.weather import WeatherFile
from single_commit_1c import solve_1c_single_commit
records=[]
for label,ta,tm,ext,low,high,oldflag in [
    ("off_to_heat",15.,15.,0.,22.,50.,"off"),
    ("heat_to_off",22.,22.,25.,19.,50.,"heating on"),
    ("off_to_cool",30.,30.,35.,19.,26.,"off"),
    ("cool_to_heat",15.,15.,0.,22.,50.,"cooling on"),
    ("equilibrium",20.,20.,20.,19.,50.,"off")]:
    z=SimpleNamespace(Htr_is=300.,Htr_w=50.,Htr_ms=1000.,Htr_em=250.,Cm=2e7,
        _air_thermal_capacity=3e5,Ta0=ta,Tm0=[tm,tm],heating_sigma={"1C":[.2,.8]},
        cooling_sigma={"1C":[.2,.8]},natural_ventilation=None,
        air_handling_unit=SimpleNamespace(air_flow_rate_kg_S=[0.]),latent_load=[0.],
        infiltration_air_flow_rate=[.03],phi_ia=[0.],phi_st=[0.],phi_m=[0.],
        _temperature_setpoint=SimpleNamespace(schedule_lower=SimpleNamespace(schedule=[low]),
            schedule_upper=SimpleNamespace(schedule=[high])),sensible_load_hourly=[0.],
        external_temperature=[0.],flag=oldflag)
    z.sensible_balance_1C=MethodType(ThermalZone.sensible_balance_1C,z)
    weather=SimpleNamespace(hourly_data={"out_air_db_temperature":[ext]})
    d=solve_1c_single_commit(z,0,weather)
    h=d["infiltration_H_W_K"];dt=3600
    # Independently solve backward-Euler nodes with np.linalg.solve (not engine inverse).
    A=np.array([[300+h+3e5/dt,-300,0],[-300,1350,-1000],[0,-1000,1250+2e7/dt]])
    b=np.array([h*ext+3e5/dt*ta,50*ext,250*ext+2e7/dt*tm])
    free=np.linalg.solve(A,b)
    if free[0]<low or free[0]>high:
        tset=low if free[0]<low else high
        # Unknowns q, surface T, mass T.
        B=np.array([[.8,300,0],[.2,-1350,1000],[0,1000,-1250-2e7/dt]])
        rhs=np.array([(300+h+3e5/dt)*tset-h*ext-3e5/dt*ta,-300*tset-50*ext,-250*ext-2e7/dt*tm])
        q,ts,m=np.linalg.solve(B,rhs);ref=np.array([q,tset,ts,m])
    else:ref=np.r_[0.,free]
    actual=np.array([z.sensible_zone_load,z.zone_air_temperature,z.zone_mean_radiant_temperature,z.Tm0[0]])
    records.append({"test":label,"mode":z.flag,"max_reference_difference":float(np.max(np.abs(actual-ref))),
                    "old_states_exported_correctly":d["old_air_C"]==ta and d["old_mass_C"]==tm,
                    "committed_once_to_reference":bool(np.allclose(actual,ref,rtol=0,atol=1e-8))})
weather_checks={}
for site in ("Padova","Bolzano","Roma","Palermo"):
    w=WeatherFile(str(ROOT/f"05_simulations/climate_inputs_v2/{site}.epw"),year=2023,time_steps=1,irradiances_calculation=False)
    weather_checks[site]={"runtime_mean_air_minus_sky_K":float(w.general_data["average_dt_air_sky"]),
                         "opaque_values":np.unique(w.hourly_data["opaque_sky_coverage"]).tolist(),
                         "first_timestamp":str(w.hourly_data["time_index"][0]),
                         "last_timestamp":str(w.hourly_data["time_index"][-1])}
# Actual GRINS schedule functions (separate AST extraction; no author's helper).
source=Path(r"C:\Works\GRINS\GRINS-Tool\classes\EndUse.py")
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="User")
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {"create_random_prof","create_prof","substitute_istat_heatingTSP"}]
assign=[n for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="periodi_risc" for t in n.targets)]
cls.body=assign+methods
import random
env={"np":np,"random":random}
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),"exec"),env)
response=json.loads((ROOT/"05_simulations/climate_campaign_v3/Padova/design.json").read_text())["response"]
profiles={}
for case in ("seeded","early","late"):
    u=env["User"]()
    if case!="seeded":
        def block(n,seed,mode=case):
            n=int(n);start=0 if mode=="early" else 8-n
            return np.r_[np.zeros(start),np.ones(n),np.zeros(8-start-n)]
        u.create_random_prof=block
    u.substitute_istat_heatingTSP(response)
    with (ROOT/f"05_simulations/climate_campaign_v3/Padova/{case}_hourly.csv").open() as f:
        h=list(csv.DictReader(f))
    profiles[case]={"all_8760_setpoints_equal_source":bool(np.array_equal(u.heatingTSP,np.array([float(r["heating_setpoint_C"]) for r in h]))),"daily_profile":u.create_prof(4,4,1,3523).tolist()}
out={"transition_checks":records,"weather":weather_checks,"profiles":profiles}
(HERE/"runtime_checks.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
