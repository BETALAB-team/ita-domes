from pathlib import Path
import importlib.util,sys,json,collections
import numpy as np
root=Path.cwd(); out=root/'07_reviews/reviewer_01/dispatcher_diagnostic';out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,r'C:\Works\EUReCA\EUReCA')
from eureca_building.config import load_config
load_config(str(root/'05_simulations/padova_pilot_v1/eureca_config.json'))
from eureca_building.thermal_zone import ThermalZone
from eureca_building.fluids_properties import air_properties
original=ThermalZone.solve_timestep
records=[];depth=0

def observed(self,t,weather,flag,model='2C',pot=0,Ta=0):
 global depth
 top=depth==0; oldta=self.Ta0; oldtm=self.Tm0[0];oldflag=self.flag
 depth+=1
 try: value=original(self,t,weather,flag,model,pot,Ta)
 finally: depth-=1
 if top and t>=0:
  ta=self.Ta0;tm=self.Tm0[0];ts=self.zone_mean_radiant_temperature;q=self.sensible_zone_load;ext=weather.hourly_data['out_air_db_temperature'][t]
  hi=self.infiltration_air_flow_rate[t]*air_properties['specific_heat']; rad,conv=self.heating_sigma['1C']
  residual=[q*conv+self.phi_ia[t]+self.Htr_is*(ts-ta)+hi*(ext-ta)-self._air_thermal_capacity*(ta-oldta)/3600,
   q*rad+self.phi_st[t]+self.Htr_is*(ta-ts)+self.Htr_w*(ext-ts)+self.Htr_ms*(tm-ts),
   self.phi_m[t]+self.Htr_ms*(ts-tm)+self.Htr_em*(ext-tm)-self.Cm*(tm-oldtm)/3600]
  records.append({'hour':t,'old_flag':oldflag,'new_flag':self.flag,'old_ta':float(oldta),'old_tm':float(oldtm),'ta':float(ta),'tm':float(tm),'load_W':float(q),'residual_W':[float(x) for x in residual]})
 return value
ThermalZone.solve_timestep=observed
spec=importlib.util.spec_from_file_location('pilot',root/'05_simulations/padova_pilot_v1/adapter_snapshot.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.ROOT=root;mod.RUN=out
mod.main()
summary=[]
for i,name in enumerate(['seeded','early','late','repeat_seeded','warmup_seeded']):
 rs=records[i*8760:(i+1)*8760]; bad=[r for r in rs if max(abs(v) for v in r['residual_W'])>1e-6]
 summary.append({'case':name,'n_bad_hourly_state_balances':len(bad),'max_abs_node_residual_W':max(abs(v) for r in rs for v in r['residual_W']),'first_bad':bad[:3],'largest_bad':sorted(bad,key=lambda r:max(abs(v) for v in r['residual_W']),reverse=True)[:2]})
(out/'state_balance_diagnostic.json').write_text(json.dumps(summary,indent=2));print('DIAGNOSTIC',json.dumps(summary,indent=2))
