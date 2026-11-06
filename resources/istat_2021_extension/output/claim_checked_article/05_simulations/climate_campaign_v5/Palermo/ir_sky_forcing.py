"""Local horizontal-IR sky correction; retains native mean-delta formulation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
SIGMA=5.6697e-8
class GuardUnusedInvalidFields(dict):
    """Prove invalid loaded fields cannot enter the active simulation path."""
    def __getitem__(self,key):
        if key in ['wind_speed','opaque_sky_coverage']:
            raise RuntimeError(f'Inactive invalid weather field unexpectedly consumed: {key}')
        return super().__getitem__(key)

def apply_ir_sky_forcing(weather,epw_path,output_directory,site,audit_path):
    audit=json.loads(Path(audit_path).read_text())
    assert audit['pass'] and audit['sites'][site]['active_forcing_QC_pass']
    assert hashlib.sha256(Path(epw_path).read_bytes()).hexdigest()==audit['sites'][site]['epw_sha256']
    raw=pd.read_csv(epw_path,skiprows=8,header=None)
    ir=raw[12].to_numpy(float);air=raw[6].to_numpy(float)
    assert len(ir)==8760 and np.isfinite(ir).all() and ((ir>0)&(ir<9999)).all()
    assert np.array_equal(air,weather.hourly_data['out_air_db_temperature'])
    skyK=(ir/SIGMA)**.25;delta=air-(skyK-273.15);mean=float(delta.mean())
    assert abs(mean-audit['sites'][site]['IR_derived_mean_air_minus_sky_K'])<1e-12
    assert 0<mean<50 and np.max(np.abs(SIGMA*skyK**4-ir))<1e-9
    old=float(weather.general_data['average_dt_air_sky'])
    weather.general_data['average_dt_air_sky']=mean
    # No later call can accidentally reuse invalid opaque cover or wind.
    weather.hourly_data=GuardUnusedInvalidFields(weather.hourly_data)
    record={'site':site,'old_invalid_cloud_derived_mean_delta_K':old,'corrected_IR_mean_delta_K':mean,
      'sigma_W_m2_K4':SIGMA,'interval_hours':1,'source':'valid raw horizontal IR; no cloud filling',
      'formulation':'native annual-mean air-minus-effective-sky scalar retained',
      'active_path_guarded_invalid_keys':['wind_speed','opaque_sky_coverage'],
      'active_forcing_gate_pass':True,'whole_weather_file_universally_valid':False,
      'raw_epw_sha256':hashlib.sha256(Path(epw_path).read_bytes()).hexdigest(),
      'pre_run_audit_sha256':hashlib.sha256(Path(audit_path).read_bytes()).hexdigest()}
    Path(output_directory,'weather_forcing_correction.json').write_text(json.dumps(record,indent=2))
    return record

def verify_native_longwave(zone,weather,output_directory,case):
    """Recompute signed longwave coefficients from actual surfaces, no calibration."""
    items=[]
    for s in zone._surface_list:
        if s.surface_type not in ['ExtWall','Roof']:continue
        coefficient=float(s._sky_view_factor*s._construction._R_se*s._construction._u_value_net*s._opaque_area*s.get_surface_external_radiative_coefficient())
        items.append({'surface_type':s.surface_type,'coefficient_W_K':coefficient})
    total=sum(x['coefficient_W_K'] for x in items)
    delta=float(weather.general_data['average_dt_air_sky'])
    loss=total*delta
    assert total>0 and loss>0
    Path(output_directory,f'{case}_longwave_check.json').write_text(json.dumps({'surface_coefficients':items,'total_coefficient_W_K':total,'mean_delta_K':delta,'opaque_longwave_term_W':-loss,'sign_is_loss':True,'pass':True},indent=2))
