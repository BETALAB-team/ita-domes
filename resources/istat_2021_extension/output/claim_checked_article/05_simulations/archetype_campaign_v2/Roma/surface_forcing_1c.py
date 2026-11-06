"""Explicit local 1C preprocessing: every external surface initializes its inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import interpolate

def preprocess_1c_surface_loads(zone,weather,output_directory=None):
    from eureca_building._auxiliary_function_for_monthly_calc import get_monthly_value_from_annual_vector
    internal=zone.extract_convective_radiative_latent_electric_load()
    delta=float(weather.general_data['average_dt_air_sky'])
    count=len(weather.hourly_data['out_air_db_temperature'])
    solar=np.zeros(count);records={};parameters={}
    for surface in zone._surface_list:
        if surface.surface_type not in ['ExtWall','Roof']:continue
        name=surface.name
        own=weather.hourly_data_irradiances[float(surface._azimuth_round)][float(surface._height_round)]
        direct=np.asarray(own['direct'],float);total=np.asarray(own['global'],float);aoi=np.asarray(own['AOI'],float)
        opaque=float(surface._opaque_area);glazed=float(surface._glazed_area)
        shade=float(getattr(surface,'shading_coefficient',1.))
        construction=surface._construction
        alpha=float(construction.ext_absorptance);rse=float(construction._R_se);unet=float(construction._u_value_net)
        sky_factor=float(surface._sky_view_factor);hr=float(surface.get_surface_external_radiative_coefficient())
        opaque_sw=shade*total*alpha*rse*unet*opaque
        opaque_lw=np.full(count,-sky_factor*rse*unet*opaque*hr*delta)
        glass_sw=np.zeros(count);gdir=np.zeros(count);gdiff=0.
        p={'type':surface.surface_type,'opaque_area_m2':opaque,'glazed_area_m2':glazed,'shading':shade,'absorptance':alpha,'Rse_m2_K_W':rse,'U_net_W_m2_K':unet,'sky_view_factor':sky_factor,'h_r_W_m2_K':hr,'mean_air_minus_sky_K':delta,'azimuth_round':float(surface._azimuth_round),'height_round':float(surface._height_round)}
        if glazed>0:
            window=surface.window
            fsh=float(window._shading_coef_ext);fw=float(window._shading_coef_int);frame=float(window._frame_factor)
            gdir=interpolate.splev(aoi,window.solar_heat_gain_coef_profile,der=0)
            gdiff=float(interpolate.splev(70,window.solar_heat_gain_coef_profile,der=0))
            glass_sw=shade*direct*fsh*fw*(1-frame)*gdir*glazed+(total-direct)*fsh*fw*(1-frame)*gdiff*glazed
            t,c,k=window.solar_heat_gain_coef_profile
            p.update({'window_shading_external':fsh,'window_shading_internal':fw,'frame_factor':frame,'shgc_spline':{'knots':np.asarray(t).tolist(),'coefficients':np.asarray(c).tolist(),'degree':int(k)}})
        # Zero-glazed surfaces never access a window descriptor and never reuse
        # state from a preceding surface.
        solar+=opaque_sw+opaque_lw+glass_sw
        parameters[name]=p
        for key,a in [('irradiance_global_W_m2',total),('irradiance_direct_W_m2',direct),('AOI_degree',aoi),('SHGC_direct',gdir),('SHGC_diffuse',np.full(count,gdiff)),('opaque_shortwave_W',opaque_sw),('opaque_longwave_W',opaque_lw),('window_shortwave_W',glass_sw)]:records[f'{name}::{key}']=a
    weights=[1-zone.Am/zone.Atot-zone.Htr_w/(9.1*zone.Atot),zone.Am/zone.Atot]
    zone.phi_ia=internal['convective [W]']
    zone.phi_st=weights[0]*(internal['radiative [W]']+solar)
    zone.phi_m=weights[1]*(internal['radiative [W]']+solar)
    zone.monthly_int_gain=get_monthly_value_from_annual_vector(internal['convective [W]']+internal['radiative [W]'],method='integral')
    zone.monthly_sol_gain=get_monthly_value_from_annual_vector(solar,method='integral')
    records.update({'hour':np.arange(count),'internal_convective_W':internal['convective [W]'],'internal_radiative_W':internal['radiative [W]'],'total_surface_gain_W':solar,'phi_air_W':zone.phi_ia,'phi_surface_W':zone.phi_st,'phi_mass_W':zone.phi_m})
    frame=pd.DataFrame(records)
    if output_directory is not None:
        output=Path(output_directory);output.mkdir(parents=True,exist_ok=True)
        # Inputs/forcing are identical across schedule cases at a site. Preserve
        # the first immutable export and verify later constructions match it.
        target=output/'surface_forcing_hourly.csv'
        content=frame.to_csv(index=False).encode('utf-8')
        if target.exists():
            assert target.read_bytes()==content,'Unexpected case-specific surface forcing'
        else:
            target.write_bytes(content)
            (output/'surface_forcing_parameters.json').write_text(json.dumps({'surfaces':parameters,'node_surface_weight':weights[0],'node_mass_weight':weights[1]},indent=2),encoding='utf-8')
    return frame,parameters

def check_surface_order_invariance(zone,weather):
    original=list(zone._surface_list)
    reference=[zone.phi_ia.copy(),zone.phi_st.copy(),zone.phi_m.copy()]
    errors=[]
    try:
        for order in [list(reversed(original)),original[1:]+original[:1]]:
            zone._surface_list=order
            preprocess_1c_surface_loads(zone,weather)
            errors.append(max(float(np.max(np.abs(a-b))) for a,b in zip(reference,[zone.phi_ia,zone.phi_st,zone.phi_m])))
    finally:
        zone._surface_list=original
        preprocess_1c_surface_loads(zone,weather)
    assert max(errors)<1e-8
    return {'reverse_rotate_max_node_forcing_error_W':errors,'surface_count':len(original),'pass':True}
