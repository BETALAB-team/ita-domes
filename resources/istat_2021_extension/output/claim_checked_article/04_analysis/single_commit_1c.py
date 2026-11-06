"""Adapter-local ideal sensible 1C stepping; EUReCA checkout stays unchanged.

The pure EUReCA network algebra is reused. Each trial reads the same old state;
only the selected physical solution is committed, exactly once per timestep.
Supported here: no mechanical/natural ventilation or internal latent gains.
"""
import numpy as np

def select_solution(zone, lower_C, upper_C, Hve, outdoor_C, supply_C, gains):
    """No state writes: return one solution, mode and radiant/convective split."""
    if not lower_C < upper_C:
        raise ValueError('Heating lower setpoint must be below cooling upper setpoint')
    free=zone.sensible_balance_1C('phiset',Hve,outdoor_C,supply_C,gains,sigma=[.5,.5],phi_HC_set=0.)
    if free[1] < lower_C:
        sigma=zone.heating_sigma['1C']
        chosen=zone.sensible_balance_1C('Tset',Hve,outdoor_C,supply_C,gains,sigma=sigma,T_set=lower_C)
        assert chosen[0] >= -1e-7
        mode='heating on'
    elif free[1] > upper_C:
        sigma=zone.cooling_sigma['1C']
        chosen=zone.sensible_balance_1C('Tset',Hve,outdoor_C,supply_C,gains,sigma=sigma,T_set=upper_C)
        assert chosen[0] <= 1e-7
        mode='cooling on'
    else:
        chosen=free;sigma=[.5,.5];mode='off'
    return chosen,mode,sigma

def solve_1c_single_commit(zone,t,weather):
    from eureca_building.fluids_properties import air_properties
    # Fail closed if this bounded sensible-only adapter is used beyond its scope.
    if zone.natural_ventilation is not None:
        raise ValueError('Adapter requires zero natural ventilation')
    if abs(zone.air_handling_unit.air_flow_rate_kg_S[t])>1e-14:
        raise ValueError('Adapter requires zero mechanical ventilation')
    if abs(zone.latent_load[t])>1e-14:
        raise ValueError('Adapter requires zero internal latent gain')
    oldta=float(zone.Ta0); oldtm=float(zone.Tm0[0])
    ext=float(weather.hourly_data['out_air_db_temperature'][t])
    hinf=float(zone.infiltration_air_flow_rate[t]*air_properties['specific_heat'])
    gains=[float(zone.phi_ia[t]),float(zone.phi_st[t]),float(zone.phi_m[t])]
    lower=float(zone._temperature_setpoint.schedule_lower.schedule[t])
    upper=float(zone._temperature_setpoint.schedule_upper.schedule[t])
    chosen,mode,sigma=select_solution(zone,lower,upper,[0.,hinf],ext,ext,gains)
    # Both free and constrained trials must have left the old state untouched.
    assert zone.Ta0==oldta and zone.Tm0[0]==oldtm
    q,ta,ts,tm=map(float,chosen)
    # The only physical-state commit in this function.
    zone.Ta0=ta;zone.Tm0=[tm,tm]
    zone.zone_air_temperature=ta
    zone.zone_mean_radiant_temperature=ts
    zone.zone_operative_temperature=(ta+ts)/2
    zone.sensible_zone_load=q;zone.flag=mode
    zone.sensible_load_hourly[t]=q
    zone.external_temperature[t]=ext
    # Moisture is uncoupled from this sensible-only network and is not stepped
    # or reported. No latent HVAC, DHW, equipment or final-energy result follows.
    return {'old_air_C':oldta,'old_mass_C':oldtm,'surface_C':ts,'mass_C':tm,
            'phi_air_W':gains[0],'phi_surface_W':gains[1],'phi_mass_W':gains[2],
            'infiltration_H_W_K':hinf,'sigma_radiant':float(sigma[0]),
            'sigma_convective':float(sigma[1])}
