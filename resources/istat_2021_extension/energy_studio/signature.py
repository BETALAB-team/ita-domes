"""Reference-trained energy signature with static exposure and ISTAT corrections.

No indoor temperature state, thermal capacitance, time integration or target EP
outcome is used. Daily weather features serve an empirical change-point model;
only annual target predictions are published.
"""
import copy
import numpy as np
import pandas as pd
from .physics import geometry, solar_gain

USES=('heating_kwh','cooling_kwh')


def features(cfg, scenario, weather, reference=False, reference_features=None):
    s,h,p=cfg['simulation'],cfg['hybrid'],scenario['profile']
    context=scenario['context']
    if reference_features is not None and not h['adjust_exposure']:context=reference_features['context']
    geo=geometry(s,context); conductance=geo['h_out']+geo['h_ground']
    if conductance<=0:raise ValueError('Signature requires positive envelope/ventilation conductance.')
    members=p['members'] if reference else p['members_index']
    appliances=p['appliance_kwh'] if reference else p['appliance_index']
    heat=p['heating_hours'] if reference else p['heating_index']
    cool=p['cooling_hours'] if reference else p['cooling_index']
    occupancy=p.get('occupancy_profile',s['occupancy_profile']) if reference else s['occupancy_profile']
    # Independent known-summary tests may supply mean presence explicitly.
    # ISTAT profiles do not contain this field and retain the original assumption.
    presence=float(np.mean(occupancy)) if reference else p.get('presence_index',float(np.mean(occupancy)))
    gains=members*s['people_sensible_w']*presence+appliances*s['appliance_heat_fraction']*1000/8760
    setpoints={'heating_kwh':s['heat_setback_c']+(s['heat_setpoint_c']-s['heat_setback_c'])*heat/24,
               'cooling_kwh':s['cool_setback_c']+(s['cool_setpoint_c']-s['cool_setback_c'])*cool/24}
    if reference_features is not None:
        if not h['adjust_internal_gains']:gains=reference_features['gains_w']
        if not h['adjust_operating_hours']:setpoints=reference_features['setpoint_c'].copy()
    dry=np.asarray(weather['dry']).reshape(365,24).mean(axis=1)
    # Static weather index: floor-ground coupling and projected window irradiation.
    equivalent=(geo['h_out']*dry+geo['h_ground']*s['ground_temperature_c'])/conductance
    solar=solar_gain(weather,s,context).reshape(365,24).mean(axis=1)/conductance if h['adjust_solar'] else np.zeros(365)
    return {'context':context,'conductance_w_k':conductance,'gains_w':float(gains),
            'setpoint_c':setpoints,'equivalent_c':equivalent,'solar_equivalent_k':solar,
            'outdoor_c':dry,'hdd18':float(np.maximum(18-dry,0).sum()),'cdd22':float(np.maximum(dry-22,0).sum())}


def degree_features(equivalent,solar,balance,coefficient,use):
    effective=equivalent+coefficient*solar
    return np.maximum(balance-effective,0) if use=='heating_kwh' else np.maximum(effective-balance,0)


def fit_reference(cfg, ref, hourly):
    """Select balance point/solar response using reference daily EP demand only."""
    if len(hourly)!=8760:raise ValueError('Reference signature needs 8760 hourly EP records.')
    fits={};diagnostics=[];h=cfg['hybrid']
    for use in USES:
        observed=hourly[use].to_numpy(float).reshape(365,24).sum(axis=1)
        if not np.isfinite(observed).all() or np.any(observed<0):raise ValueError('Invalid reference EP energy.')
        active=int((observed>1e-4).sum())
        if active<h['minimum_active_days'] or observed.sum()<=1e-6:
            fits[use]={'status':'insufficient_reference_activity','active_days':active};continue
        prefix=use.split('_')[0];lo,hi=h[prefix+'_balance_min_c'],h[prefix+'_balance_max_c']
        balances=np.arange(lo,hi+h['balance_step_c']*.5,h['balance_step_c'])
        solar_grid=np.arange(0,h['solar_coefficient_max']+h['solar_step']*.5,h['solar_step']) if h['adjust_solar'] else np.array([0.])
        best=None
        for coefficient in solar_grid:
            effective=ref['equivalent_c']+coefficient*ref['solar_equivalent_k']
            x=np.maximum(balances[:,None]-effective,0) if use=='heating_kwh' else np.maximum(effective-balances[:,None],0)
            denom=np.sum(x*x,axis=1)
            slopes=np.divide(x@observed,denom,out=np.zeros_like(denom),where=denom>1e-12)
            losses=np.mean((x*slopes[:,None]-observed)**2,axis=1);losses[denom<=1e-12]=np.inf
            i=int(np.argmin(losses));candidate=(float(losses[i]),float(balances[i]),float(coefficient),float(slopes[i]))
            if best is None or candidate[0]<best[0]:best=candidate
        if best is None or not np.isfinite(best[0]):
            fits[use]={'status':'unidentifiable_reference_signature','active_days':active};continue
        loss,balance,coefficient,slope=best
        degrees=degree_features(ref['equivalent_c'],ref['solar_equivalent_k'],balance,coefficient,use)
        fits[use]={'status':'fitted','active_days':active,'balance_c':balance,'solar_coefficient':coefficient,
                   'daily_slope_kwh_k':slope,'daily_fit_rmse_kwh':float(np.sqrt(loss)),
                   'reference_degree_sum':float(degrees.sum()),'reference_energy_kwh':float(observed.sum()),
                   'annual_normalisation':float(observed.sum()/degrees.sum()),
                   'balance_at_search_edge':bool(np.isclose(balance,balances[0]) or np.isclose(balance,balances[-1])),
                   'solar_at_search_edge':bool(h['adjust_solar'] and (np.isclose(coefficient,0) or np.isclose(coefficient,solar_grid[-1]))),
                   'equivalent_temperature_min_c':float(ref['equivalent_c'].min()),'equivalent_temperature_max_c':float(ref['equivalent_c'].max())}
        for i,(obs,degree) in enumerate(zip(observed,degrees),1):
            diagnostics.append({'day':i,'end_use':use,'observed_reference_kwh':obs,'fitted_reference_kwh':slope*degree,
                                'outdoor_c':ref['outdoor_c'][i-1],'equivalent_c':ref['equivalent_c'][i-1]})
    return fits,pd.DataFrame(diagnostics,columns=['day','end_use','observed_reference_kwh','fitted_reference_kwh','outdoor_c','equivalent_c'])


def estimate_one(cfg,sc,anchor,feat,ref,fits):
    values={};audit=[];p=sc['profile'];is_anchor=sc['id']==anchor['id']
    for use in USES:
        fit=fits[use];status=fit['status'];estimate=np.nan;balance=np.nan;degree_sum=np.nan;ratio=np.nan;shift=np.nan
        scale=feat['conductance_w_k']/ref['conductance_w_k']
        if status=='fitted':
            required=[feat['setpoint_c'][use],feat['gains_w'],p[use.split('_')[0]+'_index'],p['members_index'],p['appliance_index']]
            if not all(np.isfinite(required)):status='missing_or_unsupported_istat_inputs'
            else:
                shift=(feat['setpoint_c'][use]-ref['setpoint_c'][use])-(feat['gains_w']/feat['conductance_w_k']-ref['gains_w']/ref['conductance_w_k'])
                balance=fit['balance_c']+shift
                degree_sum=float(degree_features(feat['equivalent_c'],feat['solar_equivalent_k'],balance,fit['solar_coefficient'],use).sum())
                ratio=scale*degree_sum/fit['reference_degree_sum']
                estimate=fit['reference_energy_kwh']*ratio;status='estimated'
                if is_anchor:estimate=fit['reference_energy_kwh'];ratio=1.;status='anchor_identity'
        values[use]=float(estimate)
        audit.append({'scenario_id':sc['id'],'scenario':sc['key'],'end_use':use,'status':status,
                      'reference_scenario':anchor['key'],'reference_energy_kwh':fit.get('reference_energy_kwh',np.nan),
                      'reference_conductance_w_k':ref['conductance_w_k'],'target_conductance_w_k':feat['conductance_w_k'],
                      'conductance_ratio':scale,'reference_gains_w':ref['gains_w'],'target_gains_w':feat['gains_w'],
                      'reference_setpoint_c':ref['setpoint_c'][use],'target_setpoint_c':feat['setpoint_c'][use],
                      'reference_balance_c':fit.get('balance_c',np.nan),'balance_shift_c':shift,'target_balance_c':balance,
                      'solar_coefficient':fit.get('solar_coefficient',np.nan),'reference_degree_sum':fit.get('reference_degree_sum',np.nan),
                      'target_degree_sum':degree_sum,'total_transfer_ratio':ratio,'estimate_kwh':estimate,
                      'hdd18':feat['hdd18'],'cdd22':feat['cdd22'],
                      'weather_outside_reference_fraction':float(np.mean((feat['equivalent_c']<ref['equivalent_c'].min())|(feat['equivalent_c']>ref['equivalent_c'].max()))),
                      'target_hours_support_n':p[use.split('_')[0]+'_index_n'],'target_members_support_n':p['members_index_n'],
                      'target_appliance_support_n':p['appliance_index_n']})
    return values,audit


def predict(cfg,scenarios,anchor,weather,reference_hourly,baseline):
    """Function accepts precisely one EP series; target outcomes have no input path."""
    ref=features(cfg,anchor,weather[anchor['climate']['name']],reference=True)
    fits,diagnostics=fit_reference(cfg,ref,reference_hourly)
    results={};audit=[];ablations=[]
    stages=[('Weather only',False,False,False),('Weather + exposure',True,False,False),
            ('+ ISTAT hours',True,True,False),('+ ISTAT gains',True,True,True)]
    for sc in scenarios:
        feat=ref if sc['id']==anchor['id'] else features(cfg,sc,weather[sc['climate']['name']],reference_features=ref)
        values,records=estimate_one(cfg,sc,anchor,feat,ref,fits)
        values['appliance_kwh']=baseline[sc['id']]['appliance_kwh'];results[sc['id']]=values;audit.extend(records)
        for label,exposure,hours,gains in stages:
            variant=copy.deepcopy(cfg);variant['hybrid'].update(adjust_exposure=exposure,adjust_operating_hours=hours,adjust_internal_gains=gains)
            f=ref if sc['id']==anchor['id'] else features(variant,sc,weather[sc['climate']['name']],reference_features=ref)
            estimates,_=estimate_one(variant,sc,anchor,f,ref,fits)
            ablations.append({'scenario_id':sc['id'],'scenario':sc['key'],'stage':label,**estimates})
    fit_record={'method':'reference_energy_signature_v1','fit_data':'reference EnergyPlus daily totals only',
                'reference_scenario':anchor['key'],'reference_conductance_w_k':ref['conductance_w_k'],
                'reference_internal_gains_w':ref['gains_w'],'fits':fits,
                'definition':'Annual reference energy × conductance ratio × adjusted degree-sum ratio',
                'target_thermal_solver':False,'target_energyplus_used':False}
    return results,pd.DataFrame(audit),fit_record,diagnostics,pd.DataFrame(ablations)
