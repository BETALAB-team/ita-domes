"""Shared, source-backed chart data for scientific figures and editable PPT charts."""
import calendar
import json
from pathlib import Path
import numpy as np
import pandas as pd

COLORS={'energyplus':'#145c85','hybrid_signature':'#388670','hybrid_istat':'#cf7f32','istat_only':'#777777'}
NAMES={'energyplus':'EnergyPlus','hybrid_signature':'Climate + gains hybrid','hybrid_istat':'ISTAT-ratio baseline','istat_only':'ISTAT only'}
PALETTE=['#145c85','#cf7f32','#388670','#8b5fa1','#ae4f54','#567783']


def series(name,values,color,x=None,reference=False):
    result={'name':name,'values':[float(v) if pd.notna(v) else None for v in values], 'color':color,'reference':reference}
    if x is not None:result['x']=[float(v) for v in x]
    return result


def panel(title,kind,series_list,categories=None,xlabel='',ylabel='',limits=None):
    return {'title':title,'type':kind,'series':series_list,'categories':categories or [],'xlabel':xlabel,'ylabel':ylabel,'limits':limits}


def matched_errors(annual):
    """Pair by scenario, exclude calibration anchor, retain finite pairs only."""
    keys=['scenario_id','climate','dwelling','archetype']
    truth=annual[(annual.method=='energyplus')&~annual.is_anchor]
    rows=[]
    for method in ['hybrid_signature','hybrid_istat']:
        pred=annual[(annual.method==method)&~annual.is_anchor]
        paired=truth.merge(pred,on=keys,suffixes=('_ep','_pred'),validate='one_to_one')
        for use in ['heating_kwh','cooling_kwh']:
            for _,r in paired.iterrows():
                obs,estimate=r[use+'_ep'],r[use+'_pred']
                if np.isfinite(obs) and np.isfinite(estimate):
                    rows.append({**{k:r[k] for k in keys},'method':method,'end_use':use,'benchmark':obs,'estimate':estimate,'error':estimate-obs})
    return pd.DataFrame(rows,columns=[*keys,'method','end_use','benchmark','estimate','error'])


def aggregate_accuracy(errors):
    """Pooled annual bias alongside WAPE: expose cancellation rather than hide it."""
    rows=[]
    for (method,use),part in errors.groupby(['method','end_use']):
        for factors in [[],['climate'],['dwelling'],['archetype'],['climate','dwelling']]:
            groups=[('All',part)] if not factors else part.groupby(factors[0] if len(factors)==1 else factors)
            for label,g in groups:
                obs=float(g.benchmark.sum());pred=float(g.estimate.sum());absolute=float(g.error.abs().sum())
                rows.append({'method':method,'end_use':use,'aggregation':'+'.join(factors) or 'all',
                             'group':' / '.join(map(str,label)) if isinstance(label,tuple) else str(label),
                             'n_excluding_anchor':len(g),'benchmark_sum_kwh':obs,'estimate_sum_kwh':pred,
                             'pooled_signed_error_pct':100*(pred-obs)/obs if obs>0 else np.nan,
                             'case_wape_pct':100*absolute/obs if obs>0 else np.nan})
    return pd.DataFrame(rows,columns=['method','end_use','aggregation','group','n_excluding_anchor','benchmark_sum_kwh','estimate_sum_kwh','pooled_signed_error_pct','case_wape_pct'])


def build_plot_suite(data_root,cfg,annual,manifest,output_root=None):
    data_root=Path(data_root);out=Path(output_root or data_root);figdir=out/'plots';figdir.mkdir(parents=True,exist_ok=True)
    climates=[c['name'] for c in cfg['climates'] if c['enabled']]
    contexts=[c['name'] for c in cfg['contexts'] if c['enabled']]
    arches=sorted(annual.archetype.unique());expected=annual.scenario_id.nunique()
    complete=[m for m in COLORS if len(annual[annual.method.eq(m)])==expected]
    specs=[]
    def add(key,title,note,panels):
        panels=[p for p in panels if p['series']]
        if panels:specs.append({'key':key,'title':title,'note':note,'panels':panels})
    def grouped(factor,groups):
        panels=[]
        for use in ['heating_kwh','cooling_kwh']:
            rows=[]
            for method in complete:
                vals=annual[annual.method.eq(method)].groupby(factor)[use].mean().reindex(groups)
                if vals.notna().any():rows.append(series(NAMES[method],vals,COLORS[method]))
            panels.append(panel('Sensible '+use.split('_')[0],'bar',rows,[f'A{x}' if factor=='archetype' else str(x) for x in groups],factor.replace('dwelling','Dwelling context').capitalize(),'Mean kWh/year'))
        return panels
    add('demand_by_dwelling','Annual demand by dwelling context','Unweighted scenario means. Each context changes both exposure and its survey-derived household profile.',grouped('dwelling',contexts))
    add('demand_by_archetype','Annual demand by household archetype','Unweighted means across climates and dwelling contexts. These scenarios use the configured archetype labels.',grouped('archetype',arches))
    for use in ['heating_kwh','cooling_kwh']:
        candidates=[m for m in complete if annual.loc[annual.method.eq(m),use].notna().all()]
        if not candidates:continue
        method=candidates[0];data=annual[annual.method.eq(method)];panels=[]
        ymax=float(data[use].max())*1.13 or 1.
        for climate in climates:
            subset=data[data.climate.eq(climate)];rows=[]
            for i,context in enumerate(contexts):
                vals=subset[subset.dwelling.eq(context)].set_index('archetype')[use].reindex(arches)
                rows.append(series(context,vals,PALETTE[i%len(PALETTE)]))
            panels.append(panel(climate,'bar',rows,[f'A{x}' for x in arches],'Archetype','kWh/year',{'y':[0,ymax]}))
        add('scenario_'+use.split('_')[0],f'All scenarios: sensible {use.split("_")[0]}',f'{NAMES[method]}. Common vertical scale across climates. Building context includes exposure and survey-profile differences.',panels)
    monthly=[]
    for _,r in annual.iterrows():
        file=data_root/'hourly'/r.scenario_id/f'{r.method}_monthly.csv'
        if not file.exists():continue
        frame=pd.read_csv(file);frame['method']=r.method;frame['climate']=r.climate;frame['scenario_id']=r.scenario_id;monthly.append(frame)
    if monthly:
        months=pd.concat(monthly,ignore_index=True)
        for use in ['heating_kwh','cooling_kwh']:
            panels=[]
            for climate in climates:
                rows=[]
                for method in complete:
                    vals=months[(months.climate==climate)&(months.method==method)].groupby('month')[use].mean().reindex(range(1,13))
                    if vals.notna().any():rows.append(series(NAMES[method],vals,COLORS[method]))
                panels.append(panel(climate,'line',rows,[calendar.month_abbr[i] for i in range(1,13)],'Month','Mean kWh/month'))
            all_values=[v for p in panels for s in p['series'] for v in s['values'] if v is not None]
            for p in panels:p['limits']={'y':[0,max(all_values,default=1)*1.12 or 1]}
            add('monthly_'+use.split('_')[0],f'Monthly sensible {use.split("_")[0]}','Unweighted scenario means within each climate. EnergyPlus benchmark only; hybrid and ISTAT do not identify monthly loads.',panels)
    errors=matched_errors(annual);errors.to_csv(figdir/'matched_errors.csv',index=False)
    aggregation=aggregate_accuracy(errors);aggregation.to_csv(out/'aggregation_metrics.csv',index=False)
    panels=[]
    for use in ['heating_kwh','cooling_kwh']:
        preferred='hybrid_signature' if 'hybrid_signature' in set(aggregation.method) else 'hybrid_istat'
        selected=aggregation[(aggregation.end_use==use)&(aggregation.aggregation=='climate')&(aggregation.method==preferred)]
        if selected.empty:continue
        selected=selected.set_index('group').reindex(climates)
        panels.append(panel(use.split('_')[0].capitalize(),'bar',[
            series('Pooled signed error',selected.pooled_signed_error_pct,'#cf7f32'),
            series('Case absolute error (WAPE)',selected.case_wape_pct,'#145c85')],climates,'Climate','Error / benchmark total (%)'))
    add('aggregation_accuracy','Does aggregation improve agreement?','Selected hybrid, annual scenario totals; reference excluded. WAPE exposes case errors hidden by cancellation.',panels)
    for kind in ['parity','residual','bias']:
        panels=[]
        for use in ['heating_kwh','cooling_kwh']:
            subset=errors[errors.end_use.eq(use)];rows=[]
            for method in ['hybrid_signature','hybrid_istat']:
                part=subset[subset.method.eq(method)]
                if part.empty:continue
                if kind=='bias':rows.append(series(NAMES[method],part.groupby('climate').error.mean().reindex(climates),COLORS[method]))
                else:rows.append(series(NAMES[method],part.estimate if kind=='parity' else part.error,COLORS[method],part.benchmark))
            if not rows:continue
            high=max(float(subset.benchmark.max()),float(subset.estimate.max()),1)*1.06
            limits={'x':[0,high],'y':[0,high]} if kind=='parity' else None
            panels.append(panel('Sensible '+use.split('_')[0],'bar' if kind=='bias' else 'scatter',rows,climates if kind=='bias' else None,'Climate' if kind=='bias' else 'EnergyPlus (kWh/year)','Estimate (kWh/year)' if kind=='parity' else 'Estimate minus EnergyPlus (kWh/year)',limits))
        add({'parity':'scenario_agreement','residual':'scenario_errors','bias':'bias_by_climate'}[kind],{'parity':'Scenario agreement with EnergyPlus','residual':'Scenario errors against EnergyPlus','bias':'Mean error by climate'}[kind], 'Reference case excluded. Positive errors mean overestimation. EnergyPlus provides a model benchmark.',panels)
    anchor=annual[annual.is_anchor].iloc[0];curves={}
    for method in ['energyplus']:
        if not ((annual.scenario_id==anchor.scenario_id)&(annual.method==method)).any():continue
        file=data_root/'energyplus'/anchor.scenario_id/'hourly.csv' if method=='energyplus' else data_root/'hourly'/anchor.scenario_id/f'{method}.csv'
        if file.exists():curves[method]=pd.read_csv(file)
    panels=[]
    for use in ['heating_kwh','cooling_kwh']:
        rows=[]
        for method,frame in curves.items():
            if frame[use].isna().any():continue
            vals=np.sort(frame[use].to_numpy())[::-1];indices=np.unique(np.r_[np.arange(0,len(vals),60),len(vals)-1])
            rows.append(series(NAMES[method],vals[indices],COLORS[method],indices+1))
        if rows:panels.append(panel('Sensible '+use.split('_')[0],'duration',rows,None,'Hours ranked by load','Hourly average thermal kW'))
    add('anchor_load_duration','Reference-case load duration',f'{manifest["anchor_scenario"].replace("__"," / ")}. Curves rank hours independently and sample every 60th hour for display.',panels)
    profile_file=data_root/'scenario_profiles.csv'
    if profile_file.exists():
        profiles=pd.read_csv(profile_file)
        keys=annual.drop_duplicates('scenario')[['scenario','climate','dwelling','archetype']]
        profiles=profiles.merge(keys,on='scenario',validate='one_to_one');panels=[]
        for col,title,ylabel in [('appliance_kwh','Appliance electricity','Mean kWh/household/year'),('members','Household size','Mean people/household')]:
            rows=[series(climate,profiles[profiles.climate.eq(climate)].groupby('archetype')[col].mean().reindex(arches),PALETTE[i%len(PALETTE)]) for i,climate in enumerate(climates)]
            panels.append(panel(title,'bar',rows,[f'A{x}' for x in arches],'Archetype',ylabel))
        add('household_inputs','Household inputs to the models','Means across dwelling contexts. Appliance electricity is model-derived, not measured; its agreement is not independent validation.',panels)
    if profile_file.exists():
        panels=[]
        for col,title in [('heating_index','Heating index'),('cooling_index','Cooling index')]:
            rows=[series(climate,profiles[profiles.climate.eq(climate)].groupby('archetype')[col].mean().reindex(arches),PALETTE[i%len(PALETTE)]) for i,climate in enumerate(climates)]
            panels.append(panel(title,'bar',rows,[f'A{x}' for x in arches],'Archetype','Reported hours/day'))
        add('survey_hours','ISTAT heating and cooling indices','Training-only weighted indices; cooling includes valid reporting users only. No schedule-default imputation.',panels)
    audit_file=data_root/'transfer_audit.csv'
    if audit_file.exists():
        audit=pd.read_csv(audit_file).merge(annual.drop_duplicates('scenario_id')[['scenario_id','climate','dwelling','archetype']],on='scenario_id',validate='many_to_one')
        panels=[]
        for use in ['heating_kwh','cooling_kwh','appliance_kwh']:
            subset=audit[audit.end_use.eq(use)]
            rows=[series(climate,subset[subset.climate.eq(climate)].groupby('archetype').ratio.mean().reindex(arches),PALETTE[i%len(PALETTE)]) for i,climate in enumerate(climates)]
            panels.append(panel(use.split('_')[0].capitalize(),'bar',rows,[f'A{x}' for x in arches],'Archetype','Target / reference index'))
        add('istat_transfer_ratios','ISTAT transfer ratios','Means across dwelling contexts. Multiply reference annual EnergyPlus demand by the corresponding scenario ratio.',panels)
    fit_file=data_root/'reference_signature_daily.csv'
    if fit_file.exists():
        daily=pd.read_csv(fit_file);panels=[]
        if not daily.empty:
            for use in ['heating_kwh','cooling_kwh']:
                part=daily[daily.end_use.eq(use)]
                if part.empty:continue
                panels.append(panel(use.split('_')[0].capitalize(),'scatter',[
                    series('Reference EnergyPlus',part.observed_reference_kwh,'#145c85',part.outdoor_c),
                    series('Fitted reference signature',part.fitted_reference_kwh,'#388670',part.outdoor_c)],None,'Daily outdoor temperature (°C)','Reference daily kWh'))
            add('reference_energy_signature','Weather response learned from the reference','Reference fit only. Heating and cooling have separate balance points and solar coefficients. These are not target daily predictions.',panels)
    ablation_file=data_root/'signature_ablations_frozen.csv'
    if ablation_file.exists():
        stages=['Weather only','Weather + exposure','+ ISTAT hours','+ ISTAT gains']
        ablation=pd.read_csv(ablation_file)
        truth=annual[(annual.method=='energyplus') & ~annual.is_anchor]
        paired=ablation.merge(truth[['scenario_id','heating_kwh','cooling_kwh']],on='scenario_id',suffixes=('_pred','_ep'),validate='many_to_one')
        records=[];panels=[]
        for use in ['heating_kwh','cooling_kwh']:
            vals=[]
            for stage in stages:
                g=paired[paired.stage.eq(stage)].dropna(subset=[use+'_ep',use+'_pred'])
                err=g[use+'_pred']-g[use+'_ep'];mean=g[use+'_ep'].mean()
                value=float(100*np.sqrt(np.mean(err**2))/mean) if len(g) and mean>0 else np.nan;vals.append(value)
                records.append({'end_use':use,'stage':stage,'n_excluding_anchor':len(g),'cvrmse_percent':value})
            if np.isfinite(vals).any():panels.append(panel(use.split('_')[0].capitalize(),'bar',[series('CVRMSE',vals,'#388670')],['Weather','+ Exposure','+ Hours','+ Gains'],'Sequential additions','Annual CVRMSE (%)'))
        pd.DataFrame(records).to_csv(out/'signature_component_metrics.csv',index=False)
        add('signature_components','Contribution of the transfer corrections','One fixed reference fit for every stage. Components enter in the shown order; their apparent contributions depend on that order.',panels)
    if 'hybrid_signature' in set(annual.method):
        target_climates=[c for c in climates if c!=cfg['simulation']['anchor_climate']]
        for climate in target_climates:
            subset=annual[annual.climate.eq(climate)]
            for use in ['heating_kwh','cooling_kwh']:
                panels=[]
                for context in contexts:
                    rows=[]
                    for method in ['energyplus','hybrid_signature']:
                        part=subset[(subset.method==method)&(subset.dwelling==context)].set_index('archetype')[use].reindex(arches)
                        if part.notna().any():rows.append(series(NAMES[method],part,COLORS[method]))
                    panels.append(panel(context,'bar',rows,[f'A{x}' for x in arches],'Archetype','Annual kWh'))
                add('target_'+climate.lower()+'_'+use.split('_')[0],f'{climate}: {use.split("_")[0]} transfer',f'Reference: {manifest["anchor_scenario"].replace("__"," / ")}. Target EnergyPlus runs serve only as the benchmark.',panels)
    for spec in specs:render_figure(spec,figdir)
    payload={'schema_version':1,'source_run':str(data_root.resolve()),'scenario_count':int(expected),'figures':specs}
    (out/'plot_data.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    return payload


def render_figure(spec,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    n=len(spec['panels']);fig,axes=plt.subplots(1,n,figsize=(7*n,5.8),squeeze=False,layout='constrained');table=[]
    for ax,p in zip(axes.flat,spec['panels']):
        for i,s in enumerate(p['series']):
            y=np.array([np.nan if v is None else v for v in s['values']]);x=np.asarray(s.get('x',range(len(y))))
            if p['type']=='bar':
                width=.78/len(p['series']);ax.bar(x+(i-(len(p['series'])-1)/2)*width,y,width=width,color=s['color'],label=s['name'])
            elif p['type']=='scatter' and not s['reference']:ax.scatter(x,y,s=33,alpha=.83,color=s['color'],label=s['name'])
            else:ax.plot(x,y,color=s['color'],lw=1.5 if s['reference'] else 2.1,ls='--' if s['reference'] else '-',label=s['name'],marker='o' if p['type']=='line' else None,markersize=3)
            for j,v in enumerate(y):table.append({'panel':p['title'],'series':s['name'],'x':float(x[j]),'category':p['categories'][j] if p['categories'] else '', 'value':v,'reference_line':s['reference']})
        if p['categories']:ax.set_xticks(range(len(p['categories'])),p['categories'])
        ax.set(title=p['title'],xlabel=p['xlabel'],ylabel=p['ylabel']);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if p.get('limits'):
            if 'x' in p['limits']:ax.set_xlim(p['limits']['x'])
            if 'y' in p['limits']:ax.set_ylim(p['limits']['y'])
        if p['type']=='bar' and any((v or 0)<0 for s in p['series'] for v in s['values']):ax.axhline(0,c='#78858e',lw=.8)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,fontsize=9,frameon=False)
    fig.savefig(folder/f'{spec["key"]}.png',dpi=200);fig.savefig(folder/f'{spec["key"]}.svg');plt.close(fig)
    pd.DataFrame(table).to_csv(folder/f'{spec["key"]}.csv',index=False)
