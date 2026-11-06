"""Transparent plots and separate accuracy accounting for independent inputs."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd

NAMES={'energyplus':'EnergyPlus','known_summary':'Known aggregates','istat_profile':'ISTAT profiles'}
COLORS={'energyplus':'#145c85','known_summary':'#388670','istat_profile':'#cf7f32'}
USES=['heating_kwh','cooling_kwh','appliance_kwh']


def errors_and_metrics(annual):
    ep=annual[annual.method.eq('energyplus')].set_index('scenario_id')
    rows=[];metrics=[]
    for row in annual[~annual.method.eq('energyplus') & ~annual.is_anchor].to_dict('records'):
        if row['scenario_id'] not in ep.index:continue
        for use in USES:
            actual=float(ep.loc[row['scenario_id'],use]);pred=float(row[use])
            if not np.isfinite(pred):continue
            rows.append({**{k:row[k] for k in ['scenario_id','climate','dwelling','profile_id','method']},
                         'end_use':use,'benchmark':actual,'estimate':pred,'error':pred-actual})
    errors=pd.DataFrame(rows,columns=['scenario_id','climate','dwelling','profile_id','method','end_use','benchmark','estimate','error'])
    for (method,use),g in errors.groupby(['method','end_use']):
        # Directly supplying annual appliance kWh is not an accuracy test.
        if method=='known_summary' and use=='appliance_kwh':continue
        rmse=float(np.sqrt(np.mean(g.error**2)));mean=float(g.benchmark.mean())
        metrics.append({'method':method,'end_use':use,'n':len(g),'mae_kwh':float(g.error.abs().mean()),
                        'rmse_kwh':rmse,'bias_kwh':float(g.error.mean()),'cvrmse_percent':100*rmse/mean if mean>0 else None})
    return errors,pd.DataFrame(metrics,columns=['method','end_use','n','mae_kwh','rmse_kwh','bias_kwh','cvrmse_percent'])


def make_report(out,cfg,annual,manifest,doc,data_root=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .independent import dump
    data_root=Path(data_root or out);plots=out/'plots';plots.mkdir(exist_ok=True)
    errors,metrics=errors_and_metrics(annual)
    errors.to_csv(out/'case_errors.csv',index=False);metrics.to_csv(out/'comparison_metrics.csv',index=False)
    aggregates=pd.read_csv(out/'independent_aggregates.csv');specs=[]
    def bar(title,rows,category,value,methods=None,ylabel='kWh/year'):
        cats=list(dict.fromkeys(rows[category]));series=[]
        for method in methods or [m for m in NAMES if m in set(rows.method)]:
            if method not in set(rows.method):continue
            g=rows[rows.method.eq(method)].groupby(category)[value].mean()
            series.append({'name':NAMES.get(method,method),'values':[float(g.get(c,np.nan)) for c in cats],
                           'color':COLORS.get(method,'#145c85')})
        return {'type':'bar','title':title,'categories':cats,'series':series,'xlabel':category.replace('_',' '),'ylabel':ylabel}
    def figure(key,title,note,panels):
        specs.append({'key':key,'title':title,'note':note,'panels':panels})
        fig,axes=plt.subplots(1,len(panels),figsize=(7*len(panels),4.8),squeeze=False)
        records=[]
        for pi,(ax,panel) in enumerate(zip(axes[0],panels)):
            for si,series in enumerate(panel['series']):
                values=series['values'];x=series.get('x',list(range(len(values))))
                if panel['type']=='scatter':ax.scatter(x,values,label=series['name'],s=23,color=series['color'],alpha=.8)
                elif panel['type']=='line':ax.plot(x,values,label=series['name'],color=series['color'],linewidth=2)
                else:
                    n=len(panel['series']);width=.8/max(n,1)
                    ax.bar(np.arange(len(values))+(si-(n-1)/2)*width,values,width,label=series['name'],color=series['color'])
                    ax.set_xticks(range(len(values)),panel['categories'])
                records.extend({'panel':panel['title'],'series':series['name'],'x':panel.get('categories',x)[j],'value':v} for j,v in enumerate(values))
            ax.set(title=panel['title'],xlabel=panel['xlabel'],ylabel=panel['ylabel']);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            ax.legend(fontsize=8,loc='best')
        fig.tight_layout();fig.savefig(plots/f'{key}.png',dpi=170);fig.savefig(plots/f'{key}.svg');plt.close(fig)
        pd.DataFrame(records).to_csv(plots/f'{key}.csv',index=False)
    profiles=list(aggregates.profile_id)
    full_benchmark=int(annual.method.eq('energyplus').sum())==int(manifest['scenario_count'])
    displayed=annual if full_benchmark else annual[~annual.method.eq('energyplus')]
    palette=['#145c85','#388670','#cf7f32','#8b5fa6','#c45463']
    for kind,cols in [('controls',[('heat','Heating setpoint','°C'),('occupancy','Presence','fraction at home')]),
                      ('electricity',[('appliance_w','Appliance demand','W'),('cool','Cooling setpoint','°C')])]:
        panels=[]
        for col,title,unit in cols:
            series=[]
            for i,pid in enumerate(profiles):
                frame=pd.read_csv(data_root/'independent_inputs'/f'{pid}.csv',nrows=24)
                series.append({'name':pid,'x':list(range(24)),'values':frame[col].tolist(),'color':palette[i%len(palette)]})
            panels.append({'type':'line','title':title,'series':series,'xlabel':'Hour beginning (weekday)','ylabel':unit})
        figure('schedule_'+kind,'Independent weekday schedules',
               'Synthetic inputs, specified before simulation. Weekend schedules differ; all 8,760 input hours are saved.',panels)
    panels=[]
    for col,title,unit in [('heating_hours','Heating operation','equivalent active hours/day'),('appliance_kwh','Appliance electricity','kWh/year')]:
        panels.append({'type':'bar','title':title,'categories':profiles,
            'series':[{'name':'Specified input','values':aggregates[col].tolist(),'color':'#145c85'}],'xlabel':'Independent profile','ylabel':unit})
    figure('input_aggregates','Aggregates seen by the transfer model',
           'Equivalent hours come from annual mean thermostat settings. They are not hours with positive heating energy.',panels)
    for use in USES[:2]:
        panels=[]
        for method in ['known_summary','istat_profile']:
            g=errors[errors.method.eq(method)&errors.end_use.eq(use)]
            if g.empty:continue
            series=[]
            for i,(city,group) in enumerate(g.groupby('climate',sort=False)):
                series.append({'name':city,'x':group.benchmark.tolist(),'values':group.error.tolist(),'color':palette[i]})
            panels.append({'type':'scatter','title':NAMES[method],'series':series,'xlabel':'EnergyPlus (kWh/year)','ylabel':'Estimate minus EP (kWh/year)'})
        if panels:figure(use+'_errors',use.split('_')[0].capitalize()+' transfer errors',
            'Reference excluded. Markers only. Known aggregates test thermal transfer; ISTAT profiles add behavioural prediction error.',panels)
    for grouping,title in [('climate','Annual demand by climate'),('dwelling','Annual demand by exposure')]:
        figure('annual_'+grouping,title,'Equal-weight scenario means; not population estimates.'+('' if full_benchmark else ' Predictions only: target EP cases were not simulated.'),
               [bar(use.split('_')[0].capitalize(),displayed,grouping,use) for use in USES[:2]])
    if not errors.empty:
        er=errors[errors.end_use.isin(USES[:2])].copy();er['absolute_error']=er.error.abs()
        figure('profile_errors','Error by independent household profile','Mean absolute error, reference excluded. P2_shift is deliberately timing-shifted.',
            [bar(use.split('_')[0].capitalize(),er[er.end_use.eq(use)],'profile_id','absolute_error',ylabel='MAE (kWh/year)') for use in USES[:2]])
    pairs=[]
    for item in doc['profiles']:
        if not item.get('paired_with'):continue
        # The ISTAT test has a special observed reference but predicted targets;
        # keep that different information set out of the pure timing contrast.
        for row in annual[annual.profile_id.eq(item['id']) & annual.method.isin(['energyplus','known_summary'])].to_dict('records'):
            other=annual[annual.profile_id.eq(item['paired_with']) & annual.climate.eq(row['climate']) & annual.dwelling.eq(row['dwelling']) & annual.method.eq(row['method'])]
            if len(other)!=1:continue
            for use in USES:
                pairs.append({'climate':row['climate'],'dwelling':row['dwelling'],'case':row['climate']+' / '+row['dwelling'],
                              'method':row['method'],'end_use':use,'difference_kwh':row[use]-float(other.iloc[0][use])})
    pair_frame=pd.DataFrame(pairs);pair_frame.to_csv(out/'timing_pair_differences.csv',index=False)
    if len(pair_frame) and full_benchmark:
        figure('timing_pair','Same totals, different timing','P2_shift minus P2. Same household size, annual energy, mean presence and operation hours.',
               [bar(use.split('_')[0].capitalize(),pair_frame[pair_frame.end_use.eq(use)],'climate','difference_kwh',ylabel='Mean timing effect (kWh/year)') for use in USES[:2]])
    if 'istat_profile' in set(annual.method):
        figure('appliance_prediction','Appliance electricity prediction',
            'Known-aggregate appliance agreement is bookkeeping and is excluded from validation metrics. ISTAT supplies a separate prediction.',
            [bar('Annual appliance electricity',displayed,'profile_id','appliance_kwh',methods=['energyplus','istat_profile'])])
    fit=json.loads((out/'signature_fit.json').read_text())
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    assignment=pd.read_csv(out/'demographic_assignment.csv').to_dict('records') if (out/'demographic_assignment.csv').exists() else []
    payload=clean({'config':cfg,'manifest':manifest,'profiles':doc,'aggregates':aggregates.to_dict('records'),
        'assignments':assignment,'metrics':metrics.to_dict('records'),'plots':specs,'fit':fit})
    dump(out/'report_data.json',payload)
    sections=''.join(f'<section><h2>{html.escape(p["title"])}</h2><p>{html.escape(p["note"])}</p><img src="plots/{p["key"]}.png"><p><a href="plots/{p["key"]}.svg">SVG</a> / <a href="plots/{p["key"]}.csv">data CSV</a></p></section>' for p in specs)
    body=f'''<!doctype html><meta charset="utf-8"><title>Independent schedule benchmark</title>
    <style>body{{font:16px system-ui;max-width:1180px;margin:35px auto;padding:20px;color:#223746}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #ddd}}section{{margin:40px 0}}</style>
    <h1>Independent schedule benchmark</h1><p>{manifest['scenario_count']} cases. Reference: {html.escape(manifest['anchor_scenario'])}.</p>
    <p>Schedule evidence: <b>{html.escape(doc['evidence_kind'])}</b>. {html.escape(doc['source'])}</p>
    <p>EnergyPlus receives independent schedules. The known-aggregate hybrid receives only household size, annual appliance kWh, mean presence and equivalent thermostat hours. The ISTAT-profile hybrid receives demographic assignments and frozen survey indices. Both use only one reference EP result. No target thermal solver, 1R1C state or hourly hybrid prediction is used.</p>
    <p>Target EP outputs are evaluation only. Geometry and weather are shared experimental controls. This is a model-to-model stress test, not measured-building validation. Synthetic household cases are not a representative population sample.</p>
    <h2>Accuracy (reference excluded)</h2>{metrics.to_html(index=False,float_format=lambda x:f'{x:,.2f}')}
    <p>Known-aggregate appliance accuracy is excluded: annual appliance energy was supplied as an input. EP thermal demand is ideal sensible load, not delivered fuel or HVAC electricity.</p>
    <h2>Demographic-only assignment</h2>{pd.DataFrame(assignment).to_html(index=False)}
    {sections}<h2>Reproducibility</h2><p>Frozen schedules: independent_inputs/. Predictions: predictions_frozen.csv. Transfer terms: known_summary_transfer_audit.csv and istat_profile_transfer_audit.csv. See INDEPENDENT_BENCHMARK.md in the application for equations and input schema.</p>
    <p>EnergyPlus 25.1 documentation: <a href="https://bigladdersoftware.com/epx/docs/25-1/input-output-reference/group-schedules.html">Schedules</a> and <a href="https://bigladdersoftware.com/epx/docs/25-1/input-output-reference/group-internal-gains-people-lights-other.html">Internal gains</a>.</p>'''
    (out/'report.html').write_text(body,encoding='utf-8')
