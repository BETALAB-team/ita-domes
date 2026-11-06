"""One reference simulation -> frozen ISTAT transfer -> optional EP benchmark."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import subprocess
import threading
import time
import traceback
import uuid
import numpy as np
import pandas as pd
from . import __version__
from .config import validate, save
from .analysis import prepare, scenario_profile
from .physics import read_weather
from .energyplus import make_model, run_model
from .transfer import predict, END_USES


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cfg, log=print, cancel=None):
    validate(cfg); cancel=cancel or threading.Event()
    if cfg['benchmark']['mode']=='independent_schedules':
        from .independent import run as run_independent
        return run_independent(cfg,log,cancel)
    root=Path(cfg['paths']['output_root']).resolve(); root.mkdir(parents=True,exist_ok=True)
    out=root/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]); out.mkdir()
    save(cfg,out/'config.json')
    manifest={'version':__version__,'method_version':'reference_signature_v1',
              'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
              'python':platform.python_version(),'input_sha256':sha(cfg['paths']['households']),
              'config_sha256':sha(out/'config.json'),'output':str(out),'weather':[],
              'model_scope':'Annual sensible zone heating/cooling demand; appliance electricity separate.',
              'data_boundary':'2021 heating/cooling indices are reported daily hours, not measured thermal kWh. Appliance kWh are model-derived.',
              'hybrid_equation':'EP(reference,u) * conductance ratio * adjusted degree-sum ratio; ISTAT ratio retained as baseline',
              'selected_hybrid':cfg['hybrid']['method'],
              'assumptions':[
                  'One fixed-area hypothetical space; not a surveyed home or a metered validation case.',
                  'The baseline scales reference energy by survey hours. The extended hybrid learns its balance point and solar response from reference daily EP energy only.',
                  'Survey indices use training households, supported joint cohorts and survey weights. Missing cooling hours are excluded, not filled.',
                  'Cooling hours describe reporting cooling users; they are not an unconditional population cooling-energy index.',
                  'Target EPW weather supplies daily climate and solar indices; survey macro-regions supply household associations, not exact city observations.',
                  'Exposure and mean internal-gain adjustments are static assumptions. The hybrid has no thermal capacitance, indoor-temperature state or target thermal simulation.',
                  'Hybrid annual estimates have no inferred hourly shape, peak load or monthly distribution.',
                  'Detailed EP models use assumed daily schedules, adiabatic neighbours and constant ground temperature.',
                  'Appliance input and transfer use the same survey reconstruction; agreement is not independent validation.',
                  'Data-only reports hours and reconstructed appliance kWh; thermal kWh remain unavailable.'
              ]}
    def emit(message):
        with (out/'run.log').open('a',encoding='utf-8') as f: f.write(datetime.now().isoformat(timespec='seconds')+' '+message+'\n')
        log(message)
    emit(f'Output folder: {out}'); start=time.perf_counter()
    try:
        frame,mappings,curves=prepare(cfg,out,emit)
        manifest['selected_energy_groups']={k:int(v.group.nunique()) for k,v in mappings.items()}
        if cancel.is_set(): raise RuntimeError('Cancelled')
        s=cfg['simulation']; scenarios=[]; profiles=[]
        for climate in [c for c in cfg['climates'] if c.get('enabled')]:
            for context in [c for c in cfg['contexts'] if c.get('enabled')]:
                for a in sorted(frame.archetype.unique()):
                    profile=scenario_profile(frame,mappings,cfg,climate['location'],context['name'],int(a))
                    key=f'{climate["name"]}__{context["name"]}__A{int(a)}'
                    sc={'id':f'S{len(scenarios)+1:03d}','key':key,'climate':climate,'context':context,'archetype':int(a),'profile':profile}
                    scenarios.append(sc); profiles.append({'scenario':key,**profile})
        pd.DataFrame(profiles).to_csv(out/'scenario_profiles.csv',index=False)
        anchors=[x for x in scenarios if x['climate']['name']==s['anchor_climate'] and x['context']['name']==s['anchor_context'] and x['archetype']==s['anchor_archetype']]
        if len(anchors)!=1: raise ValueError('Reference climate, context and archetype must be an enabled scenario.')
        anchor=anchors[0]; manifest.update(anchor_scenario=anchor['key'],scenario_count=len(scenarios))
        ep={}; ep_seconds={}; warnings={}; weather={}; executed=[]; signature_results={}
        def load_weather(climate):
            name=climate['name']
            if name not in weather:
                path=Path(climate['epw']); weather[name]=read_weather(path)
                manifest['weather'].append({'name':name,'path':str(path.resolve()),'sha256':sha(path),'source':climate.get('source','user-supplied')})
        def simulate(sc):
            if cancel.is_set(): raise RuntimeError('Cancelled')
            emit(f'EnergyPlus {sc["id"]}: {sc["key"]}')
            model=make_model(cfg,sc['context'],sc['profile'],weather[sc['climate']['name']])
            executed.append(sc['id'])
            return sc['id'],run_model(model,sc['climate']['epw'],cfg['paths']['energyplus'],out/'energyplus'/sc['id'],s['timeout_seconds'],cancel)
        def accept(result):
            key,(values,seconds,warn)=result; ep[key]=values; ep_seconds[key]=seconds; warnings[key]=warn
            emit(f'Completed {key}; {len(ep)} simulation(s); {warn} warnings.')
        if s['enabled']:
            version=subprocess.run([cfg['paths']['energyplus'],'--version'],capture_output=True,text=True,timeout=15,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout.strip()
            if '25.1' not in version: raise ValueError(f'This builder is verified for EnergyPlus 25.1, found {version}.')
            manifest['energyplus_version']=version
            load_weather(anchor['climate'])
            if cfg['paths'].get('reference_run'):
                from .reuse import reuse_reference
                values,provenance=reuse_reference(cfg,anchor,weather[anchor['climate']['name']],out)
                accept((anchor['id'],(values,0.,0)));manifest['reused_reference']=provenance
            else:accept(simulate(anchor))
            manifest['reference_simulations_used']=1
        anchor_energy=ep[anchor['id']][list(END_USES)].sum().to_dict() if anchor['id'] in ep else {}
        hybrid,audit=predict(scenarios,anchor,anchor_energy)
        pd.DataFrame(audit).to_csv(out/'transfer_audit.csv',index=False)
        if s['enabled'] and cfg['hybrid']['method']=='climate_exposure_gains':
            from .signature import predict as predict_signature
            for climate in [c for c in cfg['climates'] if c.get('enabled')]:load_weather(climate)
            signature_results,signature_audit,fit,diagnostics,ablations=predict_signature(cfg,scenarios,anchor,weather,ep[anchor['id']],hybrid)
            signature_audit.to_csv(out/'signature_transfer_audit.csv',index=False)
            (out/'signature_fit.json').write_text(json.dumps(fit,indent=2,allow_nan=False),encoding='utf-8')
            diagnostics.to_csv(out/'reference_signature_daily.csv',index=False)
            ablations.to_csv(out/'signature_ablations_frozen.csv',index=False)
            manifest['signature_ablations_sha256']=sha(out/'signature_ablations_frozen.csv')
            manifest['signature_fit_sha256']=sha(out/'signature_fit.json')
            emit('Reference signature fitted. Target outcomes have not been read or simulated.')
        frozen=pd.DataFrame([{'scenario_id':sc['id'],'scenario':sc['key'],'method':method,**values[sc['id']]} for method,values in [('hybrid_istat',hybrid),('hybrid_signature',signature_results)] if values for sc in scenarios])
        frozen.to_csv(out/'hybrid_predictions_frozen.csv',index=False)
        manifest['hybrid_frozen_sha256']=sha(out/'hybrid_predictions_frozen.csv')
        manifest['hybrid_frozen_utc']=datetime.now(timezone.utc).isoformat()
        emit('ISTAT transfer predictions frozen. Target EnergyPlus outputs are evaluation only.')
        if s['enabled'] and s['run_full_benchmark']:
            targets=[sc for sc in scenarios if sc['id']!=anchor['id']]
            for climate in [c for c in cfg['climates'] if c.get('enabled')]: load_weather(climate)
            manifest['target_benchmark_started_utc']=datetime.now(timezone.utc).isoformat()
            with concurrent.futures.ThreadPoolExecutor(max_workers=s['workers']) as pool:
                futures=[pool.submit(simulate,sc) for sc in targets]
                try:
                    for future in concurrent.futures.as_completed(futures): accept(future.result())
                except Exception:
                    cancel.set()
                    for f in futures: f.cancel()
                    raise
        assert sha(out/'hybrid_predictions_frozen.csv')==manifest['hybrid_frozen_sha256']
        if 'signature_ablations_sha256' in manifest:
            assert sha(out/'signature_ablations_frozen.csv')==manifest['signature_ablations_sha256']
        manifest.update(energyplus_processes_run=len(executed),energyplus_seconds_sum=sum(ep_seconds.values()),
                        hybrid_anchor_simulation_seconds=ep_seconds.get(anchor['id']),energyplus_warning_counts=warnings)
        rows=[]
        for sc in scenarios:
            key=sc['id']; p=sc['profile']
            methods={'istat_only':{'heating_kwh':np.nan,'cooling_kwh':np.nan,'appliance_kwh':p['appliance_index']}}
            if anchor['id'] in ep: methods['hybrid_istat']=hybrid[key]
            if signature_results:methods['hybrid_signature']=signature_results[key]
            if key in ep:
                methods['energyplus']=ep[key][list(END_USES)].sum().to_dict()
                folder=out/'hourly'/key; folder.mkdir(parents=True)
                ep[key].groupby('month')[list(END_USES)].sum().to_csv(folder/'energyplus_monthly.csv')
            for method,values in methods.items():
                row={'scenario_id':key,'scenario':sc['key'],'climate':sc['climate']['name'],'dwelling':sc['context']['name'],
                     'archetype':sc['archetype'],'method':method,'is_anchor':key==anchor['id'],
                     'heating_hours_index':p['heating_index'],'cooling_hours_index':p['cooling_index']}
                for col in END_USES: row[col]=float(values[col]); row[col+'_m2']=row[col]/s['area_m2']
                rows.append(row)
        annual=pd.DataFrame(rows); annual.to_csv(out/'annual_results.csv',index=False)
        from .plots import matched_errors
        errors=matched_errors(annual); metrics=[]
        for (method,use),group in errors.groupby(['method','end_use']):
            err=group.error.to_numpy(); mean=group.benchmark.mean(); rmse=float(np.sqrt(np.mean(err**2)))
            metrics.append({'method':method,'end_use':use,'n_excluding_anchor':len(group),'mae_kwh':float(np.mean(abs(err))),
                            'rmse_kwh':rmse,'bias_kwh':float(np.mean(err)),'cvrmse_percent':float(100*rmse/mean) if mean>0 else None})
        pd.DataFrame(metrics).to_csv(out/'comparison_metrics.csv',index=False)
        make_report(out,cfg,annual,metrics,curves,manifest)
        if cfg.get('report',{}).get('powerpoint'):
            emit('Creating the PowerPoint run report…'); make_powerpoint(out,cfg,annual,metrics,curves,manifest)
        manifest.update(status='complete',elapsed_seconds=time.perf_counter()-start)
        emit(f'Finished {len(scenarios)} scenarios. Open report.html for results.')
    except Exception as exc:
        manifest.update(status='cancelled' if str(exc).lower().startswith('cancel') else 'failed',error=str(exc),traceback=traceback.format_exc())
        emit(f'FAILED: {exc}'); raise
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    return out


def make_powerpoint(out,cfg,annual,metrics,curves,manifest):
    import os
    payload={'config':cfg,'manifest':manifest,'annual':annual.replace({np.nan:None}).to_dict('records'),
             'metrics':metrics,'curves':{k:v.replace({np.nan:None}).to_dict('records') for k,v in curves.items()}}
    if (out/'plot_data.json').exists(): payload['plots']=json.loads((out/'plot_data.json').read_text(encoding='utf-8'))['figures']
    if (out/'signature_fit.json').exists():payload['signature_fit']=json.loads((out/'signature_fit.json').read_text(encoding='utf-8'))
    if (out/'signature_transfer_audit.csv').exists():payload['signature_transfer']=pd.read_csv(out/'signature_transfer_audit.csv').replace({np.nan:None}).to_dict('records')
    if (out/'transfer_audit.csv').exists(): payload['transfer']=pd.read_csv(out/'transfer_audit.csv').replace({np.nan:None}).to_dict('records')
    (out/'report_data.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    command=[cfg['report']['node'],str(Path(__file__).with_name('report_deck.mjs')),str(out.resolve())]
    with (out/'powerpoint.log').open('w',encoding='utf-8') as f:
        proc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=480,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    receipt=out/'report_build/validation.json'
    if not (out/'presentation/run_report.pptx').is_file() or not receipt.is_file():
        raise RuntimeError(f'PowerPoint export failed (exit {proc.returncode}); see {out}/powerpoint.log. Scientific CSV results remain available.')
    result=json.loads(receipt.read_text())
    if result.get('finalSha256')!=sha(out/'presentation/run_report.pptx'): raise RuntimeError('PowerPoint receipt hash mismatch.')


def make_report(out,cfg,annual,metrics,curves,manifest,data_root=None):
    from .plots import build_plot_suite
    plot_data=build_plot_suite(data_root or out,cfg,annual,manifest,out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(name,curve) in zip(axes,curves.items()):
        ax.plot(curve.k,curve.wcss/(curve.wcss.iloc[0] or 1),'o-',color='#145c85'); ax.set(xlabel='Number of groups',ylabel='Remaining between-cell WCSS',title=name.capitalize()); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'group_elbows.png',dpi=170); plt.close(fig)
    gallery=''.join(f'<section><h2>{html.escape(p["title"])}</h2><p>{html.escape(p["note"])}</p><a href="plots/{p["key"]}.png"><img loading="lazy" src="plots/{p["key"]}.png" alt="{html.escape(p["title"])}"></a><p><a href="plots/{p["key"]}.svg">Vector SVG</a> · <a href="plots/{p["key"]}.csv">Plotted data CSV</a></p></section>' for p in plot_data['figures'])
    text=f'''<!doctype html><meta charset="utf-8"><title>ISTAT transfer experiment</title>
    <style>body{{font:16px system-ui;margin:35px auto;max-width:1200px;padding:0 25px;color:#182b3b}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border:1px solid #ccd7df}}img{{max-width:100%}}h1,h2{{color:#145c85}}</style>
    <h1>One EnergyPlus reference + climate and ISTAT transfer</h1><p>{manifest['scenario_count']} scenarios · {cfg['simulation']['area_m2']:g} m² · reference: {html.escape(manifest['anchor_scenario'])}</p>
    <h2>The three methods, simply</h2><ol>
    <li>EnergyPlus: build and simulate a virtual home for each case.</li>
    <li>Hybrid: simulate one reference home. Learn its relationship between daily weather and demand. Shift that relationship using ISTAT household gains and operating hours, and scale it for the target exposure. Sum target weather features to obtain annual energy. No target heat-balance simulation. The earlier ISTAT-ratio baseline remains available.</li>
    <li>ISTAT only: report what the survey analysis supports: heating/cooling hours and reconstructed appliance electricity. Thermal kWh remain blank.</li></ol>
    <h2>Transfer process</h2><p><b>E(target,u) = E_EP(reference,u) × H(target)/H(reference) × D(target,u)/D(reference,u)</b></p><p>H is nominal exposed-envelope plus ventilation/ground conductance. D sums a fitted weather index, adjusted for ISTAT-derived mean controls and gains. Balance temperatures and solar coefficients use only reference EP daily totals. The baseline still uses direct ISTAT-index ratios.</p>
    <ol><li>Fit supported survey groups on training households. Pool predictions within each region × dwelling × archetype cohort; cooling uses valid reported hours.</li>
    <li>Run or reuse the reference EnergyPlus case once. Fit the reference weather signature.</li><li>Calculate exposure, operating-hours and gains adjustments from configured geometry and supported ISTAT profiles. Save predictions and sequential component checks before any target simulations.</li>
    <li>Optionally run target EnergyPlus cases to assess error, excluding the reference. Never feed these results back into predictions.</li></ol>
    <p>ISTAT hours constrain assumed mean thermostat settings, rather than multiplying degree days by another regional hours ratio. Exposure, gain utilisation and fixed schedule assumptions still need validation. No hybrid hourly/monthly load shape is provided. Missing inputs and insufficient reference heating/cooling activity return unavailable estimates.</p>
    <h2>Benchmark results</h2>{pd.DataFrame(metrics).to_html(index=False,float_format=lambda x:f'{x:,.2f}')}
    <h2>Group complexity</h2><img src="group_elbows.png">{gallery}
    <h2>Assumptions and limitations</h2><ul>{''.join('<li>'+html.escape(v)+'</li>' for v in manifest['assumptions'])}</ul>
    <h2>Annual results</h2>{annual.to_html(index=False,float_format=lambda x:f'{x:,.2f}')}
    <p><a href="config.json">Configuration</a> · <a href="manifest.json">Provenance</a> · <a href="transfer_audit.csv">Baseline audit</a> · <a href="signature_transfer_audit.csv">Extended transfer audit</a> · <a href="signature_fit.json">Reference fit</a> · <a href="hybrid_predictions_frozen.csv">Frozen predictions</a> · <a href="annual_results.csv">Annual CSV</a></p>'''
    (out/'report.html').write_text(text,encoding='utf-8')


def refresh_reports(run_dir,log=print,powerpoint=None):
    """New report revision for ISTAT-transfer runs; never relabel legacy RC results."""
    import shutil
    source=Path(run_dir).resolve(); manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete': raise ValueError('Choose a completed experiment to refresh reports.')
    if manifest.get('method_version')=='independent_benchmark_v1':
        from .independent import refresh_reports as refresh_independent
        return refresh_independent(source,log,powerpoint)
    if manifest.get('method_version')!='reference_signature_v1': raise ValueError('Legacy run uses a different hybrid. Run a new schema-3 experiment; these results cannot be relabelled.')
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8')); annual=pd.read_csv(source/'annual_results.csv')
    if powerpoint is not None:cfg['report']['powerpoint']=bool(powerpoint)
    try: metrics=pd.read_csv(source/'comparison_metrics.csv').replace({np.nan:None}).to_dict('records')
    except pd.errors.EmptyDataError: metrics=[]
    curves={k:pd.read_csv(source/f'{k}_elbow.csv') for k in ['heating','electricity']}
    out=source/'reports'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]); out.mkdir(parents=True)
    for name in ['config.json','annual_results.csv','comparison_metrics.csv','heating_elbow.csv','electricity_elbow.csv','scenario_profiles.csv','transfer_audit.csv','hybrid_predictions_frozen.csv','signature_fit.json','signature_transfer_audit.csv','reference_signature_daily.csv','signature_ablations_frozen.csv']:
        if (source/name).exists(): shutil.copy2(source/name,out/name)
    save(cfg,out/'config.json')
    data_root=Path(manifest.get('report_source_run',str(source)))
    manifest.update(report_source_run=str(data_root),report_version=__version__,report_created_utc=datetime.now(timezone.utc).isoformat(),report_source_annual_sha256=sha(source/'annual_results.csv'))
    manifest['report_source_config_sha256']=manifest['config_sha256']
    manifest['config_sha256']=sha(out/'config.json')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    log(f'Refreshing plots from saved results: {source}')
    make_report(out,cfg,annual,metrics,curves,manifest,data_root=data_root)
    if cfg.get('report',{}).get('powerpoint'): make_powerpoint(out,cfg,annual,metrics,curves,manifest)
    log(f'Refreshed report: {out}'); return out
