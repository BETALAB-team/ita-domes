"""Reference-first experiments with a benchmark input path isolated from ISTAT."""
import concurrent.futures
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import traceback
import uuid
import numpy as np
import pandas as pd
from . import __version__
from .config import validate,save
from .independent_inputs import load_profiles,classify_demographics
from .energyplus import make_model,run_model
from .physics import read_weather
from .transfer import predict as ratio_predict,END_USES
from .signature import predict as signature_predict


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def summary_profile(profile):
    """No target hourly schedules or target EP outputs cross this boundary."""
    return {k:copy.deepcopy(v) for k,v in profile.items() if k not in ['annual_schedules','occupancy_profile']}


def build_scenarios(cfg,doc,profiles):
    scenarios=[]
    for climate in [c for c in cfg['climates'] if c.get('enabled')]:
        for context in [c for c in cfg['contexts'] if c.get('enabled')]:
            for item in doc['profiles']:
                pid=item['id']
                scenarios.append({'id':f'I{len(scenarios)+1:03d}','key':f'{climate["name"]}__{context["name"]}__{pid}',
                    'climate':climate,'context':context,'profile_id':pid,'archetype':None,'profile':profiles[pid]})
    s=cfg['simulation'];b=cfg['benchmark']
    anchors=[x for x in scenarios if x['climate']['name']==s['anchor_climate'] and x['context']['name']==s['anchor_context'] and x['profile_id']==b['anchor_profile']]
    if len(anchors)!=1:raise ValueError('Independent reference must be an enabled case.')
    return scenarios,anchors[0]


def istat_scenarios(cfg,out,scenarios,doc,log):
    """Fit survey-only profiles. EP still receives the independent case objects."""
    from .analysis import prepare,scenario_profile
    survey=out/'istat_model';survey.mkdir()
    frame,mappings,_=prepare(cfg,survey,log)
    if cfg['analysis']['archetype_mode']=='existing':
        model=json.loads(Path(cfg['benchmark']['demographic_model']).read_text(encoding='utf-8'))
        labels,_=classify_demographics(frame.to_dict('records'),model)
        if not np.array_equal(labels,frame.archetype.to_numpy()):
            raise ValueError('Frozen demographic model does not reproduce the survey archetype labels. Choose the matching model or refit.')
    else:model=json.loads((survey/'archetype_model.json').read_text())
    dump(survey/'assignment_model_frozen.json',model)
    records=[p['demographics'] for p in doc['profiles']]
    for p,r in zip(doc['profiles'],records):
        if r['nmembers']!=p['members']:raise ValueError('Demographic nmembers must equal independent member count.')
    labels,outside=classify_demographics(records,model)
    assignments={p['id']:int(a) for p,a in zip(doc['profiles'],labels)}
    pd.DataFrame([{'profile_id':p['id'],**p['demographics'],'archetype':int(a),'outside_reference_range':bool(o)}
                  for p,a,o in zip(doc['profiles'],labels,outside)]).to_csv(out/'demographic_assignment.csv',index=False)
    result=[]
    for sc in scenarios:
        a=assignments[sc['profile_id']]
        estimated=scenario_profile(frame,mappings,cfg,sc['climate']['location'],sc['context']['name'],a)
        # Only frozen survey indices are passed on. No actual target schedule.
        profile={k:v for k,v in estimated.items() if '_index' in k}
        result.append({**sc,'archetype':a,'profile':profile})
    pd.DataFrame([{'scenario_id':sc['id'],'profile_id':sc['profile_id'],'archetype':sc['archetype'],**sc['profile']} for sc in result]).to_csv(out/'istat_target_indices.csv',index=False)
    return result


def freeze_predictions(cfg,out,scenarios,anchor,weather,reference,istat_cases=None):
    known=[{**sc,'profile':summary_profile(sc['profile'])} for sc in scenarios]
    # The actual detailed reference is available to both tests, including gains.
    def calculate(cases,name):
        case_anchor=next(x for x in cases if x['id']==anchor['id'])
        case_anchor['profile'].update({k:v for k,v in anchor['profile'].items() if k in ['members','heating_hours','cooling_hours','appliance_kwh','occupancy_profile']})
        baseline,_=ratio_predict(cases,case_anchor,reference[list(END_USES)].sum().to_dict())
        result,audit,fit,daily,ablation=signature_predict(cfg,cases,case_anchor,weather,reference,baseline)
        audit.to_csv(out/f'{name}_transfer_audit.csv',index=False)
        ablation['stage']=ablation.stage.str.replace('ISTAT','aggregate' if name=='known_summary' else 'ISTAT',regex=False)
        ablation.to_csv(out/f'{name}_ablations.csv',index=False)
        return result,fit,daily
    values,fit,daily=calculate(known,'known_summary');predictions={'known_summary':values}
    if istat_cases is not None:
        values,other_fit,_=calculate(copy.deepcopy(istat_cases),'istat_profile')
        if fit!=other_fit:raise AssertionError('Both comparisons must use exactly the same reference fit.')
        predictions['istat_profile']=values
    dump(out/'signature_fit.json',fit);daily.to_csv(out/'reference_signature_daily.csv',index=False)
    pd.DataFrame([{'scenario_id':sc['id'],'scenario':sc['key'],'method':method,**values[sc['id']]}
                  for method,values in predictions.items() for sc in scenarios]).to_csv(out/'predictions_frozen.csv',index=False)
    return predictions


def run(cfg,log=print,cancel=None):
    validate(cfg);cancel=cancel or threading.Event();start=time.perf_counter()
    out=Path(cfg['paths']['output_root']).resolve()/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    save(cfg,out/'config.json')
    manifest={'version':__version__,'method_version':'independent_benchmark_v1','status':'running',
        'started_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':sha(out/'config.json'),
        'empirical_building_validation':False,'target_outcomes_used_for_fitting':False,
        'benchmark_uses_istat_schedules':False,'target_thermal_solver_in_hybrid':False,
        'known_summary_appliance_agreement_is_validation':False,'weather':[]}
    def emit(message):
        with (out/'run.log').open('a',encoding='utf-8') as f:f.write(datetime.now().isoformat(timespec='seconds')+' '+message+'\n')
        log(message)
    try:
        doc,profiles=load_profiles(cfg);dump(out/'independent_profiles_frozen.json',doc)
        manifest['evidence_kind']=doc['evidence_kind'];manifest['schedule_source']=doc['source']
        inputs=out/'independent_inputs';inputs.mkdir();input_hashes={}
        for pid,p in profiles.items():
            path=inputs/f'{pid}.csv';pd.DataFrame({'hour':range(1,8761),**p['annual_schedules']}).to_csv(path,index=False)
            input_hashes[pid]=sha(path)
        manifest['independent_schedule_sha256']=input_hashes
        scenarios,anchor=build_scenarios(cfg,doc,profiles)
        manifest.update(scenario_count=len(scenarios),anchor_scenario=anchor['key'],anchor_id=anchor['id'],
                        independent_inputs_frozen_utc=datetime.now(timezone.utc).isoformat())
        pd.DataFrame([{'profile_id':pid,**summary_profile(p)} for pid,p in profiles.items()]).to_csv(out/'independent_aggregates.csv',index=False)
        # Deep snapshots catch accidental mutation through shared profile dicts.
        initial=json.dumps(profiles,sort_keys=True)
        cases=istat_scenarios(cfg,out,scenarios,doc,emit) if cfg['benchmark']['compare_istat'] else None
        if json.dumps(profiles,sort_keys=True)!=initial:raise AssertionError('Survey preparation altered benchmark inputs.')
        if cases is not None:
            manifest['istat_input_sha256']=sha(cfg['paths']['households'])
            manifest['demographic_model_sha256']=sha(out/'istat_model/assignment_model_frozen.json')
        weather={}
        for climate in [c for c in cfg['climates'] if c.get('enabled')]:
            weather[climate['name']]=read_weather(climate['epw'])
            manifest['weather'].append({'name':climate['name'],'sha256':sha(climate['epw']),'source':climate.get('source','user supplied')})
        ep={};seconds={};warnings={};s=cfg['simulation']
        import subprocess
        version=subprocess.run([cfg['paths']['energyplus'],'--version'],capture_output=True,text=True,timeout=15,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout.strip()
        if '25.1' not in version:raise ValueError(f'This builder requires EnergyPlus 25.1; found {version}.')
        manifest['energyplus_version']=version
        def simulate(sc):
            if cancel.is_set():raise RuntimeError('Cancelled')
            emit(f'EnergyPlus {sc["id"]}: {sc["key"]}')
            model=make_model(cfg,sc['context'],sc['profile'],weather[sc['climate']['name']])
            return sc['id'],run_model(model,sc['climate']['epw'],cfg['paths']['energyplus'],out/'energyplus'/sc['id'],s['timeout_seconds'],cancel)
        def accept(result):
            key,(df,t,w)=result;ep[key]=df;seconds[key]=t;warnings[key]=w
            emit(f'Completed {key}; {len(ep)} simulations; {w} warnings.')
        emit('Independent EP inputs frozen. No ISTAT value sets a benchmark schedule.')
        accept(simulate(anchor))
        predictions=freeze_predictions(cfg,out,scenarios,anchor,weather,ep[anchor['id']],cases)
        manifest['predictions_frozen_sha256']=sha(out/'predictions_frozen.csv')
        manifest['signature_fit_sha256']=sha(out/'signature_fit.json')
        manifest['predictions_frozen_utc']=datetime.now(timezone.utc).isoformat()
        emit('Known-summary and optional ISTAT-profile predictions frozen before target simulations.')
        if s['run_full_benchmark']:
            manifest['target_benchmark_started_utc']=datetime.now(timezone.utc).isoformat()
            with concurrent.futures.ThreadPoolExecutor(max_workers=s['workers']) as pool:
                pending=[pool.submit(simulate,sc) for sc in scenarios if sc['id']!=anchor['id']]
                try:
                    for future in concurrent.futures.as_completed(pending):accept(future.result())
                except Exception:
                    cancel.set()
                    for f in pending:f.cancel()
                    raise
        if sha(out/'predictions_frozen.csv')!=manifest['predictions_frozen_sha256']:raise AssertionError('Frozen predictions changed.')
        if sha(out/'signature_fit.json')!=manifest['signature_fit_sha256']:raise AssertionError('Reference fit changed.')
        rows=[]
        assignments={sc['id']:sc['archetype'] for sc in cases or []}
        for sc in scenarios:
            methods={name:values[sc['id']] for name,values in predictions.items()}
            if sc['id'] in ep:
                methods['energyplus']=ep[sc['id']][list(END_USES)].sum().to_dict()
                if not np.isclose(methods['energyplus']['appliance_kwh'],sc['profile']['appliance_kwh'],rtol=1e-8):
                    raise AssertionError('EP appliance schedule total differs from frozen independent input.')
            for method,values in methods.items():
                rows.append({'scenario_id':sc['id'],'scenario':sc['key'],'climate':sc['climate']['name'],
                    'dwelling':sc['context']['name'],'profile_id':sc['profile_id'],'archetype':assignments.get(sc['id']),
                    'method':method,'is_anchor':sc['id']==anchor['id'],**values})
        annual=pd.DataFrame(rows);annual.to_csv(out/'annual_results.csv',index=False)
        manifest.update(energyplus_processes_run=len(ep),energyplus_warning_counts=warnings,
            reference_seconds=seconds[anchor['id']],energyplus_seconds_sum=sum(seconds.values()),
            scientific_elapsed_seconds=time.perf_counter()-start)
        from .independent_report import make_report
        make_report(out,cfg,annual,manifest,doc)
        if cfg['report']['powerpoint']:
            emit('Creating independent-benchmark PowerPoint…');make_powerpoint(out,cfg)
        manifest.update(status='complete',elapsed_seconds=time.perf_counter()-start)
        emit(f'Completed. Results: {out}')
    except Exception as exc:
        manifest.update(status='cancelled' if cancel.is_set() else 'failed',error=str(exc),traceback=traceback.format_exc());emit(str(exc));raise
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat();dump(out/'manifest.json',manifest)
    return out


def make_powerpoint(out,cfg):
    import subprocess
    command=[cfg['report']['node'],str(Path(__file__).with_name('independent_deck.mjs')),str(out.resolve())]
    with (out/'powerpoint.log').open('w',encoding='utf-8') as f:
        proc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=480,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    result=out/'presentation/independent_benchmark.pptx';receipt=out/'report_build/validation.json'
    if not result.exists() or not receipt.exists():raise RuntimeError(f'PowerPoint failed (exit {proc.returncode}); scientific outputs saved. See {out}/powerpoint.log')
    if json.loads(receipt.read_text())['finalSha256']!=sha(result):raise RuntimeError('PowerPoint receipt mismatch.')


def refresh_reports(source,log=print,powerpoint=None):
    from .independent_report import make_report
    source=Path(source);cfg=json.loads((source/'config.json').read_text());manifest=json.loads((source/'manifest.json').read_text())
    if powerpoint is not None:cfg['report']['powerpoint']=bool(powerpoint)
    root=Path(manifest.get('report_source_run',str(source)))
    out=root/'reports'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    for name in ['independent_profiles_frozen.json','independent_aggregates.csv','demographic_assignment.csv','istat_target_indices.csv','annual_results.csv','predictions_frozen.csv','signature_fit.json','reference_signature_daily.csv','known_summary_transfer_audit.csv','istat_profile_transfer_audit.csv']:
        if (root/name).exists():shutil.copy2(root/name,out/name)
    save(cfg,out/'config.json');manifest.update(report_source_run=str(root),report_version=__version__)
    doc=json.loads((root/'independent_profiles_frozen.json').read_text())
    make_report(out,cfg,pd.read_csv(root/'annual_results.csv'),manifest,doc,data_root=root)
    if cfg['report']['powerpoint']:make_powerpoint(out,cfg)
    dump(out/'manifest.json',manifest);log(f'Report revision: {out}');return out
