"""Independent post hoc raw-hourly peak context; no model or producer analysis import."""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys,time,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PRE=ROOT/'05_simulations/archetype_campaign_v2/pre_run'
SITES=['Padova','Bolzano','Roma','Palermo']
RESTART=6912;DAY_END=6935

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out-name',default='posthoc_peak_context_v1');args=parser.parse_args()
    out=HERE/args.out_name;assert out.resolve().is_relative_to(HERE.resolve())
    if out.exists():raise FileExistsError(f'Preserving existing diagnostic destination: {out}')
    out.mkdir()
    plan=ROOT/'03_methods/archetype_extension/posthoc_peak_context_plan.md'
    (out/'diagnostic_plan.md').write_bytes(plan.read_bytes());(out/'diagnostic_script.py').write_bytes(Path(__file__).read_bytes())
    selection=json.loads((PRE/'selection_manifest.json').read_text())
    assert sha(PRE/'role_manifest.csv')==selection['output_sha256']['role_manifest.csv']
    roles=list(csv.DictReader((PRE/'role_manifest.csv').open(newline='',encoding='utf-8')))
    assert len(roles)==36 and len({r['profile_id'] for r in roles})==36
    assert sum(r['role']=='test' for r in roles)==32 and sum(r['role']=='component' for r in roles)==4
    sources=[{'path':str((PRE/'role_manifest.csv').relative_to(ROOT)),'sha256':sha(PRE/'role_manifest.csv')},{'path':str((PRE/'selection_manifest.json').relative_to(ROOT)),'sha256':sha(PRE/'selection_manifest.json')}]
    rows=[];total_rows=0
    for site in SITES:
        for role in roles:
            path=ROOT/f"05_simulations/archetype_campaign_v2/{site}/{role['profile_id']}_hourly.csv"
            sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
            with path.open(newline='',encoding='utf-8') as stream:hourly=list(csv.DictReader(stream))
            assert len(hourly)==8760
            assert [int(r['hour']) for r in hourly]==list(range(8760))
            fields=['sensible_load_W','heating_setpoint_C','old_air_C','outdoor_temperature_C']
            assert all(math.isfinite(float(r[k])) for r in hourly for k in fields)
            total_rows+=len(hourly)
            heating=[max(float(r['sensible_load_W']),0.) for r in hourly];maximum=max(heating)
            peaks=[h for h,q in enumerate(heating) if q==maximum] if maximum>0 else []
            first=peaks[0] if peaks else None
            context=hourly[first] if peaks else None
            restart=hourly[RESTART];before=hourly[RESTART-1]
            calendar=(datetime.datetime(2023,1,1,tzinfo=datetime.timezone(datetime.timedelta(hours=1)))+datetime.timedelta(hours=first)).isoformat() if first is not None else ''
            rows.append({'site':site,'role':role['role'],'role_id':role['role_id'],'household_id':int(role['household_id']),'archetype_id':int(role['archetype_id']),'control_branch':role['control_branch'],'profile_id':role['profile_id'],'annual_max_heating_kW':maximum/1000,'positive_heating_maximum':maximum>0,'first_max_hour':first if first is not None else '', 'first_max_calendar_fixed_UTCplus1':calendar,'exact_tied_max_hours':';'.join(map(str,peaks)),'exact_tied_max_n':len(peaks),'first_max_on_restart_day':first is not None and RESTART<=first<=DAY_END,'any_max_on_restart_day':any(RESTART<=h<=DAY_END for h in peaks),'all_max_on_restart_day':bool(peaks) and all(RESTART<=h<=DAY_END for h in peaks),'first_max_at_restart_hour':first==RESTART,'any_max_at_restart_hour':RESTART in peaks,'first_max_in_January':first is not None and 0<=first<744,'first_max_old_air_C':float(context['old_air_C']) if context else '', 'first_max_setpoint_C':float(context['heating_setpoint_C']) if context else '', 'first_max_outdoor_C':float(context['outdoor_temperature_C']) if context else '', 'restart_hour_heating_kW':heating[RESTART]/1000,'restart_hour_old_air_C':float(restart['old_air_C']),'restart_hour_setpoint_C':float(restart['heating_setpoint_C']),'previous_hour_setpoint_C':float(before['heating_setpoint_C'])})
    assert len(rows)==144 and total_rows==1261440
    counts=[]
    for site in SITES:
        for role,denominator in [('test',32),('component',4)]:
            cohort=[r for r in rows if r['site']==site and r['role']==role];assert len(cohort)==denominator
            count={'site':site,'role':role,'denominator':denominator}
            for key in ['positive_heating_maximum','first_max_on_restart_day','any_max_on_restart_day','all_max_on_restart_day','first_max_at_restart_hour','any_max_at_restart_hour','first_max_in_January']:
                count['n_'+key]=sum(bool(r[key]) for r in cohort)
            count['n_multiple_exact_ties']=sum(r['exact_tied_max_n']>1 for r in cohort)
            counts.append(count)
    write_csv(out/'scientific_peak_context.csv',rows);write_csv(out/'cohort_counts.csv',counts)
    write_csv(out/'input_file_hashes.csv',sources)
    manifest={'status':'post hoc descriptive peak context after R5-02; primary outcomes unchanged','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'script_sha256':sha(__file__),'plan_sha256':sha(plan),'selection_manifest_sha256':sha(PRE/'selection_manifest.json'),'role_manifest_sha256':sha(PRE/'role_manifest.csv'),'scientific_hourly_files_read':144,'scientific_rows_read':total_rows,'test_files':128,'component_files':16,'per_site_denominators':{'test':32,'component':4},'technical_cases_read':0,'annual_simulations_run':0,'exact_tie_rule':'all exactly equal positive maxima preserved; first occurrence used for first-max context','restart_hour':RESTART,'restart_day_inclusive':[RESTART,DAY_END],'calendar':'2023 nonleap fixed UTC+1, no DST','read_producer_endpoint_summaries':False,'imported_engine_or_producer_analysis':False,'primary_endpoints_changed':False,'frozen_results_changed':False,'output_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},'command_argv':[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]],'python':sys.version}
    (out/'diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'manifest_sha256':sha(out/'diagnostic_manifest.json'),'counts':counts},indent=2))
if __name__=='__main__':main()
