"""Standard-library endpoint reconstruction and frozen v2 mixture/loss analysis."""
from pathlib import Path
import csv,json,math,hashlib,argparse
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PRE=ROOT/'05_simulations/archetype_campaign_v2/pre_run'
ENDPOINTS={'annual_sensible_heating_kWh':'kWh','peak_sensible_heating_kW':'kW'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loss(rows,weight):
    weights=[float(r[weight]) for r in rows];assert abs(math.fsum(weights)-1)<1e-12
    result={}
    for method in ['conditional','pooled']:
        errors=[float(r[method+'_prediction'])-float(r['detailed_reference']) for r in rows]
        mse=math.fsum(w*e*e for w,e in zip(weights,errors))
        result[method+'_MSE']=mse;result[method+'_RMSE']=math.sqrt(mse)
        result[method+'_MAE']=math.fsum(w*abs(e) for w,e in zip(weights,errors))
        result[method+'_signed_deviation']=math.fsum(w*e for w,e in zip(weights,errors))
    result['paired_MSE_difference_conditional_minus_pooled']=result['conditional_MSE']-result['pooled_MSE']
    result['paired_MAE_difference_conditional_minus_pooled']=result['conditional_MAE']-result['pooled_MAE']
    mean=math.fsum(w*float(r['detailed_reference']) for w,r in zip(weights,rows))
    result['detailed_reference_weighted_mean']=mean
    result['detailed_reference_weighted_SD']=math.sqrt(math.fsum(w*(float(r['detailed_reference'])-mean)**2 for w,r in zip(weights,rows)))
    result['detailed_reference_min']=min(float(r['detailed_reference']) for r in rows)
    result['detailed_reference_max']=max(float(r['detailed_reference']) for r in rows)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',default='archetype_campaign_v2');p.add_argument('--out-name',default='results_v2');args=p.parse_args()
    campaign=ROOT/'05_simulations'/args.campaign;assert campaign.resolve().is_relative_to((ROOT/'05_simulations').resolve())
    out=HERE/args.out_name;assert out.resolve().is_relative_to(HERE)
    if (out/'analysis_manifest.json').exists():raise FileExistsError('Preserving completed analysis')
    out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((PRE/'selection_manifest.json').read_text())
    for filename,digest in selection['output_sha256'].items():assert sha(PRE/filename)==digest,f'Frozen selection drift: {filename}'
    roles=list(csv.DictReader((PRE/'role_manifest.csv').open(newline='')));tests=[r for r in roles if r['role']=='test'];components=[r for r in roles if r['role']=='component']
    probabilities=json.loads((PRE/'mixture_probabilities.json').read_text());individual=[];metrics=[];mixtures=[];integration=[]
    for site in ['Padova','Bolzano','Roma','Palermo']:
        run=campaign/site;design=json.loads((run/'design.json').read_text())
        frozen=json.loads((run/'frozen_manifest.json').read_text())
        assert frozen['selection_manifest_sha256']==sha(PRE/'selection_manifest.json')
        assert sha(run/'selection_manifest.json')==sha(PRE/'selection_manifest.json')
        for filename in ['selection_protocol.md','role_manifest.csv','profile_manifest.json','mixture_probabilities.json','component_manifest.csv','runtime_source_manifest.json']:
            assert sha(run/filename)==sha(PRE/filename),f'Run/analysis source mismatch: {site}/{filename}'
        runtime=json.loads((run/'runtime_source_manifest.json').read_text())
        assert runtime['sha256'][str(Path(__file__).resolve())]==sha(__file__)
        assert len(design['cases'])==38
        for name in ['checks.json','full_step_balance_checks.json','independent_surface_forcing_checks.json']:assert json.loads((run/name).read_text())['pass']
        summary={r['case']:r for r in json.loads((run/'summary.json').read_text())};endpoints={}
        for profile in sorted({r['profile_id'] for r in roles}):
            path=run/f'{profile}_hourly.csv';hourly=list(csv.DictReader(path.open(newline='')))
            assert len(hourly)==8760 and [int(r['hour']) for r in hourly]==list(range(8760))
            q=[float(r['sensible_load_W']) for r in hourly];assert all(math.isfinite(v) for v in q)
            y={'annual_sensible_heating_kWh':math.fsum(max(v,0) for v in q)/1000,'peak_sensible_heating_kW':max(max(v,0) for v in q)/1000}
            for endpoint,value in y.items():assert abs(value-summary[profile][endpoint])<1e-8
            assert sha(path)==summary[profile]['hourly_sha256']
            endpoints[profile]=y;integration.append({'site':site,'profile_id':profile,'hourly_sha256':sha(path),**y})
        for endpoint,unit in ENDPOINTS.items():
            bybranch={r['control_branch']:endpoints[r['profile_id']][endpoint] for r in components}
            pooled=math.fsum(probabilities['pooled_P_branch'][b]*bybranch[b] for b in bybranch)
            conditional={g:math.fsum(prior[b]*bybranch[b] for b in bybranch) for g,prior in probabilities['training_P_branch_given_group'].items()}
            mixtures.append({'site':site,'endpoint':endpoint,'unit':unit,'pooled_prediction':pooled,**{f'group_{g}_prediction':v for g,v in conditional.items()},'annual_maximum_definition':'mean of individual annual maxima; no coincident curve maximum'})
            group_rows=[]
            for r in tests:
                g=str(int(r['archetype_id']));ref=endpoints[r['profile_id']][endpoint];cg=conditional[g]
                row={'site':site,'endpoint':endpoint,'unit':unit,'household_id':int(r['household_id']),'archetype_id':g,'control_branch':r['control_branch'],'profile_id':r['profile_id'],'overall_empirical_weight':float(r['overall_empirical_weight']),'within_group_weight':float(r['within_group_weight']),'detailed_reference':ref,'conditional_prediction':cg,'pooled_prediction':pooled,'conditional_signed_deviation':cg-ref,'pooled_signed_deviation':pooled-ref,'paired_squared_deviation_difference':(cg-ref)**2-(pooled-ref)**2,'paired_absolute_deviation_difference':abs(cg-ref)-abs(pooled-ref)}
                individual.append(row);group_rows.append(row)
            metrics.append({'site':site,'endpoint':endpoint,'unit':unit,'domain':'all_eligible_empirical_groups','n_test':32,**loss(group_rows,'overall_empirical_weight')})
            for g in ['1','2','3','4']:
                rows=[r for r in group_rows if r['archetype_id']==g]
                metrics.append({'site':site,'endpoint':endpoint,'unit':unit,'domain':'archetype_'+g,'n_test':len(rows),**loss(rows,'within_group_weight')})
    for name,rows in [('individual_deviations.csv',individual),('weighted_metrics.csv',metrics),('mixture_predictions.csv',mixtures),('independent_endpoints.csv',integration)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    manifest={'status':'candidate model-reference results; independent review required','campaign':args.campaign,'scientific_profiles_per_site':36,'scientific_executions':144,'technical_executions':8,'test_people':32,'components':4,'selection_manifest_sha256':sha(PRE/'selection_manifest.json'),'probabilities_sha256':sha(PRE/'mixture_probabilities.json'),'analysis_script_sha256':sha(__file__),'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'independent_hourly_integration_pass':True,'no_relative_error_or_zero_division':True,'expected_individual_peak_not_aggregate_peak':True}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
