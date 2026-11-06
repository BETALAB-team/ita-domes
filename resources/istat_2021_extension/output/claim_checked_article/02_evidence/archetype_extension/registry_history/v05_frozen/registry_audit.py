"""Read-only source/producer-target audit; writes only supplemental registry artifacts.

Run with --ledger PATH --draft PATH --frozen only after author freeze.
No simulations or source-data transformations are performed here.
"""
import argparse, csv, hashlib, json, re, shutil, struct, subprocess, sys
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CWD=ROOT.parents[1]
DATE='2026-10-20'
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def absolute(p):
    p=Path(p)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
def main():
    a=argparse.ArgumentParser();a.add_argument('--ledger',required=True);a.add_argument('--draft',required=True);a.add_argument('--frozen',action='store_true');args=a.parse_args()
    ledger=absolute(args.ledger);draft=absolute(args.draft);claims=load(ledger)['claims']
    if args.frozen:
        expected={'article_claims_v05.md':'2e5889e3948a72dab10af6867b4fad193f7bfd40efecab8248c3fb3aad9773fa','claim_ledger_v05.json':'7b2f31d57d8bbadcb17a07b1249136033caa39f5381c0fb8dd985076798003b0','claim_evidence_ledger_v05.json':'53962106c455bb5e780c025878f6cee8e2b6b9de769c795a09874293d59c0f30'}
        for name,h in expected.items():assert digest(ROOT/'06_claim_drafts'/name)==h,'Author freeze identity mismatch: '+name
        index=load(ROOT/'06_claim_drafts/claim_evidence_ledger_v05.json')
        assert len(claims)==len(index['claims'])==105 and len({c['id'] for c in claims})==105
        assert all(index['claims'][c['id']]['source_ids']==c['source_ids'] and index['claims'][c['id']]['locators']==c['locators'] for c in claims),'Evidence-index mapping differs from ledger'
    files={};checks=[];identity_checks=[];unresolved=[]
    def file(p,depth='Existence and exact byte SHA256; format/selected relevant contents inspected',stability=None):
        p=absolute(p);key=str(p)
        if key not in files:
            ok=p.is_file();files[key]={'path':key,'exists':ok,'access_date':DATE,'access_depth':depth,'bytes':p.stat().st_size if ok else None,'sha256':digest(p) if ok else None,'stability':stability or ('immutable per-run/pre-run snapshot' if '05_simulations' in p.parts else 'current artifact; byte version bound here')}
        return key
    def verify(producer,field,target,expected):
        target=file(target);actual=files[target]['sha256'];checks.append({'producer':str(absolute(producer)),'field':field,'target':target,'expected_sha256':expected,'actual_sha256':actual,'match':actual==expected})
    priorpath=ROOT/'02_evidence/local_sources.json';prior=load(priorpath)
    litpath=HERE/'sources.json';lit=load(litpath)
    confpath=ROOT/'02_evidence/conference_sources.json';conf=load(confpath)
    old={s['id']:s for s in prior['sources']};external={s['id']:s for s in lit['sources']};conference={s['source_id']:s for s in conf['sources']}
    by=defaultdict(list)
    for c in claims:
        for sid in c['source_ids']:by[sid].append(c)
    manual={
      'ARCH2025-FULL':['C:/Users/khajmoh18975/Downloads/1-s2.0-S0378778825006541-main.pdf','00_context/archetype_extension/primary_sources_and_local_inventory.json'],
      'ARCH-SPEC':['00_context/archetype_extension/published_archetype_specification.md'],
      'ARCH-ASSIGN':['00_context/archetype_extension/assignment_mapping_schema.json','00_context/archetype_extension/archived_four_archetype_assignments.csv',str(CWD/'clusters.csv')],
      'ARCH-AUDIT':['00_context/archetype_extension/local_assignment_audit.json','00_context/archetype_extension/assignment_packet_checks.json'],
      'ARCH-CONTROL-AUDIT':['00_context/archetype_extension/survey_control_eligibility_audit.json','00_context/archetype_extension/survey_control_routing.md'],
      'AE-DOMAIN-AUDIT':['04_analysis/archetype_extension/archived_operating_domain_audit.json'],
      'AE-PROTOCOL':['03_methods/archetype_extension/simulation_protocol.md'],
      'AE-V2-PROTOCOL':['03_methods/archetype_extension/mixture_protocol_v2.md','05_simulations/archetype_campaign_v2/pre_run/selection_protocol.md'],
      'AE-DESIGN':['01_plan/archetype_extension/design_decision.md','01_plan/archetype_extension/launch_gate.json'],
      'AE-BRIDGE':['04_analysis/archetype_extension/profile_bridge.py'],
      'AE-LIT-PRIOR':['02_evidence/archetype_extension/nearest_work_matrix.md'],
      'AE-V1-PROTOTYPES':['05_simulations/archetype_campaign_v1/pre_run/prototype_manifest.csv'],
      'AE-V1-PROFILES':['05_simulations/archetype_campaign_v1/pre_run/profile_manifest.json'],
      'AE-V1-SELECT':['05_simulations/archetype_campaign_v1/pre_run/selection_manifest.json'],
      'AE-V2-COMPONENTS':['05_simulations/archetype_campaign_v2/pre_run/component_manifest.csv','05_simulations/archetype_campaign_v2/pre_run/component_fit_details.json'],
      'AE-V2-MIXTURE':['05_simulations/archetype_campaign_v2/pre_run/mixture_probabilities.json'],
      'AUTHOR-AE-PRERUN':['06_claim_drafts/author_archetype_prerun_check_v2.json'],
      'CONF-BIB':['00_context/conference_complete_text.txt','02_evidence/archetype_extension/conference_reference_crosswalk.md','02_evidence/archetype_extension/conference_reference_crosswalk.json'],
      'AE-CAMPAIGN':['05_simulations/archetype_campaign_v2/campaign_completion_manifest.json','04_analysis/archetype_extension/results_v2/execution_inventory.csv','04_analysis/archetype_extension/run_mixture_campaign.py','04_analysis/archetype_extension/analyze_mixtures.py'],
      'AE-METRICS':['04_analysis/archetype_extension/results_v2/weighted_metrics.csv','04_analysis/archetype_extension/results_v2/individual_deviations.csv','04_analysis/archetype_extension/results_v2/mixture_predictions.csv','04_analysis/archetype_extension/results_v2/independent_endpoints.csv','04_analysis/archetype_extension/results_v2/analysis_manifest.json','04_analysis/archetype_extension/analyze_mixtures.py'],
      'AUTHOR-AE-RESULT':['06_claim_drafts/author_archetype_result_check_v2.json']}
    for sid in ['AE-SITE-GATES','AE-RUNS']:
        manual[sid]=[]
        for site in ['Padova','Bolzano','Roma','Palermo']:
            run=ROOT/'05_simulations/archetype_campaign_v2'/site
            names=['site_gate.json','completion_manifest.json','full_step_balance_checks.json','independent_surface_forcing_checks.json','profile_preservation_checks.json'] if sid=='AE-SITE-GATES' else ['summary.json','summary.csv','frozen_manifest.json','design.json','role_manifest.csv','profile_manifest.json','runtime_source_manifest.json']
            manual[sid]+=[str(run/n) for n in names]
            if sid=='AE-RUNS':manual[sid]+=[str(p) for p in sorted(run.glob('*_hourly.csv'))]
    boundaries={
      'ARCH2025-FULL':'Published demographic groups, declared heating hours and survey-appliance estimates; not measured thermal demand. Complete 15-page PDF previously read and all rendered pages inspected. Remote UserArchetypes branch identity to local archive remains unverified; web cache miss/403 is not evidence of absence.',
      'ARCH-ASSIGN':'Archived article-consistent local four-group mapping; exact remote published-branch identity unproved. Segment 0/1/2/3 maps to group 1/2/3/4. Direct unique-ID linkage avoids polynomial probabilities and synthetic dwelling assignment. Not new clustering or demographic causal evidence.',
      'AE-PROTOCOL':'Historical input-only v1 singleton-medoid design, superseded by pre-output v2 mixture amendment; zero v1 annual executions.',
      'AE-V2-PROTOCOL':'Pre-output assumptions and declared estimator. Shared four physical components; conditional 12 versus pooled 3 free probabilities; endpoint mixtures rather than average-temperature simulation; empirical eligible-domain weights, not national expansion weights.'}
    entries=[]
    for sid,cs in sorted(by.items()):
        locators=list(dict.fromkeys(l for c in cs for l in c.get('locators',[]) if l.startswith(sid+' |')))
        entry={'id':sid,'supported_claim_ids':[c['id'] for c in cs],'claim_declared_locators':locators,'access_date':DATE,'creator':'unknown','institution':'unknown','artifacts':[],'metadata_origin':None,'evidence_boundary':boundaries.get(sid,'Source scope and limitations are retained in the cited artifact/catalog. Simulation consistency does not establish measured building accuracy. Review approval is not physical evidence.'),'permission_boundary':'Read-only research access. Registry does not authorize public redistribution of survey rows or unpublished artifacts; respect each primary source licence. No external changes or contacts.'}
        if sid in old:
            o=old[sid];entry.update({'source_kind':o.get('source_kind'),'access_depth':'Prior-delivery content audit retained; current cited artifact identities rechecked where local files are registered.','creator':o.get('creator','unknown'),'institution':o.get('institution','unknown'),'metadata_origin':str(priorpath),'primary_metadata':o,'evidence_boundary':o.get('evidence_boundary',entry['evidence_boundary'])})
            entry['artifacts']=[file(p['path']) for p in o.get('artifacts',[]) if p.get('path')]
            if not entry['artifacts'] and o.get('url'):entry['artifacts']=[file(priorpath)]
            entry['prior_delivery_record_id']=sid
        elif sid in external:
            e=external[sid];entry.update({'source_kind':e.get('type','primary literature'),'access_depth':e.get('access_depth','unknown'),'creator':e.get('authors','unknown'),'institution':'unknown','metadata_origin':str(litpath),'primary_metadata':e,'evidence_boundary':e.get('limitations',entry['evidence_boundary'])});entry['artifacts']=[file(litpath)]
        elif sid in conference:
            e=conference[sid];entry.update({'source_kind':'published primary conference chapter','access_depth':e['access_depth'],'creator':e['authors'],'primary_metadata':e,'metadata_origin':str(confpath)});entry['artifacts']=[file(e['local_path']),file(confpath)]
        else:
            entry.update({'source_kind':'study-local evidence artifact','access_depth':'Direct exact target inspection; byte hashes and selected content/producer chain checked'})
        candidates=list(manual.get(sid,[]))
        for l in locators:
            for m in re.finditer(r'(?:00_context|01_plan|02_evidence|03_methods|04_analysis|05_simulations|06_claim_drafts|07_reviews)/[\w./\-]+\.(?:json|csv|md|py|txt|epw)',l): candidates.append(m.group())
        for p in candidates:
            q=file(p)
            if q not in entry['artifacts']:entry['artifacts'].append(q)
        if sid=='ARCH2025-FULL':entry.update({'source_kind':'published primary full article','primary_metadata':load(ROOT/'00_context/archetype_extension/primary_sources_and_local_inventory.json')['primary'],'creator':['Mohamad Hasan Khajedehi','Jacopo Vivian','Angelo Zarrella'],'institution':'University of Padova; author affiliations in primary PDF'})
        if sid.startswith('AE-V1') or sid=='AE-PROTOCOL':entry['evidence_class']='historical input-only diagnostic or superseded design; no annual outputs'
        elif sid in ['AE-RUNS','AE-CAMPAIGN','AE-SITE-GATES']:entry['evidence_class']='simulation outputs and derived numerical consistency checks; not measured observations'
        elif sid=='AE-METRICS' or sid.startswith('AUTHOR-') or sid.endswith('AUDIT'):entry['evidence_class']='derived audit or arithmetic estimate'
        elif sid in ['AE-BRIDGE','AE-DESIGN','AE-V2-PROTOCOL','AE-V2-COMPONENTS','AE-V2-MIXTURE']:entry['evidence_class']='executable transformation, declared assumption or fitted input representation; no measured temperatures'
        elif sid in ['ARCH-ASSIGN']:entry['evidence_class']='archived derived household assignment'
        elif sid in external or sid in conference or sid=='ARCH2025-FULL':entry['evidence_class']='published primary source at declared access depth'
        else:entry['evidence_class']=entry.get('primary_metadata',{}).get('evidence_class','see source kind and artifact-specific boundary')
        if not entry['artifacts']:unresolved.append({'alias':sid,'reason':'No exact local target or catalog record resolved','locators':locators})
        entries.append(entry)
    # Complete v2 chain inventory, independent of alias granularity. Historical v1 stays input-only.
    campaign=ROOT/'05_simulations/archetype_campaign_v2';pre=campaign/'pre_run';result=ROOT/'04_analysis/archetype_extension/results_v2'
    for base in [campaign,result,ROOT/'07_reviews/archetype_extension/pre_run_audit']:
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:file(p,depth='Exact byte hash; manifest/schema and selected scientific content inspected; not every row semantically re-evaluated')
    for p in (ROOT/'03_methods/archetype_extension').glob('*'):
        if p.is_file():file(p,stability='mutable current method/report summary; distinct from immutable per-run protocol copies')
    for p in (ROOT/'04_analysis/archetype_extension').glob('*.py'):file(p)
    for p in [priorpath,litpath,confpath,ledger,draft,ROOT/'01_plan/archetype_extension/launch_gate.json']:file(p)
    # Verify exact named producer targets; never guess JSON for a CSV summary hash.
    sel=load(pre/'selection_manifest.json')
    for name,h in sel['output_sha256'].items():verify(pre/'selection_manifest.json','output_sha256.'+name,pre/name,h)
    runtime=load(pre/'runtime_source_manifest.json')
    for p,h in runtime['sha256'].items():verify(pre/'runtime_source_manifest.json','sha256.'+p,p,h)
    launch=load(ROOT/'01_plan/archetype_extension/launch_gate.json')
    for p,h in launch['frozen_sha256'].items():verify('01_plan/archetype_extension/launch_gate.json','frozen_sha256.'+p,p,h)
    analysis=load(result/'analysis_manifest.json')
    for p,h in analysis['output_sha256'].items():verify(result/'analysis_manifest.json','output_sha256.'+p,result/p,h)
    verify(result/'analysis_manifest.json','selection_manifest_sha256',pre/'selection_manifest.json',analysis['selection_manifest_sha256'])
    verify(result/'analysis_manifest.json','probabilities_sha256',pre/'mixture_probabilities.json',analysis['probabilities_sha256'])
    verify(result/'analysis_manifest.json','analysis_script_sha256','04_analysis/archetype_extension/analyze_mixtures.py',analysis['analysis_script_sha256'])
    completion=load(campaign/'campaign_completion_manifest.json')
    for s in completion['source_records']:verify(campaign/'campaign_completion_manifest.json','source_records.'+s['path'],s['path'],s['sha256'])
    for field,target in [('analysis_manifest_sha256',result/'analysis_manifest.json'),('selection_manifest_sha256',pre/'selection_manifest.json'),('runtime_manifest_sha256',pre/'runtime_source_manifest.json'),('inventory_sha256',result/'execution_inventory.csv')]:verify(campaign/'campaign_completion_manifest.json',field,target,completion[field])
    for site in ['Padova','Bolzano','Roma','Palermo']:
        run=campaign/site;c=load(run/'completion_manifest.json')
        verify(run/'completion_manifest.json','summary_sha256 (producer writes summary.csv)',run/'summary.csv',c['summary_sha256'])
        verify(run/'completion_manifest.json','script_sha256','04_analysis/archetype_extension/run_mixture_campaign.py',c['script_sha256'])
        for row in load(run/'summary.json'):verify(run/'summary.json',row['case']+'.hourly_sha256',run/(row['case']+'_hourly.csv'),row['hourly_sha256'])
        for key in ['selection_manifest.json','role_manifest.csv','profile_manifest.json','mixture_probabilities.json','component_manifest.csv','runtime_source_manifest.json','selection_protocol.md']:verify(run/'frozen_manifest.json','pre_run copy identity '+key,run/key,digest(pre/key))
        fields={'config_sha256':'eureca_config.json','design_sha256':'design.json','protocol_sha256':'simulation_protocol.md','protocol_v2_sha256':'simulation_protocol_v2_addendum.md','correction_protocol_sha256':'correction_protocol_v3.md','correction_module_sha256':'single_commit_1c.py','transition_gate_sha256':'transition_checks_pre_run.json','surface_preprocessing_sha256':'surface_forcing_1c.py','surface_protocol_sha256':'surface_correction_protocol_v5.md','fixture_sha256':'surface_fixture_checks_pre_run.json','weather_correction_sha256':'ir_sky_forcing.py','weather_protocol_sha256':'weather_correction_protocol_v4.md','forcing_audit_sha256':'weather_forcing_audit_pre_run.json','selection_manifest_sha256':'selection_manifest.json','archetype_protocol_sha256':'selection_protocol.md','profile_bridge_sha256':'profile_bridge.py','role_manifest_sha256':'role_manifest.csv'}
        for field,name in fields.items():verify(run/'completion_manifest.json',field,run/name,c[field])
        for p,h in c['package_source_sha256'].items():verify(run/'completion_manifest.json','package_source_sha256.'+p,p,h)
        gate=load(run/'site_gate.json')
        for field,target in [('launch_gate_sha256',ROOT/'01_plan/archetype_extension/launch_gate.json'),('selection_manifest_sha256',pre/'selection_manifest.json'),('frozen_manifest_sha256',run/'frozen_manifest.json'),('checker_sha256',run/'site_gate_checker_snapshot.py')]:verify(run/'site_gate.json',field,target,gate[field])
    if load(ledger).get('draft_sha256'):verify(ledger,'draft_sha256',draft,load(ledger)['draft_sha256'])
    if args.frozen:
        evidence_index=ROOT/'06_claim_drafts/claim_evidence_ledger_v05.json'
        file(evidence_index)
        verify(evidence_index,'draft_sha256',draft,index['draft_sha256'])
        verify(evidence_index,'claim_ledger_sha256',ledger,index['claim_ledger_sha256'])
    # Observed raw responses versus preprocessing sentinels/source-derived controls.
    rawpath=CWD/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt';compact=CWD/'istat_microdata_2013.csv'
    for p in [rawpath,compact]:file(p,depth='Prior complete raw/compact eligibility audit retained; current exact byte hash; this registry does not rerun all survey mappings')
    schema=load(ROOT/'00_context/archetype_extension/assignment_mapping_schema.json')
    for field,target in [('assignment_sha256',schema['assignment_artifact']),('input_sha256',schema['input_path'])]:verify('00_context/archetype_extension/assignment_mapping_schema.json',field,target,schema[field])
    verify(pre/'selection_manifest.json','input_sha256.compact_survey',compact,sel['input_sha256']['compact_survey'])
    verify(pre/'selection_manifest.json','input_sha256.EndUse.py',CWD.parents[1]/'classes/EndUse.py',sel['input_sha256']['EndUse.py'])
    git=subprocess.run(['git','show',schema['git_revision']+':'+schema['git_path']],cwd=schema['git_repository'],capture_output=True,check=True)
    gh=hashlib.sha256(git.stdout).hexdigest()
    identity_checks.append({'kind':'Git blob exact bytes','source':schema['git_repository']+'@'+schema['git_revision']+':'+schema['git_path'],'expected_sha256':schema['git_blob_sha256'],'actual_sha256':gh,'match':gh==schema['git_blob_sha256'],'text_equal_after_CRLF_to_LF':Path(schema['input_path']).read_bytes().replace(b'\r\n',b'\n')==git.stdout.replace(b'\r\n',b'\n')})
    for pid,p in load(pre/'profile_manifest.json').items():
        with (pre/(pid+'.csv')).open(newline='') as f:values=[float(row['lower_C']) for row in csv.DictReader(f)]
        h=hashlib.sha256(b''.join(struct.pack('<d',v) for v in values)).hexdigest()
        identity_checks.append({'kind':'little-endian float64 full annual temperature-array identity','source':str(pre/(pid+'.csv')),'rows':len(values),'expected_sha256':p['profile_sha256'],'actual_sha256':h,'match':h==p['profile_sha256']})
    # Supplemental upstream and technical records do not pretend to be author-declared aliases.
    extra=[
      ('AE-UPSTREAM-SURVEY','official raw input / preprocessed compact input',[rawpath,compact,CWD/'istat/ECH_2013_IT/METADATI/ECH_Questionario_Anno 2013.pdf'],['DAT-007','MET-012'],'Raw public-use responses are observations of declared survey quantities. Compact blank-to-zero conversions are preprocessing sentinels, not affirmative answers or measured controls. Use only declared mean-winter operating bands, not exact presence/burner duty.'),
      ('AE-UPSTREAM-ARCHIVE','archived derived assignment',[CWD/'clusters.csv',ROOT/'00_context/archetype_extension/assignment_mapping_schema.json',ROOT/'00_context/archetype_extension/local_assignment_audit.json'],['DAT-006','LIM-009'],'Local byte and Git blob identities recorded in the original audit; exact equality with inaccessible published branch unverified.'),
      ('AE-V2-FROZEN-INPUTS','immutable pre-run input/assumption/role/probability/weight artifacts',list(pre.glob('*')),['MET-013','MET-014','MET-015'],'Profile identities are arrays; actual serialized CSV hashes live in output_sha256. Evaluation weights target the eligible unweighted empirical domain; four component endpoints are mixed after separate simulation.'),
      ('AE-V2-RUNTIME-SOURCES','executable code and acquired weather',list(map(Path,runtime['sha256'].keys())),['MET-012','MET-014','RES-007'],'Exact byte versions executed, source-derived control temperatures and fixed gains remain modelling assumptions. No empirical measurement or complete external model benchmark is inferred.'),
      ('AE-V2-TECHNICAL-AUDITS','technical input/fixture audit code and derived results',list((ROOT/'07_reviews/archetype_extension/pre_run_audit').glob('*')),['MET-013','MET-014','MET-015'],'Technical scripts/checks support stated input/freeze/arithmetic checks; review narratives and approval are not physical truth.')]
    for sid,kind,paths,cids,boundary in extra:
        entries.append({'id':sid,'supplemental_upstream_record':True,'source_kind':kind,'evidence_class':kind,'creator':'Istat' if sid=='AE-UPSTREAM-SURVEY' else 'unknown','institution':'Istituto nazionale di statistica' if sid=='AE-UPSTREAM-SURVEY' else 'unknown','access_date':DATE,'access_depth':'Current byte hashes and direct selected schema/content checks; upstream complete questionnaire audit retained','supported_claim_ids':[x for x in cids if x in {c['id'] for c in claims}],'claim_declared_locators':[],'artifacts':[file(p) for p in paths if Path(p).is_file()],'evidence_boundary':boundary,'permission_boundary':'Read-only local research; no redistribution permission inferred.'})
    # Snapshot exact registry predecessors before any replacement.
    history=[]
    for name in ['local_sources.json','local_source_ledger.md','registry_audit_v05.json','registry_audit_v05.md']:
        p=HERE/name
        if p.exists():
            target=HERE/'registry_history'/(p.stem+'_'+digest(p)[:12]+p.suffix);target.parent.mkdir(exist_ok=True)
            if not target.exists():shutil.copy2(p,target)
            history.append({'path':str(target),'sha256':digest(target)})
    missing=[p for p,x in files.items() if not x['exists']];mismatch=[x for x in checks+identity_checks if not x['match']]
    registry={'schema_version':'1.0','registry_version':'archetype-extension-v05','status':'frozen author claim binding' if args.frozen else 'preparation against mutable author preview; no final claim freeze implied','access_date':DATE,'requested_model':'gpt-6.1-sol','confirmed_model':'unknown','based_on_claim_ledger':{'path':str(ledger),'sha256':digest(ledger),'claim_count':len(claims),'draft_path':str(draft),'draft_sha256':digest(draft)},'prior_delivery_registry':{'path':str(priorpath),'sha256':digest(priorpath),'preserved':True},'literature_catalog':{'path':str(litpath),'sha256':digest(litpath),'source_count':len(lit['sources']),'access_qualification':'Catalog retains source-specific access depths; 48 registered records does not mean 48 complete full texts read.'},'history':history,'sources':entries,'artifact_inventory':list(files.values()),'producer_target_hash_checks':checks,'array_and_git_identity_checks':identity_checks,'hash_semantics':{'summary_sha256':'Site completion manifests hash summary.csv, while summary.json is separately inventoried.','profile_sha256':'Thermal-profile ndarray identity, not serialized CSV bytes. selection_manifest.output_sha256 hashes actual profile CSV bytes.','review_evidence':'Technical scripts/check outputs can support arithmetic/input/solver checks; verdict narratives are not measured physical truth.','historical_runs':'Root prior delivery and quarantined runs remain historical. Archetype v1 singleton collapse and preflight failure.txt had zero annual executions. Only v2 has 152 completed annual executions.','current_summaries':'Current method/results reports may change with procedural status; immutable pre_run/per-site snapshots preserve executed inputs.'},'counts':{'claim_count':len(claims),'alias_count':len(by),'registered_alias_count':len(entries),'file_count':len(files),'producer_target_hash_check_count':len(checks),'array_and_git_identity_check_count':len(identity_checks),'hash_mismatch_count':len(mismatch),'missing_file_count':len(missing),'unresolved_alias_count':len(unresolved)},'unresolved_aliases':unresolved,'missing_files':missing,'hash_mismatches':mismatch,'limits':['Source existence/hash integrity is not independent validation of every scientific claim.','Published archive identity to inaccessible remote UserArchetypes branch remains unverified; aggregate agreement is not proof of identical household membership.','Detailed individual outputs are model references, not measured energy; fixed gains and source-control temperatures remain assumptions.']}
    if args.frozen:registry['frozen_author_bundle']=[{'path':str(ROOT/'06_claim_drafts'/name),'sha256':h,'identity_checked':True} for name,h in expected.items()]
    (HERE/'local_sources.json').write_text(json.dumps(registry,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# Archetype extension supplemental local source ledger','',registry['status']+'. Access date: '+DATE+'.','',f"Claim ledger `{ledger}` SHA256 `{digest(ledger)}`. Draft `{draft}` SHA256 `{digest(draft)}`.",'','The prior root registry is preserved and referenced. Source-specific literature access qualifications are retained; catalog inclusion does not imply full-text reading. Study artifacts separate raw observations, preprocessing, assumptions, executable transformations and within-model outputs.','', '| Alias | Kind / metadata origin | Formal claim IDs | Exact artifacts / locators |','|---|---|---|---|']
    for e in entries:lines.append('| '+e['id']+' | '+str(e['source_kind'])+' | '+', '.join(e['supported_claim_ids'])+' | '+'; '.join(e['artifacts']).replace('|','\\|')+' |')
    lines+=['','All exact artifact hashes, declared claim locators, access depths, evidence limits and producer-target comparisons are in `local_sources.json`.','',json.dumps(registry['counts'],ensure_ascii=False),'','Site `summary_sha256` targets CSV; thermal `profile_sha256` identifies the array, while frozen output maps identify CSV bytes. V1/preflight diagnostics are input-only. Mutable current summaries are distinct from immutable per-run snapshots. Review verdicts do not establish physical accuracy.','', 'Unresolved aliases: '+json.dumps(unresolved,ensure_ascii=False), 'Missing files: '+json.dumps(missing,ensure_ascii=False)]
    (HERE/'local_source_ledger.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit={'access_date':DATE,'status':registry['status'],'counts':registry['counts'],'registry_sha256':digest(HERE/'local_sources.json'),'ledger_sha256':digest(HERE/'local_source_ledger.md'),'script_sha256':digest(__file__),'claim_ledger':registry['based_on_claim_ledger'],'unresolved_aliases':unresolved,'missing_files':missing,'hash_mismatches':mismatch}
    (HERE/'registry_audit_v05.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (HERE/'registry_audit_v05.md').write_text('# Archetype extension registry audit\n\n'+registry['status']+'.\n\n'+json.dumps(registry['counts'],indent=2)+'\n\nRegistry SHA256 `'+audit['registry_sha256']+'`. Ledger SHA256 `'+audit['ledger_sha256']+'`.\n\nExact byte identity and producer-target mapping checked; this audit does not declare scientific review closure. Unresolved primary provenance: exact local archive identity to published remote branch remains unverified. No physical result is treated as an observation.\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
