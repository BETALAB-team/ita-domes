from pathlib import Path
import copy
import json
import math
import sys


def repository_root():
    start = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    for p in [start, *start.parents]:
        if (p / 'revised_paper_outputs').is_dir():
            return p
    return start


def default_config():
    root = repository_root()
    climates = []
    for city, station, region in [('Milano', 'Milano-Linate.160800', 'North-west'),
                                  ('Roma', 'Roma-Ciampino.162390', 'Centre'),
                                  ('Palermo', 'Palermo-Boccadifalco.164000', 'South/islands')]:
        name = f'ITA_{station}_IGDG'
        climates.append({'name': city, 'location': region, 'enabled': True,
                         'epw': str(root / 'energy_studio' / 'weather' / f'{city}.epw'),
                         'source': f'https://energyplus-weather.s3.amazonaws.com/europe_wmo_region_6/ITA/{name}/{name}.epw'})
    return {
        'schema_version': 3,
        'benchmark': {'mode':'shared_istat', 'profiles_file':'', 'anchor_profile':'P2',
                      'compare_istat':True,
                      'demographic_model':str(root/'revised_paper_outputs/reference_model.json')},
        'hybrid': {'method':'climate_exposure_gains', 'index_source':'group_means',
                   'heating_balance_min_c':-5., 'heating_balance_max_c':30.,
                   'cooling_balance_min_c':5., 'cooling_balance_max_c':45.,
                   'balance_step_c':.25, 'solar_coefficient_max':1.5, 'solar_step':.1,
                   'minimum_active_days':20, 'adjust_exposure':True,
                   'adjust_operating_hours':True, 'adjust_internal_gains':True, 'adjust_solar':True},
        'report': {'powerpoint': True,
                   'node': str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'),
                   'node_modules': str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'),
                   'python': str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'),
                   'skill_dir': str(Path.home()/'.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations')},
        'paths': {'households': str(root/'revised_paper_outputs/2021/households.csv'),
                  'output_root': str(root/'energy_studio_runs'),
                  'energyplus': 'C:/EnergyPlusV25-1-0/energyplus.exe', 'reference_run':''},
        'analysis': {'archetype_mode': 'existing', 'n_archetypes': 4,
                     'features': ['agemin', 'agemax', 'nmembers', 'education'],
                     'pca_components': 0, 'kmeans_starts': 4, 'seed': 42,
                     'group_factors': ['location', 'dwelling_type', 'archetype'],
                     'electricity_include_cooling': True,
                     'heating_groups': 4, 'electricity_groups': 5,
                     'group_selection': 'fixed', 'elbow_max_k': 12,
                     'minimum_train_cell': 30},
        'simulation': {'enabled': True, 'run_full_benchmark': False, 'workers': 2,
                       'timeout_seconds': 180, 'timestep': 4,
                       'area_m2': 100., 'aspect_ratio': 1., 'height_m': 2.7,
                       'north_axis_deg': 0., 'window_wall_ratio': .2,
                       'wall_u': .6, 'roof_u': .4, 'floor_u': .5, 'window_u': 2.4,
                       'window_shgc': .6, 'infiltration_ach': .4,
                       'ground_temperature_c': 18., 'heat_setpoint_c': 20.,
                       'heat_setback_c': 16., 'cool_setpoint_c': 26., 'cool_setback_c': 30.,
                       'cooling_enabled': True, 'default_cooling_hours': 6.,
                       'people_sensible_w': 75., 'appliance_heat_fraction': 1.,
                       'anchor_climate': 'Milano', 'anchor_context': 'Multi-family', 'anchor_archetype': 2,
                       'occupancy_profile': [.95,.95,.95,.95,.95,.95,.85,.7,.4,.35,.35,.45,.55,.5,.4,.4,.55,.75,.9,.95,.95,.95,.95,.95],
                       'appliance_profile': [.3,.25,.25,.25,.25,.35,.7,1.,.6,.5,.5,.8,1.,.7,.6,.6,.8,1.2,1.5,1.6,1.5,1.2,.8,.5],
                       'archetype_overrides': {}},
        'contexts': [
            {'name': 'Apartment', 'enabled': True, 'exposed_walls': ['South','North'], 'roof': 'Adiabatic', 'floor': 'Adiabatic'},
            {'name': 'Multi-family', 'enabled': True, 'exposed_walls': ['South','North','West'], 'roof': 'Outdoors', 'floor': 'Ground'},
            {'name': 'Single-family', 'enabled': True, 'exposed_walls': ['South','East','North','West'], 'roof': 'Outdoors', 'floor': 'Ground'}],
        'climates': climates}


def validate(cfg):
    errors = []
    if cfg.get('schema_version') != 3:
        raise ValueError('This release requires schema 3 (reference energy signature). Reset defaults and reapply your settings; older configurations are not changed automatically.')
    a, s = cfg['analysis'], cfg['simulation']
    cfg.setdefault('benchmark', default_config()['benchmark'])
    b=cfg['benchmark']
    if b['mode'] not in ['shared_istat','independent_schedules']:errors.append('Unknown benchmark mode.')
    independent=b['mode']=='independent_schedules'
    if independent:
        if not s['enabled']:errors.append('Independent benchmark requires an EnergyPlus reference.')
        if cfg['hybrid']['method']!='climate_exposure_gains':errors.append('Independent benchmark uses the climate/exposure/gains hybrid.')
        if cfg['paths'].get('reference_run'):errors.append('Clear reference reuse for independent benchmarks; each experiment freezes new inputs.')
        if b['profiles_file'] and not Path(b['profiles_file']).is_file():errors.append('Independent profiles JSON does not exist.')
        if b['compare_istat'] and a['archetype_mode']=='existing' and not Path(b['demographic_model']).is_file():errors.append('Frozen demographic model does not exist.')
    a.setdefault('electricity_include_cooling',True)
    if a['archetype_mode'] not in ['existing','refit']: errors.append('Archetype mode must be existing or refit.')
    if a['group_selection'] not in ['fixed','elbow']: errors.append('Group selection must be fixed or elbow.')
    if type(a['pca_components']) is not int or not 0<=a['pca_components']<=len(a['features']): errors.append('Invalid PCA component count.')
    if type(a['seed']) is not int or a['seed']<0: errors.append('Seed must be a nonnegative integer.')
    for key in ['n_archetypes','heating_groups','electricity_groups','elbow_max_k','minimum_train_cell','kmeans_starts']:
        if type(a[key]) is not int or a[key] < 1: errors.append(f'{key} must be a positive integer.')
    if not a['features'] or not a['group_factors']: errors.append('Select at least one feature and one group factor.')
    if len(set(a['features'])) != len(a['features']) or len(set(a['group_factors'])) != len(a['group_factors']): errors.append('Duplicate feature/factor names.')
    if set(a['features'])-{'agemin','agemax','nmembers','education','area_proxy','age_proxy'}: errors.append('Unsupported demographic feature (outcomes are prohibited).')
    if set(a['group_factors'])-{'location','dwelling_type','archetype','cooling','insulation_improvement'}: errors.append('Unsupported grouping factor.')
    if cfg.get('hybrid',{}).get('index_source') not in ['group_means','cohort_means']: errors.append('Hybrid index source must be group_means or cohort_means.')
    hy=cfg['hybrid']
    if hy['method'] not in ['climate_exposure_gains','istat_ratio']:errors.append('Unknown hybrid method.')
    for use in ['heating','cooling']:
        lo,hi=hy[use+'_balance_min_c'],hy[use+'_balance_max_c']
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in [lo,hi]) or not -30<=lo<hi<=60:errors.append('Invalid balance-temperature bounds.')
    for key,lo,hi in [('balance_step_c',.05,5),('solar_step',.01,1),('solar_coefficient_max',0,3)]:
        if not isinstance(hy[key],(int,float)) or not math.isfinite(hy[key]) or not lo<=hy[key]<=hi:errors.append(f'Invalid {key}.')
    if type(hy['minimum_active_days']) is not int or not 5<=hy['minimum_active_days']<=365:errors.append('Minimum active days must be an integer from 5 to 365.')
    for key in ['adjust_exposure','adjust_operating_hours','adjust_internal_gains','adjust_solar']:
        if type(hy[key]) is not bool:errors.append(f'{key} must be true or false.')
    if cfg['paths'].get('reference_run') and not (Path(cfg['paths']['reference_run'])/'manifest.json').is_file():errors.append('Reference reuse requires a completed Energy Studio run directory.')
    for key in ['area_m2','aspect_ratio','height_m','wall_u','roof_u','floor_u','window_u','timeout_seconds']:
        if not isinstance(s[key],(int,float)) or not math.isfinite(s[key]) or s[key] <= 0: errors.append(f'{key} must be finite and positive.')
    for key in ['window_wall_ratio','window_shgc','appliance_heat_fraction']:
        if not 0 <= s[key] <= 1: errors.append(f'{key} must be between 0 and 1.')
    if s['window_wall_ratio'] > .8: errors.append('Window-wall ratio above 0.8 is not supported by this geometry.')
    if s['wall_u'] >= 5 or s['roof_u'] >= 5 or s['floor_u'] >= 5: errors.append('Opaque U-values must be below 5 W/m²K.')
    if s['infiltration_ach'] < 0: errors.append('Infiltration cannot be negative.')
    if not 0<=s['default_cooling_hours']<=24: errors.append('Default cooling hours must be in [0,24].')
    if s['people_sensible_w']<=0: errors.append('Sensible heat per person must be positive.')
    for key,value in s.items():
        if isinstance(value,float) and not math.isfinite(value): errors.append(f'{key} must be finite.')
    for name,values in s['archetype_overrides'].items():
        for key in ['heating_hours','cooling_hours']:
            if key in values and not 0<=values[key]<=24: errors.append(f'Archetype {name}: {key} must be in [0,24].')
        for key in ['members','appliance_kwh']:
            if key in values and (not math.isfinite(values[key]) or values[key]<0):errors.append(f'Archetype {name}: invalid {key}.')
    if not s['heat_setback_c'] <= s['heat_setpoint_c'] < s['cool_setpoint_c'] <= s['cool_setback_c']: errors.append('Setpoints must satisfy setback heat ≤ heat < cool ≤ setback cool.')
    if s['timestep'] not in [1,2,3,4,5,6,10,12,15,20,30,60]: errors.append('Invalid EnergyPlus timesteps per hour.')
    if not 1 <= s['workers'] <= 8: errors.append('Use 1–8 workers.')
    for key in ['occupancy_profile','appliance_profile']:
        arr=s[key]
        if len(arr)!=24 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in arr) or sum(arr)<=0: errors.append(f'{key} must have 24 nonnegative values, not all zero.')
    if max(s['occupancy_profile'])>1: errors.append('Occupancy fractions cannot exceed 1.')
    active=[x for x in cfg['contexts'] if x.get('enabled')]
    for c in active:
        if c['name'] not in ['Apartment','Multi-family','Single-family']: errors.append('Context name must match a survey dwelling type.')
        if not set(c['exposed_walls']) <= {'North','East','South','West'}: errors.append('Unknown exposed wall.')
        if c['roof'] not in ['Outdoors','Adiabatic'] or c['floor'] not in ['Ground','Adiabatic']: errors.append('Unsupported roof/floor boundary.')
    if not active or not any(c.get('enabled') for c in cfg['climates']): errors.append('Enable at least one climate and context.')
    if len({x['name'] for x in cfg['climates']})!=len(cfg['climates']): errors.append('Climate names must be unique.')
    if len({x['name'] for x in cfg['contexts']})!=len(cfg['contexts']): errors.append('Context names must be unique.')
    if not any(x['name']==s['anchor_climate'] and x.get('enabled') for x in cfg['climates']): errors.append('Reference climate must be enabled.')
    if not any(x['name']==s['anchor_context'] and x.get('enabled') for x in cfg['contexts']): errors.append('Reference context must be enabled.')
    if not 1<=s['anchor_archetype']<=a['n_archetypes']: errors.append('Reference archetype must exist.')
    if (not independent or b['compare_istat']) and not Path(cfg['paths']['households']).is_file(): errors.append('Household CSV does not exist.')
    if s['enabled'] and not Path(cfg['paths']['energyplus']).is_file(): errors.append('EnergyPlus executable does not exist.')
    for climate in cfg['climates']:
        required=s['enabled'] and climate.get('enabled') and (s['run_full_benchmark'] or hy['method']=='climate_exposure_gains' or climate['name']==s['anchor_climate'])
        if required and not Path(climate['epw']).is_file(): errors.append(f"Missing weather file: {climate['epw']}")
    if cfg.get('report',{}).get('powerpoint'):
        rt=cfg['report']
        for key in ['node','python']:
            if not Path(rt.get(key,'')).is_file():errors.append(f'Missing PowerPoint runtime: {key}. Disable PowerPoint or update report paths.')
        for path in [Path(rt.get('node_modules',''))/'@oai/artifact-tool/dist/artifact_tool.mjs',Path(rt.get('skill_dir',''))/'container_tools/artifact_tool_utils.mjs']:
            if not path.is_file():errors.append(f'Missing PowerPoint dependency: {path}')
    if errors: raise ValueError('\n'.join(errors))
    return cfg


def load(path):
    return validate(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def save(cfg,path):
    Path(path).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
