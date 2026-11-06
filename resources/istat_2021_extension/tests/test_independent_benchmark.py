import copy
import json
import tempfile
import threading
import unittest
from datetime import date,timedelta
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from energy_studio.config import default_config,validate
from energy_studio.independent_inputs import default_profiles,load_profiles,build_profile,classify_demographics
from energy_studio.independent import summary_profile,run
from energy_studio.energyplus import make_model
from energy_studio.independent_report import errors_and_metrics


class IndependentTests(unittest.TestCase):
    def setUp(self):
        self.cfg=default_config();self.cfg['benchmark'].update(mode='independent_schedules',compare_istat=False)
        self.cfg['report']['powerpoint']=False

    def test_no_istat_inputs_required_and_presets_fixed(self):
        self.cfg['paths']['households']='does_not_exist.csv';validate(self.cfg)
        doc,p=load_profiles(self.cfg)
        changed=copy.deepcopy(self.cfg);changed['analysis']['n_archetypes']=20
        changed['simulation']['archetype_overrides']={'2':{'appliance_kwh':99999}}
        self.assertEqual(load_profiles(changed), (doc,p))
        for key in ['members','heating_hours','cooling_hours','appliance_kwh','presence_index']:
            self.assertAlmostEqual(p['P2'][key],p['P2_shift'][key])
        self.assertNotEqual(p['P2']['annual_schedules'],p['P2_shift']['annual_schedules'])

    def test_aggregate_boundary_excludes_target_hourly(self):
        _,p=load_profiles(self.cfg);clean=summary_profile(p['P2'])
        self.assertNotIn('annual_schedules',clean);self.assertNotIn('occupancy_profile',clean)
        self.assertEqual(clean['members_index'],4)

    def test_ep_schedule_roundtrip_all_8760_hours(self):
        _,p=load_profiles(self.cfg);model=make_model(self.cfg,self.cfg['contexts'][0],p['P2'],{'latitude':45.,'longitude':9.,'timezone':1.,'elevation':100.})
        for name,key in [('Heating','heat'),('Cooling','cool'),('Occupancy','occupancy'),('Appliances','appliance_w')]:
            result=[]
            for segment in model['Schedule:Year'][name]['schedule_weeks']:
                first=date(2001,segment['start_month'],segment['start_day']);last=date(2001,segment['end_month'],segment['end_day'])
                week=model['Schedule:Week:Daily'][segment['schedule_week_name']]
                for d in range((last-first).days+1):
                    day=first+timedelta(days=d);dn=week[day.strftime('%A').lower()+'_schedule_day_name']
                    values=model['Schedule:Day:Hourly'][dn];result.extend(values[f'hour_{h}'] for h in range(1,25))
            np.testing.assert_allclose(result,p['P2']['annual_schedules'][key])

    def test_frozen_assignment_matches_existing_labels(self):
        root=Path(self.cfg['paths']['households'])
        frame=pd.read_csv(root);model=json.loads(Path(self.cfg['benchmark']['demographic_model']).read_text())
        label,_=classify_demographics(frame.to_dict('records'),model)
        np.testing.assert_array_equal(label,frame.archetype)
        doc=default_profiles();before=classify_demographics([p['demographics'] for p in doc['profiles']],model)[0]
        doc['profiles'][1]['weekday']['appliance_w']=[99999]*24
        after=classify_demographics([p['demographics'] for p in doc['profiles']],model)[0]
        np.testing.assert_array_equal(before,after)

    def test_import_validation(self):
        p=default_profiles()['profiles'][0];p['weekday']['heating_on_hours']=[1,1]
        with self.assertRaises(ValueError):build_profile(p,self.cfg)
        p=default_profiles()['profiles'][0];p['weekday']['occupancy'][0]=2
        with self.assertRaises(ValueError):build_profile(p,self.cfg)
        with tempfile.TemporaryDirectory() as temp:
            p={'members':1,'hourly_csv':'input.csv'}
            pd.DataFrame({'hour':[1,3]}).to_csv(Path(temp)/'input.csv',index=False)
            with self.assertRaisesRegex(ValueError,'1..8760'):build_profile(p,self.cfg,Path(temp))

    def test_appliance_identity_not_counted_as_validation(self):
        rows=[]
        for method in ['energyplus','known_summary','istat_profile']:
            rows.append({'method':method,'scenario_id':'x','is_anchor':False,'climate':'M','dwelling':'A','profile_id':'P',
                         'heating_kwh':100,'cooling_kwh':100,'appliance_kwh':100})
        _,m=errors_and_metrics(pd.DataFrame(rows))
        self.assertEqual(len(m),5)
        self.assertFalse(((m.method=='known_summary')&(m.end_use=='appliance_kwh')).any())

    def test_frozen_before_targets_and_target_outputs_cannot_change_predictions(self):
        cfg=copy.deepcopy(self.cfg);cfg['simulation']['run_full_benchmark']=True
        for x in cfg['climates'][1:]:x['enabled']=False
        for x in cfg['contexts']:x['enabled']=x['name']=='Multi-family'
        cfg['paths']['households']='missing.csv';cfg['hybrid'].update(balance_step_c=2.,solar_step=.5)
        with tempfile.TemporaryDirectory() as temp:
            cfg['paths']['output_root']=temp;frozen=[]
            for multiplier in [1,100]:
                def fake(model,epw,exe,folder,timeout,cancel):
                    root=folder.parent.parent
                    if folder.name!='I002':
                        self.assertTrue((root/'predictions_frozen.csv').is_file())
                    # Appliance totals reproduced from model schedule inputs.
                    annual=0.
                    for seg in model['Schedule:Year']['Appliances']['schedule_weeks']:
                        a=date(2001,seg['start_month'],seg['start_day']);b=date(2001,seg['end_month'],seg['end_day']);w=model['Schedule:Week:Daily'][seg['schedule_week_name']]
                        for d in range((b-a).days+1):
                            dn=w[(a+timedelta(days=d)).strftime('%A').lower()+'_schedule_day_name'];annual+=sum(model['Schedule:Day:Hourly'][dn][f'hour_{h}'] for h in range(1,25))/1000
                    factor=1 if folder.name=='I002' else multiplier
                    return pd.DataFrame({'heating_kwh':np.tile(np.linspace(.1,2,365)*factor,24).reshape(24,365).T.ravel(),
                                         'cooling_kwh':np.ones(8760)*.5*factor,'appliance_kwh':np.full(8760,annual/8760)}),.1,0
                with patch('energy_studio.analysis.prepare',side_effect=AssertionError('ISTAT accessed')), patch('energy_studio.independent.run_model',side_effect=fake),patch('energy_studio.independent_report.make_report'):
                    out=run(cfg,lambda _:None,threading.Event())
                frozen.append(pd.read_csv(out/'predictions_frozen.csv'))
            pd.testing.assert_frame_equal(*frozen)


if __name__=='__main__':unittest.main()
