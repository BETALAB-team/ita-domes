import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from energy_studio.config import default_config, validate
from energy_studio.transfer import transfer_one
from energy_studio.analysis import scenario_profile
from energy_studio import engine
from energy_studio.plots import aggregate_accuracy


class TransferTests(unittest.TestCase):
    def test_aggregation_distinguishes_cancellation(self):
        frame=pd.DataFrame([{'method':'hybrid_istat','end_use':'heating_kwh','climate':'Palermo','dwelling':'Apartment',
                             'archetype':i,'benchmark':100.,'estimate':100.+err,'error':err} for i,err in [(1,50),(2,-50)]])
        g=aggregate_accuracy(frame).query('aggregation == "all"').iloc[0]
        self.assertEqual(g.pooled_signed_error_pct,0)
        self.assertEqual(g.case_wape_pct,50)

    def test_ratio_identity_and_edge_cases(self):
        self.assertEqual(transfer_one(1000,5,8),(1600,1.6,'estimated'))
        self.assertEqual(transfer_one(1000,5,5,True),(1000,1,'anchor_identity'))
        for args in [(1000,0,4),(1000,5,math.nan),(0,5,8)]:
            self.assertTrue(math.isnan(transfer_one(*args)[0]))
        self.assertEqual(transfer_one(1000,5,0)[0],0)
        self.assertEqual(transfer_one(0,5,5,True)[0],0)

    def test_indices_ignore_overrides_missing_and_test_households(self):
        cfg=default_config();cfg['analysis']['minimum_train_cell']=2
        cfg['hybrid']['index_source']='cohort_means'
        cfg['simulation']['archetype_overrides']={'1':{'heating_hours':22,'cooling_hours':21}}
        frame=pd.DataFrame([dict(split=s,location='Centre',dwelling_type='Apartment',archetype=1,weight=1,
                                 nmembers=2,heating=h,electricity=1000,cooling_hours=c,cooling='Present')
                            for s,h,c in [('train',4,2),('train',8,6),('train',6,math.nan),('test',24,24)]])
        maps={k:pd.DataFrame({'cell':['Centre|Apartment|1'+('|Present' if k=='electricity' else '')],'prediction':[6 if k=='heating' else 1000]}) for k in ['heating','electricity']}
        p=scenario_profile(frame,maps,cfg,'Centre','Apartment',1)
        self.assertEqual(p['heating_index'],6); self.assertEqual(p['cooling_index'],4)
        self.assertEqual(p['cooling_index_n'],2);self.assertEqual(p['heating_hours'],22)
        p=scenario_profile(frame,maps,cfg,'North-west','Apartment',1)
        self.assertTrue(math.isnan(p['heating_index']))
        self.assertEqual(p['profile_source'],'archetype fallback')

    def test_reference_only_requires_no_target_weather(self):
        c=default_config();c['simulation']['run_full_benchmark']=False;c['report']['powerpoint']=False;c['hybrid']['method']='istat_ratio'
        for climate in c['climates']:
            if climate['name']!='Milano':climate['epw']='no-such-file.epw'
        validate(c)
        c['schema_version']=1
        with self.assertRaisesRegex(ValueError,'schema 3'):validate(c)

    def test_engine_freezes_before_benchmark_and_never_uses_rc(self):
        cfg=default_config();cfg['report']['powerpoint']=False
        cfg['contexts']=cfg['contexts'][1:2];cfg['analysis']['n_archetypes']=2;cfg['hybrid']['method']='istat_ratio'
        cfg['climates']=cfg['climates'][2:];cfg['simulation']['anchor_archetype']=2;cfg['simulation']['anchor_climate']='Palermo'
        profile={k:v for i in ['heating','cooling','appliance'] for k,v in [(i+'_index',5.),(i+'_index_n',40),(i+'_index_coverage',1.),(i+'_index_source','test')]}
        profile.update(members=2,heating_hours=5,cooling_hours=5,appliance_kwh=1000)
        frame=pd.DataFrame({'archetype':[1,2]});mapping={k:pd.DataFrame({'group':['1']}) for k in ['heating','electricity']}
        def fake_prepare(*args):return frame,mapping,{}
        predictions=[]
        for full,multiplier in [(False,1),(True,999)]:
            with tempfile.TemporaryDirectory() as tmp:
                cfg['paths']['output_root']=tmp;cfg['simulation']['run_full_benchmark']=full
                def fake_run(model,epw,exe,folder,*args):
                    if folder.name=='S001':
                        self.assertTrue((folder.parent.parent/'hybrid_predictions_frozen.csv').is_file())
                    scale=multiplier if folder.name=='S001' else 1
                    return pd.DataFrame({'month':[1,2],'heating_kwh':[100*scale,100*scale],
                                         'cooling_kwh':[20*scale,20*scale],'appliance_kwh':[500,500]}),.01,0
                with patch.object(engine,'prepare',side_effect=fake_prepare),patch.object(engine,'scenario_profile',return_value=profile), \
                     patch.object(engine,'read_weather',return_value=None) as weather,patch.object(engine,'make_model',return_value={}), \
                     patch.object(engine,'run_model',side_effect=fake_run),patch.object(engine,'make_report'), \
                     patch.object(engine.subprocess,'run') as proc,patch('energy_studio.physics.simulate_rc',side_effect=AssertionError('RC prohibited')):
                    proc.return_value.stdout='EnergyPlus 25.1'
                    out=engine.run(copy.deepcopy(cfg),log=lambda _:None)
                manifest=json.loads((out/'manifest.json').read_text())
                self.assertEqual(manifest['energyplus_processes_run'],2 if full else 1)
                self.assertEqual(weather.call_count,1)
                predictions.append((out/'hybrid_predictions_frozen.csv').read_bytes())
                annual=pd.read_csv(out/'annual_results.csv')
                self.assertTrue(annual.loc[annual.method.eq('istat_only'),'heating_kwh'].isna().all())
                self.assertFalse(list(out.glob('hourly/*/hybrid*.csv')))
        self.assertEqual(predictions[0],predictions[1])

    def test_legacy_reports_are_not_relabelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'manifest.json').write_text(json.dumps({'status':'complete','version':'0.2.0'}))
            with self.assertRaisesRegex(ValueError,'Legacy'):engine.refresh_reports(tmp)


if __name__=='__main__':unittest.main()
