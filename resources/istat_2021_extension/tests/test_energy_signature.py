import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from energy_studio.config import default_config,validate
from energy_studio.signature import fit_reference,degree_features,estimate_one,predict,features
from energy_studio.reuse import reuse_reference,digest


class SignatureTests(unittest.TestCase):
    def setUp(self):self.cfg=default_config()

    def test_reference_fit_recovers_synthetic_signature(self):
        outdoor=np.linspace(-8,35,365);solar=2+np.cos(np.arange(365)*.27)
        ref={'equivalent_c':outdoor,'solar_equivalent_k':solar,'outdoor_c':outdoor}
        hourly=pd.DataFrame({use:np.repeat(3*degree_features(outdoor,solar,b,.6,use)/24,24) for use,b in [('heating_kwh',15.),('cooling_kwh',23.)]})
        fits,diagnostics=fit_reference(self.cfg,ref,hourly)
        for use,b in [('heating_kwh',15.),('cooling_kwh',23.)]:
            self.assertAlmostEqual(fits[use]['balance_c'],b)
            self.assertAlmostEqual(fits[use]['solar_coefficient'],.6)
            self.assertLess(fits[use]['daily_fit_rmse_kwh'],1e-10)
        self.assertEqual(len(diagnostics),730)

    def test_no_active_reference_is_not_invented(self):
        ref={'equivalent_c':np.ones(365)*18,'solar_equivalent_k':np.zeros(365),'outdoor_c':np.ones(365)*18}
        fits,diagnostics=fit_reference(self.cfg,ref,pd.DataFrame({'heating_kwh':np.zeros(8760),'cooling_kwh':np.zeros(8760)}))
        self.assertTrue(diagnostics.empty)
        self.assertEqual(fits['heating_kwh']['status'],'insufficient_reference_activity')

    def fixture(self):
        p={'members_index':2.,'appliance_index':1500.,'heating_index':8.,'cooling_index':6.,
           'members_index_n':40,'appliance_index_n':40,'heating_index_n':40,'cooling_index_n':40}
        sc={'id':'target','key':'Palermo__Apartment__A4','profile':p};anchor={'id':'ref','key':'Milano__Multi-family__A2'}
        ref={'conductance_w_k':100.,'gains_w':100.,'setpoint_c':{'heating_kwh':18.,'cooling_kwh':26.},
             'equivalent_c':np.ones(365)*10,'solar_equivalent_k':np.zeros(365),'hdd18':1000.,'cdd22':200.}
        fits={use:{'status':'fitted','balance_c':b,'reference_energy_kwh':1000.,'reference_degree_sum':730.,'solar_coefficient':0.} for use,b in [('heating_kwh',12.),('cooling_kwh',8.)]}
        return sc,anchor,ref,fits

    def test_gains_reduce_heating_and_increase_cooling(self):
        sc,anchor,ref,fits=self.fixture();target=copy.deepcopy(ref);target['gains_w']=200.
        values,_=estimate_one(self.cfg,sc,anchor,target,ref,fits)
        self.assertAlmostEqual(values['heating_kwh'],500.)
        self.assertAlmostEqual(values['cooling_kwh'],1500.)

    def test_reference_identity_and_unsupported_target(self):
        sc,anchor,ref,fits=self.fixture();anchor.update(profile=sc['profile'])
        values,audit=estimate_one(self.cfg,anchor,anchor,ref,ref,fits)
        self.assertEqual(values['heating_kwh'],1000.);self.assertEqual(audit[0]['status'],'anchor_identity')
        sc['profile']['members_index']=float('nan')
        values,audit=estimate_one(self.cfg,sc,anchor,ref,ref,fits)
        self.assertTrue(np.isnan(values['heating_kwh']))

    def test_exposure_and_hours_have_explicit_effects(self):
        sc,anchor,ref,fits=self.fixture();target=copy.deepcopy(ref);target['conductance_w_k']=200.
        values,audit=estimate_one(self.cfg,sc,anchor,target,ref,fits)
        self.assertAlmostEqual(values['heating_kwh'],2500.)
        target=copy.deepcopy(ref);target['setpoint_c']['heating_kwh']+=1
        values,_=estimate_one(self.cfg,sc,anchor,target,ref,fits)
        self.assertAlmostEqual(values['heating_kwh'],1500.)

    def test_public_input_excludes_target_ep_and_no_rc(self):
        self.assertEqual(list(inspect.signature(predict).parameters),['cfg','scenarios','anchor','weather','reference_hourly','baseline'])
        import energy_studio.signature as module
        self.assertNotIn('simulate_rc',inspect.getsource(module))
        cfg=default_config();cfg['simulation']['run_full_benchmark']=False
        cfg['climates'][2]['epw']='missing.epw'
        with self.assertRaisesRegex(ValueError,'weather'):validate(cfg)

    def test_reuse_rejects_changed_model_and_weather(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();dest=root/'dest';dest.mkdir()
            folder=source/'energyplus'/'S006';folder.mkdir(parents=True)
            epw=root/'test.epw';epw.write_text('weather')
            cfg=default_config();cfg['paths']['reference_run']=str(source)
            anchor={'id':'S006','key':'Milano__Multi-family__A2','climate':{'name':'Milano','epw':str(epw)},'context':{},'profile':{}}
            (source/'manifest.json').write_text(json.dumps({'status':'complete','weather':[{'name':'Milano','sha256':digest(epw)}]}))
            pd.DataFrame([{'method':'energyplus','scenario':anchor['key'],'scenario_id':'S006','heating_kwh':8760.,'cooling_kwh':8760.,'appliance_kwh':8760.}]).to_csv(source/'annual_results.csv',index=False)
            pd.DataFrame({'heating_kwh':np.ones(8760),'cooling_kwh':np.ones(8760),'appliance_kwh':np.ones(8760)}).to_csv(folder/'hourly.csv',index=False)
            (folder/'model.epJSON').write_text(json.dumps({'model':1}))
            with patch('energy_studio.reuse.make_model',return_value={'model':1}):
                data,provenance=reuse_reference(cfg,anchor,{},dest)
                self.assertEqual(len(data),8760)
            with patch('energy_studio.reuse.make_model',return_value={'model':2}):
                with self.assertRaisesRegex(ValueError,'differs'):reuse_reference(cfg,anchor,{},dest)


if __name__=='__main__':unittest.main()
