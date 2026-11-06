import copy
import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
from energy_studio.config import default_config,validate
from energy_studio.analysis import exact_partitions,grouping_factors
from energy_studio.physics import geometry,daily_operation,schedules,read_weather,simulate_rc
from energy_studio.energyplus import make_model


class EnergyStudioTests(unittest.TestCase):
    def setUp(self):
        self.cfg=default_config();self.s=self.cfg['simulation']
        self.profile={'members':2.,'heating_hours':8.5,'cooling_hours':6.,'appliance_kwh':2000.}

    def test_partition_objective(self):
        x=np.array([1.,2.,4.,9.,10.,15.]);w=np.array([2.,1.,4.,1.,3.,2.])
        _,losses=exact_partitions(x,w,6)
        for k in range(1,7):
            brute=[]
            for cuts in itertools.combinations(range(1,6),k-1):
                edges=[0,*cuts,6]
                brute.append(sum(np.sum(w[l:r]*(x[l:r]-np.average(x[l:r],weights=w[l:r]))**2) for l,r in zip(edges[:-1],edges[1:])))
            self.assertAlmostEqual(losses[k-1],min(brute))

    def test_cooling_factor_is_electricity_only_by_default(self):
        self.assertNotIn('cooling',grouping_factors(self.cfg,'heating'))
        self.assertIn('cooling',grouping_factors(self.cfg,'electricity'))
        self.cfg['analysis']['electricity_include_cooling']=False
        self.assertNotIn('cooling',grouping_factors(self.cfg,'electricity'))

    def test_appliances_close_annual_energy(self):
        s=schedules(self.s,self.profile)
        self.assertAlmostEqual(float(s['appliance_w'].sum()*365/1000),2000.)
        self.assertAlmostEqual(float(daily_operation(8.5,'heat').sum()),8.5)
        self.assertTrue(np.all(s['heat']<s['cool']))

    def test_exposure_changes_heat_loss(self):
        g=[geometry(self.s,c) for c in self.cfg['contexts']]
        self.assertGreater(g[2]['h_out'],g[1]['h_out']);self.assertGreater(g[1]['h_out'],g[0]['h_out'])
        self.assertEqual(g[0]['h_ground'],0)

    def test_weather_and_physics_outputs(self):
        w=read_weather(self.cfg['climates'][1]['epw'])
        legacy={**self.s,'rc_capacity_kj_m2k':165.}
        out=simulate_rc(w,legacy,self.cfg['contexts'][0],self.profile)
        self.assertEqual(len(out),8760);self.assertFalse(out.isna().any().any())
        self.assertTrue((out[['heating_kwh','cooling_kwh']]>=0).all().all())
        self.assertAlmostEqual(out.appliance_kwh.sum(),2000.)
        off=copy.deepcopy(legacy);off['cooling_enabled']=False
        no_cool=simulate_rc(w,off,self.cfg['contexts'][0],self.profile)
        self.assertAlmostEqual(no_cool.cooling_kwh.sum(),0.)

    def test_model_boundaries_and_zones(self):
        weather=read_weather(self.cfg['climates'][1]['epw'])
        model=make_model(self.cfg,self.cfg['contexts'][0],self.profile,weather)
        self.assertEqual(len(model['Zone']),1)
        surfaces=model['BuildingSurface:Detailed']
        self.assertEqual(surfaces['East']['outside_boundary_condition'],'Adiabatic')
        self.assertEqual(surfaces['South']['outside_boundary_condition'],'Outdoors')
        self.assertEqual(len(model['FenestrationSurface:Detailed']),2)
        self.assertEqual(model['GlobalGeometryRules']['Rules']['coordinate_system'],'Relative')
        self.cfg['simulation']['cooling_enabled']=False
        model=make_model(self.cfg,self.cfg['contexts'][0],self.profile,weather)
        self.assertEqual(model['Schedule:Constant']['Cooling available']['hourly_value'],0)

    def test_bad_configuration_rejected(self):
        c=copy.deepcopy(self.cfg);c['simulation']['area_m2']=-1
        with self.assertRaises(ValueError):validate(c)
        c=copy.deepcopy(self.cfg);c['analysis']['features']=['heating']
        with self.assertRaises(ValueError):validate(c)
        c=copy.deepcopy(self.cfg);c['simulation']['cool_setpoint_c']=18
        with self.assertRaises(ValueError):validate(c)


if __name__=='__main__':unittest.main()
