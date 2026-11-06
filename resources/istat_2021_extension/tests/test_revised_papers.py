"""Invariants of the corrected analysis; no published coefficients are targets."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import patch
from scipy.special import softmax
import revised_analysis as r


class CorrectedMethodsTests(unittest.TestCase):
    def test_no_insulation_model_does_not_require_insulation_columns(self):
        rng = np.random.default_rng(82)
        n = 600
        frame = pd.DataFrame({
            'split': ['test'] * n, 'heating': rng.uniform(1, 20, n),
            'weight': rng.uniform(1, 3, n),
            'age_proxy': rng.uniform(5, 100, n),
            'area_proxy': rng.uniform(30, 200, n),
            'archetype': rng.choice([1, 2, 3, 4], n),
            'dwelling_type': rng.choice(r.TYPES, n),
            'location': rng.choice(r.LOCATIONS, n),
            'cooling': rng.choice(['Present', 'Absent'], n)})
        with tempfile.TemporaryDirectory() as directory:
            result = r.energy_anova(frame, 'heating', 2021, Path(directory),
                                    include_insulation=False)
            self.assertEqual(result['n'], n)
            self.assertNotIn('insulation', result['formula'])
            self.assertEqual(result['rank'], result['columns'])

    def test_2021_routing_keeps_renovation_and_cooling_measures_distinct(self):
        raw = pd.DataFrame({
            'ID_RECORD':['1','2','3','4'], 'v1_1':['1']*4,
            'v5_6':['1','5',' ','6'],
            **{f'v2_11{c}':['1','1','2','1'] if c=='a' else ['2']*4 for c in 'abcde'},
            'v3_13':['1','1',' ','2'], 'v3_14gh_new':['1',' ',' ',' '],
            **{f'v5_7{c}':['2',' ',' ',' '] for c in 'abc'},
            'v2_11conta':['1','1','0','1'], 'v5_0':['1','1',' ','1'],
            'v5_4_mfr':['1','1',' ','1']})
        with patch.object(r.pd, 'read_csv', return_value=raw):
            result = r.supplement(Path('unused'),2021)
        self.assertEqual(result.insulation_improvement.tolist(),
                         ['Reported','Not reported','Unknown','Not reported'])
        self.assertTrue(result.insulation.eq('Unavailable').all())
        self.assertEqual(result.cooling.tolist(),['Present','Present','Absent','Present'])
        self.assertEqual(result.cooling_hours.iloc[0],6)
        self.assertTrue(result.cooling_hours.iloc[1:].isna().all())

    def setUp(self):
        self.cfg=json.loads((r.ROOT/'revised_paper_config.json').read_text())

    def test_weighted_pca_is_orthonormal_and_explains_all_variance(self):
        x=np.random.default_rng(3).normal(size=(100,3))
        p=r.fit_pca(x,np.linspace(1,2,100));v=np.array(p['loadings'])
        np.testing.assert_allclose(v@v.T,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(sum(p['explained_ratio']),1)

    def test_education_mapping_is_not_recursive_and_ignores_child_codes(self):
        raw=pd.DataFrame([{'id':1,'q_1_1_sq1':2,'cletad_1':35,'titstu_1':3,'cletad_2':0,'titstu_2':0}])
        f,e=r.features_from_codes(raw,2013,self.cfg)
        self.assertEqual(len(e),0)
        self.assertAlmostEqual(f.education.iloc[0],2.5/18)
        self.assertEqual(f.agemin.iloc[0],7)

    def test_missing_adult_education_is_excluded_not_zero(self):
        raw=pd.DataFrame([{'id':1,'q_1_1_sq1':1,'cletad_1':35,'titstu_1':0}])
        f,e=r.features_from_codes(raw,2013,self.cfg)
        self.assertEqual(len(f),0);self.assertEqual(len(e),1)

    def test_split_is_disjoint_and_reproducible(self):
        s=r.assign_splits(np.arange(100),self.cfg,2013)
        np.testing.assert_array_equal(s,r.assign_splits(np.arange(100),self.cfg,2013))
        self.assertEqual(dict(zip(*np.unique(s,return_counts=True))),{'train':70,'tune':15,'test':15})

    def test_softmax_probabilities_form_simplex_without_clipping(self):
        x=np.column_stack([np.ones(80),np.tile([0.,1.],40)])
        y=np.tile([1,2,3,4],20)
        beta=r.fit_softmax(x,y,np.ones(80),.001)
        p=softmax(x@beta,axis=1)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
        self.assertTrue(np.all(p>0));np.testing.assert_array_equal(beta[:,-1],0)

    def test_weight_rescaling_does_not_change_fit(self):
        x=np.column_stack([np.ones(80),np.tile([0.,1.],40)])
        y=np.tile([1,2,3,4],20);w=np.linspace(1,2,80)
        np.testing.assert_allclose(r.fit_softmax(x,y,w,.001),r.fit_softmax(x,y,100*w,.001),atol=1e-5)

    def test_reference_assignment_does_not_depend_on_batch(self):
        rng=np.random.default_rng(7)
        frame=pd.DataFrame({'agemin':rng.uniform(15,65,120),'agemax':rng.uniform(65,90,120),
                            'nmembers':rng.integers(1,6,120),'education':rng.uniform(.1,1,120),'weight':1.})
        model=r.fit_demographics(frame,self.cfg)
        all_labels=r.assign_archetypes(frame,model).archetype
        subset=r.assign_archetypes(frame.iloc[:20],model).archetype
        np.testing.assert_array_equal(all_labels.iloc[:20],subset)


@unittest.skipUnless((r.ROOT/'revised_paper_outputs/results.json').exists(),'Run the corrected analysis first')
class CorrectedArtifactTests(unittest.TestCase):
    def test_probability_fitting_and_evaluation_are_disjoint(self):
        for year in [2013,2021]:
            frame=pd.read_csv(r.ROOT/f'revised_paper_outputs/{year}/probability_households.csv')
            self.assertTrue(frame.id.is_unique)
            for a,b in [('train','test'),('tune','test'),('train','tune')]:
                self.assertFalse(set(frame.loc[frame.split.eq(a),'id'])&set(frame.loc[frame.split.eq(b),'id']))
            np.testing.assert_allclose(frame[[f'prob_{a}' for a in range(1,5)]].sum(axis=1),1,atol=1e-12)

    def test_all_energy_test_records_are_from_the_test_subset(self):
        for year,outcomes in [(2013,['heating','electricity']),(2021,['heating','electricity'])]:
            for outcome in outcomes:
                data=pd.read_csv(r.ROOT/f'revised_paper_outputs/{year}/{outcome}_test.csv')
                self.assertTrue(data.split.eq('test').all())
                self.assertTrue(np.isfinite(data.prediction).all())

    def test_new_wave_outcomes_are_not_fabricated(self):
        data=pd.read_csv(r.ROOT/'revised_paper_outputs/2021/households.csv')
        self.assertTrue(data.electricity.notna().all())
        self.assertTrue(data.electricity.gt(0).all())
        self.assertTrue(data.insulation.eq('Unavailable').all())
        self.assertEqual(set(data.dwelling_type),{'Apartment','Single-family','Multi-family'})
        self.assertEqual(int(data.insulation_improvement.eq('Reported').sum()),5687)
        self.assertEqual(int(data.cooling_hours.notna().sum()),17493)
        self.assertTrue(data.loc[~data.cooling_frequency.isin([1,2,3]),'cooling_hours'].isna().all())
        tests=pd.read_csv(r.ROOT/'revised_paper_outputs/2021/heating_anova.csv')
        self.assertEqual(len(tests),14)
        self.assertTrue(tests.F.notna().all())
        self.assertIn('Insulation improvement',tests.term.tolist())
        pairs=pd.read_csv(r.ROOT/'revised_paper_outputs/2021/heating_insulation_pairs.csv')
        self.assertTrue(pairs.p_holm.between(0,1).all())

    def test_robust_energy_designs_are_full_rank(self):
        result=json.loads((r.ROOT/'revised_paper_outputs/results.json').read_text())
        for year in result['years'].values():
            for outcome in year['energy'].values():
                self.assertEqual(outcome['anova']['rank'],outcome['anova']['columns'])

    def test_electricity_reconstruction_and_absent_appliances(self):
        folder=r.ROOT/'revised_paper_outputs/appliance_audit'
        components=pd.read_csv(folder/'components_2013.csv',index_col='id')
        saved=pd.read_csv(r.ROOT/'results_grins.csv',index_col='id')
        total=components[['lights','small','fridge_ce','big_ce','screens','standby']].sum(axis=1)
        np.testing.assert_allclose(total,saved.loc[total.index,'appliances_el_kWh'],rtol=0,atol=1e-8)
        model=pd.read_csv(folder/'electricity_2021.csv',index_col='id')
        path=next((r.ROOT/'istat/Microdati21').rglob('*Microdati_2021_DELIMITED.txt'))
        raw=pd.read_csv(path,sep='^',index_col='ID_RECORD')
        for field,col in [('v7_10','washing_kwh'),('v7_16','drying_kwh'),('v7_19','dishwashing_kwh'),('v7_32','tv_kwh'),('v7_36','computers_kwh')]:
            self.assertTrue(model.loc[raw[field].eq(2),col].eq(0).all())
        regular=pd.to_numeric(raw.v7_12a,errors='coerce').notna()
        temperatures=raw[['v7_13a','v7_13b','v7_13c']].apply(pd.to_numeric,errors='coerce')
        expected=47*1.8*(temperatures*np.array([.3,1.,2.56])).sum(axis=1)+raw.v7_15.astype(str).str.strip().eq('1').astype(int)
        np.testing.assert_allclose(model.loc[regular,'washing_kwh'],expected[regular])


if __name__=='__main__':unittest.main()
