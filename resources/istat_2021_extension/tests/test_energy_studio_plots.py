import unittest
import pandas as pd
from energy_studio.plots import matched_errors


class PlotDataTests(unittest.TestCase):
    def test_pairs_exclude_anchor_missing_and_unmatched(self):
        rows=[]
        for method in ['energyplus','hybrid_istat']:
            for i in range(3):
                rows.append(dict(scenario_id=f'S{i}',climate='Roma',dwelling='Apartment',archetype=i+1,
                                 method=method,is_anchor=i==0,heating_kwh=100+i+(10 if method!='energyplus' else 0),cooling_kwh=20.))
        frame=pd.DataFrame(rows)
        frame.loc[(frame.method=='hybrid_istat')&(frame.scenario_id=='S1'),'heating_kwh']=float('nan')
        frame=frame[~((frame.method=='energyplus')&(frame.scenario_id=='S2'))]
        errors=matched_errors(frame)
        self.assertEqual(set(errors.scenario_id),{'S1'})
        self.assertEqual(len(errors),1)
        heat=errors[errors.end_use=='heating_kwh']
        self.assertTrue(heat.empty)
        self.assertEqual(errors.iloc[0]['method'],'hybrid_istat')

    def test_no_benchmark_has_no_errors(self):
        frame=pd.DataFrame([dict(scenario_id='S1',climate='Roma',dwelling='Apartment',archetype=1,
                                method='istat_only',is_anchor=True,heating_kwh=float('nan'),cooling_kwh=float('nan'))])
        self.assertTrue(matched_errors(frame).empty)


if __name__=='__main__':unittest.main()
