"""Scientific invariants for the isolated comparison, not figure-count tests."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_replicability import (
    assign_reference, common_dwelling_groups, common_features,
    dwelling_probabilities, paper_equation_audit,
)


def household(year, ages, education, count=None):
    row = {"id": 1, "q_1_1_sq1": len(ages) if count is None else count}
    row.update({f"cletad_{i}": ages[i-1] if i <= len(ages) else 0 for i in range(1,13)})
    row.update({f"titstu_{i}": education[i-1] if i <= len(education) else 0 for i in range(1,13)})
    return pd.DataFrame([row])


class ScientificComparisonTests(unittest.TestCase):
    def test_primary_and_no_qualification_are_coarsened_in_both_waves(self):
        for year, age, codes in [(2013,45,[1,3]), (2021,7,[1])]:
            for code in codes:
                f,_=common_features(household(year,[age],[code]),year)
                self.assertAlmostEqual(f.education.iloc[0],5/21)

    def test_children_are_not_silently_aged_into_adolescents(self):
        f,_=common_features(household(2013,[0,45],[1,4]),2013)
        self.assertEqual(f.agemin.iloc[0],7)
        self.assertAlmostEqual(f.education.iloc[0],8/21)

    def test_same_household_size_limit_for_both_years(self):
        for year in [2013,2021]:
            f,ex=common_features(household(year,[1]*6,[1]*6),year)
            self.assertTrue(f.empty)
            self.assertEqual(ex.reason.iloc[0],"outside_common_1_to_5_member_sample")

    def test_transfer_assignment_is_batch_independent(self):
        model={"minimum":[-100,0,0],"span":[200,200,1],
               "centers":[[.1,.1,.1],[.3,.5,.5],[.5,.6,.9],[.8,.3,.3]]}
        f=pd.DataFrame({"id":[1,2],"agemin":[30,5],"agemax":[40,80],"nmembers":[2,5],"education":[.5,.8]})
        full=assign_reference(f,model)
        single=assign_reference(f.iloc[:1],model)
        self.assertEqual(full.archetype.iloc[0],single.archetype.iloc[0])

    def test_unknown_construction_is_not_assigned_a_fake_age(self):
        for year in [2013,2021]:
            f=common_dwelling_groups(pd.DataFrame({"id":[1],"q_2_7_class":[3],"q_2_4_ric":[9],"reg":[1]}),year)
            self.assertTrue(f.construction_period.isna().all())

    def test_weighted_probabilities_are_counted_from_weights(self):
        f=pd.DataFrame({"area_band":["40-59"]*4,"construction_period":["1990+"]*4,
                        "archetype":[1,2,3,4],"weight":[1,2,3,4]})
        p=dwelling_probabilities(f,2021)
        np.testing.assert_allclose(p[["prob_1","prob_2","prob_3","prob_4"]],[ [.1,.2,.3,.4] ])
        self.assertFalse(p.meets_support_30.iloc[0])

    def test_published_equation_discrepancy_is_preserved(self):
        checks,_=paper_equation_audit()
        row=checks.query("source=='Equation_3' and age==100 and area==100").iloc[0]
        self.assertAlmostEqual(row["sum"],1.14)


if __name__ == "__main__":
    unittest.main()
