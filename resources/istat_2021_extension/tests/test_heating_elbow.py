import itertools
import unittest

import numpy as np
import pandas as pd

from run_heating_elbow import exact_partitions, knee


class HeatingElbowTests(unittest.TestCase):
    def test_exact_objective_matches_exhaustive_partitions(self):
        x = np.array([1., 2., 4., 9., 10., 15.])
        w = np.array([2., 1., 4., 1., 3., 2.])
        partitions, losses = exact_partitions(x, w, len(x))
        for k in range(1, len(x)+1):
            candidates = []
            for cuts in itertools.combinations(range(1, len(x)), k-1):
                edges = [0, *cuts, len(x)]
                loss = 0.
                for i, j in zip(edges[:-1], edges[1:]):
                    mean = np.average(x[i:j], weights=w[i:j])
                    loss += np.sum(w[i:j]*(x[i:j]-mean)**2)
                candidates.append(loss)
            self.assertAlmostEqual(losses[k-1], min(candidates))
            self.assertEqual(sum(j-i for i, j in partitions[k]), len(x))
        self.assertTrue(np.all(np.diff(losses) <= 1e-10))

    def test_knee_identifies_synthetic_bend(self):
        curve = pd.DataFrame({'k': [1, 2, 3, 4, 5], 'cell_wcss': [100, 40, 10, 8, 6]})
        self.assertEqual(knee(curve, 5), 3)


if __name__ == '__main__':
    unittest.main()
