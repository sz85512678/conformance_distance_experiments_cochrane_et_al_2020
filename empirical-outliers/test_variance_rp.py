import unittest
import numpy as np

from variance_rp import _sort_and_truncate_arrays_descending, variance_rp

class TestVarianceRP(unittest.TestCase):

    def test_sort_and_truncate_arrays_descending(self):
        eigs = np.array([1, 1e-1, 1e-3, 1e-2])
        eigvecs = np.array([[1, 0,0,0], [0,1,0,0], [0,0,0,1],[0,0,1,0]])

        a, b =_sort_and_truncate_arrays_descending(eigs, eigvecs, 2e-3)

        self.assertTrue(np.array_equal(a, np.array([1, 1e-1, 1e-2])))
        self.assertTrue(np.array_equal(
            b, np.array([[1, 0,0,0], [0,1,0,0], [0,0,1,0]])))

    def test_variance_rp(self):
        np.random.seed(0)
        order = 2
        corpus = np.array([
            [[0,0], [0, 1], [1,1], [0,0] ], 
            [[0,0], [0, 2], [2,2], [0,0] ], 
            [[0,0], [0, 1], [3,1], [0,0] ] , 
            [[0,0], [1,1], [2,2], [0,0]] 
            ])
        paths = np.array(
            [[[0.73911896, 0.87414819],
             [0.32503197, 0.90596157],
             [0.94639122, 0.00261653],
             [0.54151759, 0.45043385]], 
            [[0,0], [0,3], [3,4], [0,0]]]
             )
        projected_dim = 2

        variances = variance_rp(paths, corpus, order, projected_dim)
        print(variances)
        self.assertTrue(variances[0] > variances[1] )

        corpus = np.array([
            [[0,0], [0, 1], [1,1], [0,0] ], 
            [[0,0], [0, 2], [2,2], [0,0] ], 
            [[0,0], [0, -1], [-1,-1], [0,0] ] , 
            [[0,0], [1,1], [2,2], [0,0]] 
            ])

        variances = variance_rp(paths, corpus, order, projected_dim)
        print(variances)
        self.assertTrue(variances[1] > variances[0] )

if __name__ == '__main__':
    unittest.main()