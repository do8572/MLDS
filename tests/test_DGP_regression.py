import unittest

from copy import deepcopy
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from imodels import HSTreeRegressor


def DGP_regression_1(n: int = 10, m: int = 2):
    # y = 0.5 * x1
    x = np.random.uniform(0, 4, (n, m))
    y = 0.5 * x[:,0] + np.random.normal(0, 0.5, n)
    return x, y

def DGP_regression_2(n: int = 10, m: int = 2):
    # y = 2 * x1 - 1.5
    x = np.random.uniform(0, 4, (n, m))
    y = 2 * x[:,0] - 1.5 + np.random.normal(0, 0.5, n)
    return x, y

def DGP_regression_3(n: int = 10, m: int = 2):
    # y = 0.5 * x1^2 - 2 * x1 + 1
    x = np.random.uniform(0, 4, (n, m))
    y = 0.5 * x[:,0] * x[:,0] - 2 * x[:,0] + 1 + np.random.normal(0, 0.5, n)
    return x, y


class TestDGPRegression(unittest.TestCase):
    
    def test_tree_shape(self):
        X, y = DGP_regression_1(20)

        for max_leaf_nodes in [3, 5, 7, 9]:
            for reg_param in [0.1, 1, 10, 100]:
                skmodel = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
                skmodel.fit(X, y)
                imodel = HSTreeRegressor(deepcopy(skmodel), reg_param=reg_param)

                sk_tree = skmodel.tree_
                i_tree = imodel.estimator_.tree_

                # Test that the underlying tree structure is unchanged.
                np.testing.assert_array_equal(sk_tree.children_left, i_tree.children_left)
                np.testing.assert_array_equal(sk_tree.children_right, i_tree.children_right)
                np.testing.assert_array_equal(sk_tree.weighted_n_node_samples, i_tree.weighted_n_node_samples)
                np.testing.assert_array_equal(sk_tree.impurity, i_tree.impurity)
                np.testing.assert_array_equal(sk_tree.feature, i_tree.feature)
                np.testing.assert_equal(sk_tree.node_count, i_tree.node_count)
                np.testing.assert_equal(sk_tree.n_leaves, i_tree.n_leaves)
                np.testing.assert_equal(sk_tree.max_depth, i_tree.max_depth)

    def test_DGP1_lambda2(self):
        # Reproducible dataset
        np.random.seed(0)
        X, y = DGP_regression_1(20)

        skmodel = DecisionTreeRegressor(max_leaf_nodes=3)
        skmodel.fit(X, y)
        imodel = HSTreeRegressor(deepcopy(skmodel), reg_param=2)
        i_tree = imodel.estimator_.tree_

        # Tree matches our manual calculation
        manual_calculation = np.array([
            [[0.790]], # root (1)
            [[0.520]], # left child of 1 (2)
            [[1.870]], # right child of 1 (3)
            [[-0.009]], # left child of 2 (4)
            [[0.761]] # right child of 2 (5)
        ])
        np.testing.assert_almost_equal(manual_calculation, i_tree.value, decimal=3)
    
    def test_DGP1_lambda10(self):
        pass
    
    def test_DGP2_lambda1(self):
        pass
    
    def test_DGP2_lambda5(self):
        pass
    
    def test_DGP3_lambda3(self):
        pass
    
    def test_DGP3_lambda7(self):
        pass


if __name__ == "__main__":
    unittest.main()