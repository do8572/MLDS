import unittest

from copy import deepcopy
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from imodels import HSTreeRegressor


def DGP_regression_1(n: int = 10, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # y = 0.5 * x1
    x = np.random.uniform(0, 4, (n, m))
    y = 0.5 * x[:, 0] + np.random.normal(0, 0.5, n)
    return x, y


def DGP_regression_2(n: int = 10, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # y = 0.5 * x1^2 - 2 * x1 + 1
    x = np.random.uniform(0, 4, (n, m))
    y = 0.5 * x[:, 0] * x[:, 0] - 2 * x[:, 0] + 1 + np.random.normal(0, 0.5, n)
    return x, y


def DGP_limit_test(n: int = 10, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # y = 3 * x1 * x2 - 7 * x2 + 1.5 * x1 - 2
    x = np.random.uniform(0, 4, (n, m))
    y = 3 * x[:, 0] * x[:, 1] - 7 * x[:, 1] + 1.5 * x[:, 0] - 2 + np.random.normal(0, 0.5, n)
    return x, y


class TestDGPRegression(unittest.TestCase):
    
    def test_tree_shape(self):
        """
        This test checks that the underlying structure of the tree is still the
        same after Hierarchical Shrinkage has been done.
        """
        for max_leaf_nodes in [3, 5, 7, 9, 11]:
            for reg_param in [0.1, 1, 10, 100]:
                for X, y in [DGP_regression_1(20), DGP_regression_2(20)]:
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
    
    def test_lambda_infinity(self):
        """
        Hierarchical Shrinkage is supposedly equivalent to ridge regression.
        This test checks that as the lambda goes to infinity, the predictions
        in each tree node approach the data mean.

        Interesting side observation:
           the lambda that is large enough to be considered infinity is linked
        to the number of data points `n`.
        """
        for n in [10, 100, 1000, 10000]:
            for max_leaf_nodes in [3, 5, 7, 9, 11]:
                X, y = DGP_limit_test(n)
                dist = np.mean(y)

                skmodel = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
                skmodel.fit(X, y)
                imodel = HSTreeRegressor(deepcopy(skmodel), reg_param=3e4*n)
                i_tree = imodel.estimator_.tree_

                distribution = np.tile(dist, (i_tree.value.shape[0], 1, 1))

                np.testing.assert_almost_equal(distribution, i_tree.value, decimal=3)
    
    def test_lambda_zero(self):
        """
        Hierarchical Shrinkage is supposedly equivalent to ridge regression.
        This test checks that as the lambda goes to zero, the predictions
        in each tree node should be unchanged and the same as before HS
        """
        for n in [10, 100, 1000, 10000]:
            for max_leaf_nodes in [3, 5, 7, 9, 11]:
                X, y = DGP_limit_test(n)

                skmodel = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
                skmodel.fit(X, y)
                imodel = HSTreeRegressor(deepcopy(skmodel), reg_param=0)
                sk_tree = skmodel.tree_
                i_tree = imodel.estimator_.tree_

                np.testing.assert_almost_equal(sk_tree.value, i_tree.value)

    def test_DGP1_lambda2(self):
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
    
    def test_DGP1_lambda5(self):
        np.random.seed(1)
        X, y = DGP_regression_1(20)

        skmodel = DecisionTreeRegressor(max_leaf_nodes=4)
        skmodel.fit(X, y)
        imodel = HSTreeRegressor(deepcopy(skmodel), reg_param=5)
        i_tree = imodel.estimator_.tree_

        # Tree matches our manual calculation
        manual_calculation = np.array([
            [[0.643]], # root (1)
            [[0.379]], # left child of 1 (2)
            [[1.038]], # right child of 1 (3)
            [[0.150]], # left child of 2 (4)
            [[0.543]], # right child of 2 (5)
            [[0.979]], # left child of 3 (6)
            [[1.459]] # right child of 3 (7)
        ])
        np.testing.assert_almost_equal(manual_calculation, i_tree.value, decimal=3)
    
    def test_DGP2_lambda10(self):
        np.random.seed(2)
        X, y = DGP_regression_2(20)

        skmodel = DecisionTreeRegressor(max_leaf_nodes=3)
        skmodel.fit(X, y)
        imodel = HSTreeRegressor(deepcopy(skmodel), reg_param=10)
        i_tree = imodel.estimator_.tree_

        # Tree matches our manual calculation
        manual_calculation = np.array([
            [[-0.448]], # root (1)
            [[-0.700]], # left child of 1 (2)
            [[-0.069]], # right child of 1 (3)
            [[ 0.192]], # left child of 3 (4)
            [[-0.156]], # right child of 3 (5)
        ])
        np.testing.assert_almost_equal(manual_calculation, i_tree.value, decimal=3)
    
    def test_DGP2_lambda20(self):
        np.random.seed(3)
        X, y = DGP_regression_2(20)

        skmodel = DecisionTreeRegressor(max_leaf_nodes=4)
        skmodel.fit(X, y)
        imodel = HSTreeRegressor(deepcopy(skmodel), reg_param=20)
        i_tree = imodel.estimator_.tree_

        # Tree matches our manual calculation
        manual_calculation = np.array([
            [[-0.533]], # root (1)
            [[ 0.151]], # left child of 1 (2)
            [[-0.654]], # right child of 1 (3)
            [[-0.823]], # left child of 3 (4)
            [[-0.246]], # right child of 3 (5)
            [[-0.626]], # left child of 4 (4)
            [[-0.965]], # right child of 4 (5)
        ])
        np.testing.assert_almost_equal(manual_calculation, i_tree.value, decimal=3)


if __name__ == "__main__":
    unittest.main()