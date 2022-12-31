import unittest

from copy import deepcopy
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from imodels import HSTreeClassifier


def DGP_classification_1(n: int = 10, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # y = x1 > 2
    x = np.random.uniform(0, 4, (n, m))
    y = (x[:,0] > 2)*1
    x += np.random.normal(0, 1, (n, m))
    return x, y


def DGP_classification_2(n: int = 10, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # y := 2 ; x1 >= 3
    #      1 ; 3 > x1 >= 1
    #      0 ; 1 > x1
    x = np.random.uniform(0, 4, (n, m))
    y = (1 <= x[:,0])*1 + (3 <= x[:,0])*1
    x += np.random.normal(0, 0.5, (n, m))
    return x, y


def DGP_limit_test(n: int = 10, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # y := 3 ; x1 >= 3 and x2 >= 2
    #      2 ; (x1 >= 3 and 2 > x2) or (x1 >= 1 and x2 >= 2)
    #      1 ; (3 > x1 >= 1 and 2 > x2) or (1 > x1 and x2 >= 2)
    #      0 ; 1 > x1 and 2 > x2
    m = min(2, m) # check that m is at least 2
    x = np.random.uniform(0, 4, (n, m))
    y = (1 <= x[:,0])*1 + (2 <= x[:,1])*1 + (3 <= x[:,0])*1
    x += np.random.normal(0, 0.5, (n, m))
    return x, y


class TestDGPClassification(unittest.TestCase):

    def test_tree_shape(self):
        """
        This test checks that the underlying structure of the tree is still the
        same after Hierarchical Shrinkage has been done.
        """
        for max_leaf_nodes in [3, 5, 7, 9, 11]:
            for reg_param in [0.1, 1, 10, 100]:
                for X, y in [DGP_classification_1(20), DGP_classification_2(20)]:
                    skmodel = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
                    skmodel.fit(X, y)
                    imodel = HSTreeClassifier(deepcopy(skmodel), reg_param=reg_param)

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
        in each tree node approach the data distribution.

        Interesting side observation:
           the lambda that is large enough to be considered infinity is linked
        to the number of data points `n`.
        """
        for n in [10, 100, 1000, 10000]:
            for max_leaf_nodes in [3, 5, 7, 9, 11]:
                X, y = DGP_limit_test(n)
                _, values = np.unique(y, return_counts=True)
                dist = values / n

                skmodel = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
                skmodel.fit(X, y)
                imodel = HSTreeClassifier(deepcopy(skmodel), reg_param=1e4*n)
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

                skmodel = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
                skmodel.fit(X, y)
                imodel = HSTreeClassifier(deepcopy(skmodel), reg_param=0)
                sk_tree = skmodel.tree_
                i_tree = imodel.estimator_.tree_

                sk_tree_value = sk_tree.value / sk_tree.weighted_n_node_samples[:,None,None]
                np.testing.assert_almost_equal(sk_tree_value, i_tree.value)
    
    def test_DGP1_lambda2(self):
        np.random.seed(0)
        X, y = DGP_classification_1(20)

        skmodel = DecisionTreeClassifier(max_leaf_nodes=3)
        skmodel.fit(X, y)
        imodel = HSTreeClassifier(deepcopy(skmodel), reg_param=2)
        i_tree = imodel.estimator_.tree_

        # The probability in each node sums to 1
        np.testing.assert_allclose(np.ones(5), i_tree.value.sum(axis=2).flatten())

        # Tree matches our manual calculation
        manual_calculation = np.array([
            [[0.500, 0.500]], # root (1)
            [[0.955, 0.045]], # left child of 1 (2)
            [[0.305, 0.695]], # right child of 1 (3)
            [[0.152, 0.848]], # left child of 3 (4)
            [[0.580, 0.420]], # right child of 3 (5)
        ])
        np.testing.assert_almost_equal(manual_calculation, i_tree.value, decimal=3)
    
    def test_DGP1_lambda5(self):
        np.random.seed(1)
        X, y = DGP_classification_1(20)

        skmodel = DecisionTreeClassifier(max_leaf_nodes=4)
        skmodel.fit(X, y)
        imodel = HSTreeClassifier(deepcopy(skmodel), reg_param=5)
        i_tree = imodel.estimator_.tree_

        # The probability in each node sums to 1
        np.testing.assert_allclose(np.ones(7), i_tree.value.sum(axis=2).flatten())

        # Tree matches our manual calculation
        manual_calculation = np.array([
            [[0.750, 0.250]], # root (1)
            [[0.950, 0.050]], # left child of 1 (2)
            [[0.283, 0.717]], # right child of 1 (3)
            [[0.465, 0.535]], # left child of 3 (4)
            [[0.192, 0.808]], # right child of 3 (5)
            [[0.608, 0.392]], # left child of 4 (6)
            [[0.322, 0.678]], # right child of 4 (7)
        ])
        np.testing.assert_almost_equal(manual_calculation, i_tree.value, decimal=3)
    
    def test_DGP2_lambda10(self):
        np.random.seed(2)
        X, y = DGP_classification_2(20)

        skmodel = DecisionTreeClassifier(max_leaf_nodes=3)
        skmodel.fit(X, y)
        imodel = HSTreeClassifier(deepcopy(skmodel), reg_param=10)
        i_tree = imodel.estimator_.tree_

        # The probability in each node sums to 1
        np.testing.assert_allclose(np.ones(5), i_tree.value.sum(axis=2).flatten())

        # Tree matches our manual calculation
        manual_calculation = np.array([
            [[0.350, 0.500, 0.150]], # root (1)
            [[0.783, 0.167, 0.050]], # left child of 1 (2)
            [[0.206, 0.611, 0.183]], # right child of 1 (3)
            [[0.226, 0.711, 0.063]], # left child of 3 (4)
            [[0.126, 0.211, 0.663]], # right child of 3 (5)
        ])
        np.testing.assert_almost_equal(manual_calculation, i_tree.value, decimal=3)
    
    def test_DGP2_lambda20(self):
        np.random.seed(3)
        X, y = DGP_classification_2(100)

        skmodel = DecisionTreeClassifier(max_leaf_nodes=4)
        skmodel.fit(X, y)
        imodel = HSTreeClassifier(deepcopy(skmodel), reg_param=20)
        i_tree = imodel.estimator_.tree_

        # The probability in each node sums to 1
        np.testing.assert_allclose(np.ones(7), i_tree.value.sum(axis=2).flatten())

        # Tree matches our manual calculation
        manual_calculation = np.array([
            [[0.210, 0.560, 0.230]], # root (1)
            [[0.749, 0.212, 0.038]], # left child of 1 (2)
            [[0.067, 0.652, 0.281]], # right child of 1 (3)
            [[0.088, 0.846, 0.066]], # left child of 3 (4)
            [[0.036, 0.383, 0.581]], # right child of 3 (5)
            [[0.036, 0.449, 0.514]], # left child of 5 (6)
            [[0.036, 0.175, 0.788]], # right child of 5 (7)
        ])
        np.testing.assert_almost_equal(manual_calculation, i_tree.value, decimal=3)


if __name__ == "__main__":
    unittest.main()