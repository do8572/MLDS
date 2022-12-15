import unittest

from copy import deepcopy
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from imodels import HSTreeClassifier


def DGP_classification_1(n: int = 10, m: int = 2):
    # y = x1 > 2
    x = np.random.uniform(0, 4, (n, m))
    y = (x[:,0] > 2)*1
    x += np.random.normal(0, 1, (n, m))
    return x, y

def DGP_classification_2(n: int = 10, m: int = 2):
    # y := 2 ; x1 > 3
    #      1 ; 3 >= x1 > 1
    #      0 ; 1 >= x1
    x = np.random.uniform(0, 4, (n, m))
    y = (1 <= x[:,0])*1 + (3 <= x[:,0])*1
    x += np.random.normal(0, 0.5, (n, m))
    return x, y

class TestDGPClassification(unittest.TestCase):

    def test_tree_shape(self):
        X, y = DGP_classification_1(20)

        for max_leaf_nodes in [3, 5, 7, 9]:
            for reg_param in [0.1, 1, 10, 100]:
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
            [[0.580, 0.420]] # right child of 3 (5)
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
            [[0.322, 0.678]] # right child of 4 (7)
        ])
        np.testing.assert_almost_equal(manual_calculation, i_tree.value, decimal=3)


if __name__ == "__main__":
    unittest.main()