import math

import numpy as np
import pytest

from woodelf.core.cube_metric import ShapleyValues
from woodelf.core.path_to_s_vectors.archive.lts_polynomial_multiplication import quadrature_tree_shap_batched_approach, linear_tree_shap_magic_blocked
from woodelf.core.path_to_s_vectors.archive.quadrature_shap_p2s import QuadratureSHAPPathToSVectors
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import shapley_values_f_w, improved_linear_tree_shap_magic


@pytest.mark.parametrize("D", list(range(5, 61, 5)))
def test_linear_tree_shap_magic_blocked(D):
    rng = np.random.default_rng(42)
    leaf_weight = 5
    consumer_size = 6 # Test many consumers while this is still fast
    r = rng.integers(low=1, high=9999, size=D) / 10000
    p = np.concat([rng.integers(low=0, high=(2 ** D) - 2, size=consumer_size), np.array([(2 ** D) - 1])])
    f_w = shapley_values_f_w(D)

    shap_matrix = improved_linear_tree_shap_magic(
        r=r, p=p.astype(np.uint64),f_w=f_w, w=leaf_weight
    )
    shap_matrix_using_blocks = linear_tree_shap_magic_blocked(
        r=r, p=p.astype(np.uint64),f_w=f_w, leaf_weight=leaf_weight
    )
    tolerance = 0.000001
    np.testing.assert_allclose(
        shap_matrix_using_blocks, shap_matrix, atol=tolerance
    )


@pytest.mark.parametrize("D", list(range(1,10)) + list(range(10, 61, 5)) + [36])
def test_quadrature_shap_high_depth(D):
    rng = np.random.default_rng(42)
    leaf_weight = 5
    consumer_size = 6 # Test many consumers while this is still fast

    p2m = QuadratureSHAPPathToSVectors(ShapleyValues(), max_depth=D)
    required_n_quads = min(max(int(math.ceil(D / 2)), 2), 16)
    # This tends to be numerically unstable when there are many close to 1 ratios
    for low, high in [(1, 10000), (5000, 10000), (9000, 10000), (9990, 10000), (1, 5000), (1, 10)]:
        for i in range(10):
            # generate many random ratios vectors (many r vectors)
            r = rng.integers(low=low, high=high, size=D) / 10000
            p = np.concat([rng.integers(low=0, high=(2 ** D) - 1, size=consumer_size), np.array([(2 ** D) - 1])])

            shap_matrix = quadrature_tree_shap_batched_approach(
                r=r, p=p, leaf_value=leaf_weight,
                quad_nodes=p2m.quad_nodes[required_n_quads], quad_weights=p2m.quad_weights[required_n_quads]
            )
            all_missing_prediction = np.prod(r)*leaf_weight

            # Due to the efficiency property, the sum of all the features shapley values of each pattern must be equal to
            # the prediction when all features participate minus the prediction when all features are missing.
            # When the pattern is 7 when all features participate the prediction reaches the leaf and is equal to "leaf_weight"
            # on other patterns the prediction does not reach the leaf and the prediction is 0
            tolerance = 0.000001
            np.testing.assert_allclose(
                shap_matrix.sum(axis=1),
                np.array([0 - all_missing_prediction] * consumer_size + [leaf_weight - all_missing_prediction]),
                atol=tolerance
            )
