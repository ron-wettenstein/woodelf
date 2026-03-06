import time

import numpy as np
import pytest

from woodelf.lts_algo import compute_P, bits_matrix, continue_P_compute, improved_linear_tree_shap_magic, linear_tree_shap_magic_blocked
from woodelf.vectorized_linear_tree_shap import shapley_values_f_w, linear_tree_shap_magic
from woodelf.vectorized_linear_tree_shap import linear_tree_shap_magic_blocked as linear_tree_shap_magic_blocked_orig


def test_continue_P_compute():
    D=10
    rng = np.random.default_rng(42)
    consumer_size = 8 # Test many consumers while this is still fast
    r = rng.integers(low=1, high=9999, size=D) / 10000
    p = np.concat([rng.integers(low=0, high=(2 ** D) - 2, size=consumer_size), np.array([(2 ** D) - 1])])

    q_M = bits_matrix(p.astype(np.uint64), len(r)) * (1 / r.reshape(-1, 1))

    whole_P_way1 = compute_P(r, q_M, start_index=0, end_index=len(r))
    top_P = compute_P(r, q_M, start_index=0, end_index=len(r) // 2)
    bottom_P = compute_P(r, q_M, start_index=len(r) // 2, end_index=len(r))

    whole_P_way2 = continue_P_compute(top_P, q_M, len(r) // 2, len(r))

    tolerance = 0.000001
    np.testing.assert_allclose(
        whole_P_way1, whole_P_way2, atol=tolerance
    )

    whole_P_way3 = continue_P_compute(bottom_P, q_M, 0, len(r) // 2)
    np.testing.assert_allclose(
        whole_P_way1, whole_P_way3, atol=tolerance
    )

    middle_way = continue_P_compute(bottom_P, q_M, 0, len(r) // 4)
    whole_P_way4 = continue_P_compute(middle_way, q_M, len(r) // 4, len(r) // 2)
    np.testing.assert_allclose(
        whole_P_way1, whole_P_way4, atol=tolerance
    )


@pytest.mark.parametrize("D", list(range(5, 61, 5)))
def test_linear_tree_shap_magic_longer_high_depth(D):
    rng = np.random.default_rng(42)
    leaf_weight = 5
    consumer_size = 1000000 // D # Test many consumers while this is still fast
    r = rng.integers(low=1, high=9999, size=D) / 10000
    p = np.concat([rng.integers(low=0, high=(2 ** D) - 2, size=consumer_size), np.array([(2 ** D) - 1])])
    f_w = shapley_values_f_w(D)

    start_time = time.time()
    shap_matrix = linear_tree_shap_magic(
        r=r, p=p.astype(np.uint64),f_w=f_w, leaf_weight=leaf_weight
    )
    print(f"linear_tree_shap_magic took {time.time() - start_time}")
    start_time = time.time()
    improved_shap_matrix = improved_linear_tree_shap_magic(
        r=r, patterns=p.astype(np.uint64),f_w=f_w, w=leaf_weight
    )
    print(f"improved_shap_matrix took {time.time() - start_time}")
    all_missing_prediction = np.prod(r)*leaf_weight

    # Due to the efficiency property, the sum of all the features shapley values of each pattern must be equal to
    # the prediction when all features participate minus the prediction when all features are missing.
    # When the pattern is 7 when all features participate the prediction reaches the leaf and is equal to "leaf_weight"
    # on other patterns the prediction does not reach the leaf and the prediction is 0
    tolerance = 0.000001
    np.testing.assert_allclose(
        improved_shap_matrix.sum(axis=1),
        np.array([0 - all_missing_prediction] * consumer_size + [leaf_weight - all_missing_prediction]),
        atol=tolerance
    )

@pytest.mark.parametrize("D", list(range(5, 61, 5)))
def test_linear_tree_shap_magic_blocked(D):
    rng = np.random.default_rng(42)
    leaf_weight = 5
    consumer_size = 1000000 // D # Test many consumers while this is still fast
    r = rng.integers(low=1, high=9999, size=D) / 10000
    p = np.concat([rng.integers(low=0, high=(2 ** D) - 2, size=consumer_size), np.array([(2 ** D) - 1])])
    f_w = shapley_values_f_w(D)

    start_time = time.time()
    shap_matrix = improved_linear_tree_shap_magic(
        r=r, patterns=p.astype(np.uint64),f_w=f_w, w=leaf_weight
    )
    print(f"improved_shap_matrix took {time.time() - start_time}")
    start_time = time.time()
    shap_matrix_using_blocks = linear_tree_shap_magic_blocked(
        r=r, p=p.astype(np.uint64),f_w=f_w, leaf_weight=leaf_weight
    )
    print(f"linear_tree_shap_magic_blocked took {time.time() - start_time}")
    start_time = time.time()
    shap_matrix_using_blocks = linear_tree_shap_magic_blocked_orig(
        r=r, p=p.astype(np.uint64),f_w=f_w, leaf_weight=leaf_weight
    )
    print(f"linear_tree_shap_magic_blocked_orig took {time.time() - start_time}")
    tolerance = 0.000001
    np.testing.assert_allclose(
        shap_matrix_using_blocks, shap_matrix, atol=tolerance
    )