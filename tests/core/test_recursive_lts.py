import numpy as np
import pytest

from woodelf.core.cube_metric import ShapleyValues
from woodelf.core.path_to_s_vectors.archive.lts_polynomial_multiplication import linear_tree_shap_magic, linear_tree_shap_magic_for_neighbors, \
    linear_tree_shap_division_forward_for_neighbors
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import continue_P_compute, compute_P, linear_tree_shap_magic_for_banzhaf, \
    improved_linear_tree_shap_magic_for_neighbors, shapley_values_f_w, improved_linear_tree_shap_magic, banzhaf_values_f_w, \
    LTSRecursivePathToSVectors, AlwaysParticipatingLTSPathToSVectors
from woodelf.core.utils import bits_matrix


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


def test_linear_tree_shap_magic():
    leaf_weight = 5
    shap_matrix = linear_tree_shap_magic(
        r= np.array([0.5, 0.6, 0.9]), p=np.array([2, 4, 5, 6, 7]).astype(np.uint16),
        f_w=np.array([[1/3], [1/6], [1/3]]), leaf_weight=leaf_weight
    )
    print(shap_matrix)
    print(shap_matrix.sum(axis=1))
    all_missing_prediction = 0.5*0.6*0.9*leaf_weight

    # Due to the efficiency property, the sum of all the features shapley values of each pattern must be equal to
    # the prediction when all features participate minus the prediction when all features are missing.
    # When the pattern is 7 when all features participate the prediction reaches the leaf and is equal to "leaf_weight"
    # on other patterns the prediction does not reach the leaf and the prediction is 0
    np.testing.assert_allclose(
        shap_matrix.sum(axis=1),
        np.array([0 - all_missing_prediction] * 4 + [leaf_weight - all_missing_prediction])
    )


def test_linear_tree_shap_fast_banzhaf():
    leaf_weight = 5
    original_code_matrix = linear_tree_shap_magic(
        r= np.array([0.5, 0.6, 0.9]), p=np.array([0, 1, 2, 3, 4, 5, 6, 7]).astype(np.uint16),
        f_w=np.array([[0.25], [0.25], [0.25]]), leaf_weight=leaf_weight
    )
    print(original_code_matrix)
    print(original_code_matrix.sum(axis=1))

    fast_code_matrix = linear_tree_shap_magic_for_banzhaf(
        r=np.array([0.5, 0.6, 0.9]), p=np.array([0, 1, 2, 3, 4, 5, 6, 7]).astype(np.uint16),
        leaf_weight=leaf_weight
    )
    print(fast_code_matrix)
    print(fast_code_matrix.sum(axis=1))

    np.testing.assert_allclose(
        original_code_matrix.sum(axis=1),
        fast_code_matrix.sum(axis=1)
    )

@pytest.mark.parametrize("lts_method", [linear_tree_shap_magic_for_neighbors, linear_tree_shap_division_forward_for_neighbors, improved_linear_tree_shap_magic_for_neighbors])
def test_linear_tree_shap_for_neighbors(lts_method):
    for D in [2,3,4,5,10,15,20,25,30,35,40]:
        rng = np.random.default_rng(42)
        left_leaf_weight = 5
        right_leaf_weight = 3
        consumer_size = 10 # Test many consumers while this is still fast
        r = rng.integers(low=1, high=9999, size=D) / 10000
        p = np.concat([rng.integers(low=0, high=(2 ** D) - 1, size=consumer_size), np.array([(2 ** D) - 1])])
        f_w = shapley_values_f_w(D)

        left_shap_matrix_using_neighbors = lts_method(
            r=r, p=p.astype(np.uint64),f_w=f_w, left_leaf_weight=left_leaf_weight, right_leaf_weight=0
        )
        left_shap_matrix = linear_tree_shap_magic(
            r=r, p=p.astype(np.uint64), f_w=f_w, leaf_weight=left_leaf_weight
        )

        tolerance = 0.000001
        np.testing.assert_allclose(
            left_shap_matrix_using_neighbors,
            left_shap_matrix,
            atol=tolerance
        )

        right_shap_matrix_using_neighbors = lts_method(
            r=r, p=p.astype(np.uint64),f_w=f_w, left_leaf_weight=0, right_leaf_weight=right_leaf_weight
        )
        r_of_right = np.array(list(r[:-1]) + [1 - r[-1]])
        p_right = p.copy()
        p_right[p % 2 == 0] += 1
        p_right[p % 2 == 1] -= 1
        right_shap_matrix = linear_tree_shap_magic(
            r=r_of_right, p=p_right.astype(np.uint64), f_w=f_w, leaf_weight=right_leaf_weight
        )

        np.testing.assert_allclose(
            right_shap_matrix_using_neighbors,
            right_shap_matrix,
            atol=tolerance
        )

        shap_matrix_using_neighbors = lts_method(
            r=r, p=p.astype(np.uint64),f_w=f_w, left_leaf_weight=left_leaf_weight, right_leaf_weight=right_leaf_weight
        )

        np.testing.assert_allclose(
            left_shap_matrix_using_neighbors + right_shap_matrix_using_neighbors,
            shap_matrix_using_neighbors,
            atol=tolerance
        )


@pytest.mark.parametrize("D", list(range(1,10)) + list(range(10, 61, 5)) + [36])
def test_linear_tree_shap_magic_longer_high_depth(D):
    rng = np.random.default_rng(42)
    leaf_weight = 5
    consumer_size = 6 # Test many consumers while this is still fast
    # This tends to be numerically unstable when there are many close to 1 ratios
    for low, high in [(1, 10000), (5000, 10000), (9000, 10000), (9990, 10000), (1, 5000), (1, 10)]:
        for i in range(10):
            # generate many random ratios vectors (many r vectors)
            r = rng.integers(low=low, high=high, size=D) / 10000
            p = np.concat([rng.integers(low=0, high=(2 ** D) - 1, size=consumer_size), np.array([(2 ** D) - 1])])
            f_w = shapley_values_f_w(D)


            shap_matrix = improved_linear_tree_shap_magic(r=r, p=p.astype(np.uint64),f_w=f_w, w=leaf_weight)
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


@pytest.mark.parametrize("D", list(range(5, 61, 5)))
def test_linear_tree_shap_fast_banzhaf_many_depths(D):
    # The technics are the same, also - no numerical errors in Banzhaf!
    rng = np.random.default_rng(42)
    leaf_weight = 5
    r = rng.integers(low=1, high=100, size=D) / 100
    p = np.concat([rng.integers(low=0, high=(2 ** D) - 2, size=80), np.array([(2 ** D) - 1])])
    f_w = banzhaf_values_f_w(D)
    original_code_matrix = linear_tree_shap_magic(
        r=r, p=p.astype(np.uint64),f_w=f_w, leaf_weight=leaf_weight
    )

    fast_code_matrix = linear_tree_shap_magic_for_banzhaf(
        r=r, p=p.astype(np.uint64), leaf_weight=leaf_weight
    )

    np.testing.assert_allclose(
        original_code_matrix.sum(axis=1),
        fast_code_matrix.sum(axis=1),
        rtol=10**-7
    )


@pytest.mark.parametrize("features_in_path,always_participating", [
    (["f0"], []),
    (["f0"], ["f0"]),
    (["f0", "f1", "f2", "f3"], []),
    (["f0", "f1", "f2", "f3"], ["f0", "f3"]),
    (["f0", "f1", "f2", "f3"], ["f1", "f2"]),
    (["f0", "f1", "f2", "f3"], ["f0", "f1", "f2", "f3"]),
    (["f0", "f1", "f2", "f3", "f4", "f5"], ["f1", "f4"]),
])
def test_always_participating_efficiency_axiom(features_in_path, always_participating):
    """
    Efficiency: for each consumer pattern p, the sum of ShapleyValues across all features equals
    v(all, p) - v(always_participating_only, p).

    When all consumer_pattern bits are 1 (consumer reaches the leaf):
      v(all)                  = w
      v(always_participating) = prod_{i in regular} r_i * w
    Efficiency: sum of Shapley values = w - prod_{regular} r_i * w.
    """
    rng = np.random.default_rng(42)
    D = len(features_in_path)
    always_participating_set = set(always_participating)
    regular_indices = [i for i, f in enumerate(features_in_path) if f not in always_participating_set]

    covers = rng.integers(low=1, high=9999, size=D) / 10000
    w = 3.7

    all_ones = np.array([2 ** D - 1], dtype=np.uint64)
    p2s = AlwaysParticipatingLTSPathToSVectors(
        metric=ShapleyValues(), max_depth=D, always_participating=always_participating
    )
    result = p2s.get_path_dependent_s_matrix(features_in_path, all_ones, covers, w)

    total = sum(result.get(f, np.zeros(1))[0] for f in features_in_path)

    # v(all) = w (consumer reaches leaf), v(always_participating only) = prod_{regular} r_i * w
    v_regular_empty = np.prod(covers[regular_indices]) * w  # empty prod = 1 when no regular features
    np.testing.assert_allclose(total, w - v_regular_empty, atol=1e-5)


def test_no_always_participating_is_identical_to_base():
    """With an empty always_participating list the result must be identical to LTSRecursivePathToSVectors."""
    D = 5
    rng = np.random.default_rng(42)
    features_in_path = [f"f{i}" for i in range(D)]
    covers = rng.integers(low=1, high=9999, size=D) / 10000
    consumer_patterns = rng.integers(low=0, high=2 ** D, size=20).astype(np.uint64)
    w = 3.7
    metric = ShapleyValues()

    base = LTSRecursivePathToSVectors(metric=metric, max_depth=D)
    restricted = AlwaysParticipatingLTSPathToSVectors(metric=metric, max_depth=D, always_participating=[])

    base_result = base.get_path_dependent_s_matrix(features_in_path, consumer_patterns, covers, w)
    restricted_result = restricted.get_path_dependent_s_matrix(features_in_path, consumer_patterns, covers, w)

    assert set(base_result.keys()) == set(restricted_result.keys())
    for f in base_result:
        np.testing.assert_array_equal(base_result[f], restricted_result[f])
