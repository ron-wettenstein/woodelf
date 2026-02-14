import time

import numpy as np
import pytest
import shap

from shared_fixtures_and_utils import testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22, assert_shap_package_is_same_as_woodelf
from woodelf.cube_metric import ShapleyValues, BanzhafValues
from woodelf.simple_woodelf import calculate_path_dependent_metric
from woodelf.vectorized_linear_tree_shap import linear_tree_shap_magic, shapley_values_f_w, \
    linear_tree_shap_magic_for_banzhaf, banzhaf_values_f_w, vectorized_linear_tree_shap, \
    linear_tree_shap_magic_not_numerically_stable

FIXTURES = [testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22]

TOLERANCE = 0.00001


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



@pytest.mark.parametrize("D", list(range(5, 41, 5)))
def test_linear_tree_shap_magic_high_depth(D):
    # TODO fix numerical problems in high depths and low covers
    rng = np.random.default_rng(42)
    leaf_weight = 5
    # Use minimum 0.7 cover so the np.prod(r) will stay large (the all_missing_prediction uses it)
    r = rng.integers(low=70, high=100, size=D) / 100
    p = np.concat([rng.integers(low=0, high=(2 ** D) - 2, size=80), np.array([(2 ** D) - 1])])
    f_w = shapley_values_f_w(D)

    shap_matrix = linear_tree_shap_magic(
        r=r, p=p.astype(np.uint64),f_w=f_w, leaf_weight=leaf_weight
    )
    print(shap_matrix)
    print(shap_matrix.sum(axis=1))
    all_missing_prediction = np.prod(r)*leaf_weight

    # Due to the efficiency property, the sum of all the features shapley values of each pattern must be equal to
    # the prediction when all features participate minus the prediction when all features are missing.
    # When the pattern is 7 when all features participate the prediction reaches the leaf and is equal to "leaf_weight"
    # on other patterns the prediction does not reach the leaf and the prediction is 0
    tolerance = 0.000001 if D <= 30 else 0.0001 # Numerical problems after 30
    np.testing.assert_allclose(
        shap_matrix.sum(axis=1),
        np.array([0 - all_missing_prediction] * 80 + [leaf_weight - all_missing_prediction]),
        atol=tolerance
    )


@pytest.mark.parametrize("D", list(range(5, 61, 5)))
def test_linear_tree_shap_magic_longer_high_depth(D):
    rng = np.random.default_rng(42)
    leaf_weight = 5
    r = rng.integers(low=1, high=99, size=D) / 100
    p = np.concat([rng.integers(low=0, high=(2 ** D) - 2, size=80), np.array([(2 ** D) - 1])])
    f_w = shapley_values_f_w(D)

    shap_matrix = linear_tree_shap_magic(
        r=r, p=p.astype(np.uint64),f_w=f_w, leaf_weight=leaf_weight
    )
    print(shap_matrix)
    print(shap_matrix.sum(axis=1))
    all_missing_prediction = np.prod(r)*leaf_weight

    # Due to the efficiency property, the sum of all the features shapley values of each pattern must be equal to
    # the prediction when all features participate minus the prediction when all features are missing.
    # When the pattern is 7 when all features participate the prediction reaches the leaf and is equal to "leaf_weight"
    # on other patterns the prediction does not reach the leaf and the prediction is 0
    tolerance = 0.000001
    np.testing.assert_allclose(
        shap_matrix.sum(axis=1),
        np.array([0 - all_missing_prediction] * 80 + [leaf_weight - all_missing_prediction]),
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


# @pytest.mark.parametrize("D", list(range(5, 61, 5)))
# def test_linear_tree_shap_fast_banzhaf_many_depths_timings(D):
#     # Testing the timing of the linear_tree_shap_magic_for_banzhaf and linear_tree_shap_magic functions
#     rng = np.random.default_rng(42)
#     leaf_weight = 5
#     r = rng.integers(low=70, high=100, size=D) / 100
#     p = np.concat([rng.integers(low=0, high=(2 ** D) - 2, size=100000), np.array([(2 ** D) - 1])])
#
#     # f_w = banzhaf_values_f_w(D)
#     # original_code_matrix = linear_tree_shap_magic(
#     #     r=r, p=p.astype(np.uint64),f_w=f_w, leaf_weight=leaf_weight
#     # )
#
#     fast_code_matrix = linear_tree_shap_magic_for_banzhaf(
#         r=r, p=p.astype(np.uint64), leaf_weight=leaf_weight
#     )


def test_linear_tree_shap_on_a_model(testset, xgb_model):

    simple_woodelf_shap_values = calculate_path_dependent_metric(
        xgb_model, testset, metric=ShapleyValues()
    )

    vectorized_linear_tree_shap_values = vectorized_linear_tree_shap(
        xgb_model, testset, is_shapley=True, GPU=False
    )

    for feature in simple_woodelf_shap_values:
        np.testing.assert_allclose(
            simple_woodelf_shap_values[feature], vectorized_linear_tree_shap_values[feature], atol=0.00001
        )


def test_linear_tree_banzhaf_on_a_model(testset, xgb_model):

    simple_woodelf_shap_values = calculate_path_dependent_metric(
        xgb_model, testset, metric=BanzhafValues()
    )

    vectorized_linear_tree_shap_values = vectorized_linear_tree_shap(
        xgb_model, testset, is_shapley=False, GPU=False
    )

    for feature in simple_woodelf_shap_values:
        np.testing.assert_allclose(
            simple_woodelf_shap_values[feature], vectorized_linear_tree_shap_values[feature], atol=TOLERANCE
        )

def test_linear_tree_shap_on_high_depth_models(testset, xgb_model_depth_16, xgb_model_depth_22):
    for model in [xgb_model_depth_16, xgb_model_depth_22]:
        start_time = time.time()
        explainer = shap.TreeExplainer(model)
        shap_package_values = explainer.shap_values(testset)
        print("shap took: ", time.time() - start_time)

        print(testset.shape)
        start_time = time.time()
        linear_tree_shap_values = vectorized_linear_tree_shap(
            model, testset, is_shapley=True, GPU=False
        )
        print("high depth woodelf took: ", time.time() - start_time)

        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values, shap_package_values, testset, TOLERANCE)
