from itertools import combinations
from math import comb

import numpy as np
import pytest
import shap

from shared_fixtures_and_utils import trainset, testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22, \
    assert_shap_package_is_same_as_woodelf, assert_shap_package_is_same_as_woodelf_on_interaction_values
from woodelf.core.cube_metric import ShapleyValues, BanzhafValues, ShapleyInteractionValues, \
    GeneralShapleyInteractionValues, GeneralBanzhafInteractionValues, BanzhafInteractionValues
from woodelf.core.cube_metric import (
    MobiusCoefficients, FaithfulShapleyInteractionValues, FaithfulBanzhafInteractionValues
)
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode, DecisionTreesEnsemble
from woodelf.high_depth_woodelf import woodelf_for_high_depth
from woodelf.woodelf_sparse import woodelf_sparse, hybrid_woodelf
from woodelf.simple_woodelf import calculate_path_dependent_metric, calculate_background_metric

FIXTURES = [trainset, testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22]

TOLERANCE = 0.00001

@pytest.mark.parametrize("metric",
                         [ShapleyValues(), BanzhafValues()],
                         ids=["shapley", "banzhaf"])
def test_linear_tree_metric_on_a_model(testset, xgb_model, metric):

    simple_woodelf_values = calculate_path_dependent_metric(
        xgb_model, testset, metric=metric
    )

    vectorized_linear_tree_values = woodelf_sparse(
        xgb_model, testset, None, metric, GPU=False
    )

    for feature in simple_woodelf_values:
        np.testing.assert_allclose(
            simple_woodelf_values[feature], vectorized_linear_tree_values[feature], atol=TOLERANCE
        )

@pytest.mark.parametrize("metric",
                         [ShapleyInteractionValues(), GeneralShapleyInteractionValues(3, 3)],
                         ids=["shapley_iv", "shapley_order_3"])
def test_hybrid_woodelf_on_path_dependent_iv_metric(testset, xgb_model, metric):

    woodelf_hd_values = woodelf_for_high_depth(
        xgb_model, testset, background_data=None, metric=metric
    )

    vectorized_linear_tree_values = hybrid_woodelf(
        xgb_model, testset, None, metric, GPU=False
    )

    for feature in woodelf_hd_values:
        np.testing.assert_allclose(                                                                            
            woodelf_hd_values[feature], vectorized_linear_tree_values[feature], atol=TOLERANCE
        )


def test_hybrid_woodelf_on_path_dependent_iv_metric_order_1_and_2_together(testset, xgb_model):

    woodelf_hd_values = woodelf_for_high_depth(
        xgb_model, testset, background_data=None, metric=ShapleyValues()
    )

    woodelf_hd_iv = woodelf_for_high_depth(
        xgb_model, testset, background_data=None, metric=ShapleyInteractionValues()
    )

    vectorized_linear_tree_values = hybrid_woodelf(
        xgb_model, testset, None, GeneralShapleyInteractionValues(1, 2, shap_convention=True), GPU=False
    )

    for features in vectorized_linear_tree_values:
        if len(features) == 1:
            np.testing.assert_allclose(
                woodelf_hd_values[features[0]], vectorized_linear_tree_values[features], atol=TOLERANCE
            )
        else:
            np.testing.assert_allclose(
                woodelf_hd_iv[features], vectorized_linear_tree_values[features], atol=TOLERANCE
            )

def test_linear_tree_shap_on_high_depth_models(testset, xgb_model_depth_16, xgb_model_depth_22):
    for model in [xgb_model_depth_16, xgb_model_depth_22]:

        explainer = shap.TreeExplainer(model)
        shap_package_values = explainer.shap_values(testset)

        linear_tree_shap_values = woodelf_sparse(
            model, testset, None, ShapleyValues(), GPU=False
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values, shap_package_values, testset, TOLERANCE)

        linear_tree_shap_values_neighbor_leaf_trick = woodelf_sparse(
            model, testset, None, ShapleyValues(), GPU=False, use_neighbor_leaf_trick=True
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values_neighbor_leaf_trick, shap_package_values, testset, TOLERANCE)


@pytest.mark.parametrize("metric", [ShapleyValues(), BanzhafValues()], ids=["shapley", "banzhaf"])
def test_mn_background_metric_on_a_model(trainset, testset, xgb_model, metric):

    simple_woodelf_values = calculate_background_metric(
        xgb_model, testset, trainset, metric=metric
    )

    mn_values = woodelf_sparse(
        xgb_model, testset, trainset, metric, GPU=False
    )

    for feature in simple_woodelf_values:
        np.testing.assert_allclose(
            simple_woodelf_values[feature], mn_values[feature], atol=TOLERANCE
        )


def test_mn_background_shap_on_high_depth_models(trainset, testset, xgb_model_depth_16, xgb_model_depth_22):
    background = trainset.head(10)
    for model in [xgb_model_depth_16, xgb_model_depth_22]:

        explainer = shap.TreeExplainer(model, background, feature_perturbation='interventional')
        shap_package_values = explainer.shap_values(testset)

        mn_values = woodelf_sparse(
            model, testset, background, ShapleyValues(), GPU=False
        )
        assert_shap_package_is_same_as_woodelf(mn_values, shap_package_values, testset, TOLERANCE)


@pytest.mark.parametrize("metric", [
    ShapleyInteractionValues(),
    BanzhafInteractionValues(),
    ShapleyValues(),
    BanzhafValues(),
    GeneralShapleyInteractionValues(3, 3)
], ids=["shap_iv", "banzhaf_iv", "shap", "banzhaf", "order_3_shap"])
def test_mn_background_vs_woodelf_hd(trainset, testset, xgb_model, metric):
    cii_values = woodelf_sparse(xgb_model, testset, trainset, metric)
    dense_values = woodelf_for_high_depth(xgb_model, testset, trainset, metric)

    assert set(cii_values) == set(dense_values)
    for key in set(cii_values):
        np.testing.assert_allclose(cii_values[key], dense_values[key], atol=TOLERANCE)


def test_mn_background_neighbor_leaf_trick_consistency(trainset, testset, xgb_model_depth_16):
    metric = GeneralBanzhafInteractionValues(1, 2)

    values_with_trick = woodelf_sparse(
        xgb_model_depth_16, testset, trainset, metric, use_neighbor_leaf_trick=True
    )
    values_without_trick = woodelf_sparse(
        xgb_model_depth_16, testset, trainset, metric, use_neighbor_leaf_trick=False
    )

    zeros = np.zeros(len(testset))
    for key in set(values_with_trick) | set(values_without_trick):
        np.testing.assert_allclose(
            values_with_trick.get(key, zeros), values_without_trick.get(key, zeros), atol=TOLERANCE
        )


def test_single_leaf_tree(testset):

    leaf = DecisionTreeNode(feature_name=None, value=5, right=None, left=None, index=0, cover=1)
    leaf.depth = 1
    leaf.parent = None
    single_leaf_tree = DecisionTreesEnsemble(trees=[leaf])

    values = woodelf_sparse(
        single_leaf_tree, testset, None, ShapleyValues(), model_was_loaded=True
    )
    for feature in values:
        assert np.sum(np.abs(values[feature])) == 0


def test_linear_tree_shap_iv_on_high_depth_models(testset, xgb_model):
    testset_head = testset.head(20)
    linear_tree_shap_iv_values = woodelf_sparse(
        xgb_model, testset_head, None, ShapleyInteractionValues(), GPU=False, use_neighbor_leaf_trick=False
    )
    explainer = shap.TreeExplainer(xgb_model)
    shap_iv_package_values = explainer.shap_interaction_values(testset_head)

    assert_shap_package_is_same_as_woodelf_on_interaction_values(linear_tree_shap_iv_values, shap_iv_package_values, testset_head, TOLERANCE)


def fsii_mobius_weight(t, order, max_order):
    """
    The weight the FSII Mobius representation gives m_T, for |T| = t > max_order and a subset of size
    order (Proposition A.13's proof in arXiv:2605.22738):
        phi_S = m_S + (-1)^(k-s) s/(k+s) C(k,s) sum over T strictly containing S, |T|>k, of
                C(|T|-1, k) / C(|T|+k-1, k+s) m_T
    """
    return (
        ((-1) ** (max_order - order)) * (order / (max_order + order)) * comb(max_order, order)
        * comb(t - 1, max_order) / comb(t + max_order - 1, max_order + order)
    )


def fbii_mobius_weight(t, order, max_order):
    """
    The weight the FBII Mobius representation gives m_T, for |T| = t > max_order and a subset of size
    order (Proposition A.12's proof in arXiv:2605.22738):
        phi_S = m_S + (-1)^(k-s) sum over T strictly containing S, |T|>k, of
                (1/2)^(|T|-s) C(|T|-s-1, k-s) m_T
    """
    return (
        ((-1) ** (max_order - order)) * (0.5 ** (t - order))
        * comb(t - order - 1, max_order - order)
    )


def faithful_index_from_mobius_coefficients(mobius_values, max_order, mobius_weight):
    """
    Rebuild a faithful interaction index out of the Mobius coefficients of the same game.

    A game is the sum of the unanimity games of its Mobius coefficients, v = sum_T m(T) u_T, and the
    faithful indices are linear in the game, so index(S) = sum over T containing S of m(T) u_T's index.
    A max_order-additive surrogate reproduces u_T exactly whenever |T| <= max_order, so those T give the
    Mobius coefficient itself to S = T and 0 to any other subset - which is the m(S) each subset starts
    from below. The |T| > max_order ones contribute their closed form unanimity weight.
    """
    values = {
        subset: mobius_vector.astype(np.float64)
        for subset, mobius_vector in mobius_values.items() if len(subset) <= max_order
    }
    for subset, mobius_vector in mobius_values.items():
        if len(subset) <= max_order:
            continue
        for order in range(1, max_order + 1):
            weight = mobius_weight(len(subset), order, max_order)
            if weight == 0:
                continue
            for sub_subset in combinations(subset, order):
                values[sub_subset] += weight * mobius_vector
    return values


@pytest.mark.parametrize("max_order", [1, 2], ids=["order_1", "order_2"])
def test_faithful_indices_match_the_ones_rebuilt_from_the_mobius_coefficients(
        trainset, testset, xgb_model, max_order
):
    # The faithful metrics collapse the Mobius sum analytically on every cube, inside the contribution
    # tables of the path-to-s-vectors. This test takes the other route: it computes the Mobius
    # coefficients of the whole ensemble and rebuilds the indices from them subset by subset.
    consumer_data, background_data = testset.head(5), trainset.head(100)
    mobius_values = woodelf_sparse(
        xgb_model, consumer_data, background_data, MobiusCoefficients(1, None)
    )

    for metric, mobius_weight in [
        (FaithfulShapleyInteractionValues(1, max_order), fsii_mobius_weight),
        (FaithfulBanzhafInteractionValues(1, max_order), fbii_mobius_weight),
    ]:
        woodelf_values = woodelf_sparse(xgb_model, consumer_data, background_data, metric)
        rebuilt_values = faithful_index_from_mobius_coefficients(
            mobius_values, max_order, mobius_weight
        )
        assert set(woodelf_values) == set(rebuilt_values)
        for subset in woodelf_values:
            np.testing.assert_allclose(
                woodelf_values[subset], rebuilt_values[subset], atol=TOLERANCE
            )
