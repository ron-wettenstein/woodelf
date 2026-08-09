import numpy as np
import pytest
import shap

from shared_fixtures_and_utils import trainset, testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22, \
    assert_shap_package_is_same_as_woodelf, assert_shap_package_is_same_as_woodelf_on_interaction_values
from woodelf.core.cube_metric import ShapleyValues, BanzhafValues, ShapleyInteractionValues, \
    GeneralShapleyInteractionValues, GeneralBanzhafInteractionValues, BanzhafInteractionValues
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode, DecisionTreesEnsemble
from woodelf.high_depth_woodelf import woodelf_for_high_depth
from woodelf.woodelf_sparse import woodelf_sparse
from woodelf.simple_woodelf import calculate_path_dependent_metric, calculate_background_metric

FIXTURES = [trainset, testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22]

TOLERANCE = 0.00001

@pytest.mark.parametrize("metric", [ShapleyValues(), BanzhafValues()], ids=["shapley", "banzhaf"])
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

