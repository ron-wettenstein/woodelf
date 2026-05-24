import time

import numpy as np
import pytest
import shap

from shared_fixtures_and_utils import testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22, assert_shap_package_is_same_as_woodelf, \
    assert_shap_package_is_same_as_woodelf_on_interaction_values
from woodelf.core.cube_metric import ShapleyValues, BanzhafValues, ShapleyInteractionValues
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode, DecisionTreesEnsemble
from woodelf.hybrid_woodelf import hybrid_woodelf

from woodelf.simple_woodelf import calculate_path_dependent_metric

FIXTURES = [testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22]

TOLERANCE = 0.00001

def test_linear_tree_shap_on_a_model(testset, xgb_model):

    simple_woodelf_shap_values = calculate_path_dependent_metric(
        xgb_model, testset, metric=ShapleyValues()
    )

    vectorized_linear_tree_shap_values = hybrid_woodelf(
        xgb_model, testset, None, ShapleyValues(), GPU=False, use_sparse_approaches=True
    )

    for feature in simple_woodelf_shap_values:
        np.testing.assert_allclose(
            simple_woodelf_shap_values[feature], vectorized_linear_tree_shap_values[feature], atol=0.00001
        )


def test_linear_tree_banzhaf_on_a_model(testset, xgb_model):

    simple_woodelf_shap_values = calculate_path_dependent_metric(
        xgb_model, testset, metric=BanzhafValues()
    )

    vectorized_linear_tree_shap_values = hybrid_woodelf(
        xgb_model, testset, None, BanzhafValues(), GPU=False, use_sparse_approaches=True
    )

    for feature in simple_woodelf_shap_values:
        np.testing.assert_allclose(
            simple_woodelf_shap_values[feature], vectorized_linear_tree_shap_values[feature], atol=TOLERANCE
        )

def test_linear_tree_shap_on_high_depth_models(testset, xgb_model_depth_16, xgb_model_depth_22):
    for model in [xgb_model_depth_16, xgb_model_depth_22]:

        explainer = shap.TreeExplainer(model)
        shap_package_values = explainer.shap_values(testset)

        linear_tree_shap_values = hybrid_woodelf(
            model, testset, None, ShapleyValues(), GPU=False, use_sparse_approaches=True
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values, shap_package_values, testset, TOLERANCE)

        linear_tree_shap_values_neighbor_leaf_trick = hybrid_woodelf(
            model, testset, None, ShapleyValues(), GPU=False, use_neighbor_leaf_trick=True, use_sparse_approaches=True
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values_neighbor_leaf_trick, shap_package_values, testset, TOLERANCE)


def test_single_leaf_tree(testset):

    leaf = DecisionTreeNode(feature_name=None, value=5, right=None, left=None, index=0, cover=1)
    leaf.depth = 1
    leaf.parent = None
    single_leaf_tree = DecisionTreesEnsemble(trees=[leaf])

    values = hybrid_woodelf(
        single_leaf_tree, testset, None, ShapleyValues(), model_was_loaded=True, use_sparse_approaches=True
    )
    for feature in values:
        assert np.sum(np.abs(values[feature])) == 0


def test_linear_tree_shap_iv_on_high_depth_models(testset, xgb_model):
    testset_head = testset.head(20)
    linear_tree_shap_iv_values = hybrid_woodelf(
        xgb_model, testset_head, None, ShapleyInteractionValues(), GPU=False, use_neighbor_leaf_trick=False, use_sparse_approaches=True
    )
    start_time = time.time()
    explainer = shap.TreeExplainer(xgb_model)
    shap_iv_package_values = explainer.shap_interaction_values(testset_head)
    print(time.time() - start_time)

    start_time = time.time()
    assert_shap_package_is_same_as_woodelf_on_interaction_values(linear_tree_shap_iv_values, shap_iv_package_values, testset_head, TOLERANCE)
    print(time.time() - start_time)


def test_lts_on_high_depth_models(testset, xgb_model_depth_16, xgb_model_depth_22):
    for model in [xgb_model_depth_16, xgb_model_depth_22]:

        explainer = shap.TreeExplainer(model)
        shap_package_values = explainer.shap_values(testset)

        linear_tree_shap_values = hybrid_woodelf(
            model, testset, None, ShapleyValues(), GPU=False, use_sparse_approaches=True
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values, shap_package_values, testset, TOLERANCE)

        linear_tree_shap_values_neighbor_leaf_trick = hybrid_woodelf(
            model, testset, None, ShapleyValues(), GPU=False, use_neighbor_leaf_trick=True, use_sparse_approaches=True
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values_neighbor_leaf_trick, shap_package_values, testset, TOLERANCE)
