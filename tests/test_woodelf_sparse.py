import numpy as np
import shap

from shared_fixtures_and_utils import trainset, testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22, \
    assert_shap_package_is_same_as_woodelf, assert_shap_package_is_same_as_woodelf_on_interaction_values
from woodelf.core.cube_metric import ShapleyValues, BanzhafValues, ShapleyInteractionValues
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode, DecisionTreesEnsemble
from woodelf.woodelf_sparse import woodelf_sparse
from woodelf.simple_woodelf import calculate_path_dependent_metric, calculate_background_metric

FIXTURES = [trainset, testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22]

TOLERANCE = 0.00001

def test_linear_tree_shap_on_a_model(testset, xgb_model):

    simple_woodelf_shap_values = calculate_path_dependent_metric(
        xgb_model, testset, metric=ShapleyValues()
    )

    vectorized_linear_tree_shap_values = woodelf_sparse(
        xgb_model, testset, None, ShapleyValues(), GPU=False
    )

    for feature in simple_woodelf_shap_values:
        np.testing.assert_allclose(
            simple_woodelf_shap_values[feature], vectorized_linear_tree_shap_values[feature], atol=0.00001
        )


def test_linear_tree_banzhaf_on_a_model(testset, xgb_model):

    simple_woodelf_shap_values = calculate_path_dependent_metric(
        xgb_model, testset, metric=BanzhafValues()
    )

    vectorized_linear_tree_shap_values = woodelf_sparse(
        xgb_model, testset, None, BanzhafValues(), GPU=False
    )

    for feature in simple_woodelf_shap_values:
        np.testing.assert_allclose(
            simple_woodelf_shap_values[feature], vectorized_linear_tree_shap_values[feature], atol=TOLERANCE
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


def test_mn_background_shap_on_a_model(trainset, testset, xgb_model):

    simple_woodelf_values = calculate_background_metric(
        xgb_model, testset, trainset, metric=ShapleyValues()
    )

    mn_values = woodelf_sparse(
        xgb_model, testset, trainset, ShapleyValues(), GPU=False
    )

    for feature in simple_woodelf_values:
        np.testing.assert_allclose(
            simple_woodelf_values[feature], mn_values[feature], atol=TOLERANCE
        )


def test_mn_background_banzhaf_on_a_model(trainset, testset, xgb_model):

    simple_woodelf_values = calculate_background_metric(
        xgb_model, testset, trainset, metric=BanzhafValues()
    )

    mn_values = woodelf_sparse(
        xgb_model, testset, trainset, BanzhafValues(), GPU=False
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

