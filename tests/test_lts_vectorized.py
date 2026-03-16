import numpy as np
import pytest
import shap

from shared_fixtures_and_utils import testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22, assert_shap_package_is_same_as_woodelf, \
    assert_shap_package_is_same_as_woodelf_on_interaction_values
from woodelf.cube_metric import ShapleyValues, BanzhafValues
from woodelf.lts_vectorized import vectorized_linear_tree_shap, LinearTreeShapPathToMatrices, LinearTreeShapPathToMatricesSimple, \
    LinearTreeShapPathToMatricesImproved
from woodelf.simple_woodelf import calculate_path_dependent_metric

FIXTURES = [testset, xgb_model, xgb_model_depth_16, xgb_model_depth_22]

TOLERANCE = 0.00001

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
        xgb_model, testset, is_shapley=False, is_banzhaf=True, GPU=False
    )

    for feature in simple_woodelf_shap_values:
        np.testing.assert_allclose(
            simple_woodelf_shap_values[feature], vectorized_linear_tree_shap_values[feature], atol=TOLERANCE
        )

def test_linear_tree_shap_on_high_depth_models(testset, xgb_model_depth_16, xgb_model_depth_22):
    for model in [xgb_model_depth_16, xgb_model_depth_22]:

        explainer = shap.TreeExplainer(model)
        shap_package_values = explainer.shap_values(testset)

        linear_tree_shap_values = vectorized_linear_tree_shap(
            model, testset, is_shapley=True, GPU=False
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values, shap_package_values, testset, TOLERANCE)

        linear_tree_shap_values_neighbor_leaf_trick = vectorized_linear_tree_shap(
            model, testset, is_shapley=True, GPU=False, use_neighbor_leaf_trick=True
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values_neighbor_leaf_trick, shap_package_values, testset, TOLERANCE)


def test_linear_tree_shap_iv_on_high_depth_models(testset, xgb_model_depth_16, xgb_model_depth_22):
    for model in [xgb_model_depth_16, xgb_model_depth_22]:

        linear_tree_shap_iv_values = vectorized_linear_tree_shap(
            model, testset, is_shapley=True, GPU=False, is_interaction_values=True, use_neighbor_leaf_trick=False
        )

        explainer = shap.TreeExplainer(model)
        shap_iv_package_values = explainer.shap_interaction_values(testset)

        # for (f1, f2) in linear_tree_shap_iv_values:
        #     linear_tree_shap_iv_values[(f1, f2)] = linear_tree_shap_iv_values[(f1, f2)] / 100

        assert_shap_package_is_same_as_woodelf_on_interaction_values(linear_tree_shap_iv_values, shap_iv_package_values, testset, TOLERANCE)


@pytest.mark.parametrize("p2m_class", [LinearTreeShapPathToMatrices, LinearTreeShapPathToMatricesSimple, LinearTreeShapPathToMatricesImproved])
def test_lts_on_different_ploy_mult_algos_on_high_depth_models(testset, xgb_model_depth_16, xgb_model_depth_22, p2m_class):
    for model in [xgb_model_depth_16, xgb_model_depth_22]:

        explainer = shap.TreeExplainer(model)
        shap_package_values = explainer.shap_values(testset)

        linear_tree_shap_values = vectorized_linear_tree_shap(
            model, testset, is_shapley=True, GPU=False, p2m_class=p2m_class
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values, shap_package_values, testset, TOLERANCE)

        linear_tree_shap_values_neighbor_leaf_trick = vectorized_linear_tree_shap(
            model, testset, is_shapley=True, GPU=False, use_neighbor_leaf_trick=True, p2m_class=p2m_class
        )
        assert_shap_package_is_same_as_woodelf(linear_tree_shap_values_neighbor_leaf_trick, shap_package_values, testset, TOLERANCE)
