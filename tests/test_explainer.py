import time

import numpy as np
import pandas as pd
import shap

from woodelf.explainer import WoodelfExplainer
from shared_fixtures_and_utils import testset, trainset, xgb_model, xgb_model_depth_16

FIXTURES = [testset, trainset, xgb_model, xgb_model_depth_16]

TOLERANCE = 0.00001

def test_background_shap_using_shap_package_is_same_as_using_woodelf_explainer(trainset, testset, xgb_model):
    start_time = time.time()
    explainer = shap.TreeExplainer(xgb_model, trainset, feature_perturbation='interventional')
    shap_package_values = explainer.shap_values(testset)
    print("shap took: ", time.time() - start_time)

    start_time = time.time()
    woodelf_explainer = WoodelfExplainer(xgb_model, trainset, feature_perturbation='interventional')
    woodelf_values = woodelf_explainer.shap_values(testset)
    print("woodelf took: ", time.time() - start_time)

    np.testing.assert_allclose(woodelf_values, shap_package_values, atol=TOLERANCE, strict=True)


def test_path_dependent_shap_using_shap_package_is_same_as_using_woodelf_explainer(trainset, testset, xgb_model):
    start_time = time.time()
    explainer = shap.TreeExplainer(xgb_model)
    shap_package_values = explainer.shap_values(testset)
    print("shap took: ", time.time() - start_time)

    start_time = time.time()
    woodelf_explainer = WoodelfExplainer(xgb_model, feature_perturbation='tree_path_dependent')
    woodelf_values = woodelf_explainer.shap_values(testset)
    print("woodelf took: ", time.time() - start_time)

    np.testing.assert_allclose(woodelf_values, shap_package_values, atol=TOLERANCE, strict=True)


# def test_background_shap_iv_using_shap_package_is_same_as_using_woodelf_explainer(trainset, testset, xgb_model):
#     # Not possible as shap package does not support background interaction values


def test_path_dependent_shap_iv_using_shap_package_is_same_as_using_woodelf_explainer(trainset, testset, xgb_model):
    start_time = time.time()
    explainer = shap.TreeExplainer(xgb_model)
    shap_package_values = explainer.shap_interaction_values(testset.head(10))
    print("shap took: ", time.time() - start_time)

    start_time = time.time()
    woodelf_explainer = WoodelfExplainer(xgb_model)
    woodelf_values = woodelf_explainer.shap_interaction_values(testset.head(10))
    print("woodelf took: ", time.time() - start_time)

    woodelf_values_not_including_self = woodelf_values.copy()
    shap_package_values_not_including_self = shap_package_values.copy()
    for i in range(len(trainset.columns)):
        np.testing.assert_allclose(
            woodelf_values_not_including_self[:,i,i], shap_package_values_not_including_self[:,i,i],
            atol=TOLERANCE, strict=True,
            err_msg=f"failed on interaction with itself of feature {list(trainset.columns)[i]}"
        )

    np.testing.assert_allclose(woodelf_values, shap_package_values, atol=TOLERANCE, strict=True)


def test_woodelf_explainer_with_xgboost_model_of_depth_16(trainset, testset, xgb_model_depth_16):
    start_time = time.time()
    explainer = shap.TreeExplainer(xgb_model_depth_16)
    shap_package_values = explainer.shap_values(testset)
    print("shap took: ", time.time() - start_time)

    start_time = time.time()
    woodelf_explainer = WoodelfExplainer(xgb_model_depth_16, feature_perturbation='tree_path_dependent')
    woodelf_values = woodelf_explainer.shap_values(testset)
    print("woodelf took: ", time.time() - start_time)

    np.testing.assert_allclose(woodelf_values, shap_package_values, atol=TOLERANCE, strict=True)

def test_excluding_only_zero_contributions(trainset, testset, xgb_model):
    woodelf_explainer = WoodelfExplainer(xgb_model, trainset, feature_perturbation='tree_path_dependent')
    woodelf_df = woodelf_explainer.shap_values(
        testset.head(5), as_df=True, exclude_zero_contribution_features=False
    )

    woodelf_explainer = WoodelfExplainer(xgb_model, trainset, feature_perturbation='tree_path_dependent')
    woodelf_no_zeros_df = woodelf_explainer.shap_values(
        testset.head(5), as_df=True, exclude_zero_contribution_features=True
    )

    for f in woodelf_df:
        if f not in woodelf_no_zeros_df:
            # assert all zeros
            assert woodelf_df[f].abs().max() == 0


def test_explainer_cache(trainset, testset, xgb_model):
    woodelf_head_5_df_1 = WoodelfExplainer(xgb_model, trainset.head(5), feature_perturbation='interventional').shap_values(
        testset.head(5), as_df=True, exclude_zero_contribution_features=False
    )
    woodelf_tail_5_df_1 = WoodelfExplainer(xgb_model, trainset.tail(5), feature_perturbation='interventional').shap_values(
        testset.head(5), as_df=True, exclude_zero_contribution_features=False
    )

    # When there is no cache one can modify the background data on the explainer, run shap and it will use
    # the new background data (this is for testing, please never actually do it in an important code)
    woodelf_explainer = WoodelfExplainer(
        xgb_model, trainset.head(5), feature_perturbation='interventional', cache_option="no"
    )
    assert not woodelf_explainer.use_cache()
    woodelf_head_5_df_2 = woodelf_explainer.shap_values(
        testset.head(5), as_df=True, exclude_zero_contribution_features=False
    )
    pd.testing.assert_frame_equal(woodelf_head_5_df_1, woodelf_head_5_df_2)

    woodelf_explainer.background_data = trainset.tail(5)
    woodelf_tail_5_df_2 = woodelf_explainer.shap_values(
        testset.head(5), as_df=True, exclude_zero_contribution_features=False
    )
    pd.testing.assert_frame_equal(woodelf_tail_5_df_1, woodelf_tail_5_df_2)

    # When the cache is enabled setting the background data will have no effect as the object
    # already extracted the needed information from the background data and does not need to reuse it
    woodelf_explainer = WoodelfExplainer(
        xgb_model, trainset.head(5), feature_perturbation='interventional', cache_option="yes"
    )
    assert woodelf_explainer.use_cache()
    woodelf_head_5_df_3 = woodelf_explainer.shap_values(
        testset.head(5), as_df=True, exclude_zero_contribution_features=False
    )
    assert woodelf_explainer.cache is not None and woodelf_explainer.cache_filled
    pd.testing.assert_frame_equal(woodelf_head_5_df_1, woodelf_head_5_df_3)

    woodelf_explainer.background_data = trainset.tail(5)
    woodelf_head_5_df_4 = woodelf_explainer.shap_values(
        testset.head(5), as_df=True, exclude_zero_contribution_features=False
    ) # This is the test! return the background shap of head(5) although we just override the background data to tail(5)
    pd.testing.assert_frame_equal(woodelf_head_5_df_1, woodelf_head_5_df_4)


def test_expected_value_property(trainset, testset, xgb_model):
    explainer = shap.TreeExplainer(xgb_model)
    woodelf_explainer = WoodelfExplainer(xgb_model)
    assert abs(explainer.expected_value - woodelf_explainer.expected_value) < TOLERANCE

    explainer = shap.TreeExplainer(xgb_model, trainset, feature_perturbation='interventional')
    woodelf_explainer = WoodelfExplainer(xgb_model, trainset, feature_perturbation='interventional')
    assert abs(explainer.expected_value - woodelf_explainer.expected_value) < TOLERANCE

    explainer = shap.TreeExplainer(xgb_model, trainset)
    woodelf_explainer = WoodelfExplainer(xgb_model, trainset)
    assert abs(explainer.expected_value - woodelf_explainer.expected_value) < TOLERANCE

    # Test object dtype
    trainset["card1"] = trainset["card1"].astype(object)
    woodelf_explainer = WoodelfExplainer(xgb_model, trainset)
    assert abs(explainer.expected_value - woodelf_explainer.expected_value) < TOLERANCE


def test_call(trainset, testset, xgb_model):
    # TODO slightly different base_values results on path-dependent
    explainer = shap.TreeExplainer(xgb_model, trainset, feature_perturbation='interventional')
    shap_package_explanation = explainer(testset)
    woodelf_explainer = WoodelfExplainer(xgb_model, trainset, feature_perturbation='interventional')
    woodelf_explanation = woodelf_explainer(testset)

    np.testing.assert_allclose(woodelf_explanation.values, shap_package_explanation.values, atol=TOLERANCE, strict=True)
    np.testing.assert_allclose(
        woodelf_explanation.base_values, shap_package_explanation.base_values, atol=TOLERANCE, strict=True
    )
    np.testing.assert_allclose(
        woodelf_explanation.data, shap_package_explanation.data, atol=TOLERANCE, strict=True
    )
    assert woodelf_explanation.feature_names == shap_package_explanation.feature_names
