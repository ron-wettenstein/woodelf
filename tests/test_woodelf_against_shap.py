import os
import time

import numpy as np
import pytest
import shap
import xgboost as xgb
import pandas as pd

from woodelf.cube_metric import ShapleyValues, ShapleyInteractionValues
from woodelf.simple_woodelf import calculate_background_metric, calculate_path_dependent_metric

RESOURCES_PATH = os.path.join(__file__, "..", "resources")
TOLERANCE = 0.00001

XGB_PARAMS = {
    "objective": "reg:squarederror",  # Regression task with mean squared error loss
    "eval_metric": "rmse",  # Evaluation metric is root mean squared error
    "max_depth": 6,  # Maximum depth of each tree
    "learning_rate": 0.1,  # Learning rate (step size shrinkage)
    "subsample": 1,  # Subsample ratio of the training instances
    "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
    "seed": 123,
    "nthread": 1, # The python shap package use parallel in path dependent if the nthread is bigger then 1. Use nthread=1 to compare the approaches when both don't utilize parallelism.
}


def train_xgboost_model(X_train, y_train, params, num_rounds=100):
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    return xgb.train(params, train_dmatrix, num_rounds)

@pytest.fixture
def trainset() -> pd.DataFrame:
    fraud_trainset = pd.read_csv(os.path.join(RESOURCES_PATH, "IEEE-CIS_trainset_sample.csv"))
    return fraud_trainset[[c for c in fraud_trainset.columns if c != 'isFraud' and c != 'Unnamed: 0']]

@pytest.fixture
def testset() -> pd.DataFrame:
    fraud_testset = pd.read_csv(os.path.join(RESOURCES_PATH, "IEEE-CIS_testset_sample.csv"))
    return fraud_testset[[c for c in fraud_testset.columns if c != 'isFraud' and c != 'Unnamed: 0']]

@pytest.fixture
def xgb_model() -> xgb.Booster:
    # Load the model from a JSON file
    loaded_model = xgb.Booster()  # Initialize an empty Booster object
    loaded_model.load_model(os.path.join(RESOURCES_PATH, "IEEE-CIS_xgboost_model.json"))
    return loaded_model


def assert_shap_package_is_same_as_woodelf(
        woodelf_result, shap_package_result, testset: pd.DataFrame, tolerance: float
):
    shap_package_shapley_values = pd.DataFrame(
        shap_package_result, index=testset.index, columns=list(testset.columns)
    )
    woodelf_shapley_values = pd.DataFrame(
        {f: woodelf_result.get(f, np.zeros(len(testset))) for f in testset.columns},
        index=testset.index
    )
    pd.testing.assert_frame_equal(
        woodelf_shapley_values, shap_package_shapley_values,
        check_dtype=False, check_names=False, check_exact=False, atol=tolerance
    )


def assert_shap_package_is_same_as_woodelf_on_interaction_values(
        woodelf_result, shap_package_result, testset: pd.DataFrame, tolerance: float
):
    shap_package_shapley_values = pd.DataFrame(
        {fi + "_" + fj: shap_package_result[:, i,j]
         for i, fi in enumerate(testset.columns) for j, fj in enumerate(testset.columns) if fi != fj
         }, index=testset.index
    )
    woodelf_shapley_values = pd.DataFrame(
        {fi + "_" + fj: woodelf_result.get((fi, fj), np.zeros(len(testset)))
         for fi in testset.columns for fj in testset.columns if fi != fj
        },
        index=testset.index
    )
    pd.testing.assert_frame_equal(
        woodelf_shapley_values, shap_package_shapley_values,
        check_dtype=False, check_names=False, check_exact=False, atol=tolerance
    )


def test_background_shap_using_shap_package_is_same_as_using_woodelf(trainset, testset, xgb_model):
    start_time = time.time()
    explainer = shap.TreeExplainer(xgb_model, trainset, feature_perturbation='interventional')
    shap_package_values = explainer.shap_values(testset)
    print("shap took: ", time.time() - start_time)

    start_time = time.time()
    woodelf_values = calculate_background_metric(
        xgb_model, testset, trainset, metric=ShapleyValues()
    )
    print("woodelf took: ", time.time() - start_time)

    assert_shap_package_is_same_as_woodelf(woodelf_values, shap_package_values, testset, TOLERANCE)


def test_path_dependent_shap_using_shap_package_is_same_as_using_woodelf(trainset, testset, xgb_model):
    start_time = time.time()
    explainer = shap.TreeExplainer(xgb_model)
    shap_package_values = explainer.shap_values(testset)
    print("shap took: ", time.time() - start_time)

    start_time = time.time()
    woodelf_values = calculate_path_dependent_metric(xgb_model, testset, metric=ShapleyValues())
    print("woodelf took: ", time.time() - start_time)

    assert_shap_package_is_same_as_woodelf(woodelf_values, shap_package_values, testset, TOLERANCE)


# def test_background_shap_iv_using_shap_package_is_same_as_using_woodelf(trainset, testset, xgb_model):
#     # Not possible as shap package does not support background interaction values


def test_path_dependent_shap_iv_using_shap_package_is_same_as_using_woodelf(trainset, testset, xgb_model):
    start_time = time.time()
    explainer = shap.TreeExplainer(xgb_model)
    shap_package_values = explainer.shap_interaction_values(testset.head(10))
    print("shap took: ", time.time() - start_time)

    start_time = time.time()
    woodelf_values = calculate_path_dependent_metric(xgb_model, testset.head(10), metric=ShapleyInteractionValues())
    print("woodelf took: ", time.time() - start_time)

    assert_shap_package_is_same_as_woodelf_on_interaction_values(
        woodelf_values, shap_package_values, testset.head(10), TOLERANCE
    )