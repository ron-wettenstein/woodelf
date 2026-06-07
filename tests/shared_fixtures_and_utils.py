import os

import joblib
import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb

# Include many shared fixtures, most of them are related to data creation & loading and model training & loading

RESOURCES_PATH = os.path.join(__file__, "..", "resources")

# The parameters used for the model, for documentation
XGB_PARAMS = {
    "objective": "reg:squarederror",  # Regression task with mean squared error loss
    "eval_metric": "rmse",  # Evaluation metric is root mean squared error
    "max_depth": 6,  # Maximum depth of each tree
    "learning_rate": 0.1,  # Learning rate (step size shrinkage)
    "subsample": 1,  # Subsample ratio of the training instances
    "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
    "seed": 123,
    "nthread": 1, # The python shap package use parallel in path dependent if the nthread is bigger than 1. Use nthread=1 to compare the approaches when both don't utilize parallelism.
}

# The parameters used for the depth 22 model, for documentation
# The max_depth is 22 and pruning parameters are used to create a model with few deep leaves but not too much - so tests will still be fast
XGB_PARAMS_DEPTH_22_EXTRA_PRUNING = XGB_PARAMS.copy()
XGB_PARAMS_DEPTH_22_EXTRA_PRUNING["max_depth"] = 22
XGB_PARAMS_DEPTH_22_EXTRA_PRUNING["min_child_weight"] = 100
XGB_PARAMS_DEPTH_22_EXTRA_PRUNING["gamma"] = 2
XGB_PARAMS_DEPTH_22_EXTRA_PRUNING["lambda"] = 10

# The parameters used for the depth 16 model, for documentation
# The max_depth is 16 and pruning parameters are used to create a model with few deep leaves but not too much - so tests will still be fast
XGB_PARAMS_DEPTH_16_EXTRA_PRUNING = XGB_PARAMS.copy()
XGB_PARAMS_DEPTH_16_EXTRA_PRUNING["max_depth"] = 16
XGB_PARAMS_DEPTH_16_EXTRA_PRUNING["min_child_weight"] = 500
XGB_PARAMS_DEPTH_16_EXTRA_PRUNING["gamma"] = 3
XGB_PARAMS_DEPTH_16_EXTRA_PRUNING["lambda"] = 10


def train_xgboost_model(X_train, y_train, params, num_rounds=100):
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    return xgb.train(params, train_dmatrix, num_rounds)

def load_xgboost(json_filename):
    # Load the model from a JSON file
    loaded_model = xgb.Booster()  # Initialize an empty Booster object
    loaded_model.load_model(os.path.join(RESOURCES_PATH, json_filename))
    return loaded_model

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
    return load_xgboost(os.path.join(RESOURCES_PATH, "IEEE-CIS_xgboost_model.json"))

@pytest.fixture
def hist_gradient_boosting_model() -> HistGradientBoostingRegressor:
    return joblib.load(os.path.join(RESOURCES_PATH, "hgb_model.joblib"))

@pytest.fixture
def xgb_model_depth_16() -> xgb.Booster:
    return load_xgboost("IEEE-CIS_xgboost_model_depth_16.json")

@pytest.fixture
def xgb_model_depth_22() -> xgb.Booster:
    return load_xgboost("IEEE-CIS_xgboost_model_depth_22.json")

def assert_woodelf_dicts_equal(
        result_a: dict, result_b: dict, testset: pd.DataFrame, tolerance: float
):
    zeros = np.zeros(len(testset))
    for f in testset.columns:
        np.testing.assert_allclose(
            result_a.get(f, zeros), result_b.get(f, zeros),
            atol=tolerance, err_msg=f"Mismatch for feature {f!r}",
        )


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
    columns = list(testset.columns)
    n_features, n_rows = len(columns), len(testset)

    # Build a (rows, features, features) array from the woodelf dict, leaving the diagonal as zero.
    woodelf_array = np.zeros((n_rows, n_features, n_features), dtype=np.float64)
    zeros = np.zeros(n_rows)
    for i, fi in enumerate(columns):
        for j, fj in enumerate(columns):
            if fi != fj:
                woodelf_array[:, i, j] = woodelf_result.get((fi, fj), zeros)

    # Compare off-diagonal entries in one vectorised call.
    off_diag = ~np.eye(n_features, dtype=bool)
    np.testing.assert_allclose(
        woodelf_array[:, off_diag], shap_package_result[:, off_diag], atol=tolerance
    )


