import os

import pandas as pd
import pytest
import shap
import numpy as np
import xgboost as xgb

from woodelf.parse_models import load_decision_tree_ensemble_model

from sklearn.ensemble import (
    HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor,
    ExtraTreesRegressor
)

TOLERANCE = 1e-7 # 0.00001

def predict_of_loaded_model(model, df):
    first_tree = model[0]
    preds = first_tree.predict(df)
    for tree in model[1:]:
        preds += tree.predict(df)
    return preds

def assert_predictions_equal(original_pred, loaded_model_pred, base_score):
    """
    The prediction of the loaded models does not include the model base score.
    """
    # All the diffs should be the base score, so they should have a small std
    assert (loaded_model_pred - original_pred).std() < 0.00001

    assert np.allclose(original_pred, loaded_model_pred + base_score, atol=1e-7)


def test_load_and_predict_xgboost():
    X, y = shap.datasets.california(n_points=100)
    base_score =  0.5
    model = xgb.train({"learning_rate": 0.01, "base_score": base_score}, xgb.DMatrix(X, label=y), 10)
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=list(X.columns))
    assert_predictions_equal(
        original_pred=model.predict(xgb.DMatrix(X)),
        loaded_model_pred=predict_of_loaded_model(tree_ensemble,X),
        base_score=base_score
    )

@pytest.mark.parametrize("model_type, params, base_score_func", [
    (HistGradientBoostingRegressor, dict(max_iter=10,max_depth=6,max_leaf_nodes=None,random_state=42),
     lambda m: m._baseline_prediction[0][0]),
    (GradientBoostingRegressor, dict(n_estimators=10,max_depth=6,random_state=42),
     lambda m: m.init_.constant_[0][0]),
    (xgb.sklearn.XGBRegressor, dict(n_estimators=10,max_depth=6,random_state=42, learning_rate=0.01, base_score=0.5),
     lambda m: 0.5),
    (ExtraTreesRegressor, dict(n_estimators=10,max_depth=6,random_state=42),
     lambda m: 0),
    # (AdaBoostRegressor, dict(n_estimators=10, random_state=42), lambda m: 0) TODO
], ids=["HistGradientBoostingRegressor", "GradientBoostingRegressor", "xgb.sklearn.XGBRegressor", "ExtraTreesRegressor"])
def test_load_and_predict_sklearn_regressor_model(model_type, params, base_score_func):
    X, y = shap.datasets.california(n_points=10000)
    model = model_type(**params)
    model.fit(X, y)
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=list(X.columns))
    assert_predictions_equal(
        original_pred=model.predict(X),
        loaded_model_pred=predict_of_loaded_model(tree_ensemble, X),
        base_score=base_score_func(model)
    )


@pytest.mark.parametrize("model_type, params, base_score_func", [
    (HistGradientBoostingRegressor, dict(max_iter=10,max_depth=6,max_leaf_nodes=None,random_state=42),
     lambda m: m._baseline_prediction[0][0]),
    (GradientBoostingRegressor, dict(n_estimators=10,max_depth=6,random_state=42),
     lambda m: m.init_.constant_[0][0]),
    (xgb.sklearn.XGBRegressor, dict(n_estimators=10,max_depth=6,random_state=42, learning_rate=0.01, base_score=0.5),
     lambda m: 0.5),
    (ExtraTreesRegressor, dict(n_estimators=10,max_depth=6,random_state=42),
     lambda m: 0),
    # (AdaBoostRegressor, dict(n_estimators=10, random_state=42), lambda m: 0) TODO
], ids=["HistGradientBoostingRegressor", "GradientBoostingRegressor", "xgb.sklearn.XGBRegressor", "ExtraTreesRegressor"])
def test_load_and_predict_sklearn_regressor_model_ieee_data(model_type, params, base_score_func):
    resources_path = os.path.join(__file__, "..", "resources")
    fraud_trainset = pd.read_csv(os.path.join(resources_path, "IEEE-CIS_trainset_sample.csv"))
    X = fraud_trainset[[c for c in fraud_trainset.columns if c != 'isFraud' and c != 'Unnamed: 0']]
    y = fraud_trainset['isFraud']
    model = model_type(**params)
    if isinstance(model,GradientBoostingRegressor):
        X.fillna(0, inplace=True)
    model.fit(X, y)
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=list(X.columns))
    assert_predictions_equal(
        original_pred=model.predict(X),
        loaded_model_pred=predict_of_loaded_model(tree_ensemble, X),
        base_score=base_score_func(model)
    )


# TODO why prediction doesn't work on RandomForrest when using n_points=1000 ...
# It seems to be due to the type of the threshold float (float with up to 5 digits behind the decimal points, or more)
def test_load_and_predict_random_forest_model():
    X, y = shap.datasets.california(n_points=100)
    model = RandomForestRegressor(n_estimators=10,max_depth=6, random_state=42)
    model.fit(X, y)
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=list(X.columns))
    tree_ensemble[0].pretty_print()
    assert_predictions_equal(
        original_pred=model.predict(X),
        loaded_model_pred=predict_of_loaded_model(tree_ensemble, X),
        base_score=0
    )
