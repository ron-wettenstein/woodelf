import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import scipy
import shap
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from woodelf.parse_models import load_decision_tree_ensemble_model

TOLERANCE = 1e-7 # 0.00001


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
        loaded_model_pred=tree_ensemble.predict(X),
        base_score=base_score
    )


def test_load_and_predict_xgboost_classifier():
    X, y = shap.datasets.california(n_points=100)
    model = xgb.train(
        {"learning_rate": 0.01, "base_score": 0.5, "eval_metric": "logloss", "objective": "binary:logistic"},
        xgb.DMatrix(X, label=(y > 2)), num_boost_round=10)
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=list(X.columns))
    assert_predictions_equal(
        original_pred=logit(model.predict(xgb.DMatrix(X))),
        loaded_model_pred=tree_ensemble.predict(X),
        base_score=0 # TODO why not base_score 0.5 ?
    )

@pytest.mark.parametrize("model_type, params, base_score_func", [
    (HistGradientBoostingRegressor, dict(max_iter=10,max_depth=6,max_leaf_nodes=None,random_state=42), lambda m: m._baseline_prediction[0][0]),
    (GradientBoostingRegressor, dict(n_estimators=10,max_depth=6,random_state=42), lambda m: m.init_.constant_[0][0]),
    (xgb.sklearn.XGBRegressor, dict(n_estimators=10,max_depth=6,random_state=42, learning_rate=0.01, base_score=0.5), lambda m: 0.5),
    (ExtraTreesRegressor, dict(n_estimators=10,max_depth=6,random_state=42), lambda m: 0),
    (DecisionTreeRegressor, dict(max_depth=6, random_state=42), lambda m: 0),
    # (AdaBoostRegressor, dict(n_estimators=10, random_state=42), lambda m: 0) TODO
], ids=["HistGradientBoostingRegressor", "GradientBoostingRegressor", "xgb.sklearn.XGBRegressor", "ExtraTreesRegressor", "DecisionTreeRegressor"])
def test_load_and_predict_sklearn_regressor_model(model_type, params, base_score_func):
    X, y = shap.datasets.california(n_points=10000)
    model = model_type(**params)
    model.fit(X, y)
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=list(X.columns))
    assert_predictions_equal(
        original_pred=model.predict(X),
        loaded_model_pred=tree_ensemble.predict(X),
        base_score=base_score_func(model)
    )


def test_load_and_predict_lightgbm_regressor():
    X, y = shap.datasets.california(n_points=10000)
    model = lgb.LGBMRegressor(n_estimators=10, max_depth=6, random_state=42, learning_rate=0.01)
    model.fit(X, y)
    base_score = float(model.booster_.dump_model().get("average_output", 0.0))
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=list(X.columns))
    assert_predictions_equal(
        original_pred=model.predict(X),
        loaded_model_pred=tree_ensemble.predict(X),
        base_score=base_score
    )

    # Test lightgbm.basic.Booster
    tree_ensemble = load_decision_tree_ensemble_model(model=model.booster_, features=list(X.columns))
    assert_predictions_equal(
        original_pred=model.predict(X),
        loaded_model_pred=tree_ensemble.predict(X),
        base_score=base_score
    )

def test_load_and_predict_lightgbm_classifier():
    X, y = make_classification(
        n_samples=5000, n_features=12, n_informative=6, n_redundant=2, n_classes=2, class_sep=1.0, random_state=42,
    )
    model = lgb.LGBMClassifier(n_estimators=10, max_depth=6, random_state=42, learning_rate=0.01)
    model.fit(X, y)
    base_score = float(model.booster_.dump_model().get("average_output", 0.0))
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=list(X_df.columns))
    assert_predictions_equal(
        original_pred=model.predict(X, raw_score=True),
        loaded_model_pred=tree_ensemble.predict(X_df),
        base_score=base_score
    )


def logit(p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# TODO why prediction doesn't work on RandomForrest when using n_points=1000 ...
# It seems to be due to the type of the threshold float (float with up to 5 digits behind the decimal points, or more)
def test_load_and_predict_random_forest_model():
    X, y = shap.datasets.california(n_points=100)
    model = RandomForestRegressor(n_estimators=10,max_depth=6, random_state=42)
    model.fit(X, y)
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=list(X.columns))
    tree_ensemble.trees[0].pretty_print()
    assert_predictions_equal(
        original_pred=model.predict(X),
        loaded_model_pred=tree_ensemble.predict(X),
        base_score=0
    )


@pytest.mark.parametrize(
    "model_type, params, predict_func, base_score_func",
    [
        # HistGB: baseline prediction is stored; in binary it's typically log-odds.
        (
            HistGradientBoostingClassifier,
            dict(max_iter=10, max_depth=6,max_leaf_nodes=None,random_state=42),
            lambda m, X: m.decision_function(X),
            lambda m: float(np.ravel(m._baseline_prediction)[0]),
        ),
        # GBC: init_.prior is the class prior; in log-odds (binary) that's the raw baseline.
        (
            GradientBoostingClassifier,
            dict(n_estimators=10, max_depth=6, random_state=42),
            lambda m, X: m.decision_function(X),
            lambda m: scipy.special.logit(m.init_.class_prior_[1]) ,
        ),
        # ExtraTreesClassifier: sklearn usually treats base score as 0 for raw margin.
        (
            ExtraTreesClassifier,
            dict(n_estimators=10, max_depth=6, random_state=42),
            lambda m, X: m.predict_proba(X)[:, 0],
            lambda m: 0.0,
        ),
        # XGBoost: explicit base_score (probability) is set.
        (
        xgb.sklearn.XGBClassifier,
        dict(
            n_estimators=10, max_depth=6, random_state=42, learning_rate=0.01, base_score=0.5,
            eval_metric="logloss",use_label_encoder=False
        ),
        lambda m, X: logit(m.predict_proba(X)[:, 1]),
        lambda m: 0,
        ),
        # DecisionTreeClassifier: same situation as ExtraTrees -> use probability output.
        (
            DecisionTreeClassifier,
            dict(max_depth=6, random_state=42),
            lambda m, X: m.predict_proba(X)[:, 0],
            lambda m: 0.0,
        ),
    # (IsolationForest, dict(n_estimators=10,contamination=0.2,random_state=42),
    #  lambda m, X: m.score_samples(X), lambda m: 0), # TODO doesn't work with IsolationForest
    ],
    ids=[
        "HistGradientBoostingClassifier",
        "GradientBoostingClassifier",
        "ExtraTreesClassifier",
        "xgb.sklearn.XGBClassifier",
        "DecisionTreeClassifier",
    ],
)
def test_load_and_predict_sklearn_classifier_model(model_type, params, predict_func, base_score_func):
    # Toy binary classification task
    X, y = make_classification(
        n_samples=5000, n_features=12, n_informative=6, n_redundant=2, n_classes=2, class_sep=1.0, random_state=42,
    )
    model = model_type(**params)
    model.fit(X, y)

    # If your loader expects feature names, provide some.
    features = [f"x{i}" for i in range(X.shape[1])]
    tree_ensemble = load_decision_tree_ensemble_model(model=model, features=features)

    # Compare positive-class probabilities
    original_pred = predict_func(model, X)
    loaded_pred = tree_ensemble.predict(pd.DataFrame(X, columns=features))

    assert_predictions_equal(
        original_pred=original_pred,
        loaded_model_pred=loaded_pred,
        base_score=base_score_func(model),
    )