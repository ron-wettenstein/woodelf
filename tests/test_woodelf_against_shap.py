import os
import time

import numpy as np
import pytest
import shap
import xgboost as xgb
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor,
    ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, IsolationForest
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from woodelf.cube_metric import ShapleyValues, ShapleyInteractionValues
from woodelf.simple_woodelf import calculate_background_metric, calculate_path_dependent_metric
from shared_fixtures_and_utils import testset, trainset, xgb_model, assert_shap_package_is_same_as_woodelf, \
    assert_shap_package_is_same_as_woodelf_on_interaction_values

FIXTURES = [testset, trainset, xgb_model]

TOLERANCE = 0.00001


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


@pytest.mark.parametrize("model_type, params", [
    (HistGradientBoostingRegressor, dict(max_iter=10,max_depth=6,max_leaf_nodes=None,random_state=42)),
    (GradientBoostingRegressor, dict(n_estimators=10,max_depth=6,random_state=42)),
    (RandomForestRegressor, dict(n_estimators=10,max_depth=6, random_state=42)),
    (xgb.sklearn.XGBRegressor, dict(n_estimators=10,max_depth=6, random_state=42, learning_rate=0.01)),
    (ExtraTreesRegressor, dict(n_estimators=10,max_depth=6,random_state=42)),
    (DecisionTreeRegressor, dict(max_depth=6, random_state=42))
], ids=["HistGradientBoostingRegressor", "GradientBoostingRegressor",
        "RandomForestRegressor", "xgb.sklearn.XGBRegressor", "ExtraTreesRegressor", "DecisionTreeRegressor"])
def test_woodelf_against_shap_on_sklearn_regressor_model(model_type, params):
    X, y = shap.datasets.california(n_points=110)
    X_train = X.head(100)
    y_train = y[:100]
    X_test = X.tail(10)
    model = model_type(**params)
    model.fit(X_train, y_train)

    # background shap
    explainer = shap.TreeExplainer(model, X_test, model_output="raw")
    shap_package_values = explainer.shap_values(X_test)
    woodelf_values = calculate_background_metric(model, X_test, X_test, metric=ShapleyValues())
    assert_shap_package_is_same_as_woodelf(
        woodelf_values, shap_package_values, X_test, TOLERANCE
    )

    # path dependent shap
    explainer = shap.TreeExplainer(model)
    shap_package_values = explainer.shap_values(X_test)
    woodelf_values = calculate_path_dependent_metric(model, X_test, metric=ShapleyValues())
    assert_shap_package_is_same_as_woodelf(
        woodelf_values, shap_package_values, X_test, TOLERANCE
    )

    # path dependent iv shap
    explainer = shap.TreeExplainer(model)
    shap_package_values = explainer.shap_interaction_values(X_test)
    woodelf_values = calculate_path_dependent_metric(model, X_test, metric=ShapleyInteractionValues())

    assert_shap_package_is_same_as_woodelf_on_interaction_values(
        woodelf_values, shap_package_values, X_test, TOLERANCE
    )


@pytest.mark.parametrize("model_type, params", [
    (HistGradientBoostingClassifier,dict(max_iter=10,max_depth=6,max_leaf_nodes=None,random_state=42)),
    (GradientBoostingClassifier,dict(n_estimators=10, max_depth=6, random_state=42)),
    (ExtraTreesClassifier,dict(n_estimators=10, max_depth=6, random_state=42)),
    (xgb.sklearn.XGBClassifier, dict(n_estimators=10,max_depth=6,random_state=42,learning_rate=0.01,
        base_score=0.5,eval_metric="logloss",use_label_encoder=False)),
    (IsolationForest, dict(n_estimators=10,contamination=0.2,random_state=42)),
    (DecisionTreeClassifier,  dict(max_depth=6, random_state=42)),
], ids=["HistGradientBoostingClassifier",
        "GradientBoostingClassifier",
        "ExtraTreesClassifier",
        "xgb.sklearn.XGBClassifier",
        "IsolationForest",
        "DecisionTreeClassifier"])
def test_woodelf_high_depths_against_shap_on_sklearn_classifier_model(model_type, params):
    # Toy binary classification task
    X, y = make_classification(n_samples=100, n_features=12, n_informative=6, n_redundant=2, n_classes=2, class_sep=1.0, random_state=42)
    model = model_type(**params)
    model.fit(X, y)

    features = [f"x{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=features)

    # background shap
    explainer = shap.TreeExplainer(model, X, model_output="raw")
    shap_package_values = explainer.shap_values(X)
    woodelf_values = calculate_background_metric(model, X, X, metric=ShapleyValues())

    # these models are treated a mutli target classifiers and get Shapley value for their 0 class and 1 class. I choose the values of the 0 class
    if isinstance(model, ExtraTreesClassifier) or isinstance(model, DecisionTreeClassifier):
        shap_package_values = shap_package_values[:, :, 0]
    assert_shap_package_is_same_as_woodelf(
        woodelf_values, shap_package_values, X, TOLERANCE
    )

    # path dependent shap
    explainer = shap.TreeExplainer(model)
    shap_package_values = explainer.shap_values(X)
    woodelf_values = calculate_path_dependent_metric(model, X, metric=ShapleyValues())
    if isinstance(model, ExtraTreesClassifier) or isinstance(model, DecisionTreeClassifier):
        shap_package_values = shap_package_values[:, :, 0]
    assert_shap_package_is_same_as_woodelf(
        woodelf_values, shap_package_values, X, TOLERANCE
    )

    # path dependent iv shap
    explainer = shap.TreeExplainer(model)
    shap_package_values = explainer.shap_interaction_values(X)
    woodelf_values = calculate_path_dependent_metric(model, X, metric=ShapleyInteractionValues())

    if isinstance(model, ExtraTreesClassifier) or isinstance(model, DecisionTreeClassifier):
        shap_package_values = shap_package_values[:, :, :, 0]
    assert_shap_package_is_same_as_woodelf_on_interaction_values(
        woodelf_values, shap_package_values, X, TOLERANCE
    )