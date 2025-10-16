import numpy as np
import pytest
import shap
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor

from woodelf.cube_metric import (
    BanzahfValues, BanzhafInteractionValues, ShapleyInteractionValues, ShapleyValues, CubeMetric
)
from woodelf.direct_computation import (
    BanzhafDirectComputation, BanzhafIVDirectComputation, ShapleyIVDirectComputation,
    ShapleyDirectComputation, DirectComputation, BackgroundModelCF, PathDependentModelCF
)
from woodelf.parse_models import load_decision_tree_ensemble_model
from woodelf.simple_woodelf import calculate_background_metric, calculate_path_dependent_metric


class BackgroundXGBoostCF(BackgroundModelCF):
    def mean_model_prediction(self, data):
        return np.mean(self.model.predict(xgb.DMatrix(data)))

TOLERANCE = 1e-5


def train_xgboost(n_cols):
    X, y = shap.datasets.california(n_points=110)
    # drop some columns so it will be fast enough
    columns_to_focus_on = list(X.columns)[:n_cols]
    X = X[columns_to_focus_on]
    X_train = X.head(100)
    y_train = y[:100]
    X_test = X.tail(10)
    model = xgb.train({"learning_rate": 0.01}, xgb.DMatrix(X_train, label=y_train), 10)
    return X_test, X_train, model


@pytest.mark.parametrize("metric, direct_computation", [
    (BanzahfValues(), BanzhafDirectComputation()),
    (BanzhafInteractionValues(), BanzhafIVDirectComputation()),
    (ShapleyInteractionValues(), ShapleyIVDirectComputation()),
    (ShapleyValues(), ShapleyDirectComputation()),
], ids=["BanzahfValues", "BanzhafInteractionValues", "ShapleyInteractionValues", "ShapleyValues"])
def test_background_metric_computation_xgboost(metric: CubeMetric, direct_computation: DirectComputation):
    X_test, X_train, model = train_xgboost(n_cols=4)
    woodelf_values = calculate_background_metric(model, consumer_data=X_test, background_data=X_train,metric=metric)

    i = 0
    for index, df_row in X_test.iterrows():
        row = {k: float(v) for k,v in dict(df_row).items()}
        cf = BackgroundXGBoostCF(model, row, X_train)
        values_using_direct_computation = direct_computation.compute(cf)
        for feature in woodelf_values:
            assert abs(woodelf_values[feature][i] - values_using_direct_computation[feature]) < TOLERANCE
        i+=1


@pytest.mark.parametrize("metric, direct_computation", [
    (BanzahfValues(), BanzhafDirectComputation()),
    (BanzhafInteractionValues(), BanzhafIVDirectComputation()),
    (ShapleyInteractionValues(), ShapleyIVDirectComputation()),
    (ShapleyValues(), ShapleyDirectComputation()),
], ids=["BanzahfValues", "BanzhafInteractionValues", "ShapleyInteractionValues", "ShapleyValues"])
def test_path_dependent_metric_computation_xgboost(metric: CubeMetric, direct_computation: DirectComputation):
    X_test, X_train, model = train_xgboost(n_cols=5)
    model_obj = load_decision_tree_ensemble_model(model, list(X_train.columns))
    woodelf_values = calculate_path_dependent_metric(model, consumer_data=X_test,metric=metric)

    i = 0
    for index, df_row in X_test.iterrows():
        row = {k: float(v) for k,v in dict(df_row).items()}
        cf = PathDependentModelCF(model_obj, row)
        values_using_direct_computation = direct_computation.compute(cf)
        for feature in woodelf_values:
            assert abs(woodelf_values[feature][i] - values_using_direct_computation[feature]) < TOLERANCE
        i+=1


@pytest.mark.parametrize("model_type, params", [
    (HistGradientBoostingRegressor, dict(max_iter=10,max_depth=6,max_leaf_nodes=None,random_state=42)),
    (GradientBoostingRegressor, dict(n_estimators=10,max_depth=6,random_state=42)),
    (RandomForestRegressor, dict(n_estimators=10,max_depth=6, random_state=42)) # TODO fix RandomForrest
], ids=["RandomForestRegressor"]) # "HistGradientBoostingRegressor", "GradientBoostingRegressor",
def test_background_shap_computation_sklearn(model_type, params):
    X, y = shap.datasets.california(n_points=110)
    columns_to_focus_on = list(X.columns)[:4]
    X = X[columns_to_focus_on]
    X_train = X.head(100)
    y_train = y[:100]
    X_test = X.tail(10)
    model = model_type(**params)
    model.fit(X_train, y_train)
    woodelf_values = calculate_background_metric(
        model, consumer_data=X_test, background_data=X_train,metric=ShapleyValues()
    )

    direct_computation_obj = ShapleyDirectComputation()
    i = 0
    for index, df_row in X_test.iterrows():
        row = {k: float(v) for k,v in dict(df_row).items()}
        cf = BackgroundModelCF(model, row, X_train)
        values_using_direct_computation = direct_computation_obj.compute(cf)
        for feature in woodelf_values:
            assert abs(woodelf_values[feature][i] - values_using_direct_computation[feature]) < TOLERANCE
        i+=1