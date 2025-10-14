import pytest
import shap
import numpy as np

from woodelf.parse_models import load_decision_tree_ensamble_model

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

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
    # All the diffs should be the base socre, so they should have a small std
    assert (loaded_model_pred - original_pred).std() < 0.00001

    assert np.allclose(original_pred, loaded_model_pred + base_score, atol=1e-7)


def test_load_and_predict_xgboost():
    xgboost = pytest.importorskip("xgboost")
    X, y = shap.datasets.california(n_points=100)
    base_score =  0.5
    model = xgboost.train({"learning_rate": 0.01, "base_score": base_score}, xgboost.DMatrix(X, label=y), 10)
    tree_ensemble = load_decision_tree_ensamble_model(model=model, features=list(X.columns))
    assert_predictions_equal(
        original_pred=model.predict(xgboost.DMatrix(X)),
        loaded_model_pred=predict_of_loaded_model(tree_ensemble,X),
        base_score=base_score
    )

@pytest.mark.parametrize("model_type, params, base_score_func", [
    (HistGradientBoostingRegressor, dict(max_iter=10,max_depth=6,max_leaf_nodes=None,random_state=42),
     lambda m: m._baseline_prediction[0][0]),
    (GradientBoostingRegressor, dict(n_estimators=10,max_depth=6,random_state=42),
     lambda m: m.init_.constant_[0][0]),
    (RandomForestRegressor, dict(n_estimators=10,max_depth=6, random_state=42), lambda m: 0)
    # (AdaBoostRegressor, dict(n_estimators=10, random_state=42), lambda m: 0) TODO
], ids=["HistGradientBoostingRegressor", "GradientBoostingRegressor", "RandomForestRegressor"])
def test_load_and_predict_sklearn_regressor_model(model_type, params, base_score_func):
    X, y = shap.datasets.california(n_points=100)
    model = model_type(**params)
    model.fit(X, y)
    tree_ensemble = load_decision_tree_ensamble_model(model=model, features=list(X.columns))
    print(base_score_func(model))
    assert_predictions_equal(
        original_pred=model.predict(X),
        loaded_model_pred=predict_of_loaded_model(tree_ensemble, X),
        base_score=base_score_func(model)
    )
