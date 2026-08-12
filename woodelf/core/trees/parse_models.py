from woodelf.core.trees.custom_parsing import (
    SKLEARN_DECISION_TREE_CLASSIFIER_CLASSES,
    SKLEARN_DECISION_TREE_REGRESSOR_CLASSES,
    load_sklearn_single_decision_tree_model,
)
from woodelf.core.trees.decision_trees_ensemble import DecisionTreesEnsemble
from woodelf.core.trees.parse_models_using_shap import load_model_using_shap
from woodelf.core.trees.parse_models_using_treelite import (
    LIGHTGBM_BOOSTER_CLASSES,
    LIGHTGBM_SKLEARN_CLASSES,
    XGBOOST_BOOSTER_CLASSES,
    XGBOOST_SKLEARN_CLASSES,
    load_model_using_treelite,
)
from woodelf.core.trees.parsing_utils import DEFAULT_CLASS_INDEX, DEFAULT_TARGET_INDEX, safe_isinstance

SKLEARN_SINGLE_DECISION_TREE_CLASSES = (
    SKLEARN_DECISION_TREE_REGRESSOR_CLASSES + SKLEARN_DECISION_TREE_CLASSIFIER_CLASSES
)

# skopt's forests subclass scikit-learn's, so safe_isinstance matches them against the scikit-learn names
# below. They have to be looked up first to keep them on the shap engine, where they have always been
# parsed, rather than silently moving to an engine we have never tested them against.
SKOPT_MODEL_CLASSES = [
    "skopt.learning.forest.ExtraTreesRegressor",
    "skopt.learning.forest.RandomForestRegressor",
]

# The scikit-learn ensembles treelite's importer accepts. A lone decision tree is deliberately not here.
SKLEARN_ENSEMBLE_CLASSES_TREELITE_SUPPORTS = [
    "sklearn.ensemble.RandomForestRegressor",
    "sklearn.ensemble.RandomForestClassifier",
    "sklearn.ensemble.ExtraTreesRegressor",
    "sklearn.ensemble.ExtraTreesClassifier",
    "sklearn.ensemble.GradientBoostingRegressor",
    "sklearn.ensemble.GradientBoostingClassifier",
    "sklearn.ensemble.HistGradientBoostingRegressor",
    "sklearn.ensemble.HistGradientBoostingClassifier",
    "sklearn.ensemble.IsolationForest",
]

TREELITE_SUPPORTED_MODEL_CLASSES = (
    XGBOOST_BOOSTER_CLASSES
    + XGBOOST_SKLEARN_CLASSES
    + LIGHTGBM_BOOSTER_CLASSES
    + LIGHTGBM_SKLEARN_CLASSES
    + SKLEARN_ENSEMBLE_CLASSES_TREELITE_SUPPORTS
)

def load_decision_tree_ensemble_model(
    model, features, class_index: int = DEFAULT_CLASS_INDEX, target_id: int = DEFAULT_TARGET_INDEX
) -> DecisionTreesEnsemble:
    """
    Load a decision tree ensemble model, using whichever parsing engine can read it.

    A model can predict more than one number per row - one per class for a classifier, one per target for
    a multi output regressor - and woodelf explains one of them at a time. class_index and target_id choose
    the one the loaded trees carry, so the values woodelf then explains are that output's. A model that has
    no such output (a single target regressor asked for a target, a regressor asked for a class, or an
    index past the last one) falls back to index 0 silently rather than failing.

    A model stores those outputs one of two ways, and both are handled: the scikit-learn ensembles hold
    every output in every leaf, of which the requested one is read, while xgboost and lightgbm grow a
    separate tree per output, of which only the requested one's trees are kept.
    """
    if safe_isinstance(model, SKLEARN_SINGLE_DECISION_TREE_CLASSES):
        return load_sklearn_single_decision_tree_model(model, features, class_index, target_id)
    if safe_isinstance(model, SKOPT_MODEL_CLASSES):
        return load_model_using_shap(model, features, class_index, target_id)
    if safe_isinstance(model, TREELITE_SUPPORTED_MODEL_CLASSES):
        return load_model_using_treelite(model, features, class_index, target_id)
    return load_model_using_shap(model, features, class_index, target_id)
