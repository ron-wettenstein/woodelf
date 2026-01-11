from typing import List

from woodelf.decision_trees_ensemble import DecisionTreeNode, LeftIsSmallerEqualDecisionTreeNode
from woodelf.utils import safe_isinstance

MODEL_CLASS_TO_DECISION_TREE_CLASS = {
    # sklearn regressors
    "sklearn.ensemble.RandomForestRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.forest.RandomForestRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.GradientBoostingRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.ExtraTreesRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.forest.ExtraTreesRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.HistGradientBoostingRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.IsolationForest": LeftIsSmallerEqualDecisionTreeNode,

    # xgboost regressors
    "xgboost.core.Booster": DecisionTreeNode,
    "xgboost.sklearn.XGBClassifier": DecisionTreeNode,
    "xgboost.sklearn.XGBRegressor": DecisionTreeNode,

    # sklearn classifiers
    "sklearn.ensemble.RandomForestClassifier": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.forest.RandomForestClassifier": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.GradientBoostingClassifier": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble._gb.GradientBoostingClassifier": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.ExtraTreesClassifier": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.forest.ExtraTreesClassifier": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.HistGradientBoostingClassifier": LeftIsSmallerEqualDecisionTreeNode,

    # TODO test these models (in one time notebook is enough)
    "skopt.learning.forest.ExtraTreesRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "skopt.learning.forest.RandomForestRegressor": LeftIsSmallerEqualDecisionTreeNode,
    # TODO support all decision trees supported in shap and more
}


# TODO all models supported by shap:
# sklearn.tree.DecisionTreeRegressor
# sklearn.tree.tree.DecisionTreeRegressor
# sklearn.ensemble.IsolationForest
# sklearn.ensemble._iforest.IsolationForest
# sklearn.tree.DecisionTreeClassifier
# sklearn.tree.tree.DecisionTreeClassifier
# sklearn.ensemble.ExtraTreesClassifier
# sklearn.ensemble.forest.ExtraTreesClassifier
# sklearn.ensemble.RandomForestClassifier
# sklearn.ensemble.forest.RandomForestClassifier
# sklearn.ensemble.MeanEstimator
# sklearn.ensemble.gradient_boosting.MeanEstimator
# sklearn.ensemble.QuantileEstimator
# sklearn.ensemble.gradient_boosting.QuantileEstimator
# sklearn.dummy.DummyRegressor
# sklearn.ensemble.HistGradientBoostingClassifier
# sklearn.ensemble.GradientBoostingClassifier
# sklearn.ensemble._gb.GradientBoostingClassifier
# sklearn.ensemble.gradient_boosting.GradientBoostingClassifier
# sklearn.ensemble.LogOddsEstimator
# sklearn.ensemble.gradient_boosting.LogOddsEstimator
# sklearn.dummy.DummyClassifier
# xgboost.sklearn.XGBRanker
# lightgbm.basic.Booster
# lightgbm.sklearn.LGBMRegressor
# lightgbm.sklearn.LGBMRanker
# lightgbm.sklearn.LGBMClassifier
# catboost.core.CatBoostRegressor
# catboost.core.CatBoostClassifier
# catboost.core.CatBoost
# pyspark.ml.classification.RandomForestClassificationModel
# pyspark.ml.regression.RandomForestRegressionModel
# pyspark.ml.classification.GBTClassificationModel
# pyspark.ml.regression.GBTRegressionModel
# pyspark.ml.classification.DecisionTreeClassificationModel
# pyspark.ml.regression.DecisionTreeRegressionModel
# ngboost.ngboost.NGBoost
# ngboost.api.NGBRegressor
# ngboost.api.NGBClassifier
# imblearn.ensemble._forest.BalancedRandomForestClassifier
# gpboost.basic.Booster
# pyod.models.iforest.IForest
# econml.grf._base_grf.BaseGRF
# causalml.inference.tree.CausalRandomForestRegressor

# TODO support mutli target classification

# TODO Support models that are not supported in shap
# sklearn.ensemble.AdaBoostRegressor   https://github.com/shap/shap/issues/4093
# sklearn.ensemble.AdaBoostClassifier
# lightGBM with linear regression instead of weights in the leaves
# Xgboost with categorical features



def load_decision_tree(tree, features, decision_tree_class):
    """
    Given an XGBoost Regressor tree, parse it and build a DecisionTreeNode object with it structure.
    Use the Tree object returned by the shap package's XGBTreeModelLoader class (given as the 'tree' parameter).
    The function also gets the training features.
    """
    nodes = {}
    for index in range(len(tree.thresholds)):
        threshold = tree.thresholds[index]
        leaf_value = tree.values[index][0]
        child_left = tree.children_left[index]
        child_right = tree.children_right[index]
        if child_left == -1 and child_right == -1:
            value = leaf_value
        else:
            value = threshold
        nan_go_left = (tree.children_left[index] == tree.children_default[index])
        cover = tree.node_sample_weight[index]
        feature_index = tree.features[index]
        feature_name = features[feature_index] if feature_index >= 0 else None
        nodes[index] = decision_tree_class(
            feature_name=feature_name, value=value, right=None, left=None,
            nan_go_left=nan_go_left, index=index, cover=cover
        )

    for index in range(len(tree.thresholds)):
        child_left = tree.children_left[index]
        child_right = tree.children_right[index]

        if child_left != -1:
            nodes[index].left = nodes[child_left]
            nodes[child_left].parent = nodes[index]
        if child_right != -1:
            nodes[index].right = nodes[child_right]
            nodes[child_right].parent = nodes[index]

    nodes[0].depth = tree.max_depth
    return nodes[0]

def find_the_right_decision_tree_class(model):
    for class_name in MODEL_CLASS_TO_DECISION_TREE_CLASS:
        if safe_isinstance(model, class_name):
            return MODEL_CLASS_TO_DECISION_TREE_CLASS[class_name]
    return DecisionTreeNode

def load_decision_tree_ensemble_model(model, features) -> List[DecisionTreeNode]:
    """
    Load an XGBoost regressor tree (utilizing the shap python package parsing object)
    """
    # Use the shap package's Decision Tree loading. this is cheating, I know...
    from shap.explainers._tree import TreeEnsemble
    decision_tree_cls = find_the_right_decision_tree_class(model)
    return [load_decision_tree(t, features, decision_tree_cls) for t in TreeEnsemble(model).trees]
