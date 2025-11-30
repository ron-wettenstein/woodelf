from typing import List

from woodelf.decision_trees_ensemble import DecisionTreeNode, LeftIsSmallerEqualDecisionTreeNode
from woodelf.utils import safe_isinstance

MODEL_CLASS_TO_DECISION_TREE_CLASS = {
    "sklearn.ensemble.RandomForestRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.forest.RandomForestRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.GradientBoostingRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.ExtraTreesRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.forest.ExtraTreesRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "skopt.learning.forest.ExtraTreesRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "skopt.learning.forest.RandomForestRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "sklearn.ensemble.HistGradientBoostingRegressor": LeftIsSmallerEqualDecisionTreeNode,
    "xgboost.core.Booster": DecisionTreeNode,
    "xgboost.sklearn.XGBClassifier": DecisionTreeNode,
    "xgboost.sklearn.XGBRegressor": DecisionTreeNode,
    # TODO support all decision trees supported in shap and more
}


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
