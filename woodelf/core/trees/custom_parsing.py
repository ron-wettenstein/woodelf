"""
Parse models with nothing but numpy, for the models woodelf's other parsing engines cannot read.

Right now that is scikit-learn's standalone DecisionTreeRegressor and DecisionTreeClassifier. They are a
single tree rather than an ensemble, which is why treelite's scikit-learn importer refuses them (see
parse_models_using_treelite.py). Their internal tree_ object already holds the structure as numpy arrays,
so reading it needs no parsing engine at all.
"""
import numpy as np

from woodelf.core.trees.decision_trees_ensemble import LeftIsSmallerEqualDecisionTreeNode, DecisionTreesEnsemble
from woodelf.core.trees.parsing_utils import safe_isinstance

SKLEARN_DECISION_TREE_REGRESSOR_CLASSES = [
    "sklearn.tree.DecisionTreeRegressor",
    "sklearn.tree.tree.DecisionTreeRegressor",
]
SKLEARN_DECISION_TREE_CLASSIFIER_CLASSES = [
    "sklearn.tree.DecisionTreeClassifier",
    "sklearn.tree.tree.DecisionTreeClassifier",
]

# scikit-learn classifiers store one value per class in every node. We keep this class' probability, which
# is the column the shap and the treelite based parsers keep too.
VECTOR_LEAF_CLASS_LABEL = 0


def read_values(sklearn_tree, is_classifier):
    """
    The value of every node, as a single number per node.

    A regressor's node holds one number already. A classifier's node holds one number per class, which we
    turn into that class' probability. Recent scikit-learn versions already store those normalised, and
    dividing by the total again leaves them untouched.
    """
    values = sklearn_tree.value.reshape(len(sklearn_tree.value), -1).astype(np.float64)
    if is_classifier:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
    return values[:, VECTOR_LEAF_CLASS_LABEL]


def read_nan_go_left(sklearn_tree):
    """
    Whether a missing feature value goes to the left child, for every node.

    scikit-learn only grew missing value support in version 1.4, and its trees sent missing values left
    before that, which is the same thing the shap based parser assumes for the older versions.
    """
    if hasattr(sklearn_tree, "missing_go_to_left"):
        return np.asarray(sklearn_tree.missing_go_to_left).astype(bool)
    return np.ones(sklearn_tree.node_count, dtype=bool)


def load_decision_tree(sklearn_tree, features, is_classifier):
    """
    Given the tree_ object of a fitted scikit-learn decision tree, parse it and build a DecisionTreeNode
    object with its structure. The function also gets the training features.
    """
    values = read_values(sklearn_tree, is_classifier)
    nan_go_left_flags = read_nan_go_left(sklearn_tree)

    nodes = {}
    for index in range(sklearn_tree.node_count):
        threshold = sklearn_tree.threshold[index]
        leaf_value = values[index]
        child_left = sklearn_tree.children_left[index]
        child_right = sklearn_tree.children_right[index]
        if child_left == -1 and child_right == -1:
            value = leaf_value
        else:
            value = threshold
        nan_go_left = nan_go_left_flags[index]
        cover = sklearn_tree.weighted_n_node_samples[index]
        # scikit-learn marks a leaf's feature with TREE_UNDEFINED (-2) rather than a real feature index
        feature_index = sklearn_tree.feature[index]
        feature_name = features[feature_index] if feature_index >= 0 else None
        nodes[index] = LeftIsSmallerEqualDecisionTreeNode(
            feature_name=feature_name, value=value, right=None, left=None,
            nan_go_left=nan_go_left, index=index, cover=cover
        )

    for index in range(sklearn_tree.node_count):
        child_left = sklearn_tree.children_left[index]
        child_right = sklearn_tree.children_right[index]

        if child_left != -1:
            nodes[index].left = nodes[child_left]
            nodes[child_left].parent = nodes[index]
        if child_right != -1:
            nodes[index].right = nodes[child_right]
            nodes[child_right].parent = nodes[index]

    nodes[0].depth = nodes[0].get_depth()
    return nodes[0]


def load_sklearn_single_decision_tree_model(model, features) -> DecisionTreesEnsemble:
    """
    Load a standalone scikit-learn decision tree as an ensemble holding that one tree.
    """
    is_classifier = safe_isinstance(model, SKLEARN_DECISION_TREE_CLASSIFIER_CLASSES)
    if not is_classifier and not safe_isinstance(model, SKLEARN_DECISION_TREE_REGRESSOR_CLASSES):
        raise ValueError(
            f"custom_parsing only parses a scikit-learn DecisionTreeRegressor or DecisionTreeClassifier, "
            f"got a {type(model).__name__}"
        )
    trees = [load_decision_tree(model.tree_, features, is_classifier)]
    assert len(trees) > 0, "Did not load the model properly"
    return DecisionTreesEnsemble(trees)
