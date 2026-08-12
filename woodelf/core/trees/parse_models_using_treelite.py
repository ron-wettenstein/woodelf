import numpy as np

from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode, LeftIsSmallerEqualDecisionTreeNode, DecisionTreesEnsemble
from woodelf.core.trees.parsing_utils import DEFAULT_CLASS_INDEX, resolve_class_index, safe_isinstance

# treelite's Operator enum. Split nodes carry one of these, leaves carry kNone (0).
TREELITE_OPERATOR_LT = 2  # go left if x < threshold. Used by XGBoost.
TREELITE_OPERATOR_LE = 3  # go left if x <= threshold. Used by LightGBM and scikit-learn.

TREELITE_OPERATOR_TO_DECISION_TREE_CLASS = {
    TREELITE_OPERATOR_LT: DecisionTreeNode,
    TREELITE_OPERATOR_LE: LeftIsSmallerEqualDecisionTreeNode,
}

XGBOOST_BOOSTER_CLASSES = ["xgboost.core.Booster"]
XGBOOST_SKLEARN_CLASSES = [
    "xgboost.sklearn.XGBRegressor",
    "xgboost.sklearn.XGBClassifier",
    "xgboost.sklearn.XGBRanker",
]
LIGHTGBM_BOOSTER_CLASSES = ["lightgbm.basic.Booster"]
LIGHTGBM_SKLEARN_CLASSES = [
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.sklearn.LGBMRanker",
]

# TODO models treelite cannot parse, they still need parse_models_using_shap:
# catboost.core.CatBoost / CatBoostRegressor / CatBoostClassifier
# skopt.learning.forest.ExtraTreesRegressor / RandomForestRegressor (untested, they subclass the sklearn ones)
# pyspark, ngboost, imblearn, gpboost, pyod, econml, causalml

# TODO support categorical splits. treelite keeps them (category_list / category_list_right_child) but
# DecisionTreeNode only knows how to compare a feature against a single threshold.


def get_field(accessor, name):
    """
    Read one treelite field as a flat numpy array.

    treelite's optional fields (data_count, sum_hess, gain, leaf_vector) come back as empty arrays when
    the model's loader did not populate them, so callers have to check the length rather than assume.
    """
    return np.asarray(accessor.get_field(name)).ravel()


def read_node_sample_weight(tree_accessor, num_nodes):
    """
    Read the cover (how many training rows reached each node) of every node in the tree.

    treelite has two fields for this and no loader populates both meaningfully: XGBoost fills sum_hess
    (the sum of the hessians, which is XGBoost's own "cover"), LightGBM and HistGradientBoosting fill
    data_count, and the other scikit-learn ensembles fill both. Preferring sum_hess reproduces the cover
    the shap based parser uses for every one of those model families.
    """
    for field_name in ("sum_hess", "data_count"):
        covers = get_field(tree_accessor, field_name)
        if len(covers) == num_nodes:
            return covers.astype(np.float64)
    raise ValueError(
        "The treelite model has no per node cover (neither sum_hess nor data_count is populated). "
        "woodelf cannot compute Shapley values without it."
    )


def read_leaf_values(tree_accessor, leaf_mask, vector_length, leaf_scaling, class_index=DEFAULT_CLASS_INDEX):
    """
    Read the value of every leaf, returning a per-node array whose inner nodes hold 0.

    Leaves hold either a single number (boosters, and scikit-learn regressors) or one number per class
    (scikit-learn classifiers), in which case treelite packs all the vectors of a tree into one flat
    leaf_vector array that leaf_vector_begin/leaf_vector_end index into. class_index picks which of those
    per class numbers we keep - the same column the shap based parser keeps, so the two parsers agree -
    and a leaf holding a single number has no class to pick, so it ignores class_index entirely.
    """
    if vector_length <= 1:
        return np.where(leaf_mask, get_field(tree_accessor, "leaf_value").astype(np.float64), 0.0) * leaf_scaling

    leaf_vector = get_field(tree_accessor, "leaf_vector").astype(np.float64)
    begins = get_field(tree_accessor, "leaf_vector_begin").astype(np.int64)
    ends = get_field(tree_accessor, "leaf_vector_end").astype(np.int64)
    leaf_indexes = np.flatnonzero(leaf_mask)

    # Every leaf holds one value per class, so its window is always vector_length wide and we can gather
    # all of the windows in one go rather than slicing leaf_vector one leaf at a time.
    if not np.all(ends[leaf_indexes] - begins[leaf_indexes] == vector_length):
        raise ValueError("woodelf expects treelite to give every leaf exactly one value per class")
    gather_indexes = begins[leaf_indexes][:, None] + np.arange(vector_length)[None, :]
    class_values = leaf_vector[gather_indexes]  # (number of leaves, vector_length)

    # treelite already normalises the scikit-learn class votes, we normalise again so that a loader
    # that hands us raw counts still produces the probabilities the shap based parser produces.
    totals = class_values.sum(axis=1)
    values = np.zeros(len(leaf_mask), dtype=np.float64)
    values[leaf_indexes] = np.divide(
        class_values[:, resolve_class_index(class_index, vector_length)], totals,
        out=np.zeros(len(leaf_indexes)), where=totals != 0,
    )
    return values * leaf_scaling


def select_class_trees(header, num_tree, vector_length, class_index):
    """
    The indexes of the trees that carry the requested class, as the model happens to store its classes.
    Will return all the models trees except in multi class XGboost and LightGBM
    """
    class_ids = get_field(header, "class_id").astype(np.int64)
    number_of_classes = int(class_ids.max()) + 1
    grows_a_tree_per_class = (
        vector_length <= 1 and len(class_ids) == num_tree and number_of_classes > 1
    )
    if not grows_a_tree_per_class:
        return range(num_tree)
    wanted_class = resolve_class_index(class_index, number_of_classes)
    return [int(tree_index) for tree_index in np.flatnonzero(class_ids == wanted_class)]


def find_the_comparison_operator(comparison_operators):
    """
    The one comparison operator the tree's splits use, given its per node 'cmp' field.

    treelite records the operator on every split node (leaves carry kNone), so unlike the shap based
    parser we read the answer off the model instead of looking the model's class up in a table.
    Returns the kLE operator for a tree with no splits at all, where there is nothing to compare.
    """
    operators = set(int(operator) for operator in comparison_operators) - {0}
    if len(operators) == 0:
        return TREELITE_OPERATOR_LE
    if len(operators) > 1:
        raise ValueError(f"woodelf does not support a tree mixing several comparison operators, got {operators}")
    operator = operators.pop()
    if operator not in TREELITE_OPERATOR_TO_DECISION_TREE_CLASS:
        raise ValueError(
            f"woodelf does not support treelite comparison operator {operator}, only "
            f"'<' ({TREELITE_OPERATOR_LT}) and '<=' ({TREELITE_OPERATOR_LE})"
        )
    return operator


def read_thresholds(tree_accessor, comparison_operator):
    """
    Read the split threshold of every node in the tree.

    XGBoost keeps its thresholds in float32 and compares them against features it has also cast to
    float32, but woodelf compares them against the float64 values of the consumer's DataFrame. A feature
    value that only equals the threshold once rounded to float32 is therefore routed left here and right
    by XGBoost. Moving every threshold down one float32 step restores XGBoost's routing, and it is what
    shap does as well (see XGBTreeModelLoader in shap/explainers/_tree.py), so the two parsers agree.
    """
    thresholds = get_field(tree_accessor, "threshold")
    if comparison_operator == TREELITE_OPERATOR_LT:
        thresholds = np.nextafter(thresholds.astype(np.float32), -np.float32(np.inf))
    return thresholds.astype(np.float64)


def load_decision_tree(tree_accessor, features, vector_length, leaf_scaling, class_index=DEFAULT_CLASS_INDEX):
    """
    Given one tree of a parsed treelite model, parse it and build a DecisionTreeNode object with its structure.
    Use the accessor returned by the treelite model's get_tree_accessor method (given as the 'tree_accessor'
    parameter). The function also gets the training features.
    """
    if get_field(tree_accessor, "has_categorical_split").any():
        raise ValueError("woodelf does not support trees with categorical splits")

    children_left = get_field(tree_accessor, "cleft").astype(np.int64)
    children_right = get_field(tree_accessor, "cright").astype(np.int64)
    feature_indexes = get_field(tree_accessor, "split_index").astype(np.int64)
    comparison_operator = find_the_comparison_operator(get_field(tree_accessor, "cmp"))
    thresholds = read_thresholds(tree_accessor, comparison_operator)
    # treelite marks the missing value direction with a boolean, where shap gives the default child's index
    nan_go_left_flags = get_field(tree_accessor, "default_left").astype(bool)
    covers = read_node_sample_weight(tree_accessor, len(children_left))
    values = read_leaf_values(tree_accessor, children_left == -1, vector_length, leaf_scaling, class_index)
    decision_tree_class = TREELITE_OPERATOR_TO_DECISION_TREE_CLASS[comparison_operator]

    nodes = {}
    for index in range(len(thresholds)):
        threshold = thresholds[index]
        leaf_value = values[index]
        child_left = children_left[index]
        child_right = children_right[index]
        if child_left == -1 and child_right == -1:
            value = leaf_value
        else:
            value = threshold
        nan_go_left = nan_go_left_flags[index]
        cover = covers[index]
        feature_index = feature_indexes[index]
        feature_name = features[feature_index] if feature_index >= 0 else None
        nodes[index] = decision_tree_class(
            feature_name=feature_name, value=value, right=None, left=None,
            nan_go_left=nan_go_left, index=index, cover=cover
        )

    for index in range(len(thresholds)):
        child_left = children_left[index]
        child_right = children_right[index]

        if child_left != -1:
            nodes[index].left = nodes[child_left]
            nodes[child_left].parent = nodes[index]
        if child_right != -1:
            nodes[index].right = nodes[child_right]
            nodes[child_right].parent = nodes[index]

    nodes[0].depth = nodes[0].get_depth()
    return nodes[0]


def load_treelite_model(model):
    """
    Hand the model to the treelite frontend that knows how to read it.
    """
    import treelite

    if safe_isinstance(model, XGBOOST_BOOSTER_CLASSES):
        return treelite.frontend.from_xgboost(model)
    if safe_isinstance(model, XGBOOST_SKLEARN_CLASSES):
        return treelite.frontend.from_xgboost(model.get_booster())
    if safe_isinstance(model, LIGHTGBM_BOOSTER_CLASSES):
        return treelite.frontend.from_lightgbm(model)
    if safe_isinstance(model, LIGHTGBM_SKLEARN_CLASSES):
        return treelite.frontend.from_lightgbm(model.booster_)
    try:
        return treelite.sklearn.import_model(model)
    except Exception as error:
        raise ValueError(
            f"treelite cannot parse a {type(model).__name__}. Use "
            f"woodelf.core.trees.parse_models_using_shap.load_model_using_shap instead, "
            f"which supports more model types (it needs the shap package installed)."
        ) from error


def load_model_using_treelite(model, features, class_index: int = DEFAULT_CLASS_INDEX) -> DecisionTreesEnsemble:
    """
    Load a decision tree ensemble model (utilizing the treelite python package parsing object)
    """
    treelite_model = load_treelite_model(model)
    header = treelite_model.get_header_accessor()

    # Leaves hold one value per class for scikit-learn classifiers, and a single value otherwise.
    leaf_vector_shape = get_field(header, "leaf_vector_shape")
    vector_length = int(leaf_vector_shape[-1]) if len(leaf_vector_shape) > 0 else 1

    average_tree_output = get_field(header, "average_tree_output")
    averages_trees = len(average_tree_output) > 0 and bool(average_tree_output[0])
    leaf_scaling = 1.0 / treelite_model.num_tree if averages_trees else 1.0

    tree_indexes = select_class_trees(header, treelite_model.num_tree, vector_length, class_index)
    trees = [
        load_decision_tree(
            treelite_model.get_tree_accessor(tree_index), features, vector_length, leaf_scaling, class_index
        )
        for tree_index in tree_indexes
    ]
    return DecisionTreesEnsemble(trees)
