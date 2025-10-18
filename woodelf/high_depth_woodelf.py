from typing import List, Dict, Any, Optional

from tqdm import tqdm

from woodelf.cube_metric import CubeMetric
from woodelf.decision_trees_ensemble import DecisionTreeNode
from woodelf.parse_models import load_decision_tree_ensemble_model
from woodelf.path_to_matrices import PathToMatricesAbstractCls, SimplePathToMatrices

import numpy as np
import pandas as pd

from woodelf.simple_woodelf import (
    GPU_get_int_dtype_from_depth, get_int_dtype_from_depth, get_cupy_data, fill_mirror_pairs
)

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False



def init_patterns_dict(tree: DecisionTreeNode, data: pd.DataFrame, GPU: bool):
    # Use a tight uint type for efficiency. This is improvement 5 of Sec. 9.1
    int_dtype = GPU_get_int_dtype_from_depth(tree.depth) if GPU else get_int_dtype_from_depth(tree.depth)
    if GPU:
        data_length = len(data[list(data.keys())[0]])
        root_pattern = cp.zeros(data_length, dtype=int_dtype)
    else:
        root_pattern = pd.Series(0, index=data.index).to_numpy().astype(int_dtype)
    return {tree.index: root_pattern}

def add_children_patterns(patterns_dict, node, path_features, data: pd.DataFrame, GPU: bool, int_dtype=int):
    if node.is_leaf():
        return

    left_condition = node.shall_go_left(data, GPU)
    right_condition = ~left_condition
    if not GPU:
        left_condition = left_condition.to_numpy().astype(int_dtype)
        right_condition = right_condition.to_numpy().astype(int_dtype)
    my_pattern = patterns_dict[node.index]

    if node.feature_name not in path_features:
        shifted_my_pattern = (my_pattern << 1)
        patterns_dict[node.left.index] = shifted_my_pattern + left_condition
        patterns_dict[node.right.index] = shifted_my_pattern + right_condition
    else:
        unique_features = []
        for feature in path_features:
            if feature not in unique_features:
                unique_features.append(feature)

        current_feature_bit = 2 ** (len(unique_features) - 1 - unique_features.index(node.feature_name))
        mask = ((2 ** len(unique_features) - 1) - current_feature_bit)
        patterns_dict[node.left.index] = my_pattern & (mask + left_condition*current_feature_bit)
        patterns_dict[node.right.index] = my_pattern & (mask + right_condition*current_feature_bit)

def clean_old_patterns(patterns_dict, node):
    if node is not None and patterns_dict is not None:
        for n in node.bfs():
            if n.index in patterns_dict:
                patterns_dict.pop(n.index)


def compute_f(patterns, path_depth: int, GPU: bool = False):
    if GPU:
        f = cp.bincount(patterns, minlength=2 ** path_depth) / len(patterns)
        return cp.asnumpy( f )[:2 ** path_depth]

    return np.bincount(patterns, minlength=2 ** path_depth) / len(patterns)

def compute_path_dependent_f(path: List[DecisionTreeNode], unique_features_in_path: List[Any]):
    proceed_covers = {f: 1 for f in unique_features_in_path}
    for i in range(len(path)-1):
        proceed_covers[path[i].feature_name] *= (path[i+1].cover / path[i].cover)

    f_size = 2 ** len(unique_features_in_path)
    f = np.ones(f_size)
    for i, feature in enumerate(unique_features_in_path):
        proceed_cover = proceed_covers[feature]
        f = f * np.tile(
            np.array([1-proceed_cover] * (f_size // 2 ** (1 + i)) +  [proceed_cover] * (f_size // 2 ** (1 + i))),
            2 ** i
        )
    return f


def compute_f_of_neighbor(neighbor_f):
    # neighbor leaves have similar patterns (only the last bit is different)
    # For efficiency we reuse the frequencies computed for the neighbor.

    # Given leaves l_i, l_{i+1} s.t. there is an inner node n where n.left = l_i and n.right=l_{i+1}.
    # The decision pattern of any consumer c in leaf l_i is the same as in leaf l_{i+1} except for the last bit which is different.
    # For example if the pattern of c and l_i is 010011011101 then the pattern of c and l_{i+1} is 010011011100 (the 1 in the end is replaced with 0)
    # Let the frequencies of l_i be [f1,f2,f3,f4,....,f_{n-1}, f_n], we can these conclude that the frequencies of l_{i+1} are [f2,f1,f4,f3,....,f_n, f_{n-1}].
    # We can find them by swapping any pair of numbers in the array.
    # The code below utilize this fact for efficiency - this saved half of the bincount opperations.
    # This trick is part of improvement 3 in Sec. 9.1 (this is the improvement to line 4)
    frqs = []
    for i in range(0, len(neighbor_f), 2):
        frqs.append(neighbor_f[i + 1])
        frqs.append(neighbor_f[i])
    return np.array(frqs, dtype=np.float32)

def compute_values_using_s_vectors(values, s_vectors, consumer_patterns, GPU: bool):
    for feature, s_vector in s_vectors.items():
        if GPU:
            replacements_array = cp.asarray(s_vector)
        else:
            replacements_array = np.ascontiguousarray(s_vector)

        # This is where the numpy indexing occur (improvement 6 of Sec. 9.1):
        current_contribution = replacements_array[consumer_patterns]

        if feature not in values:
            values[feature] = current_contribution
        else:
            values[feature] += current_contribution

def combine_neighbor_leaves_s_vectors(s_left, s_right):
    s_combined = {}
    for feature in s_left:
        s_right_vec = s_right[feature]
        swapped_s_right = []
        for i in range(0, len(s_right_vec), 2):
            swapped_s_right.append(s_right_vec[i + 1])
            swapped_s_right.append(s_right_vec[i])
        s_combined[feature] = s_left[feature] + np.array(swapped_s_right, dtype=s_right_vec.dtype)
    return s_combined

def compute_background_shap_for_leaf_node(
        values: Dict[Any, float], leaf: DecisionTreeNode, consumer_patterns: np.array, background_patterns: np.array,
        unique_features_in_path: List[Any], path_to_matrices_calculator: PathToMatricesAbstractCls, GPU: bool
):
    depth = len(unique_features_in_path)
    f = compute_f(background_patterns, depth, GPU)
    s_vectors = path_to_matrices_calculator.get_s_matrices(unique_features_in_path, f, leaf.value)
    compute_values_using_s_vectors(values, s_vectors, consumer_patterns, GPU)


def compute_path_dependent_shap_for_leaf_node(
        values: Dict[Any, float], leaf: DecisionTreeNode, consumer_patterns: np.array, path: List[DecisionTreeNode],
        unique_features_in_path: List[Any], path_to_matrices_calculator: PathToMatricesAbstractCls, GPU: bool
):
    f = compute_path_dependent_f(path + [leaf], unique_features_in_path)
    s_vectors = path_to_matrices_calculator.get_s_matrices(unique_features_in_path, f, leaf.value)
    compute_values_using_s_vectors(values, s_vectors, consumer_patterns, GPU)

def compute_background_shap_for_two_neighbor_leaves(
        values: Dict[Any, float], left_leaf: DecisionTreeNode, right_leaf: DecisionTreeNode,
        left_consumer_patterns: np.array, left_background_patterns: np.array,
        unique_features_in_path: List[Any], path_to_matrices_calculator: PathToMatricesAbstractCls, GPU: bool
):
    depth = len(unique_features_in_path)
    left_f = compute_f(left_background_patterns, depth, GPU)
    left_s_vectors = path_to_matrices_calculator.get_s_matrices(unique_features_in_path, left_f, left_leaf.value)

    right_f = compute_f_of_neighbor(left_f)
    right_s_vectors = path_to_matrices_calculator.get_s_matrices(unique_features_in_path, right_f, right_leaf.value)

    combined_vectors = combine_neighbor_leaves_s_vectors(left_s_vectors, right_s_vectors)
    compute_values_using_s_vectors(values, combined_vectors, left_consumer_patterns, GPU)

def compute_path_dependent_shap_two_neighbor_leaves(
        values: Dict[Any, float], left_leaf: DecisionTreeNode, right_leaf: DecisionTreeNode,
        left_consumer_patterns: np.array, path: List[DecisionTreeNode],
        unique_features_in_path: List[Any], path_to_matrices_calculator: PathToMatricesAbstractCls, GPU: bool
):
    left_f = compute_path_dependent_f(path + [left_leaf], unique_features_in_path)
    left_s_vectors = path_to_matrices_calculator.get_s_matrices(unique_features_in_path, left_f, left_leaf.value)
    right_f = compute_path_dependent_f(path + [right_leaf], unique_features_in_path)
    right_s_vectors = path_to_matrices_calculator.get_s_matrices(unique_features_in_path, right_f, right_leaf.value)

    combined_vectors = combine_neighbor_leaves_s_vectors(left_s_vectors, right_s_vectors)
    compute_values_using_s_vectors(values, combined_vectors, left_consumer_patterns, GPU)


def compute_patterns_generator(tree: DecisionTreeNode, data: pd.DataFrame, GPU: bool = False):
    nodes_to_visit_left = [tree]
    nodes_to_visit_right = []

    int_dtype = GPU_get_int_dtype_from_depth(tree.depth) if GPU else get_int_dtype_from_depth(tree.depth)
    patterns = init_patterns_dict(tree, data, GPU)
    nodes_to_path = tree.get_nodes_to_path_dict()
    while len(nodes_to_visit_left) > 0 or len(nodes_to_visit_right) > 0:
        if len(nodes_to_visit_left) > 0:
            node = nodes_to_visit_left.pop()
        else:
            node = nodes_to_visit_right.pop()
            clean_old_patterns(patterns, node.parent.left)

        path = nodes_to_path[node.index]
        path_features = [n.feature_name for n in path]

        add_children_patterns(patterns, node, path_features, data, GPU, int_dtype)
        if node.is_leaf():
            yield node.index, patterns[node.index]
        else:
            nodes_to_visit_left.append(node.left)
            nodes_to_visit_right.append(node.right)

def woodelf_for_high_depth_single_tree(
        tree: DecisionTreeNode, consumer_data: pd.DataFrame, background_data: pd.DataFrame,
        values: Dict[Any, float], path_to_matrices_calculator: PathToMatricesAbstractCls, GPU: bool = False,
        use_neighbor_leaf_trick: bool = True
):
    nodes_to_visit_left = [tree]
    nodes_to_visit_right = []

    is_background = background_data is not None
    int_dtype = GPU_get_int_dtype_from_depth(tree.depth) if GPU else get_int_dtype_from_depth(tree.depth)
    consumer_patterns = init_patterns_dict(tree, consumer_data, GPU)
    if background_data is not None:
        background_patterns = init_patterns_dict(tree, background_data, GPU)
    else:
        background_patterns = None

    nodes_to_path = tree.get_nodes_to_path_dict()
    while len(nodes_to_visit_left) > 0 or len(nodes_to_visit_right) > 0:
        if len(nodes_to_visit_left) > 0:
            node = nodes_to_visit_left.pop()
        else:
            node = nodes_to_visit_right.pop()
            clean_old_patterns(consumer_patterns, node.parent.left)
            clean_old_patterns(background_patterns, node.parent.left)

        path = nodes_to_path[node.index]
        path_features = [n.feature_name for n in path]

        add_children_patterns(consumer_patterns, node, path_features, consumer_data, GPU, int_dtype)
        if is_background:
            add_children_patterns(background_patterns, node, path_features, background_data, GPU, int_dtype)

        unique_features_in_path = []
        for n in path:
            if n.feature_name not in unique_features_in_path:
                unique_features_in_path.append(n.feature_name)

        current_node_feature_does_not_repeat_in_the_path = all(n.feature_name != node.feature_name for n in path)
        if node.is_leaf():
            if is_background:
                compute_background_shap_for_leaf_node(
                    values, node, consumer_patterns[node.index], background_patterns[node.index],
                    unique_features_in_path, path_to_matrices_calculator, GPU
                )
            else:
                compute_path_dependent_shap_for_leaf_node(
                    values, node, consumer_patterns[node.index], path,
                    unique_features_in_path, path_to_matrices_calculator, GPU
                )
        elif (
                node.right.is_leaf() and node.left.is_leaf() and
                current_node_feature_does_not_repeat_in_the_path and use_neighbor_leaf_trick
        ):
            if node.feature_name not in unique_features_in_path:
                unique_features_in_path.append(node.feature_name)

            if is_background:
                compute_background_shap_for_two_neighbor_leaves(
                    values, node.left, node.right, consumer_patterns[node.left.index],
                    background_patterns[node.left.index],
                    unique_features_in_path, path_to_matrices_calculator, GPU
                )
            else:
                compute_path_dependent_shap_two_neighbor_leaves(
                    values, node.left, node.right, consumer_patterns[node.left.index],
                    path + [node], unique_features_in_path, path_to_matrices_calculator, GPU
                )
        else:
            nodes_to_visit_left.append(node.left)
            nodes_to_visit_right.append(node.right)

def woodelf_for_high_depth(
        model, consumer_data: pd.DataFrame, background_data: Optional[pd.DataFrame], metric: CubeMetric,
        GPU: bool=False, use_neighbor_leaf_trick: bool=True,
        path_to_matrices_calculator: PathToMatricesAbstractCls = None
):
    model_objs = load_decision_tree_ensemble_model(model, list(consumer_data.columns))
    if path_to_matrices_calculator is None:
        path_to_matrices_calculator = SimplePathToMatrices(metric=metric, max_depth=model_objs[0].depth, GPU=GPU)
    if GPU:
        consumer_data = get_cupy_data(model_objs, consumer_data)
        background_data = get_cupy_data(model_objs, background_data)

    values = {}
    for tree in tqdm(model_objs, desc="Preprocessing the trees and computing SHAP"):
        woodelf_for_high_depth_single_tree(
            tree, consumer_data, background_data, values, path_to_matrices_calculator, GPU,
            use_neighbor_leaf_trick
        )

    if not metric.INTERACTION_VALUES_ORDER_MATTERS and metric.INTERACTION_VALUE:
        fill_mirror_pairs(values)

    return values
