from typing import List

from tqdm import tqdm

from woodelf.cube_metric import CubeMetric
from woodelf.decision_trees_ensemble import DecisionTreeNode
from woodelf.parse_models import load_decision_tree_ensemble_model
from woodelf.path_to_matrices import PathToMatricesAbstractCls, SimplePathToMatrices

import numpy as np
import pandas as pd

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False


def get_int_dtype_from_depth(depth):
    """
    The decision pattern, when encoded as a number, have a bit for each node of the root-to-leaf-path.
    Choose the dtype according to the tree depth (a.k.a the max pattern length).

    This is improvement 5 of Sec. 9.1
    """
    if depth <= 8:
        return np.uint8
    if depth <= 16:
        return np.uint16
    if depth <= 32:
        return np.uint32
    return np.uint64


def GPU_get_int_dtype_from_depth(depth):
    """
    Like get_int_dtype_from_depth but return CuPy types.
    """
    if depth <= 8:
        return cp.uint8
    if depth <= 16:
        return cp.uint16
    if depth <= 32:
        return cp.uint32
    return cp.uint64


def calc_decision_patterns(tree, data, depth, GPU=False):
    """
    An effiecent implementation of the CalcDecisionPatterns from Sec. 4 of the paper.
    """
    # Use a tight uint type for efficiency. This is improvement 5 of Sec. 9.1
    int_dtype = GPU_get_int_dtype_from_depth(depth) if GPU else get_int_dtype_from_depth(depth)

    leaves_patterns_dict = {}  # This is the P_leaves mentioned in the paper
    inner_nodes_patterns_dict = {}  # This is P_all
    if GPU:
        data_length = len(data[list(data.keys())[0]])
        inner_nodes_patterns_dict[tree.index] = cp.zeros(data_length, dtype=int_dtype)
    else:
        inner_nodes_patterns_dict[tree.index] = pd.Series(0, index=data.index).to_numpy().astype(int_dtype)

    for current_node in tree.bfs():
        if current_node.is_leaf():
            leaves_patterns_dict[current_node.index] = inner_nodes_patterns_dict[current_node.index]
            continue

        left_condition = current_node.shall_go_left(data, GPU)
        right_condition = ~left_condition
        if not GPU:
            left_condition = left_condition.to_numpy().astype(int_dtype)
            right_condition = right_condition.to_numpy().astype(int_dtype)
        my_pattern = inner_nodes_patterns_dict[current_node.index]
        shifted_my_pattern = (my_pattern << 1)
        inner_nodes_patterns_dict[current_node.left.index] = shifted_my_pattern + left_condition
        inner_nodes_patterns_dict[current_node.right.index] = shifted_my_pattern + right_condition
    return leaves_patterns_dict


def preprocess_tree_background(tree: DecisionTreeNode, background_data: pd.DataFrame, depth: int,
                               path_to_matrixes_calculator: PathToMatricesAbstractCls, GPU: bool = False,
                               unique_features_decision_pattern: bool = False):
    """
    Run all the preprocessing needed given a tree and a background_data.
    Include lines 2-21 of the pseudo-code.
    """
    background_patterns_matrix = calc_decision_patterns(tree, background_data, depth, GPU)

    # Build f, implements lines 3-4 of the pseudo-code
    Frq_b = {}
    visited_leaves_parents = {}
    data_length = len(background_data) if not GPU else len(background_data[list(background_data.keys())[0]])
    for leaf, features_in_path in tree.get_all_leaves_with_paths():
        use_neighbor_trick = (leaf.parent.index in visited_leaves_parents) and (
                (not unique_features_decision_pattern) or (features_in_path[-1] not in features_in_path[:-1])
        )
        if not use_neighbor_trick:
            # np.bincount is a faster way to implement value_counts that uses the fact all decision patterns are integers between 0 and 2**depth
            if GPU:
                Frq_b[leaf.index] = cp.bincount(background_patterns_matrix[leaf.index],
                                                minlength=2 ** len(features_in_path))
                Frq_b[leaf.index] = Frq_b[leaf.index] / data_length
                Frq_b[leaf.index] = cp.asnumpy(Frq_b[leaf.index])
            else:
                Frq_b[leaf.index] = np.bincount(background_patterns_matrix[leaf.index],
                                                minlength=2 ** len(features_in_path))
                Frq_b[leaf.index] = Frq_b[leaf.index] / data_length
            visited_leaves_parents[leaf.parent.index] = Frq_b[leaf.index]
        else:
            # neighbor leaves have similar patterns (only the last bit is different)
            # For efficiency we reuse the frequencies computed for the neighboor.

            # Given leaves l_i, l_{i+1} s.t. there is an inner node n where n.left = l_i and n.right=l_{i+1}.
            # The decision pattern of any consumer c in leaf l_i is the same as in leaf l_{i+1} except for the last bit which is different.
            # For example if the pattern of c and l_i is 010011011101 then the pattern of c and l_{i+1} is 010011011100 (the 1 in the end is replaced with 0)
            # Let the frequencies of l_i be [f1,f2,f3,f4,....,f_{n-1}, f_n], we can these conclude that the frequencies of l_{i+1} are [f2,f1,f4,f3,....,f_n, f_{n-1}].
            # We can find them by swapping any pair of numbers in the array.
            # The code below utilize this fact for efficiency - this saved half of the bincount opperations.
            # This trick is part of improvement 3 in Sec. 9.1 (this is the improvement to line 4)
            neighboor_frq = visited_leaves_parents[leaf.parent.index]
            frqs = []
            for i in range(0, len(neighboor_frq), 2):
                frqs.append(neighboor_frq[i + 1])
                frqs.append(neighboor_frq[i])
            Frq_b[leaf.index] = np.array(frqs, dtype=np.float32)

    for leaf, features_in_path in tree.get_all_leaves_with_paths():
        fl = Frq_b[leaf.index]
        if GPU and 2 ** len(features_in_path) < len(fl):  # this trim is needed only on GPU
            fl = fl[:2 ** len(features_in_path)]

        # Build M, implements lines 7-16 of the pseudo-code and Build s, implements lines 17-21 of the pseudo-code
        features_to_values = path_to_matrixes_calculator.get_s_matrices(
            features_in_path, fl, leaf.value, path_dependent=False
        )
        leaf.feature_contribution_replacement_values = features_to_values
    return tree


def get_cupy_data(trees: List[DecisionTreeNode], df: pd.DataFrame):
    """
    Cast the dataframe to cupy dict mapping between columns of CuPy arrays.
    We only do this for feature partisipating in the trees.
    """
    data = {}
    for tree in trees:
        for feature in tree.get_all_features():
            if feature not in data:
                data[feature] = cp.asarray(df[feature].to_numpy())
    return data


def calculation_given_preprocessed_tree(tree: DecisionTreeNode, data: pd.DataFrame, values=None, depth: int = 6,
                                        GPU=False):
    """
    Use the preprocessing to efficiently calculate the desired metric (Shapley/Banzahf values or interaction values)
    Implements lines 22-27 of the pseudo-code
    """
    # line 22 of the pseudo-code
    decision_patterns = calc_decision_patterns(tree, data, depth, GPU)

    # lines 23-27 of the pseudo-code
    if values is None:
        values = {}

    for almost_leaf in tree.get_all_almost_leaves():
        if not almost_leaf.right.is_leaf() or not almost_leaf.left.is_leaf():
            # If only the right or the left node is a leaf use s as is
            leaf = almost_leaf.right if almost_leaf.right.is_leaf() else almost_leaf.left
            current_edp_indexes = decision_patterns[leaf.index]
            replacements_arrays = leaf.feature_contribution_replacement_values
        else:
            # If both the right and left nodes are leaves use improvement 3 of Sec. 9.1 (improvement of line 26)
            # See also the comment in preprocess_tree_background.
            # Given leaves l_i, l_{i+1} s.t. there is an inner node n where n.left=l_i and n.right=l_{i+1}.
            # mark the s vector of feature f and leaf l_i as s_i = [a1,a2,a3,...,an]
            # mark the s vector of feature f and leaf l_{i+1} as s_{i+1} = [b1,b2,b3,...,bn]
            # A trivial numpy indexing for feature f and the two leaves is [a1,a2,a3,...,an][ patterns ] + [b1,b2,b3,...,bn][ patterns ]
            # Utilizing the property explained in comment in preprocess_tree_background, we can run the equivalent numpy indexing:
            # [a1+b2, a2+b1, a3+b4, a4+b3,...,a_{n-1}+bn, an+b_{n-1}][ patterns ]
            # This saves half of the numpy indexing opperations
            current_edp_indexes = decision_patterns[almost_leaf.left.index]
            replacements_arrays = almost_leaf.left.feature_contribution_replacement_values
            for feature, replacement_values in almost_leaf.right.feature_contribution_replacement_values.items():
                swaped_replacements_values = []
                for i in range(0, len(replacement_values), 2):
                    swaped_replacements_values.append(replacement_values[i + 1])
                    swaped_replacements_values.append(replacement_values[i])

                if feature not in replacements_arrays:
                    replacements_arrays[feature] = np.array(swaped_replacements_values, dtype=np.float32)
                else:
                    replacements_arrays[feature] = np.array(swaped_replacements_values, dtype=np.float32) + \
                                                   replacements_arrays[feature]

        for feature, replacement_values in replacements_arrays.items():
            if GPU:
                replacements_array = cp.asarray(replacement_values)
            else:
                replacements_array = np.ascontiguousarray(replacement_values)

            # This is where the numpy indexing occur (improvement 6 of Sec. 9.1):
            current_contribution = replacements_array[current_edp_indexes]

            if feature not in values:
                values[feature] = current_contribution
            else:
                values[feature] += current_contribution

    return values


def calculation_given_preprocessed_tree_ensemble(
        preprocess_trees: List[DecisionTreeNode], consumer_data: pd.DataFrame, global_importance: bool = False,
        iv_one_sized: bool = False, GPU=False):
    """
    Run desired metric calculation on a decision tree ensemble.

    @param global_importance: Interation values can quickly fill up all the machine RAM, as there are quadratic number of them.
    To be able to run the algorithm on large datasets, when global_importance=True, we save only their sum of mean absolute values across the trees.
    While it makes the result not useful it let us run WOODELF on large datasets and test its running time.
    """
    values = {}
    for tree in tqdm(preprocess_trees, desc="Computing the values"):
        if global_importance:
            current_values = {}
            calculation_given_preprocessed_tree(tree, consumer_data, values=current_values, GPU=GPU)
            for key in current_values:
                if key not in values:
                    values[key] = 0
                values[key] += np.abs(current_values[key]).sum() / len(current_values[key])
        else:
            calculation_given_preprocessed_tree(tree, consumer_data, values=values, GPU=GPU)

    # Improvement 4 of Sec. 9.1
    if iv_one_sized:
        fill_mirror_pairs(values)

    return values


def fill_mirror_pairs(values):
    all_keys = list(values.keys())
    for f1, f2 in all_keys:  # TODO support more the length 2 subsets...
        assert (f2, f1) not in values
        values[(f2, f1)] = values[(f1, f2)]


def calculate_background_metric(model, consumer_data: pd.DataFrame, background_data: pd.DataFrame,
                                metric: CubeMetric,
                                global_importance: bool = False, GPU=False,
                                path_to_matrixes_calculator: PathToMatricesAbstractCls = None):
    """
    The WOODELF algorithm!!!

    Gets a decision tree ensemble model, consumer data of size n,
    background data for size m and a desired metric to calculate.
    Compute the desired metric in O(n+m)
    """
    model_objs = load_decision_tree_ensemble_model(model, list(consumer_data.columns))
    if path_to_matrixes_calculator is None:
        max_depth = max([t.depth for t in model_objs])
        path_to_matrixes_calculator = SimplePathToMatrices(metric=metric, max_depth=max_depth, GPU=GPU)
    if GPU:
        consumer_data = get_cupy_data(model_objs, consumer_data)
        background_data = get_cupy_data(model_objs, background_data)
    preprocessed_trees = []
    for tree in tqdm(model_objs, desc="Preprocessing the trees"):
        preprocessed_trees.append(preprocess_tree_background(tree, background_data, depth=tree.depth,
                                                             path_to_matrixes_calculator=path_to_matrixes_calculator,
                                                             GPU=GPU))

    path_to_matrixes_calculator.present_statistics()
    values = calculation_given_preprocessed_tree_ensemble(
        preprocessed_trees, consumer_data, global_importance,
        iv_one_sized=not metric.INTERACTION_VALUES_ORDER_MATTERS and metric.INTERACTION_VALUE, GPU=GPU
    )
    return values

def path_dependent_frequencies(tree: DecisionTreeNode):
    """
    Estimate the frequencies of the training data using the tree cover property.
    Implement Formula 9 of the article for all the leaves in the provided tree.
    """
    if tree.is_leaf():
        return {tree.index: []}

    leaves_freq_dict = {}
    inner_nodes_freq_dict = {tree.index: [1]}
    for current_node in tree.bfs():
        current_node_freq = inner_nodes_freq_dict[current_node.index]
        if current_node.is_leaf():
            leaves_freq_dict[current_node.index] = np.array(
                inner_nodes_freq_dict[current_node.index], dtype=np.float32
            )
            continue

        freqs_l = []
        for freq in current_node_freq:
            freqs_l.append((current_node.right.cover / current_node.cover) * freq)
            freqs_l.append((current_node.left.cover / current_node.cover) * freq)
        inner_nodes_freq_dict[current_node.left.index] = freqs_l

        freqs_r = []
        for freq in current_node_freq:
            # Changed the order of the 2 lines here, now left is first.
            freqs_r.append((current_node.left.cover / current_node.cover) * freq)
            freqs_r.append((current_node.right.cover / current_node.cover) * freq)
        inner_nodes_freq_dict[current_node.right.index] = freqs_r
    return leaves_freq_dict


def fast_preprocess_path_dependent(tree: DecisionTreeNode, path_to_matrixes_calculator: PathToMatricesAbstractCls):
    """
    Implement the preprocssing needed for Path-Dependent WOODELF
    """
    freq = path_dependent_frequencies(tree)
    for leaf, features_in_path in tree.get_all_leaves_with_paths():
        leaf.feature_contribution_replacement_values = path_to_matrixes_calculator.get_s_matrices(
            features_in_path, freq[leaf.index], leaf.value, path_dependent=True
        )
    return tree


def calculate_path_dependent_metric(model, consumer_data, metric: CubeMetric, global_importance: bool = False,
                                    GPU=False, path_to_matrixes_calculator: PathToMatricesAbstractCls = None):
    """
    Path-Dependent WOODELF algorithm!!

    Given a model, a consumer data and a desired metric compute the metric under the Path-Dependent assumptions.
    """
    model_objs = load_decision_tree_ensemble_model(model, list(consumer_data.columns))
    if path_to_matrixes_calculator is None:
        path_to_matrixes_calculator = SimplePathToMatrices(metric=metric, max_depth=model_objs[0].depth, GPU=GPU)
    if GPU:
        consumer_data = get_cupy_data(model_objs, consumer_data)

    preprocessed_trees = []
    for tree in tqdm(model_objs, desc="Preprocessing the trees"):
        preprocessed_trees.append(
            fast_preprocess_path_dependent(tree, path_to_matrixes_calculator=path_to_matrixes_calculator))

    print(
        f"cache misses: {path_to_matrixes_calculator.cache_miss}, cache used: {path_to_matrixes_calculator.cached_used}")
    return calculation_given_preprocessed_tree_ensemble(
        preprocessed_trees, consumer_data, global_importance,
        iv_one_sized=not metric.INTERACTION_VALUES_ORDER_MATTERS and metric.INTERACTION_VALUE, GPU=GPU
    )