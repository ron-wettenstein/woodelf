from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.cube_metric import CubeMetric
from woodelf.decision_patterns import decision_patterns_generator, ignore_right_neighbor
from woodelf.decision_trees_ensemble import DecisionTreeNode
from woodelf.parse_models import load_decision_tree_ensemble_model
from woodelf.path_to_matrices import PathToMatricesAbstractCls, HighDepthPathToMatrices
from woodelf.simple_woodelf import get_cupy_data, fill_mirror_pairs


try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False


def compute_f(patterns, path_depth: int, GPU: bool = False):
    """
    f is a simple patterns.value_counts(normalized=True).
    Do it more efficiently using bincount
    """
    if GPU:
        return cp.bincount(patterns, minlength=2 ** path_depth) / len(patterns)
    return np.bincount(patterns, minlength=2 ** path_depth) / len(patterns)

def compute_path_dependent_f(path: List[DecisionTreeNode], unique_features_in_path: List[Any]):
    """
    Estimate the frequencies of the training data using the tree cover property.
    Implement Formula 9 of the article for the provided path.

    Note: as we want a path with only unique features, we unite repeating features by multiplying their proceed ratios.
    For example if for the split a<3 only 0.4 continue to the next node in the path, and in the split a<1 only 0.6
    continues that in total 0.4*0.6=0.24 continue in the path for the "a" feature and 1-0.24=0.76 diverge the path.
    """
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
    return f.astype(np.float32)

def neighbor_vector(f, GPU):
    """
    If both the right and left nodes are leaves use improvement 3 of Sec. 9.1 (improvement of line 26)
    See also the comment in compute_f_of_neighbor.
    Given leaves l_i, l_{i+1} s.t. there is an inner node n where n.left=l_i and n.right=l_{i+1}:
    - mark the s vector of feature f and leaf l_i as s_i = [a1,a2,a3,...,an] (In our case it is s_left)
    - mark the s vector of feature f and leaf l_{i+1} as s_{i+1} = [b1,b2,b3,...,bn] (In our case it is s_right)
    A trivial numpy indexing for feature f and the two leaves is:
        [a1,a2,a3,...,an][ patterns ] + [b1,b2,b3,...,bn][ patterns ]

    Utilizing the property explained in comment in preprocess_tree_background, we can run the equivalent numpy indexing:
        [a1+b2, a2+b1, a3+b4, a4+b3,...,a_{n-1}+bn, an+b_{n-1}][ patterns ]
    This saves half of the numpy indexing operations
    """
    if GPU:
        idx = cp.arange(len(f))
        neighbor_f = cp.zeros_like(f)
    else:
        idx = np.arange(len(f))
        neighbor_f = np.zeros_like(f)

    neighbor_f_shift_left = neighbor_f.copy()
    neighbor_f_shift_left[1:] = f[:-1]  # shift the array to the left 1 bit
    neighbor_f_shift_left[(idx & 1) == 0] = 0  # Zero all elements that are in an even place in the current division

    neighbor_f_shift_right = neighbor_f.copy()
    neighbor_f_shift_right[:-1] = f[1:]  # shift the array to the right bit
    neighbor_f_shift_right[(idx & 1) == 1] = 0  # Zero all elements that are in an odd place in the current division
    return neighbor_f_shift_left + neighbor_f_shift_right



def compute_values_using_s_vectors(values, s_vectors, consumer_patterns, GPU: bool, global_importance: bool = False):
    """
    Use numpy indexing to fetch the required values from the s_vectors and add them to the values' dict.
    """
    for feature, s_vector in s_vectors.items():
        if GPU:
            replacements_array = cp.asarray(s_vector)
        else:
            replacements_array = np.ascontiguousarray(s_vector)

        # This is where the numpy indexing occur (improvement 6 of Sec. 9.1):
        current_contribution = replacements_array[consumer_patterns]
        if global_importance:
            if GPU:
                current_contribution = cp.mean(current_contribution)
            else:
                current_contribution = np.mean(current_contribution)

        if feature not in values:
            values[feature] = current_contribution
        else:
            values[feature] += current_contribution


def compute_s_vectors_given_f_vector(leaf, f, w_neighbor, unique_features_in_path, path_to_matrices_calculator, GPU):
    if w_neighbor is None:
        return path_to_matrices_calculator.get_s_matrices(unique_features_in_path, f, leaf.value)
    else:
        s_left = path_to_matrices_calculator.get_s_matrices(unique_features_in_path, f, leaf.value)
        s_right = path_to_matrices_calculator.get_s_matrices(unique_features_in_path, neighbor_vector(f, GPU), w_neighbor)
        return {k: s_left[k] + neighbor_vector(s_right[k], GPU) for k in s_left.keys()}


def compute_background_s_vectors(
        leaf: DecisionTreeNode, background_patterns: np.array, unique_features_in_path: List[Any],
        path_to_matrices_calculator: PathToMatricesAbstractCls, w_neighbor: Optional[float] = None, GPU: bool = False,
        cache_to_use: Dict = None, cache_to_fill: Dict = None
):
    if cache_to_use is not None and leaf.index in cache_to_use:
        f = cache_to_use[leaf.index]
    else:
        depth = len(unique_features_in_path)
        f = compute_f(background_patterns, depth, GPU)

    if cache_to_fill is not None:
        cache_to_fill[leaf.index] = f

    return compute_s_vectors_given_f_vector(leaf, f, w_neighbor, unique_features_in_path, path_to_matrices_calculator, GPU)


def compute_path_dependent_s_vectors(
        leaf: DecisionTreeNode, path: List[DecisionTreeNode], unique_features_in_path: List[Any], path_to_matrices_calculator: PathToMatricesAbstractCls,
        w_neighbor: Optional[float] = None, GPU: bool = False
):
    f = compute_path_dependent_f(path + [leaf], unique_features_in_path)
    if GPU:
        f = cp.asarray(f)
    return compute_s_vectors_given_f_vector(leaf, f, w_neighbor, unique_features_in_path, path_to_matrices_calculator, GPU)


def woodelf_for_high_depth_single_tree(
        tree: DecisionTreeNode, consumer_data: pd.DataFrame, background_data: pd.DataFrame,
        values: Dict[Any, float], path_to_matrices_calculator: PathToMatricesAbstractCls, GPU: bool = False,
        use_neighbor_leaf_trick: bool = True, global_importance: bool = False, cache_to_use: Dict = None, cache_to_fill: Dict = None
):
    """
    Run the woodelf algorithm that is optimized for a high depth trees on a single tree
    """
    leaves_to_path = tree.get_nodes_to_path_dict()
    consumer_patterns_generator = decision_patterns_generator(tree, consumer_data, GPU, use_neighbor_leaf_trick)
    is_background = background_data is not None
    background_patterns_generator = None
    if is_background:
        background_patterns_generator = decision_patterns_generator(tree, background_data, GPU, use_neighbor_leaf_trick)

    for leaf, consumer_patterns in consumer_patterns_generator:
        path = leaves_to_path[leaf.index]
        unique_features_in_path = []
        for n in path:
            if n.feature_name not in unique_features_in_path:
                unique_features_in_path.append(n.feature_name)

        w_neighbor = leaf.parent.right.value if ignore_right_neighbor(leaf, path, use_neighbor_leaf_trick) else None

        if is_background:
            leaf_b, background_patterns = next(background_patterns_generator)
            assert leaf_b.index == leaf.index
            s_vectors = compute_background_s_vectors(
                leaf, background_patterns, unique_features_in_path, path_to_matrices_calculator, w_neighbor, GPU, cache_to_use, cache_to_fill
            )
        else:
            s_vectors = compute_path_dependent_s_vectors(
                leaf, path, unique_features_in_path, path_to_matrices_calculator, w_neighbor, GPU
            )
        compute_values_using_s_vectors(values, s_vectors, consumer_patterns, GPU, global_importance)


def woodelf_for_high_depth(
        model, consumer_data: pd.DataFrame, background_data: Optional[pd.DataFrame], metric: CubeMetric,
        GPU: bool=False, use_neighbor_leaf_trick: bool=True,
        path_to_matrices_calculator: PathToMatricesAbstractCls = None,
        global_importance: bool = False, cache_to_use: List[Dict] = None, cache_to_fill: List[Dict] = None, model_was_loaded: bool = False
):
    """
    WOODELF designed for higher depths decision trees.
    Save RAM and have a better complexity:
    Space Complexity: O((2^D)*D)
    Runtime complexity: O(mTL + nTLD + TL(2^D)*D + 3^D)

    @param model: The model to explain
    @param consumer_data: The data to explain its predictions
    @param background_data: A reference dataset defining the data distribution of the population.
    Using the trainset as a background is a solid choice.
    @param metric: The metric to compute: ShapleyValues(), ShapleyInteractionValues(), BanzhafValues(), ...
    @param GPU: If True accelerates the run using GPU. Make sure CuPy is installed (run: pip install cupy)
    @param use_neighbor_leaf_trick: If True save some time by using a mathematical trick around leaves that
    share a common parent. This is highly effective when the data is large. If the data is small passing False
    might provide a better results
    @param path_to_matrices_calculator: An object used to compute M matrices and s vectors, central parts of the
    algorithm. It uses cache, reusing the same object in several runs can save some time (not that significant
    on large/medium size datasets)
    @param global_importance: If true return the average value across all consumer data rows. Used to
    save RAM.
    @param cache_to_use: Cache to use and save some time (on Background approach only)
    @param cache_to_fill: Fill the given cache so next time will be faster (on Background approach only)

    @return The computed values as a dictionary that maps between features/features pairs to np.arrays with
    the values.
    """
    if model_was_loaded:
        model_obj = model
    else:
        model_obj = load_decision_tree_ensemble_model(model, list(consumer_data.columns))

    if path_to_matrices_calculator is None:
        path_to_matrices_calculator = HighDepthPathToMatrices(metric=metric, max_depth=model_obj.max_depth, GPU=GPU)
    if GPU:
        consumer_data = get_cupy_data(model_obj, consumer_data)
        if background_data is not None:
            background_data = get_cupy_data(model_obj, background_data)

    data_len = len(consumer_data) + (0 if background_data is None else len(background_data))
    if model_obj.max_depth > 12 or data_len < 10 * (2 ** model_obj.max_depth):
        # If the max depth is too large or the data size is smaller than (or within the same order of magnitude as)
        # the number of patterns - skip the neighbor leaf trick. It will be slower to apply it than to return the
        # regular computation.
        use_neighbor_leaf_trick = False


    values = {}
    for tree_index, tree in tqdm(list(enumerate(model_obj.trees)), desc="Preprocessing the trees and computing SHAP"):
        woodelf_for_high_depth_single_tree(
            tree, consumer_data, background_data, values, path_to_matrices_calculator, GPU,
            use_neighbor_leaf_trick, global_importance,
            cache_to_use[tree_index] if cache_to_use is not None else None,
            cache_to_fill[tree_index] if cache_to_fill is not None else None
        )

    if not metric.INTERACTION_VALUES_ORDER_MATTERS and metric.INTERACTION_VALUE:
        fill_mirror_pairs(values)

    path_to_matrices_calculator.present_statistics()

    return values

