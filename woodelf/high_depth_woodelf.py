from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.cube_metric import CubeMetric
from woodelf.core.decision_patterns import consumer_and_background_decision_patterns_generator, ignore_right_neighbor
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.core.path_to_s_vectors.base_p2s import PathToSVectors, compute_f_from_patterns
from woodelf.core.path_to_s_vectors.woodelf_p2s import HighDepthWoodelfPathToSVectors
from woodelf.core.utils import get_unique_features_in_path, get_covers_vector
from woodelf.simple_woodelf import get_cupy_data, fill_mirror_pairs


try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False


def woodelf_for_high_depth_single_tree(
        tree: DecisionTreeNode, consumer_data: pd.DataFrame, background_data: pd.DataFrame,
        values: Dict[Any, float], path_to_matrices_calculator: PathToSVectors, GPU: bool = False,
        use_neighbor_leaf_trick: bool = True, global_importance: bool = False, cache_to_use: Dict = None, cache_to_fill: Dict = None
):
    """
    Run the woodelf algorithm that is optimized for a high depth trees on a single tree
    """
    leaves_to_path = tree.get_nodes_to_path_dict()
    is_background = background_data is not None

    patterns_generator = consumer_and_background_decision_patterns_generator(
        tree, consumer_data, background_data, GPU, use_neighbor_leaf_trick
    )
    for leaf, consumer_patterns, background_patterns in patterns_generator:
        path = leaves_to_path[leaf.index]
        unique_features_in_path = get_unique_features_in_path(path)

        w_neighbor = leaf.parent.right.value if ignore_right_neighbor(leaf, path, use_neighbor_leaf_trick) else None

        if is_background:
            if cache_to_use is not None and leaf.index in cache_to_use and isinstance(path_to_matrices_calculator, HighDepthWoodelfPathToSVectors):
                s_matrix = path_to_matrices_calculator.compose_with_neighbor_trick(
                    unique_features_in_path, cache_to_use[leaf.index], leaf.value, w_neighbor
                )
            else:
                s_matrix = path_to_matrices_calculator.get_background_s_matrix(
                    unique_features_in_path, consumer_patterns, background_patterns, leaf.value, w_neighbor
                )
                if cache_to_fill is not None:
                    depth = len(unique_features_in_path)
                    cache_to_fill[leaf.index] = compute_f_from_patterns(background_patterns, depth, GPU)
        else:
            covers = np.array(get_covers_vector(path + [leaf], unique_features_in_path))
            s_matrix = path_to_matrices_calculator.get_path_dependent_s_matrix(
                unique_features_in_path, consumer_patterns, covers, leaf.value, w_neighbor
            )

        for feature, s_vec in s_matrix.items():
            s_vec_casted = cp.asarray(s_vec) if GPU else np.ascontiguousarray(s_vec)
            feature_values = s_vec_casted[consumer_patterns]
            if global_importance:
                feature_values = cp.mean(feature_values) if GPU else np.mean(feature_values)
            if feature not in values:
                values[feature] = feature_values
            else:
                values[feature] += feature_values


def woodelf_for_high_depth(
        model, consumer_data: pd.DataFrame, background_data: Optional[pd.DataFrame], metric: CubeMetric,
        GPU: bool=False, use_neighbor_leaf_trick: bool=True,
        path_to_matrices_calculator: PathToSVectors = None,
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

    # As we use unique feature decision patterns the length of each pattern can not be longer by the max_depth or the number of unique feature in the dataset
    effective_max_decision_pattern_length = min(model_obj.max_depth, len(consumer_data.columns))
    if path_to_matrices_calculator is None:
        path_to_matrices_calculator = HighDepthWoodelfPathToSVectors(metric=metric, max_depth=effective_max_decision_pattern_length, GPU=GPU)
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
