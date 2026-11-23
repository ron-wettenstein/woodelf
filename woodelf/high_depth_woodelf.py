from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.cube_metric import CubeMetric
from woodelf.core.decision_patterns import decision_patterns_generator, ignore_right_neighbor
from woodelf.core.path_to_s_vectors.base_p2s import PathToSVectors, WoodelfPathToSVectors
from woodelf.core.path_to_s_vectors.woodelfhd_v1_p2s import HighDepthWoodelfV1PathToSVectors
from woodelf.core.path_to_s_vectors.woodelfhd_v2_p2s import HighDepthWoodelfV2PathToSVectors
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.core.utils import get_unique_features_in_path, get_covers_vector, consumer_and_background_to_dict
from woodelf.simple_woodelf import fill_mirror_pairs

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False


def woodelf_for_high_depth_single_tree(
        tree: DecisionTreeNode, consumer_data: pd.DataFrame, background_data: pd.DataFrame,
        values: Dict[Any, float], path_to_matrices_calculator: PathToSVectors, GPU: bool = False,
        use_neighbor_leaf_trick: bool = True, global_importance: bool = False
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
        unique_features_in_path = get_unique_features_in_path(path)

        w_neighbor = leaf.parent.right.value if ignore_right_neighbor(leaf, path, use_neighbor_leaf_trick) else None

        if path_to_matrices_calculator.COMPUTES_CONSUMER_VALUES:
            if is_background:
                leaf_b, background_patterns = next(background_patterns_generator)
                assert leaf_b.index == leaf.index

                consumer_values = path_to_matrices_calculator.get_background_consumer_values(
                    unique_features_in_path, consumer_patterns, background_patterns, leaf.value, w_neighbor,
                    values=None if global_importance else values
                )
            else:
                covers = np.array(get_covers_vector(path + [leaf], unique_features_in_path))
                consumer_values = path_to_matrices_calculator.get_path_dependent_consumer_values(
                    unique_features_in_path, consumer_patterns, covers, leaf.value, w_neighbor,
                    values=None if global_importance else values
                )
            if global_importance:
                for feature, feature_values in consumer_values.items():
                    add_feature_values(values, feature, feature_values, GPU, global_importance)
            continue

        if is_background:
            leaf_b, background_patterns = next(background_patterns_generator)
            assert leaf_b.index == leaf.index

            s_matrix = path_to_matrices_calculator.get_background_s_matrix(
                unique_features_in_path, consumer_patterns, background_patterns, leaf.value, w_neighbor
            )
        else:
            covers = np.array(get_covers_vector(path + [leaf], unique_features_in_path))
            s_matrix = path_to_matrices_calculator.get_path_dependent_s_matrix(
                unique_features_in_path, consumer_patterns, covers, leaf.value, w_neighbor
            )

        for feature, s_vec in s_matrix.items():
            s_vec_casted = cp.asarray(s_vec) if GPU else np.ascontiguousarray(s_vec)
            add_feature_values(values, feature, s_vec_casted[consumer_patterns], GPU, global_importance)


def add_feature_values(values: Dict[Any, float], feature, feature_values, GPU: bool, global_importance: bool):
    """
    Add a leaf's values of a feature to the values computed so far (their mean when global_importance is True).
    """
    if global_importance:
        feature_values = cp.mean(feature_values) if GPU else np.mean(feature_values)
    if feature not in values:
        values[feature] = feature_values
    else:
        values[feature] += feature_values


def create_high_depth_woodelf_p2s(metric: CubeMetric, max_depth: int, GPU: bool = False) -> WoodelfPathToSVectors:
    """
    HighDepthWoodelfV2PathToSVectors when it supports the metric (Shapley and Banzhaf values and interaction values),
    otherwise HighDepthWoodelfV1PathToSVectors.
    """
    if HighDepthWoodelfV2PathToSVectors.supports_metric(metric):
        return HighDepthWoodelfV2PathToSVectors(metric=metric, max_depth=max_depth, GPU=GPU)
    return HighDepthWoodelfV1PathToSVectors(metric=metric, max_depth=max_depth, GPU=GPU)


def woodelf_for_high_depth(
        model, consumer_data: pd.DataFrame, background_data: Optional[pd.DataFrame], metric: CubeMetric,
        GPU: bool=False, use_neighbor_leaf_trick: bool=True,
        path_to_matrices_calculator: PathToSVectors = None,
        global_importance: bool = False, model_was_loaded: bool = False,
        verbose: bool = True
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
    @param verbose: If True show the progress bar and print the timing statistics.
    @return The computed values as a dictionary that maps between features/features pairs to np.arrays with
    the values.
    """
    if model_was_loaded:
        model_obj = model
    else:
        model_obj = load_decision_tree_ensemble_model(model, list(consumer_data.keys()))

    # As we use unique feature decision patterns the length of each pattern can not be longer by the max_depth or the number of unique feature in the dataset
    effective_max_decision_pattern_length = min(model_obj.max_depth, len(consumer_data.keys()))
    if path_to_matrices_calculator is None:
        path_to_matrices_calculator = create_high_depth_woodelf_p2s(metric=metric, max_depth=effective_max_decision_pattern_length, GPU=GPU)
    consumer_data, background_data = consumer_and_background_to_dict(consumer_data, background_data, GPU, model_obj)

    data_len = len(next(iter(consumer_data.values()))) + (0 if background_data is None else len(next(iter(background_data.values()))))
    if GPU and (model_obj.max_depth > 12 or data_len < 10 * (2 ** model_obj.max_depth)):
        # If the max depth is too large or the data size is smaller than (or within the same order of magnitude as)
        # the number of patterns - skip the neighbor leaf trick. It will be slower to apply it than to return the
        # regular computation.
        # On CPU the trick is always applied: in the benchmarks (docs/woodelf_hybrid_thresholds.html) it was faster in
        # 74 of 138 configs, within 5% in 59 and at most ~17% slower in the other 5, also for small data and deep
        # trees. The GPU keeps this cutoff until it is benchmarked on a GPU.
        use_neighbor_leaf_trick = False


    values = {}
    for tree_index, tree in tqdm(list(enumerate(model_obj.trees)),
                                 desc="Preprocessing the trees and computing SHAP",
                                 disable=not verbose):
        woodelf_for_high_depth_single_tree(
            tree, consumer_data, background_data, values, path_to_matrices_calculator, GPU,
            use_neighbor_leaf_trick, global_importance,
        )

    if metric.should_mirror():
        fill_mirror_pairs(values)

    if verbose:
        path_to_matrices_calculator.present_statistics()

    return values
