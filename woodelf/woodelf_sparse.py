from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.cube_metric import CubeMetric, ShapleyValues, BanzhafValues, ShapleyInteractionValues
from woodelf.core.decision_patterns import decision_patterns_generator, ignore_right_neighbor
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import LTSRecursivePathToSVectors
from woodelf.core.path_to_s_vectors.mn_background_p2s import MNBackgroundFasterPathToSVectors, MNBackgroundPathToSVectors
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.core.utils import get_unique_features_in_path, get_covers_vector
from woodelf.high_depth_woodelf import woodelf_for_high_depth

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError:
    cp = None
    IMPORTED_CP = False

_MAX_DEPTH_FOR_HIGH_WOODELF = 12
_MAX_DEPTH_FOR_PATH_DEPENDENT_HIGH_WOODELF = 10
_SUPPORTED_SPARSE_BACKGROUND_METRICS = (ShapleyValues, BanzhafValues)
_SUPPORTED_SPARSE_PATH_DEPENDENT_METRICS = (ShapleyValues, BanzhafValues, ShapleyInteractionValues)


def use_sparse_approach(depth, metric, is_background):
    if is_background:
        return depth > _MAX_DEPTH_FOR_HIGH_WOODELF and isinstance(metric, _SUPPORTED_SPARSE_BACKGROUND_METRICS)
    return depth > _MAX_DEPTH_FOR_PATH_DEPENDENT_HIGH_WOODELF and isinstance(metric, _SUPPORTED_SPARSE_PATH_DEPENDENT_METRICS)


def _accumulate(values: Dict, s_matrix: Dict, indices: np.ndarray, GPU: bool):
    for feature, s_vec in s_matrix.items():
        s_vec = cp.asarray(s_vec) if GPU else np.ascontiguousarray(s_vec)
        feature_values = s_vec[indices]
        if feature not in values:
            values[feature] = feature_values
        else:
            values[feature] += feature_values


def sparse_background_single_tree(
    tree: DecisionTreeNode,
    consumer_data: pd.DataFrame,
    background_data: pd.DataFrame,
    values: Dict[Any, float],
    mn_p2s: Optional[MNBackgroundFasterPathToSVectors],
    GPU: bool,
    use_neighbor_leaf_trick: bool,
):
    leaves_to_path = tree.get_nodes_to_path_dict()
    consumer_gen    = decision_patterns_generator(tree, consumer_data,    GPU, use_neighbor_leaf_trick)
    background_gen  = decision_patterns_generator(tree, background_data,  GPU, use_neighbor_leaf_trick)

    for leaf, consumer_patterns in consumer_gen:
        path = leaves_to_path[leaf.index]
        features_in_path = get_unique_features_in_path(path)
        w_neighbor = leaf.parent.right.value if ignore_right_neighbor(leaf, path, use_neighbor_leaf_trick) else None

        leaf_b, background_patterns = next(background_gen)
        # assert leaf_b.index == leaf.index

        inverse, unique_c = pd.factorize(consumer_patterns, sort=False)
        s_matrix = mn_p2s.get_background_s_matrix(
            features_in_path, unique_c,
            background_patterns, leaf.value, w_neighbor
        )
        _accumulate(values, s_matrix, inverse, GPU)


def sparse_path_dependent_single_tree(
    tree: DecisionTreeNode,
    consumer_data: pd.DataFrame,
    values: Dict[Any, float],
    lts_p2s: Optional[LTSRecursivePathToSVectors],
    GPU: bool,
    use_neighbor_leaf_trick: bool,
):
    leaves_to_path = tree.get_nodes_to_path_dict()
    consumer_gen = decision_patterns_generator(tree, consumer_data, GPU, use_neighbor_leaf_trick)

    for leaf, consumer_patterns in consumer_gen:
        path = leaves_to_path[leaf.index]
        features_in_path = get_unique_features_in_path(path)
        w_neighbor = leaf.parent.right.value if ignore_right_neighbor(leaf, path, use_neighbor_leaf_trick) else None
        covers = np.array(get_covers_vector(path + [leaf], features_in_path))

        inverse, unique_c = pd.factorize(consumer_patterns, sort=False)
        s_matrix = lts_p2s.get_path_dependent_s_matrix(
            features_in_path, unique_c, covers, leaf.value, w_neighbor
        )
        _accumulate(values, s_matrix, inverse, GPU)


def linear_tree_shap_meets_woodelf(
    model,
    consumer_data: pd.DataFrame,
    metric: CubeMetric,
    GPU: bool = False,
    use_neighbor_leaf_trick: bool = True,
    model_was_loaded: bool = False,
):
    assert isinstance(metric, _SUPPORTED_SPARSE_PATH_DEPENDENT_METRICS), (
        f"linear_tree_shap_meets_woodelf supports only {[m.__name__ for m in _SUPPORTED_SPARSE_PATH_DEPENDENT_METRICS]}. Got {type(metric).__name__}."
    )
    return woodelf_sparse(
        model, consumer_data, None, metric,
        GPU=GPU, use_neighbor_leaf_trick=use_neighbor_leaf_trick,
        model_was_loaded=model_was_loaded
    )


def woodelf_sparse(
    model,
    consumer_data: pd.DataFrame,
    background_data: Optional[pd.DataFrame],
    metric: CubeMetric,
    GPU: bool = False,
    use_neighbor_leaf_trick: bool = True,
    model_was_loaded: bool = False,
    mn_p2s_class=None,
):
    """
    Sparse WOODELF: uses pattern-factorized sparse algorithms for all leaves.

    Background mode (background_data provided): MNBackgroundFasterPathToSVectors (default) or mn_p2s_class.
    Path-dependent mode (background_data is None): LTSRecursivePathToSVectors.
    """
    if not model_was_loaded:
        model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))

    effective_depth = min(model.max_depth, len(consumer_data.columns))
    is_background = background_data is not None

    if mn_p2s_class is None:
        mn_p2s_class = MNBackgroundFasterPathToSVectors
    mn_p2s  = mn_p2s_class(metric=metric, max_depth=effective_depth) if is_background else None
    lts_p2s = LTSRecursivePathToSVectors(metric=metric, max_depth=effective_depth, GPU=GPU) if not is_background else None

    # TODO think how this connects with lts and mn approaches
    if isinstance(metric, ShapleyInteractionValues) and not is_background:
        use_neighbor_leaf_trick = False # Linear TreeSHAP doesn't support the neighbor_leaf_trick for interaction values

    values = {}
    for tree in tqdm(model.trees, desc=f"Computing {metric.__class__.__name__} using WOODELF"):
        if is_background:
            sparse_background_single_tree(
                tree, consumer_data, background_data, values,
                mn_p2s, GPU, use_neighbor_leaf_trick
            )
        else:
            sparse_path_dependent_single_tree(
                tree, consumer_data, values,
                lts_p2s, GPU, use_neighbor_leaf_trick
            )

    if mn_p2s is not None:
        mn_p2s.present_statistics()
    if lts_p2s is not None:
        lts_p2s.present_statistics()

    return values


def hybrid_woodelf(
        model,
        consumer_data: pd.DataFrame,
        background_data: Optional[pd.DataFrame],
        metric: CubeMetric,
        GPU: bool = False,
        use_neighbor_leaf_trick: bool = True,
        model_was_loaded: bool = False,
        mn_p2s_class=None,
):
    """
    Hybrid WOODELF: selects the best computation strategy (sparse woodelf or woodelf_for_high_depths) considering the tree depth and metric.
    """
    if not model_was_loaded:
        model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))

    effective_depth = min(model.max_depth, len(consumer_data.columns))
    is_background = background_data is not None

    if use_sparse_approach(effective_depth, metric, is_background):
        return woodelf_sparse(
            model, consumer_data, background_data, metric, GPU=GPU,
            use_neighbor_leaf_trick=use_neighbor_leaf_trick, model_was_loaded=True, mn_p2s_class=mn_p2s_class
        )
    else:
        data_len = len(consumer_data) + (0 if not is_background else len(background_data))
        if model.max_depth > 12 or data_len < 10 * (2 ** model.max_depth):
            use_neighbor_leaf_trick = False

        return woodelf_for_high_depth(
            model, consumer_data, background_data, metric, GPU=GPU,
            use_neighbor_leaf_trick=use_neighbor_leaf_trick, model_was_loaded=True,
        )