from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.cube_metric import CubeMetric, ShapleyValues, BanzhafValues, ShapleyInteractionValues
from woodelf.core.decision_patterns import decision_patterns_generator, ignore_right_neighbor
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import LTSRecursivePathToSVectors
from woodelf.core.path_to_s_vectors.mn_background_p2s import MNBackgroundFasterPathToSVectors
from woodelf.core.path_to_s_vectors.woodelf_p2s import HighDepthWoodelfPathToSVectors
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

_BACKGROUND_METRICS = (ShapleyValues, BanzhafValues)
_PATH_DEPENDENT_METRICS = (ShapleyValues, BanzhafValues, ShapleyInteractionValues)


def _accumulate(values: Dict, s_matrix: Dict, indices: np.ndarray, GPU: bool, add_mirror: bool = False):
    for feature, s_vec in s_matrix.items():
        s_vec = cp.asarray(s_vec) if GPU else np.ascontiguousarray(s_vec)
        feature_values = s_vec[indices]
        if feature not in values:
            values[feature] = feature_values
        else:
            values[feature] += feature_values

        if add_mirror:
            (f1,f2) = feature
            if (f2,f1) not in values:
                values[(f2,f1)] = feature_values
            else:
                values[(f2,f1)] += feature_values


def hybrid_background_single_tree(
    tree: DecisionTreeNode,
    consumer_data: pd.DataFrame,
    background_data: pd.DataFrame,
    values: Dict[Any, float],
    high_depth_p2s: Optional[HighDepthWoodelfPathToSVectors],
    mn_p2s: Optional[MNBackgroundFasterPathToSVectors],
    GPU: bool,
    use_neighbor_leaf_trick: bool,
    use_sparse_approaches: bool = False,
):
    background_len = len(background_data)
    consumer_len = len(consumer_data)
    leaves_to_path = tree.get_nodes_to_path_dict()
    consumer_gen    = decision_patterns_generator(tree, consumer_data,    GPU, use_neighbor_leaf_trick)
    background_gen  = decision_patterns_generator(tree, background_data,  GPU, use_neighbor_leaf_trick)

    for leaf, consumer_patterns in consumer_gen:
        path = leaves_to_path[leaf.index]
        features_in_path = get_unique_features_in_path(path)
        D = len(features_in_path)
        w_neighbor = leaf.parent.right.value if ignore_right_neighbor(leaf, path, use_neighbor_leaf_trick) else None

        leaf_b, background_patterns = next(background_gen)
        assert leaf_b.index == leaf.index

        depth_to_use_mn = D > 20 or (D > 10 and (2 ** D > min(10000, consumer_len) * min(10000, background_len)))
        if mn_p2s is not None and (use_sparse_approaches or depth_to_use_mn):
            inverse, unique_c = pd.factorize(consumer_patterns, sort=False)
            s_matrix = mn_p2s.get_background_s_matrix(
                features_in_path, unique_c,
                background_patterns, leaf.value, w_neighbor
            )
            _accumulate(values, s_matrix, inverse, GPU)
        else:
            s_matrix = high_depth_p2s.get_background_s_matrix(
                features_in_path, consumer_patterns,
                background_patterns, leaf.value, w_neighbor
            )
            _accumulate(values, s_matrix, consumer_patterns, GPU)


def hybrid_path_dependent_single_tree(
    tree: DecisionTreeNode,
    consumer_data: pd.DataFrame,
    values: Dict[Any, float],
    high_depth_p2s: Optional[HighDepthWoodelfPathToSVectors],
    lts_p2s: Optional[LTSRecursivePathToSVectors],
    GPU: bool,
    use_neighbor_leaf_trick: bool,
    use_sparse_approaches: bool = False,
):
    leaves_to_path = tree.get_nodes_to_path_dict()
    consumer_gen = decision_patterns_generator(tree, consumer_data, GPU, use_neighbor_leaf_trick)

    for leaf, consumer_patterns in consumer_gen:
        path = leaves_to_path[leaf.index]
        features_in_path = get_unique_features_in_path(path)
        w_neighbor = leaf.parent.right.value if ignore_right_neighbor(leaf, path, use_neighbor_leaf_trick) else None
        covers = np.array(get_covers_vector(path + [leaf], features_in_path))

        use_lts = lts_p2s is not None and (use_sparse_approaches or len(features_in_path) > 10)
        if use_lts:
            inverse, unique_c = pd.factorize(consumer_patterns, sort=False)
            s_matrix = lts_p2s.get_path_dependent_s_matrix(
                features_in_path, unique_c, covers, leaf.value, w_neighbor
            )
            _accumulate(values, s_matrix, inverse, GPU)
        else:
            s_matrix = high_depth_p2s.get_path_dependent_s_matrix(
                features_in_path, consumer_patterns, covers, leaf.value, w_neighbor
            )
            add_mirror = not high_depth_p2s.metric.INTERACTION_VALUES_ORDER_MATTERS and high_depth_p2s.metric.INTERACTION_VALUE
            _accumulate(values, s_matrix, consumer_patterns, GPU, add_mirror)


def linear_tree_shap_meets_woodelf(
    model,
    consumer_data: pd.DataFrame,
    metric: CubeMetric,
    GPU: bool = False,
    use_neighbor_leaf_trick: bool = True,
    model_was_loaded: bool = False,
):
    assert isinstance(metric, _PATH_DEPENDENT_METRICS), (
        f"linear_tree_shap_meets_woodelf supports only {[m.__name__ for m in _PATH_DEPENDENT_METRICS]}. Got {type(metric).__name__}."
    )
    return hybrid_woodelf(
        model, consumer_data, None, metric,
        GPU=GPU, use_neighbor_leaf_trick=use_neighbor_leaf_trick,
        model_was_loaded=model_was_loaded, use_sparse_approaches=True,
    )


def hybrid_woodelf(
    model,
    consumer_data: pd.DataFrame,
    background_data: Optional[pd.DataFrame],
    metric: CubeMetric,
    GPU: bool = False,
    use_neighbor_leaf_trick: bool = True,
    model_was_loaded: bool = False,
    use_sparse_approaches: bool = False,
):
    """
    Hybrid WOODELF: selects the best computation strategy per leaf based on depth and metric.

    Background mode (background_data provided):
      - Shallow paths (D <= 10) or not ShapleyValues/BanzhafValues metrics: HighDepthWoodelfPathToSVectors.
      - Deep paths (D > 10) with ShapleyValues/BanzhafValues: chooses between
        HighDepthWoodelfPathToSVectors (if 2^D <= unique_consumers * min(10000, |background|)
        and D <= 20) and MNBackgroundFasterPathToSVectors otherwise.

    Path-dependent mode (background_data is None):
      - If max_depth <= 10 or metric not in [ShapleyValues, BanzhafValues,
        ShapleyInteractionValues]: HighDepthWoodelfPathToSVectors only.
      - Otherwise: LTSRecursivePathToSVectors for D > 10 leaves, HighDepth for the rest.

    use_sparse_approaches: if True, always use MNBackgroundFasterPathToSVectors (background) or
      LTSRecursivePathToSVectors (path-dependent) and never HighDepthWoodelfPathToSVectors.
      Intended for testing.
    """
    if not model_was_loaded:
        model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))

    effective_depth = min(model.max_depth, len(consumer_data.columns))
    is_background = background_data is not None

    needs_mn  = is_background and (use_sparse_approaches or (effective_depth > 10 and isinstance(metric, _BACKGROUND_METRICS)))
    needs_lts = not is_background and (use_sparse_approaches or (effective_depth > 10 and isinstance(metric, _PATH_DEPENDENT_METRICS)))

    # On shallow trees with no sparse override the hybrid collapses to plain HighDepth — delegate directly.
    if not use_sparse_approaches and not needs_mn and not needs_lts:
        return woodelf_for_high_depth(
            model, consumer_data, background_data, metric, GPU=GPU,
            use_neighbor_leaf_trick=use_neighbor_leaf_trick, model_was_loaded=True,
        )

    high_depth_p2s = HighDepthWoodelfPathToSVectors(metric=metric, max_depth=effective_depth, GPU=GPU) if not use_sparse_approaches else None
    mn_p2s  = MNBackgroundFasterPathToSVectors(metric=metric, max_depth=effective_depth) if needs_mn  else None
    lts_p2s = LTSRecursivePathToSVectors(metric=metric, max_depth=effective_depth, GPU=GPU) if needs_lts else None

    # TODO think how this connects with lts and mn approaches
    data_len = len(consumer_data) + (0 if not is_background else len(background_data))
    if model.max_depth > 12 or data_len < 10 * (2 ** model.max_depth):
        use_neighbor_leaf_trick = False

    values = {}
    for tree in tqdm(model.trees, desc=f"Computing {metric.__class__.__name__} using WOODELF"):
        if is_background:
            hybrid_background_single_tree(
                tree, consumer_data, background_data, values,
                high_depth_p2s, mn_p2s, GPU, use_neighbor_leaf_trick, use_sparse_approaches,
            )
        else:
            hybrid_path_dependent_single_tree(
                tree, consumer_data, values,
                high_depth_p2s, lts_p2s, GPU, use_neighbor_leaf_trick, use_sparse_approaches,
            )

    if high_depth_p2s is not None:
        high_depth_p2s.present_statistics()
    if lts_p2s is not None:
        lts_p2s.present_statistics()

    return values
