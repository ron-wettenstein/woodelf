from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.cube_metric import CubeMetric
from woodelf.core.decision_patterns import decision_patterns_generator, ignore_right_neighbor
from woodelf.core.path_to_s_vectors.mn_background_p2s import PersonalizedBaselinePathToSVectors
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.core.utils import get_unique_features_in_path

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError:
    cp = None
    IMPORTED_CP = False


def _personalized_single_tree(
    tree: DecisionTreeNode,
    consumer_data: pd.DataFrame,
    background_data: pd.DataFrame,
    values: Dict[Any, np.ndarray],
    p2s: PersonalizedBaselinePathToSVectors,
    GPU: bool,
    use_neighbor_leaf_trick: bool,
):
    N = len(consumer_data)
    combined_data = pd.concat([consumer_data, background_data], ignore_index=True)

    leaves_to_path = tree.get_nodes_to_path_dict()
    for leaf, combined_patterns in decision_patterns_generator(tree, combined_data, GPU, use_neighbor_leaf_trick):
        path = leaves_to_path[leaf.index]
        features_in_path = get_unique_features_in_path(path)
        w_neighbor = leaf.parent.right.value if ignore_right_neighbor(leaf, path, use_neighbor_leaf_trick) else None

        consumer_patterns  = combined_patterns[:N]
        background_patterns = combined_patterns[N:]

        s_matrix = p2s.get_background_s_matrix(
            features_in_path, consumer_patterns, background_patterns, leaf.value, w_neighbor
        )
        for feature, s_vec in s_matrix.items():
            s_vec = cp.asarray(s_vec) if GPU else np.ascontiguousarray(s_vec)
            if feature not in values:
                values[feature] = s_vec
            else:
                values[feature] += s_vec


def personalized_baseline_woodelf(
    model,
    consumer_data: pd.DataFrame,
    background_data: pd.DataFrame,
    metric: CubeMetric,
    GPU: bool = False,
    use_neighbor_leaf_trick: bool = True,
    model_was_loaded: bool = False,
):
    """
    Compute personalized baseline SHAP/Banzhaf values for a decision tree ensemble.

    Unlike standard background-based approaches that average over all background rows,
    personalized baseline pairs each consumer row i with exactly one background row i.
    This allows per-instance baselines — useful when each consumer has a natural reference
    (e.g. a counterfactual, a matched control, or a prior state).

    @param model: A fitted decision-tree ensemble (XGBoost, LightGBM, RandomForest, ...).
    @param consumer_data: The data to explain. Each row i is explained relative to background row i.
    @param background_data: The reference (baseline) dataset. Must have the same length as consumer_data.
    @param metric: The metric to compute: ShapleyValues(), BanzhafValues(), ...
    @param GPU: If True, accelerates the run using GPU. Requires CuPy (pip install cupy).
    @param use_neighbor_leaf_trick: If True, applies a mathematical shortcut for sibling leaf pairs,
    improving performance on large datasets.
    @param model_was_loaded: If True, treats model as an already-parsed internal model object,
    skipping the parsing step.

    @return A dictionary mapping each feature name to a NumPy array of length n (one value per consumer row).
    """
    assert len(consumer_data) == len(background_data), (
        f"personalized_baseline_woodelf requires consumer and background datasets of the same length. "
        f"Got {len(consumer_data)} and {len(background_data)}."
    )

    if not model_was_loaded:
        model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))

    effective_depth = min(model.max_depth, len(consumer_data.columns))
    p2s = PersonalizedBaselinePathToSVectors(metric=metric, max_depth=effective_depth)

    values = {}
    for tree in tqdm(model.trees, desc=f"Computing {metric.__class__.__name__} using personalized baseline WOODELF"):
        _personalized_single_tree(
            tree, consumer_data, background_data, values, p2s, GPU, use_neighbor_leaf_trick
        )

    return values
