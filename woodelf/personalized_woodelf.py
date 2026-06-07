from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.cube_metric import CubeMetric
from woodelf.core.decision_patterns import decision_patterns_generator, decision_patterns_generator_for_feature_subset, ignore_right_neighbor
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
    features_subset: Optional[set] = None,
    compute_effect_on_other_features: bool = False,
):
    N = len(consumer_data)
    combined_data = pd.concat([consumer_data, background_data], ignore_index=True)

    leaves_to_path = tree.get_nodes_to_path_dict()
    if features_subset is not None:
        pattern_gen = decision_patterns_generator_for_feature_subset(
            tree, combined_data, list(features_subset), GPU, use_neighbor_leaf_trick
        )
    else:
        pattern_gen = decision_patterns_generator(tree, combined_data, GPU, use_neighbor_leaf_trick)

    for leaf, combined_patterns in pattern_gen:
        path = leaves_to_path[leaf.index]
        features_in_path = get_unique_features_in_path(path)
        w_neighbor = leaf.parent.right.value if ignore_right_neighbor(leaf, path, use_neighbor_leaf_trick) else None

        consumer_patterns  = combined_patterns[:N]
        background_patterns = combined_patterns[N:]

        s_matrix = p2s.get_background_s_matrix(
            features_in_path, consumer_patterns, background_patterns, leaf.value, w_neighbor
        )
        for feature, s_vec in s_matrix.items():
            if features_subset is not None and not compute_effect_on_other_features and feature not in features_subset:
                continue
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
    verbose: bool = True,
    features_subset: Optional[List[str]] = None,
    compute_effect_on_other_features: bool = False,
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
    @param verbose: Whether to show prints and tqdm progress bar
    @param features_subset: If provided, only computes values for leaves whose path contains at least one
    feature from this list. Values for features not in the subset are omitted from the result.
    @param compute_effect_on_other_features: When features_subset is set, also accumulate values for
    features outside the subset that appear in the same leaf paths. Useful for exact delta updates.

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

    features_set = set(features_subset) if features_subset is not None else None

    values = {}
    for tree in tqdm(model.trees, desc=f"Computing {metric.__class__.__name__} using personalized baseline WOODELF", disable=not verbose):
        _personalized_single_tree(
            tree, consumer_data, background_data, values, p2s, GPU, use_neighbor_leaf_trick,
            features_set, compute_effect_on_other_features,
        )

    return values


def apply_metric_delta(
    result: Dict[str, np.ndarray],
    prev_effect: Dict[str, np.ndarray],
    new_effect: Dict[str, np.ndarray],
    changed_features: List[str],
    n: int,
) -> None:
    """
    Apply a Group A / Group B delta update to result in-place.

    Leaves are partitioned into two groups by whether their path contains a changed feature:
      Group A — affected leaves (recomputed in prev_effect and new_effect).
      Group B — unaffected leaves (their contribution is identical before and after the change,
                so it cancels out and result is left untouched for those leaves).

    For features IN changed_features:
      They only appear in Group A leaves, so result[f] = new_effect[f] exactly.

    For features NOT IN changed_features:
      result[f] = (Group B contribution, unchanged)
                + (Group A contribution, delta-updated)
                = prev_result[f] - prev_effect[f] + new_effect[f]
    """
    zeros = np.zeros(n)

    for f in changed_features:
        if f in new_effect:
            result[f] = new_effect[f]
        elif f in result:
            del result[f]

    for f in (set(prev_effect) | set(new_effect)) - set(changed_features):
        delta = new_effect.get(f, zeros) - prev_effect.get(f, zeros)
        if f in result:
            result[f] = result[f] + delta
        else:
            result[f] = delta


def personalized_baseline_delta_update(
    model,
    consumer_data: pd.DataFrame,
    prev_background: pd.DataFrame,
    new_background: pd.DataFrame,
    features_subset: List[str],
    prev_values: Dict[str, np.ndarray],
    metric: CubeMetric,
    GPU: bool = False,
    use_neighbor_leaf_trick: bool = True,
    model_was_loaded: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Exactly updates metric values after the background of a feature subset changes,
    using two partial woodelf calls instead of one full recompute.

    The trick partitions all tree leaves into two groups:
      Group A — leaves whose path contains at least one feature from features_subset
               (affected: their background_patterns change when any subset feature's B changes).
      Group B — all other leaves
               (unaffected: no subset feature appears in their path, so their background_patterns
               are bit-for-bit identical under prev_background and new_background).

    For any feature f outside the subset:
      new_values[f] = (Group B contribution, unchanged)
                    + (Group A contribution, updated)
                    = (prev_values[f] - prev_effect[f]) + new_effect[f]

    For a feature f inside the subset:
      f only contributes from leaves where f is in the path → always Group A.
      Therefore new_values[f] = new_effect[f] exactly.

    This is exact with no approximation: Group B contributions are identical under both
    backgrounds because the subset features do not appear in those paths.

    @param model: A fitted decision-tree ensemble (XGBoost, LightGBM, RandomForest, ...).
    @param consumer_data: The data to explain (same as used to produce prev_values).
    @param prev_background: The background dataset before the change.
    @param new_background: The background dataset after the change (only subset columns differ).
    @param features_subset: Features whose background changed. Must include every feature whose
    B value differs between prev_background and new_background.
    @param prev_values: Current metric values dict (will not be mutated).
    @param metric: The metric to compute: ShapleyValues(), BanzhafValues(), ...
    @param GPU: If True, accelerates using GPU. Requires CuPy.
    @param use_neighbor_leaf_trick: Passed through to personalized_baseline_woodelf.
    @param model_was_loaded: If True, skips model parsing (model is already the internal object).

    @return Updated metric values dict with all features from prev_values adjusted by the delta.
    """
    n = len(consumer_data)
    result = dict(prev_values)

    prev_effect = personalized_baseline_woodelf(
        model, consumer_data, prev_background, metric,
        GPU=GPU, use_neighbor_leaf_trick=use_neighbor_leaf_trick,
        model_was_loaded=model_was_loaded, verbose=False,
        features_subset=features_subset, compute_effect_on_other_features=True,
    )
    new_effect = personalized_baseline_woodelf(
        model, consumer_data, new_background, metric,
        GPU=GPU, use_neighbor_leaf_trick=use_neighbor_leaf_trick,
        model_was_loaded=model_was_loaded, verbose=False,
        features_subset=features_subset, compute_effect_on_other_features=True,
    )

    apply_metric_delta(result, prev_effect, new_effect, features_subset, n)
    return result
