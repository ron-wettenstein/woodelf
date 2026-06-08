from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.cube_metric import CubeMetric
from woodelf.core.decision_patterns import decision_patterns_generator, decision_patterns_generator_for_feature_subset
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import AlwaysParticipatingLTSPathToSVectors
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.core.utils import get_unique_features_in_path, get_covers_vector
from woodelf.personalized_woodelf import apply_metric_delta


def _always_participating_single_tree(
    tree: DecisionTreeNode,
    consumer_data: pd.DataFrame,
    values: Dict,
    p2s: AlwaysParticipatingLTSPathToSVectors,
    features_subset: Optional[set] = None,
    compute_effect_on_other_features: bool = False,
):
    leaves_to_path = tree.get_nodes_to_path_dict()
    if features_subset is not None:
        pattern_gen = decision_patterns_generator_for_feature_subset(
            tree, consumer_data, list(features_subset), GPU=False, ignore_neighbor_leaf=False
        )
    else:
        pattern_gen = decision_patterns_generator(tree, consumer_data, GPU=False, ignore_neighbor_leaf=False)

    for leaf, consumer_patterns in pattern_gen:
        path = leaves_to_path[leaf.index]
        features_in_path = get_unique_features_in_path(path)
        covers = np.array(get_covers_vector(path + [leaf], features_in_path))

        inverse, unique_c = pd.factorize(consumer_patterns, sort=False)
        s_matrix = p2s.get_path_dependent_s_matrix(
            features_in_path, unique_c.astype(np.uint64), covers, leaf.value
        )
        for feature, s_vec in s_matrix.items():
            if features_subset is not None and not compute_effect_on_other_features and feature not in features_subset:
                continue
            feature_values = np.ascontiguousarray(s_vec)[inverse]
            if feature not in values:
                values[feature] = feature_values
            else:
                values[feature] += feature_values


def path_dependent_under_always_participating_features(
    model,
    consumer_data: pd.DataFrame,
    metric: CubeMetric,
    always_participating_features: List[str],
    model_was_loaded: bool = False,
    verbose: bool = True,
    features_subset: Optional[List[str]] = None,
    compute_effect_on_other_features: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute path-dependent Shapley/Banzhaf values where a fixed set of features is always
    in every coalition S (always participating).

    Always-participating features receive zero values (they have no marginal contribution
    since they are present in every coalition). Regular features receive their values in
    the restricted game.

    @param model: A fitted decision-tree ensemble.
    @param consumer_data: The data to explain.
    @param metric: The metric to compute: ShapleyValues(), BanzhafValues(), ...
    @param always_participating_features: Features fixed as always in every coalition S.
    @param model_was_loaded: If True, skips model parsing.
    @param verbose: Whether to show a tqdm progress bar.
    @param features_subset: If provided, only processes leaves whose path contains at least
    one feature from this list. Used by the delta update for efficiency.
    @param compute_effect_on_other_features: When features_subset is set, also accumulate
    values for features outside features_subset. Used by the delta update.

    @return A dictionary mapping each feature name to a NumPy array of length n.
    """
    if not model_was_loaded:
        model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))

    effective_depth = min(model.max_depth, len(consumer_data.columns))
    p2s = AlwaysParticipatingLTSPathToSVectors(
        metric=metric, max_depth=effective_depth, always_participating=always_participating_features
    )
    features_set = set(features_subset) if features_subset is not None else None

    values = {}
    for tree in tqdm(model.trees, desc=f"Computing {metric.__class__.__name__} (always-participating)", disable=not verbose):
        _always_participating_single_tree(tree, consumer_data, values, p2s, features_set, compute_effect_on_other_features)

    return values


def always_participating_delta_update(
    model,
    consumer_data: pd.DataFrame,
    prev_always_participating: List[str],
    new_always_participating: List[str],
    changed_features: List[str],
    prev_values: Dict[str, np.ndarray],
    metric: CubeMetric,
    model_was_loaded: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Exactly updates values after always_participating_features gains new members.

    Uses the same Group A / Group B partition as personalized_baseline_delta_update:
    only leaves whose path contains at least one changed feature need recomputation.

    @param changed_features: The features added to always_participating. Must cover every
    feature whose always-participating status changed between prev and new.

    @return Updated values dict with all features adjusted by the delta.
    """
    n = len(consumer_data)
    result = dict(prev_values)

    prev_effect = path_dependent_under_always_participating_features(
        model, consumer_data, metric, prev_always_participating,
        model_was_loaded=model_was_loaded, verbose=False,
        features_subset=changed_features, compute_effect_on_other_features=True,
    )
    new_effect = path_dependent_under_always_participating_features(
        model, consumer_data, metric, new_always_participating,
        model_was_loaded=model_was_loaded, verbose=False,
        features_subset=changed_features, compute_effect_on_other_features=True,
    )

    apply_metric_delta(result, prev_effect, new_effect, changed_features, n)
    return result
