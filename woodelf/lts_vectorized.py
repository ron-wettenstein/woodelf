from typing import List, Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.cube_metric import CubeMetric, ShapleyInteractionValues, ShapleyValues, BanzhafValues
from woodelf.core.decision_patterns import decision_patterns_generator, ignore_right_neighbor
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import LTSRecursivePathToSVectors
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.core.utils import get_unique_features_in_path, get_covers_vector


def vectorized_linear_tree_shap_for_a_single_tree(
        tree: DecisionTreeNode, consumer_data: pd.DataFrame, values: Dict,
        p2s: LTSRecursivePathToSVectors, GPU: bool, use_neighbor_leaf_trick: bool
):
    leaf_index_to_covers = {}
    leaf_index_to_unique_features_in_path = {}
    leaf_index_to_weight = {}
    leaf_index_to_path = {}
    for leaf, path in tree.get_all_leaves_with_paths(only_feature_names=False):
        unique_features_in_path = get_unique_features_in_path(path)
        leaf_index_to_covers[leaf.index] = np.array(get_covers_vector(path + [leaf], unique_features_in_path))
        leaf_index_to_unique_features_in_path[leaf.index] = unique_features_in_path
        leaf_index_to_weight[leaf.index] = leaf.value
        leaf_index_to_path[leaf.index] = path

    for leaf, consumer_patterns in decision_patterns_generator(tree, consumer_data, GPU, ignore_neighbor_leaf=use_neighbor_leaf_trick):
        inverse, unique_patterns = pd.factorize(consumer_patterns, sort=False)
        s_matrix = p2s.get_path_dependent_s_matrix(
            features_in_path=leaf_index_to_unique_features_in_path[leaf.index],
            consumer_patterns=unique_patterns,
            covers=leaf_index_to_covers[leaf.index],
            w=leaf_index_to_weight[leaf.index],
            w_neighbor=leaf.parent.right.value if ignore_right_neighbor(leaf, leaf_index_to_path[leaf.index], use_neighbor_leaf_trick) else None
        )
        for feature, feature_values in s_matrix.items():
            if feature not in values:
                values[feature] = feature_values[inverse]
            else:
                values[feature] += feature_values[inverse]


def vectorized_linear_tree_shap(
        model, consumer_data: pd.DataFrame, metric: CubeMetric, GPU: bool = False, use_neighbor_leaf_trick: bool = True,
        p2s_class = None, model_was_loaded: bool = False
):
    assert any(isinstance(metric, cls) for cls in [ShapleyValues, ShapleyInteractionValues, BanzhafValues])
    if not model_was_loaded:
        model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))

    if p2s_class is None:
        p2s = LTSRecursivePathToSVectors(metric, max_depth=model.max_depth, GPU=GPU)
    else:
        p2s = p2s_class(metric, max_depth=model.max_depth, GPU=GPU)
    values = {}
    for tree in tqdm(model.trees, desc="Preprocessing the trees and computing SHAP"):
        vectorized_linear_tree_shap_for_a_single_tree(
            tree, consumer_data, values, p2s, GPU, use_neighbor_leaf_trick=use_neighbor_leaf_trick
        )
    p2s.present_statistics()
    return values
