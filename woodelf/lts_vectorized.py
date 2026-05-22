import math
import time
from math import factorial
from typing import List, Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode
from woodelf.core.decision_patterns import decision_patterns_generator, ignore_right_neighbor
from woodelf.core.lts_polynomial_multiplication import (
    improved_linear_tree_shap_magic, improved_linear_tree_shap_magic_for_neighbors,
    linear_tree_shap_magic_for_banzhaf, linear_tree_shap_division_forward, linear_tree_shap_magic, improved_linear_tree_shap_iv, linear_tree_shap_v6, linear_tree_shap_v6_for_neighbors
)
from woodelf.core.path_to_matrices import PathToSVectors
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model


def nCk(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

def shapley_values_f_w(depth):
    return np.array([[1 / (depth * nCk(depth-1, s))] for s in range(depth)])

def banzhaf_values_f_w(depth):
    return np.array([[1 / 2 ** (depth - 1)] for s in range(depth)])

class LTSPathToSVectors(PathToSVectors):

    def __init__(self, is_shapley: bool, is_banzhaf: bool, max_depth: int, is_interaction_values: bool=False, GPU: bool = False):
        super().__init__(max_depth, GPU)
        self.is_shapley = is_shapley
        self.is_interaction_values = is_interaction_values
        if is_shapley:
            assert not is_banzhaf
        if is_banzhaf:
            assert not is_shapley

        self.f_ws = None
        if is_shapley:
            # None f_ws for depth 0, in the rest use shapley_values_f_w(depth)
            self.f_ws = [None] + [shapley_values_f_w(depth) for depth in range(1, max_depth + 1)]

        self.computation_time = 0

    def _get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        start_time = time.time()
        if self.is_shapley:
            # assume features in path are unique
            if self.is_interaction_values:
                assert w_neighbor is None
                if len(covers) <= 1:
                    self.computation_time += time.time() - start_time
                    return []
                f_w = self.f_ws[len(covers)-1]
                s_matrix = improved_linear_tree_shap_iv(covers, consumer_patterns, f_w, w)
            else:
                f_w = self.f_ws[len(covers)]
                if w_neighbor is None:
                    # s_matrix = linear_tree_shap_magic_faster_v2(covers, consumer_patterns, f_w, w)
                    s_matrix = improved_linear_tree_shap_magic(covers, consumer_patterns, f_w, w)
                else:
                    s_matrix = improved_linear_tree_shap_magic_for_neighbors(covers, consumer_patterns, f_w, w, w_neighbor)
        else:
            if w_neighbor is not None:
                s_matrix_left = linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)
                covers_of_right = np.array(list(covers[:-1]) + [1 - covers[-1]])
                consumer_patterns_right = consumer_patterns.copy()
                consumer_patterns_right[consumer_patterns % 2 == 0] += 1
                consumer_patterns_right[consumer_patterns % 2 == 1] -= 1
                s_matrix_right = linear_tree_shap_magic_for_banzhaf(covers_of_right, consumer_patterns_right.astype(np.uint64), w_neighbor)
                s_matrix = s_matrix_left + s_matrix_right
            else:
                s_matrix = linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)
        self.computation_time += time.time() - start_time
        return s_matrix

    def get_background_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        raise NotImplementedError("LTS uses path-dependent approach only")

    def get_path_dependent_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        if not features_in_path:
            return {}
        inverse, unique_patterns = pd.factorize(consumer_patterns, sort=False)
        s_matrix = self._get_s_matrix(covers, unique_patterns, w, w_neighbor)
        if not self.is_interaction_values:
            s_matrix = s_matrix.astype(np.float32)
            # TODO why np indexing on a matrix is slower than vector by vector! contribution_values = s_matrix[inverse]
            return {feature: s_matrix[:, i][inverse] for i, feature in enumerate(features_in_path)}
        else:
            if not s_matrix:  # < 2 unique features in path, no interactions
                return {}
            result = {}
            for i, feature1 in enumerate(features_in_path):
                s_matrix_i = s_matrix[i].astype(np.float32)
                for j, feature2 in enumerate(features_in_path):
                    if i != j:
                        f2_index = j if j < i else j - 1
                        result[(feature1, feature2)] = s_matrix_i[:, f2_index][inverse]
            return result

    def present_statistics(self):
        print(f"LTSPathToSVectors took {round(self.computation_time, 2)}")


def get_unique_features_in_path(path: List[DecisionTreeNode]):
    unique_features_in_path = []
    for n in path:
        if n.feature_name not in unique_features_in_path:
            unique_features_in_path.append(n.feature_name)
    return unique_features_in_path


def get_covers_vector(path: List[DecisionTreeNode], unique_features_in_path: List[Any]):
    if len(unique_features_in_path) == 0:
        # If the leaf is the tree's root, it has cover of 1.
        return [1]

    feature_index = {f: i for i, f in enumerate(unique_features_in_path)}

    proceed_covers = [1] * len(unique_features_in_path)
    for i in range(len(path)-1):
        proceed_covers[ feature_index[path[i].feature_name] ] *= (path[i+1].cover / path[i].cover)
    return proceed_covers


def vectorized_linear_tree_shap_for_a_single_tree(
        tree: DecisionTreeNode, consumer_data: pd.DataFrame, values: Dict,
        p2s: LTSPathToSVectors, GPU: bool, use_neighbor_leaf_trick: bool
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
        s_matrix = p2s.get_path_dependent_s_matrix(
            features_in_path=leaf_index_to_unique_features_in_path[leaf.index],
            consumer_patterns=consumer_patterns,
            covers=leaf_index_to_covers[leaf.index],
            w=leaf_index_to_weight[leaf.index],
            w_neighbor=leaf.parent.right.value if ignore_right_neighbor(leaf, leaf_index_to_path[leaf.index], use_neighbor_leaf_trick) else None
        )
        for feature, feature_values in s_matrix.items():
            if feature not in values:
                values[feature] = feature_values
            else:
                values[feature] += feature_values


def vectorized_linear_tree_shap(
        model, consumer_data: pd.DataFrame, is_shapley: bool = True, is_banzhaf: bool = False, GPU: bool = False, is_interaction_values: bool = False,
        use_neighbor_leaf_trick: bool = True, p2s_class = None, model_was_loaded: bool = False
):
    assert is_shapley or is_banzhaf

    if not model_was_loaded:
        model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))

    if p2s_class is None:
        p2s = LTSPathToSVectors(is_shapley, is_banzhaf=is_banzhaf, is_interaction_values=is_interaction_values, max_depth=model.max_depth, GPU=GPU)
    else:
        p2s = p2s_class(is_shapley, is_banzhaf=not is_shapley, max_depth=model.max_depth, GPU=GPU)
    values = {}
    for tree in tqdm(model.trees, desc="Preprocessing the trees and computing SHAP"):
        vectorized_linear_tree_shap_for_a_single_tree(
            tree, consumer_data, values, p2s, GPU, use_neighbor_leaf_trick=use_neighbor_leaf_trick
        )
    p2s.present_statistics()
    return values




############################################################################################################################################################
#
#                         For Tests and Measures
#
############################################################################################################################################################


class LTSSimpleNeighborTrickPathToSVectors(LTSPathToSVectors):

    def poly_mult_shap_func(self, covers: np.array, consumer_patterns: np.array, f_w: np.array, w: float):
        raise NotImplemented()

    def poly_mult_banzhaf_func(self, covers: np.array, consumer_patterns: np.array, w: float):
        return linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)

    def _get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        start_time = time.time()
        if self.is_shapley:
            # assume features in path are unique
            f_w = self.f_ws[len(covers)]
            if w_neighbor is None:
                s_matrix = self.poly_mult_shap_func(covers, consumer_patterns, f_w, w)
            else:
                s_matrix_left = self.poly_mult_shap_func(covers, consumer_patterns, f_w, w)
                covers_of_right = np.array(list(covers[:-1]) + [1 - covers[-1]])
                consumer_patterns_right = consumer_patterns.copy()
                consumer_patterns_right[consumer_patterns % 2 == 0] += 1
                consumer_patterns_right[consumer_patterns % 2 == 1] -= 1
                s_matrix_right = self.poly_mult_shap_func(
                    covers_of_right, consumer_patterns_right.astype(np.uint64), f_w, w_neighbor
                )
                s_matrix = s_matrix_left + s_matrix_right
        else:
            if w_neighbor is None:
                s_matrix = self.poly_mult_banzhaf_func(covers, consumer_patterns, w)
            else:
                s_matrix_left = self.poly_mult_banzhaf_func(covers, consumer_patterns, w)
                covers_of_right = np.array(list(covers[:-1]) + [1 - covers[-1]])
                consumer_patterns_right = consumer_patterns.copy()
                consumer_patterns_right[consumer_patterns % 2 == 0] += 1
                consumer_patterns_right[consumer_patterns % 2 == 1] -= 1
                s_matrix_right = self.poly_mult_banzhaf_func(covers_of_right, consumer_patterns_right.astype(np.uint64), w_neighbor)
                s_matrix = s_matrix_left + s_matrix_right
        self.computation_time += time.time() - start_time
        return s_matrix

class LTSSimplePathToSVectors(LTSSimpleNeighborTrickPathToSVectors):

    def poly_mult_shap_func(self, covers: np.array, consumer_patterns: np.array, f_w: np.array, w: float):
        return linear_tree_shap_magic(covers, consumer_patterns, f_w, w)


class LTSImprovedPathToSVectors(LTSSimpleNeighborTrickPathToSVectors):

    def poly_mult_shap_func(self, covers: np.array, consumer_patterns: np.array, f_w: np.array, w: float):
        if len(covers) <= 36:
            return linear_tree_shap_division_forward(covers, consumer_patterns, f_w, w)
        return improved_linear_tree_shap_magic(covers, consumer_patterns, f_w, w)


class LTSV6PathToSVectors(LTSPathToSVectors):

    def __init__(self, is_shapley: bool, is_banzhaf: bool, max_depth: int, GPU: bool = False):
        super().__init__(is_shapley, is_banzhaf, max_depth, GPU)
        self.quad_nodes, self.quad_weights = self.compute_quads()

    def compute_quads(self):
        max_required_quads = min(max(int(math.ceil(self.max_depth / 2)), 2), 16)
        quad_nodes = {}
        quad_weights = {}
        for n_quad in range(2, max_required_quads + 1):
            _nodes, _weights = np.polynomial.legendre.leggauss(n_quad)
            quad_nodes[n_quad] = np.float64(0.5) * (_nodes.astype(np.float64) + np.float64(1.0))  # (n_quad,)
            quad_weights[n_quad] = np.float64(0.5) * _weights.astype(np.float64)  # (n_quad,)
        return quad_nodes, quad_weights

    def _get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        start_time = time.time()
        if self.is_shapley:
            # For D<4 use 2, for D > 32 use 16 for 4<=D<=32 use 0.5*D
            n_quads = min(max(int(math.ceil(len(covers) / 2)), 2), 16)
            if w_neighbor is None:
                s_matrix = linear_tree_shap_v6(covers, consumer_patterns, w, self.quad_nodes[n_quads], self.quad_weights[n_quads])
            else:
                s_matrix = linear_tree_shap_v6_for_neighbors(covers, consumer_patterns, w, w_neighbor, self.quad_nodes[n_quads], self.quad_weights[n_quads])
        else:
            if w_neighbor is None:
                s_matrix = linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)
            else:
                s_matrix_left = linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)
                covers_of_right = np.array(list(covers[:-1]) + [1 - covers[-1]])
                consumer_patterns_right = consumer_patterns.copy()
                consumer_patterns_right[consumer_patterns % 2 == 0] += 1
                consumer_patterns_right[consumer_patterns % 2 == 1] -= 1
                s_matrix_right = linear_tree_shap_magic_for_banzhaf(covers_of_right, consumer_patterns_right.astype(np.int64), w_neighbor)
                s_matrix = s_matrix_left + s_matrix_right
        self.computation_time += time.time() - start_time
        return s_matrix
