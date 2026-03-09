import time
from math import factorial
from typing import List, Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.decision_trees_ensemble import DecisionTreeNode
from woodelf.decision_patterns import decision_patterns_generator, ignore_right_neighbor
from woodelf.lts_polynomial_multiplication import (
    improved_linear_tree_shap_magic, linear_tree_shap_division_forward_for_neighbors, improved_linear_tree_shap_magic_for_neighbors,
    linear_tree_shap_magic_for_banzhaf, linear_tree_shap_division_forward, linear_tree_shap_magic
)
from woodelf.parse_models import load_decision_tree_ensemble_model


def nCk(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

def shapley_values_f_w(depth):
    return np.array([[1 / (depth * nCk(depth-1, s))] for s in range(depth)])

def banzhaf_values_f_w(depth):
    return np.array([[1 / 2 ** (depth - 1)] for s in range(depth)])

class LinearTreeShapPathToMatrices: # doesn't inherit PathToMatricesAbstractCls as its API is different

    def __init__(self, is_shapley: bool, is_banzhaf: bool, max_depth: int, GPU: bool = False):
        self.max_depth = max_depth
        self.GPU = GPU
        self.is_shapley = is_shapley
        if is_shapley:
            assert not is_banzhaf
        if is_banzhaf:
            assert not is_shapley

        self.f_ws = None
        if is_shapley:
            # None f_ws for depth 0, in the rest use shapley_values_f_w(depth)
            self.f_ws = [None] + [shapley_values_f_w(depth) for depth in range(1, max_depth+1)]

        self.computation_time = 0

    def get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        start_time = time.time()
        if self.is_shapley:
            # assume features in path are unique
            f_w = self.f_ws[len(covers)]
            if w_neighbor is None:
                # s_matrix = linear_tree_shap_magic_faster_v2(covers, consumer_patterns, f_w, w)
                if len(covers) <= 36:
                    s_matrix = linear_tree_shap_division_forward(covers, consumer_patterns, f_w, w)
                else:
                    s_matrix = improved_linear_tree_shap_magic(covers, consumer_patterns, f_w, w)
            else:
                if len(covers) <= 36:
                    s_matrix = linear_tree_shap_division_forward_for_neighbors(covers, consumer_patterns, f_w, w, w_neighbor)
                else:
                    s_matrix = improved_linear_tree_shap_magic_for_neighbors(covers, consumer_patterns, f_w, w, w_neighbor)
        else:
            if w_neighbor is not None:
                raise NotImplemented()
            else:
                s_matrix = linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)
        self.computation_time += time.time() - start_time
        return s_matrix

    def present_statistics(self):
        print(f"LinearTreeShapPathToMatrices took {round(self.computation_time, 2)}")


class LinearTreeShapPathToMatricesSimpleNeighborTrickAbstract(LinearTreeShapPathToMatrices):

    def poly_mult_shap_func(self, covers: np.array, consumer_patterns: np.array, f_w: np.array, w: float):
        raise NotImplemented()

    def poly_mult_banzhaf_func(self, covers: np.array, consumer_patterns: np.array, w: float):
        return linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)

    def get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
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
                return s_matrix_left + s_matrix_right
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
                return s_matrix_left + s_matrix_right
        self.computation_time += time.time() - start_time
        return s_matrix


class LinearTreeShapPathToMatricesSimple(LinearTreeShapPathToMatricesSimpleNeighborTrickAbstract):

    def poly_mult_shap_func(self, covers: np.array, consumer_patterns: np.array, f_w: np.array, w: float):
        return linear_tree_shap_magic(covers, consumer_patterns, f_w, w)


class LinearTreeShapPathToMatricesImproved(LinearTreeShapPathToMatricesSimpleNeighborTrickAbstract):

    def poly_mult_shap_func(self, covers: np.array, consumer_patterns: np.array, f_w: np.array, w: float):
        if len(covers) <= 36:
            return linear_tree_shap_division_forward(covers, consumer_patterns, f_w, w)
        return improved_linear_tree_shap_magic(covers, consumer_patterns, f_w, w)


def get_unique_features_in_path(path: List[DecisionTreeNode]):
    unique_features_in_path = []
    for n in path:
        if n.feature_name not in unique_features_in_path:
            unique_features_in_path.append(n.feature_name)
    return unique_features_in_path


def get_covers_vector(path: List[DecisionTreeNode], unique_features_in_path: List[Any]):
    feature_index = {f: i for i, f in enumerate(unique_features_in_path)}

    proceed_covers = [1] * len(unique_features_in_path)
    for i in range(len(path)-1):
        proceed_covers[ feature_index[path[i].feature_name] ] *= (path[i+1].cover / path[i].cover)
    return proceed_covers


def vectorized_linear_tree_shap_for_a_single_tree(
        tree: DecisionTreeNode, consumer_data: pd.DataFrame, values: Dict, p2m: LinearTreeShapPathToMatrices, GPU: bool, use_neighbor_leaf_trick: bool
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
        # unique_patterns, inverse = np.unique(consumer_patterns, return_inverse=True)
        inverse, unique_patterns = pd.factorize(consumer_patterns, sort=False)
        s_matrix = p2m.get_s_matrix(
            covers=leaf_index_to_covers[leaf.index],
            consumer_patterns=unique_patterns,
            w=leaf_index_to_weight[leaf.index],
            w_neighbor=leaf.parent.right.value if ignore_right_neighbor(leaf, leaf_index_to_path[leaf.index], use_neighbor_leaf_trick) else None
        )

        # TODO why np indexing on a matrix is slower than vector by vector! contribution_values = s_matrix[inverse]
        for index, feature in enumerate(leaf_index_to_unique_features_in_path[leaf.index]):
            if feature not in values:
                values[feature] = s_matrix[:, index][inverse]
            else:
                values[feature] += s_matrix[:, index][inverse]


def vectorized_linear_tree_shap(
        model, consumer_data: pd.DataFrame, is_shapley: bool = True, GPU: bool = False, use_neighbor_leaf_trick: bool = False,
        p2m_class = None
):
    model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))
    if p2m_class is None:
        p2m = LinearTreeShapPathToMatrices(is_shapley, is_banzhaf=not is_shapley, max_depth=model.max_depth, GPU=GPU)
    else:
        p2m = p2m_class(is_shapley, is_banzhaf=not is_shapley, max_depth=model.max_depth, GPU=GPU)
    values = {}
    for tree in tqdm(model.trees, desc="Preprocessing the trees and computing SHAP"):
        vectorized_linear_tree_shap_for_a_single_tree(tree, consumer_data, values, p2m, GPU, use_neighbor_leaf_trick=use_neighbor_leaf_trick)
    p2m.present_statistics()
    return values
