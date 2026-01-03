import time
from math import factorial
from typing import List, Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.decision_trees_ensemble import DecisionTreeNode
from woodelf.high_depth_woodelf import compute_patterns_generator
from woodelf.parse_models import load_decision_tree_ensemble_model


def nCk(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

def shapley_values_f_w(depth):
    return np.array([[1 / (depth * nCk(depth-1, s))] for s in range(depth)])

def banzhaf_values_f_w(depth):
    return np.array([[1 / 2 ** (depth - 1)] for s in range(depth)])

def bits_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """
    x: shape (n,), integers
    returns: shape (k, n), rows are bits (k-1),...,1,0 (2^(k-1) down to LSB)
    """
    # ensure x is unsigned (np.uint) for fast bit ops
    shifts = np.arange(k-1, -1, -1, dtype=np.uint64)[:, None]  # (5,1): 4,3,2,1,0
    return ((x[None, :] >> shifts) & 1).astype(np.uint8)


def linear_tree_shap_magic(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Compute the Shapley/Banzhaf values contribution of a single leaf on all the provided
    consumer decision patterns.
    r: The vector of R0...Rk - the cover rations of traversing with the path for all the unique features in the path. (k <= D).
    p: The consumer patterns vector: Pc_1, Pc_2, ..., Pc_n. (n <= |C|).
    f_w: The Shapley values/Banzhaf values weights vector per coalition size of coalitions 0,1,...,k
    We assume |f_w| = |r| and that f_w is a row vector
    leaf_weight: The leaf weight

    Return a matrix with the Shapley/Banzhaf values contributions. The matrix rows are decision patterns (with the same
    over as in the input p) and the columns are features contributions of the path features
    (in the same order as the features cover appear in the input r)
    """
    # TODO - do not use, this is numerically unstable
    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))

    M_general = np.zeros((len(r)+1, len(p)))
    M_general[0, :] = np.prod(r) * leaf_weight
    for i, R_i in enumerate(r):
        # Multiply the polynomials by (y + q_i)
        q_part = M_general * q_M[i]
        # the y_part - shift M_general down one row, dropping the last row
        M_general[1:] = M_general[:-1]
        M_general[0] = 0
        M_general += q_part

    # Now M_general include the polynomials (y+q_0)*(y+q_1)*...*(y+q_k)

    constitutions_vectors = []
    for i in range(len(r)):
        # Divide the polynomials by (y + q_i)
        M_f_i = M_general[1:].copy()
        for d in range(len(r) - 2, -1, -1):
            M_f_i[d] = M_f_i[d] - M_f_i[d+1] * q_M[i]

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
        constitutions_vectors.append(game_theory_metric_vector)

    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T


def linear_tree_shap_magic_longer_not_optimized(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Compute the Shapley/Banzhaf values contribution of a single leaf on all the provided
    consumer decision patterns.
    r: The vector of R0...Rk - the cover rations of traversing with the path for all the unique features in the path. (k <= D).
    p: The consumer patterns vector: Pc_1, Pc_2, ..., Pc_n. (n <= |C|).
    f_w: The Shapley values/Banzhaf values weights vector per coalition size of coalitions 0,1,...,k
    We assume |f_w| = |r| and that f_w is a row vector
    leaf_weight: The leaf weight

    Return a matrix with the Shapley/Banzhaf values contributions. The matrix rows are decision patterns (with the same
    over as in the input p) and the columns are features contributions of the path features
    (in the same order as the features cover appear in the input r)
    """
    # Longer, but numerically stable
    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))
    constitutions_vectors = []
    for i in range(len(r)):
        M_f_i = np.zeros((len(r), len(p)))
        M_f_i[0, :] = np.prod(r) * leaf_weight
        for j, R_j in enumerate(r):
            if j == i:
                continue
            # Multiply the polynomials by (y + q_i)
            q_part = M_f_i * q_M[j]
            # the y_part - shift M_general down one row, dropping the last row
            M_f_i[1:] = M_f_i[:-1]
            M_f_i[0] = 0
            M_f_i += q_part

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
        constitutions_vectors.append(game_theory_metric_vector)

    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T



def linear_tree_shap_magic_longer(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Compute the Shapley/Banzhaf values contribution of a single leaf on all the provided
    consumer decision patterns.
    r: The vector of R0...Rk - the cover rations of traversing with the path for all the unique features in the path. (k <= D).
    p: The consumer patterns vector: Pc_1, Pc_2, ..., Pc_n. (n <= |C|).
    f_w: The Shapley values/Banzhaf values weights vector per coalition size of coalitions 0,1,...,k
    We assume |f_w| = |r| and that f_w is a row vector
    leaf_weight: The leaf weight

    Return a matrix with the Shapley/Banzhaf values contributions. The matrix rows are decision patterns (with the same
    over as in the input p) and the columns are features contributions of the path features
    (in the same order as the features cover appear in the input r)
    """
    # Longer, but numerically stable

    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))
    constitutions_vectors = []
    M_shared = np.zeros((len(r), len(p)))
    M_shared[0, :] = np.prod(r) * leaf_weight
    for i in range(len(r)):
        M_f_i = M_shared.copy()
        for j in range(max(i-1, 0), len(r)):
            if j == i:
                M_shared = M_f_i.copy()
                continue
            # Multiply the polynomials by (y + q_i)
            q_part = M_f_i * q_M[j]
            # the y_part - shift M_general down one row, dropping the last row
            M_f_i[1:] = M_f_i[:-1]
            M_f_i[0] = 0
            M_f_i += q_part

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
        constitutions_vectors.append(game_theory_metric_vector)

    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T


def linear_tree_shap_magic_for_banzhaf(
        r: np.array, p: np.array, leaf_weight: float
):
    """
    Compute the Shapley/Banzhaf values contribution of a single leaf on all the provided
    consumer decision patterns.
    r: The vector of R0...Rk - the cover rations of traversing with the path for all the unique features in the path. (k <= D).
    p: The consumer patterns vector: Pc_1, Pc_2, ..., Pc_n. (n <= |C|).
    f_w: The Shapley values/Banzhaf values weights vector per coalition size of coalitions 0,1,...,k
    We assume |f_w| = |r| and that f_w is a row vector
    leaf_weight: The leaf weight

    Return a matrix with the Shapley/Banzhaf values contributions. The matrix rows are decision patterns (with the same
    over as in the input p) and the columns are features contributions of the path features
    (in the same order as the features cover appear in the input r)
    """
    R_emptyset = np.prod(r) * leaf_weight
    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))
    sum_original_coefs = np.prod((1 + q_M), axis=0) * R_emptyset
    constitutions_vectors = []
    for i in range(len(r)):
        M_f_i = sum_original_coefs * (1/(1+q_M[i]))
        # Compute Banzhaf values using the constructed polynomial
        game_theory_metric_vector = M_f_i / 2 ** (len(r) - 1)
        constitutions_vectors.append(game_theory_metric_vector)

    M = np.array(constitutions_vectors)  # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T


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

    def get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float):
        start_time = time.time()
        if self.is_shapley:
            # assume features in path are unique
            f_w = self.f_ws[len(covers)]
            # s_matrix = linear_tree_shap_magic(covers, consumer_patterns, f_w, w)
            s_matrix = linear_tree_shap_magic_longer(covers, consumer_patterns, f_w, w)
        else:
            s_matrix = linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)
        self.computation_time += time.time() - start_time
        return s_matrix

    def present_statistics(self):
        print(f"LinearTreeShapPathToMatrices took {round(self.computation_time, 2)}")



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
        tree: DecisionTreeNode, consumer_data: pd.DataFrame, values: Dict, p2m: LinearTreeShapPathToMatrices, GPU: bool
):
    leaf_index_to_covers = {}
    leaf_index_to_unique_features_in_path = {}
    leaf_index_to_weight = {}
    for leaf, path in tree.get_all_leaves_with_paths(only_feature_names=False):
        unique_features_in_path = get_unique_features_in_path(path)
        leaf_index_to_covers[leaf.index] = np.array(get_covers_vector(path + [leaf], unique_features_in_path))
        leaf_index_to_unique_features_in_path[leaf.index] = unique_features_in_path
        leaf_index_to_weight[leaf.index] = leaf.value

    for leaf_index, consumer_patterns in compute_patterns_generator(tree, consumer_data, GPU):
        unique_patterns, inverse = np.unique(consumer_patterns, return_inverse=True)
        s_matrix = p2m.get_s_matrix(
            covers=leaf_index_to_covers[leaf_index],
            consumer_patterns=unique_patterns,
            w=leaf_index_to_weight[leaf_index]
        )

        contribution_values = s_matrix[inverse]
        for index, feature in enumerate(leaf_index_to_unique_features_in_path[leaf_index]):
            if feature not in values:
                values[feature] = contribution_values[:, index]
            else:
                values[feature] += contribution_values[:, index]


def vectorized_linear_tree_shap(model, consumer_data: pd.DataFrame, is_shapley: bool = True, GPU: bool = False):
    model_objs = load_decision_tree_ensemble_model(model, list(consumer_data.columns))
    max_depth = max([model_obj.depth for model_obj in model_objs])
    p2m = LinearTreeShapPathToMatrices(is_shapley, is_banzhaf=not is_shapley, max_depth=max_depth, GPU=GPU)
    values = {}
    for tree in tqdm(model_objs, desc="Preprocessing the trees and computing SHAP"):
        vectorized_linear_tree_shap_for_a_single_tree(tree, consumer_data, values, p2m, GPU)
    p2m.present_statistics()
    return values
