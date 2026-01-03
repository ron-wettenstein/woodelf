from math import factorial
from typing import List

import numpy as np

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
    # P_M = bits_matrix(p, len(r))
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

    def get_s_matrices(self, features_in_path: List, covers: List, consumer_patterns: np.array, w: float):
        r = np.array(covers)
        if self.is_shapley:
            # assume features in path are unique
            f_w = self.f_ws[len(features_in_path)]
            s_matrix = linear_tree_shap_magic(r, consumer_patterns, f_w, w)
        else:
            s_matrix = linear_tree_shap_magic_for_banzhaf(r, consumer_patterns, w)
        s_vectors = {}
        for index, feature in enumerate(features_in_path):
            s_vectors[feature] = s_matrix[:,index]
        return s_vectors
