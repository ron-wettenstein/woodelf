import time
from math import factorial
from typing import List, Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.decision_trees_ensemble import DecisionTreeNode
from woodelf.decision_patterns import decision_patterns_generator
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
    # Longer, but numerically stable

    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))
    constitutions_vectors = []
    M_shared = np.zeros((len(r), len(p)))
    M_shared[0, :] = np.prod(r)
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
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0) * leaf_weight
        constitutions_vectors.append(game_theory_metric_vector)

    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()


def linear_tree_shap_magic_for_neighbors(
        r: np.array, p: np.array, f_w: np.array, left_leaf_weight: float, right_leaf_weight: float
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
    M_shared[0, :] = np.prod(r)
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
    return (M * (q_M - 1)).T.copy()

def _poly_mul_y_plus_q_inplace(P: np.ndarray, q: np.ndarray) -> None:
    """
    In-place multiply polynomial P(y) by (y + q) for many columns at once.

    P: shape (k, n). Row t = coefficient of y^t, same convention as your code.
    q: shape (n,) broadcast across rows.

    Keeps degree capped at k-1 by dropping the last coefficient on shift (same as your code).
    """
    # q_part = P * q  (broadcast q over rows)
    q_part = P * q  # shape (k, n)

    # y_part: shift coefficients up by 1: new[t] = old[t-1]
    # do it in-place safely by shifting "down" in memory
    P[1:] = P[:-1]
    P[0] = 0.0

    # add q_part (which corresponds to q * old_P)
    P += q_part


def linear_tree_shap_magic_blocked(
    r: np.ndarray, p: np.ndarray, f_w: np.ndarray, leaf_weight: float
) -> np.ndarray:
    """
    O(k^2.5 * n) version using sqrt-blocking, while preserving the stable
    shift+add multiplication scheme.

    Returns: shape (n, k) like your original (M * (q_M - 1)).T
    """
    k = int(len(r))
    n = int(len(p))

    # Your code: q_M = bits_matrix(p, k) * (1/r.reshape(-1,1))
    # Assume bits_matrix(p, k) returns shape (k, n) with 0/1 entries.
    q_M = bits_matrix(p, k) * (1/r.reshape(-1, 1))

    # base polynomial: constant term = prod(r) * leaf_weight, rest 0
    base_const = float(np.prod(r) * leaf_weight)
    base_poly = np.zeros((k, n))
    base_poly[0, :] = base_const

    # choose block size B ~ sqrt(k)
    B = int(np.ceil(np.sqrt(k)))
    G = int(np.ceil(k / B))

    # output: rows are patterns, cols are features (k)
    result = np.empty((n, k))

    # Precompute index blocks
    blocks = [list(range(start, min(start + B, k))) for start in range(0, k, B)]

    half = G // 2
    top_blocks = blocks[:half]
    bot_blocks = blocks[half:]

    # flatten indices for each half
    top_idx = [j for blk in top_blocks for j in blk]
    bot_idx = [j for blk in bot_blocks for j in blk]

    # Precompute: multiply all factors in the top half once, and all in bottom half once
    P_top_half = base_poly.copy()
    for j in top_idx:
        _poly_mul_y_plus_q_inplace(P_top_half, q_M[j])

    P_bot_half = base_poly.copy()
    for j in bot_idx:
        _poly_mul_y_plus_q_inplace(P_bot_half, q_M[j])

    # For each block g compute P_out[g] = product over (y + q_j) for j not in block
    for block_index, block in enumerate(blocks):

        in_block = np.zeros(k, dtype=bool)
        in_block[block] = True
        if block_index < half:
            P_shared = P_bot_half.copy()
            in_block[bot_idx] = True
            to_compute_for_global_p = np.nonzero(~in_block)[0]
        else:
            P_shared = P_top_half.copy()
            in_block[top_idx] = True
            to_compute_for_global_p = np.nonzero(~in_block)[0]

        for j in to_compute_for_global_p:
            _poly_mul_y_plus_q_inplace(P_shared, q_M[j])

        # For each i in this block, finish the local multiplications excluding i
        for i in range(block[0], block[-1]+1):
            P = P_shared.copy()
            # multiply by all (y + q_j) for j in block, j != i
            for j in range(max(i-1,block[0]), block[-1]+1):
                if j == i:
                    P_shared = P.copy()
                    continue
                _poly_mul_y_plus_q_inplace(P, q_M[j])

            # game_theory_metric_vector = sum_t P[t,:] * f_w[t]
            metric = (P * f_w).sum(axis=0)  # shape (n,)

            # final multiply by (q_i - 1) as in your return (M*(q_M-1)).T
            result[:, i] = metric * (q_M[i] - 1.0)

    return result


def linear_tree_shap_magic_optimization_try(
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

    rank = len(r)
    q_M = bits_matrix(p, rank) * (1/r.reshape(-1, 1))
    constitutions_vectors = []
    M_shared = np.zeros((rank, len(p)), dtype=np.float32)
    M_shared[0, :] = np.prod(r) * leaf_weight
    for i in range(rank):
        min_effected_row = min(i+1, rank - 1)
        M_f_i = M_shared.copy()
        # Multiply the polynomials by (y + q_i)
        q_part = M_shared[0:min_effected_row] * q_M[i]
        # the y_part - shift M_general down one row, dropping the last row
        M_shared[1:min_effected_row+1] = M_shared[0:min_effected_row] # work: M_shared[1:] = M_shared[:-1]
        M_shared[0] = 0
        M_shared[0:min_effected_row] += q_part

        for j in range(i+1, rank):
            min_effected_row = min(j, rank-1)
            # Multiply the polynomials by (y + q_i)
            q_part = M_f_i[0:min_effected_row] * q_M[j]
            # the y_part - shift M_general down one row, dropping the last row
            M_f_i[1:min_effected_row+1] = M_f_i[0:min_effected_row] # work: M_f_i[1:] = M_f_i[:-1]
            M_f_i[0] = 0
            M_f_i[0:min_effected_row] += q_part


        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
        constitutions_vectors.append(game_theory_metric_vector)

    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()


def linear_tree_shap_magic_for_banzhaf(
        r: np.array, p: np.array, leaf_weight: float
):
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
    return (M * (q_M - 1)).T.copy()


def linear_tree_shap_magic_not_numerically_stable(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Not numerically stable but O(D) faster. Don't use this
    """
    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))

    M_general = np.zeros((len(r)+1, len(p)))
    M_general[0, :] = np.prod(r)
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
        M_f_i = np.zeros((len(r), len(p)))
        M_f_i[len(r) - 1] = M_general[len(r)]
        for d in range(len(r) - 2, -1, -1):
            M_f_i[d] = M_general[d+1] - M_f_i[d+1] * q_M[i]

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0) * leaf_weight
        constitutions_vectors.append(game_theory_metric_vector)

    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()


def linear_tree_shap_magic_try7(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Not numerically stable but O(D) faster. Don't use this
    """
    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))

    # constitutions_vectors = []
    # for i in range(len(r)):
    #     M_f_i = np.zeros((len(r), len(p)))
    #     M_f_i[0, :] = np.prod(r) * leaf_weight
    #     for j, R_j in enumerate(r):
    #         if i == j:
    #             continue
    #         # Multiply the polynomials by (1+y*q_i)
    #         M_f_i[1:] += M_f_i[:-1] * q_M[j]
    #     game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
    #     constitutions_vectors.append(game_theory_metric_vector)
    # M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    # return (M * (q_M - 1)).T.copy()


    M_general = np.zeros((len(r)+1, len(p)))
    M_general[0, :] = np.prod(r) * leaf_weight
    for i, R_i in enumerate(r):
        # Multiply the polynomials by (1+y*q_i)
        M_general[1:] += M_general[:-1] * q_M[i]


    # Now M_general include the polynomials (y+q_0)*(y+q_1)*...*(y+q_k)

    constitutions_vectors = []
    for i in range(len(r)):
        # Divide the polynomials by (1+y*q_i)
        M_f_i = np.zeros((len(r), len(p)))
        M_f_i[0] = M_general[0]
        for d in range(1, len(r)-1):
            M_f_i[d] = M_general[d] - M_f_i[d-1] * q_M[i]

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
        constitutions_vectors.append(game_theory_metric_vector)

    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()



def find_min_cover_for_numerical_stability_fast_calc(depth):
    if depth <= 20:
        return 0.2
    if depth <= 25:
        return 0.5
    if depth <= 30:
        return 0.7
    if depth <= 35:
        return 0.9
    return 1


def linear_tree_shap_magic_faster(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Not numerically stable but O(D) faster. Don't use this
    """
    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))

    max_cover_th = find_min_cover_for_numerical_stability_fast_calc(len(r))
    fast_calc_r_indexes = [i for i, R_i in enumerate(r) if R_i >= max_cover_th]
    slow_calc_r_indexes = [i for i, R_i in enumerate(r) if R_i < max_cover_th]
    constitutions_vectors = [None] * len(r)

    M_general = np.zeros((len(r)+1, len(p)))
    M_general[0, :] = np.prod(r) * leaf_weight
    for i, R_i in enumerate(r):
        if i in slow_calc_r_indexes:
            M_f_i = M_general[:-1].copy()
            for j in range(i+1, len(r)):
                # Multiply the polynomials by (y + q_i)
                q_part = M_f_i * q_M[j]
                # the y_part - shift M_general down one row, dropping the last row
                M_f_i[1:] = M_f_i[:-1]
                M_f_i[0] = 0
                M_f_i += q_part

            # Compute Shapley/Banzhaf values using the constructed polynomial
            game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
            constitutions_vectors[i] = game_theory_metric_vector

        # Multiply the polynomials by (y + q_i)
        q_part = M_general * q_M[i]
        # the y_part - shift M_general down one row, dropping the last row
        M_general[1:] = M_general[:-1]
        M_general[0] = 0
        M_general += q_part

    # Now M_general include the polynomials (y+q_0)*(y+q_1)*...*(y+q_k)

    for i in fast_calc_r_indexes:
        # Divide the polynomials by (y + q_i)
        M_f_i = M_general[1:].copy()
        for d in range(len(r) - 2, -1, -1):
            M_f_i[d] = M_f_i[d] - M_f_i[d+1] * q_M[i]

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
        constitutions_vectors[i] = game_theory_metric_vector


    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()


def linear_tree_shap_magic_faster_v2(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Not numerically stable but O(D) faster. Don't use this
    """
    q_M = bits_matrix(p, len(r)) * (1 / r.reshape(-1, 1))

    max_cover_th = find_min_cover_for_numerical_stability_fast_calc(len(r))
    fast_calc_r = [(i, R_i) for i, R_i in enumerate(r) if R_i >= max_cover_th]
    slow_calc_r = [(i, R_i) for i, R_i in enumerate(r) if R_i < max_cover_th]
    constitutions_vectors = [None] * len(r)

    M_general = np.zeros((len(r) + 1, len(p)))
    M_general[0, :] = np.prod(r) * leaf_weight
    for i, R_i in fast_calc_r:
        # Multiply the polynomials by (y + q_i)
        q_part = M_general * q_M[i]
        # the y_part - shift M_general down one row, dropping the last row
        M_general[1:] = M_general[:-1]
        M_general[0] = 0
        M_general += q_part

    visited_indecies = []
    for i, R_i in slow_calc_r:
        M_f_i = M_general[:-1].copy()
        visited_indecies.append(i)
        for j, R_j in slow_calc_r:
            if j in visited_indecies:
                continue
            # Multiply the polynomials by (y + q_j)
            q_part = M_f_i * q_M[j]
            # the y_part - shift M_general down one row, dropping the last row
            M_f_i[1:] = M_f_i[:-1]
            M_f_i[0] = 0
            M_f_i += q_part

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
        constitutions_vectors[i] = game_theory_metric_vector

        # Multiply the polynomials by (y + q_i)
        q_part = M_general * q_M[i]
        # the y_part - shift M_general down one row, dropping the last row
        M_general[1:] = M_general[:-1]
        M_general[0] = 0
        M_general += q_part

    # Now M_general include the polynomials (y+q_0)*(y+q_1)*...*(y+q_k)

    for i, R_i in fast_calc_r:
        # Divide the polynomials by (y + q_i)
        M_f_i = M_general[1:].copy()
        for d in range(len(r) - 2, -1, -1):
            M_f_i[d] = M_f_i[d] - M_f_i[d + 1] * q_M[i]

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
        constitutions_vectors[i] = game_theory_metric_vector

    M = np.array(constitutions_vectors)  # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()

def linear_tree_shap_magic_try6(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Not numerically stable but O(D) faster. Don't use this
    """
    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))

    max_cover_th = find_min_cover_for_numerical_stability_fast_calc(len(r))
    fast_calc_r_indexes = [i for i, R_i in enumerate(r) if R_i >= max_cover_th]
    slow_calc_r_indexes = [i for i, R_i in enumerate(r) if R_i < max_cover_th]
    constitutions_vectors = [None] * len(r)

    # TODO the code reverse the matrix
    zero_row = np.zeros((1, len(p)))
    M_general = np.full(len(p), np.prod(r) * leaf_weight)
    for i, R_i in enumerate(r):
        if i in slow_calc_r_indexes:
            M_f_i = M_general.copy()
            for j in range(i+1, len(r)):
                M_f_i = np.vstack([M_f_i, zero_row]) + np.vstack([zero_row, M_f_i * q_M[j]])

            # Compute Shapley/Banzhaf values using the constructed polynomial
            game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
            constitutions_vectors[i] = game_theory_metric_vector

        M_general = np.vstack([M_general, zero_row]) + np.vstack([zero_row, M_general * q_M[i]])

    for i in fast_calc_r_indexes:
        # Divide the polynomials by (y + q_i)
        M_f_i = M_general[:-1].copy()
        for d in range(1, len(r)):
            M_f_i[d] = M_f_i[d] - M_f_i[d-1] * q_M[i]

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0)
        constitutions_vectors[i] = game_theory_metric_vector


    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()



def linear_tree_shap_magic_longer_not_optimized(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Not optimized, the linear_tree_shap_magic is ~2x faster
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
    return (M * (q_M - 1)).T.copy()


def linear_tree_shap_magic_for_banzhaf_extra_numerically_stable(
        r: np.array, p: np.array, leaf_weight: float
):
    """
    Extra numerically stable but O(D) longer. No need to use, the current banzhaf implementation
    is stable enough
    """
    R_emptyset = np.prod(r) * leaf_weight
    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))
    M_shared = (1 + q_M) * R_emptyset
    constitutions_vectors = []
    for i in range(len(r)):
        row_i = M_shared[i]
        M_shared[i] = 1
        sum_coefs = np.prod(M_shared, axis=0)
        M_shared[i] = row_i
        game_theory_metric_vector = sum_coefs / 2 ** (len(r) - 1)
        constitutions_vectors.append(game_theory_metric_vector)
    M = np.array(constitutions_vectors)  # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()


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
            s_matrix = linear_tree_shap_magic_faster_v2(covers, consumer_patterns, f_w, w)
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

    for leaf, consumer_patterns in decision_patterns_generator(tree, consumer_data, GPU):
        # unique_patterns, inverse = np.unique(consumer_patterns, return_inverse=True)
        inverse, unique_patterns = pd.factorize(consumer_patterns, sort=False)
        s_matrix = p2m.get_s_matrix(
            covers=leaf_index_to_covers[leaf.index],
            consumer_patterns=unique_patterns,
            w=leaf_index_to_weight[leaf.index]
        )

        # TODO why np indexing on a matrix is slower than vector by vector! contribution_values = s_matrix[inverse]
        for index, feature in enumerate(leaf_index_to_unique_features_in_path[leaf.index]):
            if feature not in values:
                values[feature] = s_matrix[:, index][inverse]
            else:
                values[feature] += s_matrix[:, index][inverse]


def vectorized_linear_tree_shap(model, consumer_data: pd.DataFrame, is_shapley: bool = True, GPU: bool = False):
    model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))
    p2m = LinearTreeShapPathToMatrices(is_shapley, is_banzhaf=not is_shapley, max_depth=model.max_depth, GPU=GPU)
    values = {}
    for tree in tqdm(model.trees, desc="Preprocessing the trees and computing SHAP"):
        vectorized_linear_tree_shap_for_a_single_tree(tree, consumer_data, values, p2m, GPU)
    p2m.present_statistics()
    return values
