from __future__ import annotations

import itertools
import numpy as np


def bits_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """
    x: shape (n,), integers
    returns: shape (k, n), rows are bits (k-1),...,1,0 (2^(k-1) down to LSB)
    """
    # ensure x is unsigned (np.uint) for fast bit ops
    shifts = np.arange(k-1, -1, -1, dtype=np.uint8)[:, None]  # (5,1): 4,3,2,1,0
    return ((x[None, :] >> shifts) & 1).astype(np.uint8)

def neg_bits_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """
    identical to bits_matrix(x,k).replace({0:1, 1:0})
    """
    # ensure x is unsigned (np.uint) for fast bit ops
    shifts = np.arange(k-1, -1, -1, dtype=np.uint8)[:, None]  # (5,1): 4,3,2,1,0
    return (((x[None, :] >> shifts) + 1) & 1).astype(np.uint8)


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


############################################################################################################################################################
#
#                         The Recursive Approach (Without the Neighbor Leaf Trick)
#
############################################################################################################################################################


def poly_mul_y_plus_q_inplace(P: np.ndarray, q: np.ndarray) -> None:
    """
    In-place multiply polynomial P(y) by (y + q) for many columns at once.

    P: shape (k, n). Row t = coefficient of y^t, same convention as your code.
    q: shape (n, 1) broadcast across rows.

    Keeps degree capped at k-1 by dropping the last coefficient on shift (same as your code).
    """
    q_part = P * q

    # y_part: shift coefficients up by 1: new[t] = old[t-1]
    # do it in-place safely by shifting "down" in memory
    P[1:] = P[:-1]
    P[0] = 0.0

    # add q_part (which corresponds to q * old_P)
    P += q_part


def compute_P(
        r: np.array, q_M: np.array, start_index, end_index
):
    """
    Compute the polynomial (q_1 + y)*(q_2 + y)*..*(q_k + y).
    """
    P = np.zeros((len(r), q_M.shape[1]))
    P[0, :] = np.prod(r)
    for i in range(start_index, end_index):
        poly_mul_y_plus_q_inplace(P, q_M[i])

    return P


def continue_P_compute(P: np.array, q_M: np.array, start_index: int, end_index: int):
    # assume P is NOT zfill
    P_continued = P.copy()
    for i in range(start_index, end_index):
        poly_mul_y_plus_q_inplace(P_continued, q_M[i])
    return P_continued

 # compute_contribution_vectors_from_initial_P(P, r, q_M, f_w, w, start_index, end_index)
def compute_contribution_vectors_from_initial_P(
        P: np.array, q_M: np.array, f_w: np.array, w: float, start_index: int, end_index: int
):
    """
    Return a matrix with the Shapley/Banzhaf values contributions. The matrix rows are decision patterns (with the same
    over as in the input p) and the columns are features contributions of the path features
    (in the same order as the features cover appear in the input r)
    """
    # Longer, but numerically stable

    constitutions_vectors = []
    P_shared = P
    for i in range(start_index, end_index):
        M_f_i = P_shared.copy()
        for j in range(max(i-1, start_index), end_index):
            if j == i:
                P_shared = M_f_i.copy()
                continue
            poly_mul_y_plus_q_inplace(M_f_i, q_M[j])

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0) * w
        constitutions_vectors.append(game_theory_metric_vector)

    return np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix


def recursive_linear_tree_shap(P: np.array, r: np.array, q_M: np.array, f_w: np.array, w: float, start_index: int, end_index: int) -> np.array:
    middle_index = (start_index + end_index) // 2
    if end_index - start_index <= 4:
        return compute_contribution_vectors_from_initial_P(P, q_M, f_w, w, start_index, end_index)

    top_P = continue_P_compute(P, q_M, start_index=start_index, end_index=middle_index)
    bottom_P = continue_P_compute(P, q_M, start_index=middle_index, end_index=end_index)

    top_contribs = recursive_linear_tree_shap(bottom_P, r, q_M, f_w, w, start_index=start_index, end_index=middle_index)
    bottom_contribs = recursive_linear_tree_shap(top_P, r, q_M, f_w, w, start_index=middle_index, end_index=end_index)
    return np.vstack([top_contribs, bottom_contribs])


def improved_linear_tree_shap_magic(r: np.array, p: np.array, f_w: np.array, w: float):
    q_M = bits_matrix(p, len(r)) * (1 / r.reshape(-1, 1))
    if len(r) <= 6:
        initial_P = compute_P(r, q_M, start_index=0, end_index=0)
        M = compute_contribution_vectors_from_initial_P(initial_P, q_M, f_w, w, start_index=0, end_index=len(r))
        return (M * (q_M - 1)).T.copy()

    top_P = compute_P(r, q_M, start_index=0, end_index=len(r)//2)
    bottom_P = compute_P(r, q_M, start_index=len(r)//2, end_index=len(r))

    bottom_contribs = recursive_linear_tree_shap(top_P, r, q_M, f_w, w, start_index=len(r)//2, end_index=len(r))
    top_contribs = recursive_linear_tree_shap(bottom_P, r, q_M, f_w, w, start_index=0, end_index=len(r)//2)
    M = np.vstack([top_contribs, bottom_contribs])
    return (M * (q_M - 1)).T.copy()


############################################################################################################################################################
#
#                         The Recursive Approach that Supports The Right Leaf Neighbor Trick
#
############################################################################################################################################################


def get_neighbors_shap_from_polynomials(
        M_f_i: np.array, R_i: float, q_M_left: np.array, q_M_right: np.array, f_w: np.array, left_leaf_weight: float, right_leaf_weight: float
):
    M_left = M_f_i.copy()
    poly_mul_y_plus_q_inplace(M_left, q_M_left)
    left_game_theory_metric_vector = (M_left * f_w).sum(axis=0) * left_leaf_weight * R_i

    M_right = M_f_i.copy()
    poly_mul_y_plus_q_inplace(M_right, q_M_right)
    right_game_theory_metric_vector = (M_right * f_w).sum(axis=0) * right_leaf_weight * (1 - R_i)

    return left_game_theory_metric_vector, right_game_theory_metric_vector

def compute_contribution_vectors_from_initial_P_for_neighbors(
        P: np.array, q_M: np.array, b_M: np.array, f_w: np.array, left_leaf_weight: float, right_leaf_weight: float, R_last: float, start_index: int, end_index: int
):
    """
    Return a matrix with the Shapley/Banzhaf values contributions. The matrix rows are decision patterns (with the same
    over as in the input p) and the columns are features contributions of the path features
    (in the same order as the features cover appear in the input r)
    """
    # Longer, but numerically stable

    last_row_q_M_left = q_M[-1]
    last_row_q_M_right = (-(b_M[-1]-1)) * (1/(1-R_last)) # replace 0s and 1s and multiply by 1/(1-R_last)
    left_constitutions_vectors = []
    right_constitutions_vectors = []
    P_shared = P
    for i in range(start_index, end_index):
        M_f_i = P_shared.copy()
        for j in range(max(i-1, start_index), end_index):
            if j == i:
                P_shared = M_f_i.copy()
                continue
            poly_mul_y_plus_q_inplace(M_f_i, q_M[j])

        # Compute Shapley/Banzhaf values using the constructed polynomial
        left_game_theory_metric_vector, right_game_theory_metric_vector = get_neighbors_shap_from_polynomials(
            M_f_i, R_last, last_row_q_M_left, last_row_q_M_right, f_w, left_leaf_weight, right_leaf_weight
        )
        left_constitutions_vectors.append(left_game_theory_metric_vector)
        right_constitutions_vectors.append(right_game_theory_metric_vector)

    if end_index + 1 == len(f_w):
        poly_mul_y_plus_q_inplace(P_shared, q_M[-2])
        left_constitutions_vectors.append((P_shared * f_w).sum(axis=0) * left_leaf_weight * R_last)
        right_constitutions_vectors.append((P_shared * f_w).sum(axis=0) * right_leaf_weight * (1-R_last))

    M_left = np.array(left_constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    M_right = np.array(right_constitutions_vectors)
    return M_left, M_right


def recursive_linear_tree_shap_for_neighbors(
        P: np.array, r: np.array, q_M: np.array, b_M: np.array, f_w: np.array, left_leaf_weight: float, right_leaf_weight: float, start_index: int, end_index: int
) -> np.array:
    middle_index = (start_index + end_index) // 2
    if end_index - start_index <= 4:
        R_last = r[-1]
        return compute_contribution_vectors_from_initial_P_for_neighbors(P, q_M, b_M, f_w, left_leaf_weight, right_leaf_weight, R_last, start_index, end_index)

    top_P = continue_P_compute(P, q_M, start_index=start_index, end_index=middle_index)
    bottom_P = continue_P_compute(P, q_M, start_index=middle_index, end_index=end_index)

    top_contribs_left, top_contribs_right = recursive_linear_tree_shap_for_neighbors(bottom_P, r, q_M, b_M, f_w, left_leaf_weight, right_leaf_weight, start_index=start_index, end_index=middle_index)
    bottom_contribs_left, bottom_contribs_right = recursive_linear_tree_shap_for_neighbors(top_P, r, q_M, b_M, f_w, left_leaf_weight, right_leaf_weight, start_index=middle_index, end_index=end_index)
    return np.vstack([top_contribs_left, bottom_contribs_left]), np.vstack([top_contribs_right, bottom_contribs_right]),


def improved_linear_tree_shap_magic_for_neighbors(r: np.array, p: np.array, f_w: np.array, left_leaf_weight: float, right_leaf_weight: float):
    b_M = bits_matrix(p, len(r))
    neg_b_M = neg_bits_matrix(p, len(r))
    q_M = b_M * (1 / r.reshape(-1, 1))

    M_general = np.zeros((len(r), len(p)))
    M_general[0, :] = np.prod(r[:-1])
    if len(r) <= 6:
        R_last = r[-1]
        M_left, M_right = compute_contribution_vectors_from_initial_P_for_neighbors(
            M_general, q_M, b_M, f_w, left_leaf_weight, right_leaf_weight, R_last, start_index=0, end_index=len(r)-1
        )
    else:
        top_P = continue_P_compute(M_general, q_M, start_index=0, end_index=(len(r)-1)//2)
        bottom_P = continue_P_compute(M_general, q_M, start_index=(len(r)-1)//2, end_index=len(r)-1)

        bottom_contribs_left, bottom_contribs_right = recursive_linear_tree_shap_for_neighbors(top_P, r, q_M, b_M, f_w, left_leaf_weight, right_leaf_weight, start_index=(len(r)-1)//2, end_index=len(r)-1)
        top_contribs_left, top_contribs_right = recursive_linear_tree_shap_for_neighbors(bottom_P, r, q_M, b_M, f_w, left_leaf_weight, right_leaf_weight, start_index=0, end_index=(len(r)-1)//2)
        M_left = np.vstack([top_contribs_left, bottom_contribs_left])
        M_right = np.vstack([top_contribs_right, bottom_contribs_right])

    q_M_left = q_M
    result_left = (M_left * (q_M_left - 1)).T.copy()
    q_M_right = q_M.copy()
    last_row_q_M_right = neg_b_M[-1] * (1 / (1 - r[-1]))
    q_M_right[-1]=last_row_q_M_right
    result_right = (M_right * (q_M_right - 1)).T.copy()
    return result_left + result_right


############################################################################################################################################################
#
#  The simple approach of naive polynomial multiplication - This is used in testing and in low depth and also useful for understanding the recursive approach
#
############################################################################################################################################################

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

    patterns_M = bits_matrix(p, len(r))
    q_M = patterns_M * (1/r.reshape(-1, 1))
    R_last = r[-1]
    last_row_q_M_left = q_M[-1]
    last_row_q_M_right = (-(patterns_M[-1]-1)) * (1/(1-R_last)) # replace 0s and 1s and multiply by 1/(1-R_last)
    left_constitutions_vectors = []
    right_constitutions_vectors = []

    M_shared = np.zeros((len(r), len(p)))
    M_shared[0, :] = np.prod(r[:-1])
    for i in range(len(r)-1):
        M_f_i = M_shared.copy()
        for j in range(max(i-1, 0), len(r)-1):
            if j == i:
                M_shared = M_f_i.copy()
                continue
            poly_mul_y_plus_q_inplace(M_f_i, q_M[j])

        # Compute Shapley/Banzhaf values using the constructed polynomial
        left_game_theory_metric_vector, right_game_theory_metric_vector = get_neighbors_shap_from_polynomials(
            M_f_i, r[-1], last_row_q_M_left, last_row_q_M_right, f_w, left_leaf_weight, right_leaf_weight
        )
        left_constitutions_vectors.append(left_game_theory_metric_vector)
        right_constitutions_vectors.append(right_game_theory_metric_vector)

    poly_mul_y_plus_q_inplace(M_shared, q_M[len(r)-2])
    left_constitutions_vectors.append((M_shared * f_w).sum(axis=0) * left_leaf_weight * R_last)
    right_constitutions_vectors.append((M_shared * f_w).sum(axis=0) * right_leaf_weight * (1-R_last))

    M_left = np.array(left_constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    q_M_left = q_M
    result_left = (M_left * (q_M_left - 1)).T.copy()

    M_right = np.array(right_constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    q_M_right = q_M.copy()
    q_M_right[-1]=last_row_q_M_right
    result_right = (M_right * (q_M_right - 1)).T.copy()
    return result_left + result_right


############################################################################################################################################################
#
#         Division Forward - Do not use it as it slower than the "improved" approach above. Keep it here as it might be useful someday
#
############################################################################################################################################################


def linear_tree_shap_division_forward(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Not numerically stable but O(D) faster. Don't use this
    """
    b_M = bits_matrix(p, len(r))
    q_M = b_M * (1/r.reshape(-1, 1))
    one_over_q_M = b_M * r.reshape(-1, 1)

    M_general = np.zeros((len(r)+1, len(p)))
    M_general[0, :] = np.prod(r)
    M_general = continue_P_compute(M_general, q_M, start_index=0, end_index=len(r))

    # Now M_general include the polynomials (y+q_0)*(y+q_1)*...*(y+q_k)

    neg_b_M = neg_bits_matrix(p, len(r))
    constitutions_vectors = []
    for i in range(len(r)):
        # Divide the polynomials by (y + q_i)
        M_f_i = np.zeros((len(r), len(p)))
        M_f_i[0] = M_general[0] * one_over_q_M[i] + M_general[1] * neg_b_M[i]
        for d in range(1, len(r)):
            M_f_i[d] = (M_general[d] - M_f_i[d-1]) * one_over_q_M[i] + M_general[d+1] * neg_b_M[i]

        # Compute Shapley/Banzhaf values using the constructed polynomial
        game_theory_metric_vector = (M_f_i * f_w).sum(axis=0) * leaf_weight
        constitutions_vectors.append(game_theory_metric_vector)

    M = np.array(constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()


def linear_tree_shap_division_forward_for_neighbors(
        r: np.array, p: np.array, f_w: np.array, left_leaf_weight: float, right_leaf_weight: float
):
    """
    Not numerically stable but O(D) faster. Don't use this
    """
    b_M = bits_matrix(p, len(r))
    q_M = b_M * (1/r.reshape(-1, 1))
    one_over_q_M = b_M * r.reshape(-1, 1)

    R_last = r[-1]
    last_row_q_M_left = q_M[-1]
    last_row_q_M_right = (-(b_M[-1]-1)) * (1/(1-R_last)) # replace 0s and 1s and multiply by 1/(1-R_last)

    M_general = np.zeros((len(r), len(p)))
    M_general[0, :] = np.prod(r[:-1])
    M_general = continue_P_compute(M_general, q_M, start_index=0, end_index=len(r) - 1)

    # Now M_general include the polynomials (y+q_0)*(y+q_1)*...*(y+q_k)

    neg_b_M = neg_bits_matrix(p, len(r))
    left_constitutions_vectors = []
    right_constitutions_vectors = []
    for i in range(len(r)-1):
        # Divide the polynomials by (y + q_i)
        M_f_i = np.zeros((len(r), len(p)))
        M_f_i[0] = M_general[0] * one_over_q_M[i] + M_general[1] * neg_b_M[i]
        for d in range(1, len(r)-1):
            M_f_i[d] = (M_general[d] - M_f_i[d-1]) * one_over_q_M[i] + M_general[d+1] * neg_b_M[i]

        # Compute Shapley/Banzhaf values using the constructed polynomial
        left_game_theory_metric_vector, right_game_theory_metric_vector = get_neighbors_shap_from_polynomials(
            M_f_i, R_last, last_row_q_M_left, last_row_q_M_right, f_w, left_leaf_weight, right_leaf_weight
        )
        left_constitutions_vectors.append(left_game_theory_metric_vector)
        right_constitutions_vectors.append(right_game_theory_metric_vector)

    left_constitutions_vectors.append((M_general * f_w).sum(axis=0) * left_leaf_weight * R_last)
    right_constitutions_vectors.append((M_general * f_w).sum(axis=0) * right_leaf_weight * (1-R_last))

    M_left = np.array(left_constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    q_M_left = q_M
    result_left = (M_left * (q_M_left - 1)).T.copy()

    M_right = np.array(right_constitutions_vectors) # Now M become a |n| columns and |r| rows matrix
    q_M_right = q_M.copy()
    q_M_right[-1]=last_row_q_M_right
    result_right = (M_right * (q_M_right - 1)).T.copy()
    return result_left + result_right


############################################################################################################################################################
#
#         V6 approach
#
############################################################################################################################################################


DTYPE = np.float64
N_QUAD = 16
_nodes_15, _weights_15 = np.polynomial.legendre.leggauss(N_QUAD)
QUAD_NODES   = DTYPE(0.5) * (_nodes_15.astype(DTYPE) + DTYPE(1.0))   # (n_quad,)
QUAD_WEIGHTS = DTYPE(0.5) * _weights_15.astype(DTYPE)                 # (n_quad,)


def linear_tree_shap_v6(
    r: np.ndarray,
    p: np.ndarray,
    leaf_value: float = 1.0,
    quad_nodes = None,
    quad_weights = None
) -> np.ndarray:
    """
    SHAP values for m unique decision patterns on a root-to-leaf path of length n.

    Parameters
    ----------
    r : array, shape (n,)
        Edge weights for each depth: r[d] = child_cover / parent_cover.
        n = len(r) is the path length.
    p : int array, shape (m,)
        Unique bitmask integers, built exactly as v6 builds them:
            dp = (dp << 1) | satisfies   at each depth
        After n depths, bit (n-1-d) from the LSB is the satisfies flag at depth d.
    leaf_value : float, default 1.0
        Value stored in the leaf node.

    Returns
    -------
    phi : array, shape (m, n)
        phi[i, d] = SHAP value of the feature at depth d for pattern i.
    """
    r  = np.asarray(r, dtype=DTYPE)
    dp = np.asarray(p, dtype=np.int64)   # (m,)
    m  = len(dp)
    n  = len(r)                                          # path length
    if quad_nodes is None:
        quad_nodes=QUAD_NODES
    if quad_weights is None:
        quad_weights=QUAD_WEIGHTS
    n_quads = len(quad_nodes)

    # Unpack bitmasks → binary matrix satisfies[i, d] = 1 if pattern i goes in
    # the path direction at depth d.  bit (n-1-d) from LSB encodes depth d.
    shifts    = np.arange(n - 1, -1, -1, dtype=np.int64)             # [n-1, ..., 0]
    satisfies = ((dp[:, None] >> shifts[None, :]) & 1).astype(DTYPE) # (m, n)

    w_e    = r                                 # (n,)  r[d] is the edge weight at depth d
    w_prod = float(np.prod(r))

    # p_up = 1 always (unseen), so p_e = satisfies / w_e
    p_e     = satisfies / w_e[None, :]         # (m, n)
    alpha_e = p_e - DTYPE(1.0)                 # (m, n)

    # Linear factor at each (pattern, depth, quadrature point):
    #   linear_factors[i, d, k] = 1 + alpha_e[i, d] * t_k
    linear_factors = DTYPE(1.0) + alpha_e[:, :, None] * quad_nodes[None, None, :]  # (m, n, n_quad)

    # Exclusive prefix products: prefix[i, d, k] = prod_{d' < d} linear_factors[i, d', k]
    prefix = np.ones((m, n, n_quads), dtype=DTYPE)
    for d in range(1, n):
        prefix[:, d, :] = prefix[:, d - 1, :] * linear_factors[:, d - 1, :]

    # Exclusive suffix products: suffix[i, d, k] = prod_{d' > d} linear_factors[i, d', k]
    suffix = np.ones((m, n, n_quads), dtype=DTYPE)
    for d in range(n - 2, -1, -1):
        suffix[:, d, :] = suffix[:, d + 1, :] * linear_factors[:, d + 1, :]

    # Leave-one-out integral: prefix * suffix omits factor d → integrand without feature d
    phi = alpha_e * (leaf_value * w_prod) * ((prefix * suffix) @ quad_weights)  # (m, n)
    return phi


def linear_tree_shap_v6_for_neighbors(
    r: np.ndarray,
    decision_patterns: np.ndarray,
    w1: float = 1.0,
    w2: float = 1.0,
    quad_nodes=None,
    quad_weights=None
) -> np.ndarray:
    """
    SHAP values for two sibling leaves (sharing the same parent) simultaneously.

    The two leaves differ only at the last edge: the left leaf goes in the path
    direction at depth n-1, the right leaf goes the other way.

    Parameters
    ----------
    r : array, shape (n,)
        Edge weights for the LEFT leaf's path: r[d] = child_cover / parent_cover.
        r[n-1] is the left leaf's edge weight.
        Right leaf edge weight = 1 - r[n-1]  (covers sum at each split).
    decision_patterns : int array, shape (m,)
        Bitmasks for the LEFT leaf (n bits, same convention as shap_path).
        The right leaf's bitmasks are dp XOR 1 (LSB flipped).
    w1, w2 : float
        Leaf values for the left and right leaf respectively.

    Returns
    -------
    phi_left  : array, shape (m, n)
    phi_right : array, shape (m, n)

    Shared work
    -----------
    Both leaves share the same path for depths 0..n-2, so the following are
    computed once and reused:
      - linear_factors[:, 0:n-1, :]   (m, n-1, n_quad)
      - prefix[:, :, :]               (m, n,   n_quad)  — prefix at depth n-1
                                                           is the product of all
                                                           shared factors
      - suffix_no_last[:, :, :]       (m, n-1, n_quad)  — suffix of shared factors
      - leave_one_out_shared           (m, n-1, n_quad)  = prefix[:, :n-1] * suffix_no_last
      - integral_last                  (m,)              = prefix[:, n-1] @ weights

    Per-leaf extra cost: one (m, n_quad) multiply + one (m, n-1) matmul each.
    """
    r  = np.asarray(r, dtype=DTYPE)
    dp = np.asarray(decision_patterns, dtype=np.int64)   # (m,)
    m  = len(dp)
    n  = len(r)
    if quad_nodes is None:
        quad_nodes = QUAD_NODES
    if quad_weights is None:
        quad_weights = QUAD_WEIGHTS
    n_quads = len(quad_nodes)
    # total path length

    # ── Unpack all n bits ──────────────────────────────────────────────────
    shifts    = np.arange(n - 1, -1, -1, dtype=np.int64)              # [n-1, …, 0]
    satisfies = ((dp[:, None] >> shifts[None, :]) & 1).astype(DTYPE)  # (m, n)

    # ── Edge weights ──────────────────────────────────────────────────────
    w_e = r                                               # (n,)  r[d] IS the edge weight
    # Right leaf's last edge weight: covers sum to 1 at each split.
    w_e_right_last = DTYPE(1.0) - r[n - 1]

    w_prod_left  = float(np.prod(r))
    w_prod_right = float(np.prod(r[:n - 1])) * float(w_e_right_last)

    # ── Shared: depths 0..n-2 ─────────────────────────────────────────────
    satisfies_shared = satisfies[:, :n - 1]               # (m, n-1)
    p_e_shared       = satisfies_shared / w_e[None, :n-1] # (m, n-1)
    alpha_e_shared   = p_e_shared - DTYPE(1.0)            # (m, n-1)

    lf_shared = (DTYPE(1.0)
                 + alpha_e_shared[:, :, None] * quad_nodes[None, None, :])  # (m, n-1, n_quad)

    # Prefix over ALL n depths (based only on shared factors since last factor
    # is not included until we know which leaf we're computing).
    # prefix[i, d, k] = prod_{d'<d, d'<n-1} lf_shared[i, d', k]
    prefix = np.ones((m, n, n_quads), dtype=DTYPE)
    for d in range(1, n):
        prefix[:, d, :] = prefix[:, d - 1, :] * lf_shared[:, d - 1, :]
    # prefix[:, n-1, :] is now the full product of all shared factors.

    # Exclusive suffix of shared factors (no last factor):
    # suffix_no_last[i, d, k] = prod_{d'=d+1}^{n-2} lf_shared[i, d', k]
    suffix_no_last = np.ones((m, n - 1, n_quads), dtype=DTYPE)
    for d in range(n - 3, -1, -1):
        suffix_no_last[:, d, :] = suffix_no_last[:, d + 1, :] * lf_shared[:, d + 1, :]

    # Leave-one-out for shared depths — same for both leaves before multiplying
    # in the leaf-specific last factor.
    leave_one_out_shared = prefix[:, :n - 1, :] * suffix_no_last   # (m, n-1, n_quad)

    # Integral at the last depth reuses prefix[:, n-1, :] (suffix = 1 there).
    integral_last = prefix[:, n - 1, :] @ quad_weights              # (m,)

    # ── Per-leaf computation ───────────────────────────────────────────────
    def _phi_leaf(satisfies_last, w_e_last, leaf_value, w_prod, c_quad_nodes, c_quad_weights):
        # Depth n-1 factors (leaf-specific)
        alpha_e_last = satisfies_last / w_e_last - DTYPE(1.0)        # (m,)
        lf_last      = (DTYPE(1.0)
                        + alpha_e_last[:, None] * c_quad_nodes[None, :])  # (m, n_quad)

        scale = DTYPE(leaf_value * w_prod)

        # Shared depths: multiply leave_one_out_shared by this leaf's last factor
        integrals_shared = (
            leave_one_out_shared * lf_last[:, None, :]   # (m, n-1, n_quad)
        ) @ c_quad_weights                                              # (m, n-1)
        phi_shared = alpha_e_shared * scale * integrals_shared        # (m, n-1)

        # Last depth: suffix is 1, so integrand = prefix[:, n-1, :]
        phi_last = alpha_e_last * scale * integral_last               # (m,)

        return np.concatenate([phi_shared, phi_last[:, None]], axis=1)  # (m, n)

    phi_left  = _phi_leaf(satisfies[:, n - 1],          w_e[n - 1],    w1, w_prod_left, quad_nodes, quad_weights)
    phi_right = _phi_leaf(DTYPE(1.0) - satisfies[:, n - 1], w_e_right_last, w2, w_prod_right, quad_nodes, quad_weights)
    return phi_left + phi_right









############################################################################################################################################################
#
#                         Old Ones - Delete Them Someday
#
############################################################################################################################################################

def linear_tree_shap_magic_blocked_draft(
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
        poly_mul_y_plus_q_inplace(P_top_half, q_M[j])

    P_bot_half = base_poly.copy()
    for j in bot_idx:
        poly_mul_y_plus_q_inplace(P_bot_half, q_M[j])

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
            poly_mul_y_plus_q_inplace(P_shared, q_M[j])

        # For each i in this block, finish the local multiplications excluding i
        for i in range(block[0], block[-1]+1):
            P = P_shared.copy()
            # multiply by all (y + q_j) for j in block, j != i
            for j in range(max(i-1,block[0]), block[-1]+1):
                if j == i:
                    P_shared = P.copy()
                    continue
                poly_mul_y_plus_q_inplace(P, q_M[j])

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


def linear_tree_shap_magic_not_numerically_stable_backward(
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


def linear_tree_shap_magic_not_numerically_stable_forward(
        r: np.array, p: np.array, f_w: np.array, leaf_weight: float
):
    """
    Not numerically stable but O(D) faster. Don't use this
    """
    b_M = bits_matrix(p, len(r))
    q_M = b_M * (1/r.reshape(-1, 1))
    one_over_q_M = b_M * r.reshape(-1, 1)

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

    neg_b_M = neg_bits_matrix(p, len(r))
    constitutions_vectors = []
    for i in range(len(r)):
        # Divide the polynomials by (y + q_i)
        M_f_i = np.zeros((len(r), len(p)))
        M_f_i[0] = M_general[0] * one_over_q_M[i] + M_general[1] * neg_b_M[i]
        for d in range(1, len(r)):
            M_f_i[d] = (M_general[d] - M_f_i[d-1]) * one_over_q_M[i] + M_general[d+1] * neg_b_M[i]

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

def linear_tree_shap_magic_for_banzhaf_for_neighbors(
        r: np.array, p: np.array, left_leaf_weight: float, right_leaf_weight: float
):
    R_emptyset = np.prod(r[:-1])

    b_M = bits_matrix(p, len(r))
    neg_b_M = neg_bits_matrix(p, len(r))
    q_M = b_M * (1 / r.reshape(-1, 1))
    last_row_q_M_right = neg_b_M[-1] * (1 / (1 - r[-1]))

    sum_original_coefs = np.prod((1 + q_M), axis=0) * R_emptyset
    constitutions_vectors = []
    for i in range(len(r)):
        M_f_i = sum_original_coefs * (1/(1+q_M[i]))
            # constitutions_vectors.append((M_f_i / 2 ** (len(r) - 1)) * (left_leaf_weight * r[-1] + right_leaf_weight * (1- r[-1])))

        left_constitutions_vector = M_f_i / 2 ** (len(r) - 1) * left_leaf_weight * r[-1]
        if i == len(r) - 1:
            M_f_i_right = M_f_i
        else:
            M_f_i_excluding_last = M_f_i * (1 / (1 + q_M[-1]))
            M_f_i_right = M_f_i_excluding_last * (1+last_row_q_M_right)
        right_constitutions_vector = M_f_i_right / 2 ** (len(r) - 1) * right_leaf_weight * (1 - r[-1])
        constitutions_vectors.append(left_constitutions_vector + right_constitutions_vector)

    M = np.array(constitutions_vectors)  # Now M become a |n| columns and |r| rows matrix
    return (M * (q_M - 1)).T.copy()

def linear_tree_shap_magic_blocked(
    r: np.ndarray, p: np.ndarray, f_w: np.ndarray, leaf_weight: float
) -> np.ndarray:
    """
    O(k^2.5 * n) version using sqrt-blocking, while preserving the stable
    shift+add multiplication scheme.

    Returns: shape (n, k) like your original (M * (q_M - 1)).T
    """
    k = len(r)
    q_M = bits_matrix(p, len(r)) * (1/r.reshape(-1, 1))

    P = np.zeros((len(r), len(p)))
    P[0, :] = np.prod(r)

    # choose block size B ~ sqrt(k)
    block_size = int(np.ceil(np.sqrt(k)))
    G = int(np.ceil(k / block_size))

    # Precompute index blocks
    blocks = [list(range(start, min(start + block_size, k))) for start in range(0, k, block_size)]

    mid_index=blocks[G // 2][0]
    P_top_half = continue_P_compute(P, q_M, start_index=0, end_index=mid_index)
    P_bot_half = continue_P_compute(P, q_M, start_index=mid_index, end_index=k)

    # For each block g compute P_out[g] = product over (y + q_j) for j not in block
    contribs = []
    for block_index, block in enumerate(blocks):
        if block_index < G // 2:
            P_shared = P_bot_half.copy()
            to_compute_for_global_p = [i for i in range(0, mid_index) if i not in block]
        else:
            P_shared = P_top_half.copy()
            to_compute_for_global_p = [i for i in range(mid_index, k) if i not in block]

        for j in to_compute_for_global_p:
            poly_mul_y_plus_q_inplace(P_shared, q_M[j])

        contribs.append(compute_contribution_vectors_from_initial_P(P_shared, q_M, f_w, leaf_weight, start_index=block[0], end_index=block[-1] + 1))

    return (np.vstack(contribs) * (q_M - 1)).T.copy()