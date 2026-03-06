import numpy as np


def bits_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """
    x: shape (n,), integers
    returns: shape (k, n), rows are bits (k-1),...,1,0 (2^(k-1) down to LSB)
    """
    # ensure x is unsigned (np.uint) for fast bit ops
    shifts = np.arange(k-1, -1, -1, dtype=np.uint64)[:, None]  # (5,1): 4,3,2,1,0
    return ((x[None, :] >> shifts) & 1).astype(np.uint8)


def poly_mul_y_plus_q_inplace(P: np.ndarray, q: np.ndarray) -> None:
    """
    In-place multiply polynomial P(y) by (y + q) for many columns at once.

    P: shape (k, n). Row t = coefficient of y^t, same convention as your code.
    q: shape (n,) broadcast across rows.

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
        P: np.array, r: np.array, q_M: np.array, f_w: np.array, w: float, start_index: int, end_index: int
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
        return compute_contribution_vectors_from_initial_P(P, r, q_M, f_w, w, start_index, end_index)

    top_P = continue_P_compute(P, q_M, start_index=start_index, end_index=middle_index)
    bottom_P = continue_P_compute(P, q_M, start_index=middle_index, end_index=end_index)

    top_contribs = recursive_linear_tree_shap(bottom_P, r, q_M, f_w, w, start_index=start_index, end_index=middle_index)
    bottom_contribs = recursive_linear_tree_shap(top_P, r, q_M, f_w, w, start_index=middle_index, end_index=end_index)
    return np.vstack([top_contribs, bottom_contribs])


def improved_linear_tree_shap_magic(r: np.array, patterns: np.array, f_w: np.array, w: float):
    q_M = bits_matrix(patterns, len(r)) * (1 / r.reshape(-1, 1))

    top_P = compute_P(r, q_M, start_index=0, end_index=len(r)//2)
    bottom_P = compute_P(r, q_M, start_index=len(r)//2, end_index=len(r))

    bottom_contribs = recursive_linear_tree_shap(top_P, r, q_M, f_w, w, start_index=len(r)//2, end_index=len(r))
    top_contribs = recursive_linear_tree_shap(bottom_P, r, q_M, f_w, w, start_index=0, end_index=len(r)//2)
    M = np.vstack([top_contribs, bottom_contribs])
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

        contribs.append(compute_contribution_vectors_from_initial_P(P_shared, r, q_M, f_w, leaf_weight, start_index=block[0], end_index=block[-1]+1))

    return (np.vstack(contribs) * (q_M - 1)).T.copy()


############################################################################################################################################################
#
#                         Supports The Right Leaf Neighbor Trick
#
############################################################################################################################################################