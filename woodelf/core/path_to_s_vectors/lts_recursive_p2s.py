import time
from math import factorial
from typing import List, Dict, Optional

import numpy as np

from woodelf.core.cube_metric import CubeMetric, ShapleyInteractionValues, ShapleyValues, BanzhafValues
from woodelf.core.path_to_s_vectors.base_p2s import PathToSVectors
from woodelf.core.utils import bits_matrix, neg_bits_matrix


def nCk(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

def shapley_values_f_w(depth):
    return np.array([[1 / (depth * nCk(depth-1, s))] for s in range(depth)])

def banzhaf_values_f_w(depth):
    return np.array([[1 / 2 ** (depth - 1)] for s in range(depth)])


class LTSRecursivePathToSVectors(PathToSVectors):

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)

        self.f_ws = None
        if isinstance(metric, ShapleyInteractionValues) or isinstance(metric, ShapleyValues):
            # None f_ws for depth 0, in the rest use shapley_values_f_w(depth)
            self.f_ws = [None] + [shapley_values_f_w(depth) for depth in range(1, max_depth + 1)]

        self.computation_time = 0

    def _get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        start_time = time.time()
        # assume features in path are unique
        if isinstance(self.metric, ShapleyInteractionValues):
            assert w_neighbor is None
            if len(covers) <= 1:
                self.computation_time += time.time() - start_time
                return []
            f_w = self.f_ws[len(covers)-1]
            s_matrix = improved_linear_tree_shap_iv(covers, consumer_patterns, f_w, w)
        elif isinstance(self.metric, ShapleyValues):
            f_w = self.f_ws[len(covers)]
            if w_neighbor is None:
                # s_matrix = linear_tree_shap_magic_faster_v2(covers, consumer_patterns, f_w, w)
                s_matrix = improved_linear_tree_shap_magic(covers, consumer_patterns, f_w, w)
            else:
                s_matrix = improved_linear_tree_shap_magic_for_neighbors(covers, consumer_patterns, f_w, w, w_neighbor)
        elif isinstance(self.metric, BanzhafValues):
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
        else:
            raise ValueError(f"Unsupported metric {self.metric.__class__}. The our LinearTreeSHAP implementation currently support only Shapley values, Banzhaf values and Shapley interaction values")
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
        """consumer_patterns must be unique (pre-factorized by the caller)."""
        if not features_in_path:
            return {}
        s_matrix = self._get_s_matrix(covers, consumer_patterns, w, w_neighbor)
        if not isinstance(self.metric, ShapleyInteractionValues):
            s_matrix = s_matrix.astype(np.float32)
            # TODO why np indexing on a matrix is slower than vector by vector! contribution_values = s_matrix[inverse]
            return {feature: s_matrix[:, i] for i, feature in enumerate(features_in_path)}
        else:
            if not s_matrix:  # < 2 unique features in path, no interactions
                return {}
            result = {}
            for i, feature1 in enumerate(features_in_path):
                s_matrix_i = s_matrix[i].astype(np.float32)
                for j, feature2 in enumerate(features_in_path):
                    if i != j:
                        f2_index = j if j < i else j - 1
                        result[(feature1, feature2)] = s_matrix_i[:, f2_index]
            return result

    def present_statistics(self):
        print(f"LTSRecursivePathToSVectors took {round(self.computation_time, 2)}")


class AlwaysParticipatingLTSPathToSVectors(LTSRecursivePathToSVectors):
    """
    Extends LTSRecursivePathToSVectors for the restricted game where a fixed set of features
    is always in every coalition S (always participating).

    Implementation Details:
        Always-participating features have their cover forced to 1, making q_i = 1 and their
        marginal contribution 0. Regular features (not in always_participating) are the free
        players whose Shapley/Banzhaf values are computed.

        Consumers that do not satisfy all always-participating path conditions at a leaf contribute
        nothing at that leaf (zeroed out), because the always-participating features are fixed as
        consumer-side and the consumer does not reach the leaf through those splits.

        w_neighbor is not supported — the neighbor trick couples the last feature's cover across
        two leaves, which would break the Null-player guarantee for always-participating features.
    """

    def __init__(self, metric: CubeMetric, max_depth: int, always_participating: List[str], GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        self._always_participating: set = set(always_participating)

    def get_path_dependent_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        assert w_neighbor is None, "AlwaysParticipatingLTSPathToSVectors does not support w_neighbor"
        if not features_in_path:
            return {}

        n = len(features_in_path)
        always_participating_bit_pattern = 0
        modified_covers = covers.copy()
        for i, f in enumerate(features_in_path):
            if f in self._always_participating:
                always_participating_bit_pattern |= (1 << (n - 1 - i))
                modified_covers[i] = 1.0

        result = super().get_path_dependent_s_matrix(
            features_in_path, consumer_patterns, modified_covers, w, w_neighbor=None
        )

        if always_participating_bit_pattern == 0:
            return result

        mask = (consumer_patterns & always_participating_bit_pattern) == always_participating_bit_pattern
        return {f: s_vec * mask for f, s_vec in result.items()}


# !!!!!!!!!!!!!!!! The Recursive Linear TreeSHAP Logic !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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
#         Recursive Linear TreeSHAP for pairwise interaction values
#
############################################################################################################################################################

def extract_bit(patterns: np.ndarray, i: int):
    """
    patterns: np.array of integers
    i: bit index to extract (0 = LSB)

    Returns: patterns with bit i removed
    """
    lower = patterns & ((1 << i) - 1)     # bits below i
    upper = patterns >> (i + 1)           # bits above i
    return lower + (upper << i)

def improved_linear_tree_shap_iv(r: np.array, p: np.array, f_w: np.array, w: float):
    q_M = bits_matrix(p, len(r)) * (1 / r.reshape(-1, 1))
    assert len(f_w) == len(r) - 1
    shaps = []
    for i, ratio in enumerate(r):
        extracted_patterns = extract_bit(p, len(r) - 1 - i)
        new_r = np.concatenate([r[:i], r[i+1:]])
        shap_excluding_i = improved_linear_tree_shap_magic(new_r, extracted_patterns, f_w, w)
        q_i = (np.tile(q_M[i], (q_M.shape[0] - 1, 1))).T
        # The interaction values (i,j) are the shapley values of j in the game excluding i, times '(q_i - 1) * ratio'
        # The '/ 2' is to fit the shap package logic, splitting the interaction between the two features.
        shaps_i = (shap_excluding_i * (q_i - 1) * ratio / 2).copy()
        shaps.append(shaps_i)
    return shaps
