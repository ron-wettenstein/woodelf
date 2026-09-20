import time
from typing import List, Dict, Optional

import numpy as np

from woodelf.core.path_to_s_vectors.base_p2s import PathToSVectors
from woodelf.core.utils import bits_matrix
from woodelf.simple_woodelf import get_int_dtype_from_depth


def triu_pair_to_index(i: int, j: int, D: int) -> int:
    if i == j:
        raise ValueError("Diagonal pair has no index in triu_indices(D, k=1)")
    if i > j:
        i, j = j, i
    return i * (2 * D - i - 1) // 2 + (j - i - 1)

class PDPPathToSVectors(PathToSVectors):
    """
    PDP path-to-s-vectors calculator.
    get_background_s_matrix  — exact PDP using background data patterns.
    get_path_dependent_s_matrix — estimated PDP using tree cover ratios
                                  (was EstimatedPDPPathToMatrices).

    Requires numpy >= 2, as it uses np.bitwise_count (added in numpy 2.0).
    """

    def __init__(self, max_depth: int, GPU: bool = False):
        super().__init__(None, max_depth, GPU) # As we don't use it, we give metric=None
        self.computation_time = 0
        self.compute_f_time = 0

    def _build_f_from_patterns(self, background_patterns: np.ndarray, D: int):
        """Exact: compute f and only_positive_literals from integer background patterns."""
        x = background_patterns[np.bitwise_count(background_patterns) >= D - 1]
        int_type = get_int_dtype_from_depth(D)
        full = int_type(2 ** D - 1)
        orig_x_len = len(x)
        x = x[x != full]
        only_positive_literals = (orig_x_len - len(x)) / len(background_patterns)
        zero_bit_location = np.bitwise_count(x + 1) - 1  # which is equivalent to: D - 1 - np.log2(full ^ x).astype(np.uint16)

        f = np.bincount(zero_bit_location, minlength=D) / len(background_patterns)
        return f, only_positive_literals

        # Clearer but less optimized equivalent:
        # full = int_type(2 ** D - 1)
        # only_positive_literals = np.sum(background_patterns == full) / len(background_patterns)
        # x = background_patterns[D - np.bitwise_count(background_patterns) == 1]
        # zero_bit_location = np.bitwise_count(x + 1) - 1
        # f = np.bincount(zero_bit_location, minlength=D) / len(background_patterns)
        # return f, only_positive_literals

    def _build_f_from_covers(self, covers: np.ndarray, D: int):
        """Estimated: compute f and only_positive_literals from tree cover ratios."""
        only_positive_literals = float(np.prod(covers))
        f_list = []
        for i in range(D):
            current_covers = covers.copy()
            current_covers[i] = 1 - covers[i]
            f_list.append(np.prod(current_covers))
        return np.array(f_list), only_positive_literals

    def _compute_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        f: np.ndarray, only_positive_literals: float, w: float
    ) -> Dict:
        D = len(features_in_path)
        bm_consumer = bits_matrix(consumer_patterns, D).T
        s_matrix = bm_consumer * f
        if only_positive_literals != 0:
            s_matrix[bm_consumer == 0] = -1 * only_positive_literals
        s_matrix = w * s_matrix
        return {feature: s_matrix[:, i].astype(np.float32) for i, feature in enumerate(features_in_path)}

    def get_background_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        if not features_in_path:
            return {}
        start_time = time.time()
        D = len(features_in_path)
        f_start = time.time()
        f, only_positive_literals = self._build_f_from_patterns(background_patterns, D)
        self.compute_f_time += time.time() - f_start
        result = self._compute_s_matrix(features_in_path, consumer_patterns, f, only_positive_literals, w)
        self.computation_time += time.time() - start_time
        return result

    def get_path_dependent_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        if not features_in_path:
            return {}
        start_time = time.time()
        D = len(features_in_path)
        f_start = time.time()
        f, only_positive_literals = self._build_f_from_covers(covers, D)
        self.compute_f_time += time.time() - f_start
        result = self._compute_s_matrix(features_in_path, consumer_patterns, f, only_positive_literals, w)
        self.computation_time += time.time() - start_time
        return result

    def present_statistics(self):
        print(f"{self.__class__.__name__} took {round(self.computation_time, 2)}, computing f took {round(self.compute_f_time, 2)}")


class PDPIVPathToSVectors(PDPPathToSVectors):
    """
    PDP Interaction Values path-to-s-vectors calculator.
    get_background_s_matrix  — exact PDP-IV using background data patterns.
    get_path_dependent_s_matrix — not yet implemented (raises NotImplementedError).
    """

    def get_needed_b_vectors(self, background_patterns: np.ndarray, D: int):
        x_up_to_2_zero_bits = background_patterns[np.bitwise_count(background_patterns) >= D - 2]
        x_bitwise_count = np.bitwise_count(x_up_to_2_zero_bits)
        x_up_to_1_zero_bit = x_up_to_2_zero_bits[x_bitwise_count >= D - 1]

        int_type = get_int_dtype_from_depth(D)
        full = int_type(2 ** D - 1)
        orig_x_len = len(x_up_to_1_zero_bit)
        x_exactly_1_zero_bit = x_up_to_1_zero_bit[x_up_to_1_zero_bit != full]
        only_positive_literals = (orig_x_len - len(x_exactly_1_zero_bit)) / len(background_patterns)

        zero_bit_location = np.bitwise_count(x_exactly_1_zero_bit + 1) - 1
        f_1_zero_bit = np.bincount(zero_bit_location, minlength=D) / len(background_patterns)

        i_idx, j_idx = np.triu_indices(D, k=1)
        b_v_1 = f_1_zero_bit[j_idx]
        b_v_2 = f_1_zero_bit[i_idx]

        x_exactly_2_zero_bit = x_up_to_2_zero_bits[x_bitwise_count == D - 2]
        if len(x_exactly_2_zero_bit) == 0:
            return only_positive_literals, b_v_1, b_v_2, np.zeros(len(i_idx), dtype=float)

        z = full ^ x_exactly_2_zero_bit # exactly two 1 bits, at the zero positions of x2

        # lower 1 bit
        low = z & -z
        # upper 1 bit
        high = z ^ low

        # actual bit positions: 0 = LSB, ..., D-1 = MSB
        p_low = np.bitwise_count(low - 1)
        p_high = np.bitwise_count(high - 1)

        # convert actual bit positions to bits_matrix row indices
        r1 = D - 1 - p_high
        r2 = D - 1 - p_low

        # ensure r1 < r2
        i = np.minimum(r1, r2)
        j = np.maximum(r1, r2)

        pair_index = i * (2 * D - i - 1) // 2 + (j - i - 1)

        b_v_3 = np.bincount(pair_index, minlength=len(i_idx)) / len(background_patterns)
        return only_positive_literals, b_v_1, b_v_2, b_v_3

    def get_background_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        if not features_in_path or len(features_in_path) < 2:
            return {}
        start_time = time.time()
        D = len(features_in_path)
        bm_consumer = bits_matrix(consumer_patterns, D).T
        i_idx, j_idx = np.triu_indices(D, k=1)
        c_v = 2 * bm_consumer[:, i_idx] + bm_consumer[:, j_idx]
        f_start = time.time()
        only_positive_literals, b_v_1, b_v_2, b_v_3 = self.get_needed_b_vectors(background_patterns, D)
        self.compute_f_time += time.time() - f_start
        s_matrix = (
            only_positive_literals * (c_v == 0)
            - b_v_1 * (c_v == 1)
            - b_v_2 * (c_v == 2)
            + b_v_3 * (c_v == 3)
        )
        s_matrix = (w * s_matrix).astype(np.float32)
        self.computation_time += time.time() - start_time

        results = {}
        for index1, feature1 in enumerate(features_in_path):
            for index2, feature2 in enumerate(features_in_path):
                if index1 == index2:
                    continue
                idx = triu_pair_to_index(index1, index2, D)
                if (feature1, feature2) not in results:
                    results[(feature1, feature2)] = s_matrix[:, idx].copy()
                else:
                    results[(feature1, feature2)] += s_matrix[:, idx]
        return results

    def get_path_dependent_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        raise NotImplementedError("Estimated PDP-IV is not yet implemented")