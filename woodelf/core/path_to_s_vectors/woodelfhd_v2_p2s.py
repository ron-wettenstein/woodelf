from __future__ import annotations

import time
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np

from woodelf.core.cube_metric import CubeMetric, ShapleyValues, BanzhafValues, ShapleyInteractionValues, \
    BanzhafInteractionValues, nCk
from woodelf.core.decision_patterns import get_value_keys_and_decision_pattern_bits
from woodelf.core.path_to_s_vectors.base_p2s import WoodelfPathToSVectors, compute_f_from_patterns
from woodelf.core.utils import safe_bitwise_count

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False


def secondary_diagonal_popcounts(depth: int) -> np.ndarray:
    """
    The number of positive literals in every cube on the secondary diagonal of the M matrices.
    Row p of the secondary diagonal holds the full cube whose positive literals are the set bits of p (Rule 1)
    and whose negated literals are its zero bits (Rule 2), so this is the popcount of every p in 0..2^depth-1.
    """
    return safe_bitwise_count(np.arange(2 ** depth))


def _values_diagonals(depth: int, v_plus: Callable, v_minus: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    The secondary diagonals of M+ and M-. v_plus(p, q) and v_minus(p, q) are the values of a positive literal and of a
    negated literal on a full cube with p positive and q negated literals (p + q = depth). Following the linear
    extensions, M+ is 0 on the cube without positive literals and M- is 0 on the cube without negated literals.
    """
    plus_by_p = [v_plus(p, depth - p) if p >= 1 else 0.0 for p in range(depth + 1)]
    minus_by_p = [v_minus(p, depth - p) if depth - p >= 1 else 0.0 for p in range(depth + 1)]
    p_of_row = secondary_diagonal_popcounts(depth)
    return np.array(plus_by_p)[p_of_row], np.array(minus_by_p)[p_of_row]


def _pair_values_diagonals(
        depth: int, v_plus_plus: Callable, v_minus_minus: Callable, v_plus_minus: Callable
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The secondary diagonals of M++, M-- and M+-. v_plus_plus(p, q), v_minus_minus(p, q) and v_plus_minus(p, q) are the
    values of a pair of positive literals, a pair of negated literals and a mixed pair on a full cube with p positive and
    q negated literals (p + q = depth). Following the linear extensions, each is 0 on the cubes without its pair type.
    """
    plus_plus_by_p = [v_plus_plus(p, depth - p) if p >= 2 else 0.0 for p in range(depth + 1)]
    minus_minus_by_p = [v_minus_minus(p, depth - p) if depth - p >= 2 else 0.0 for p in range(depth + 1)]
    plus_minus_by_p = [v_plus_minus(p, depth - p) if p >= 1 and depth - p >= 1 else 0.0 for p in range(depth + 1)]
    p_of_row = secondary_diagonal_popcounts(depth)
    return np.array(plus_plus_by_p)[p_of_row], np.array(minus_minus_by_p)[p_of_row], np.array(plus_minus_by_p)[p_of_row]


# The closed form values, the formulas of the metrics calc_metric, on a full cube with p positive and q negated literals:
# (v+, v-) for values (see _values_diagonals) and (v++, v--, v+-) for pairwise interaction values (see _pair_values_diagonals)
CLOSED_FORM_VALUES = {
    ShapleyValues: (
        lambda p, q: 1 / (p * nCk(p + q, p)),
        lambda p, q: -1 / (q * nCk(p + q, q)),
    ),
    BanzhafValues: (
        lambda p, q: 1 / (2 ** (p + q - 1)),
        lambda p, q: -1 / (2 ** (p + q - 1)),
    ),
    ShapleyInteractionValues: ( # including its 0.5, the shap package convention
        lambda p, q: 0.5 / ((p - 1) * nCk(p + q - 1, p - 1)),
        lambda p, q: 0.5 / ((q - 1) * nCk(p + q - 1, q - 1)),
        lambda p, q: -0.5 / (q * nCk(p + q - 1, q)),
    ),
    BanzhafInteractionValues: (
        lambda p, q: 1 / (2 ** (p + q - 2)),
        lambda p, q: 1 / (2 ** (p + q - 2)),
        lambda p, q: -1 / (2 ** (p + q - 2)),
    ),
}


class HighDepthWoodelfV2PathToSVectors(WoodelfPathToSVectors):
    """
    WoodelfHD that exploits the symmetry of the metric. Instead of one M matrix per feature (or feature pair), all the
    features share M+ and M- (M++, M-- and M+- for pairwise interaction values), so a leaf needs only two (three)
    StrassenLike multiplications that give the shared s vectors s+ and s- (s++, s-- and s+-). Then:
        s_i[p] = s+[p] + s_R[p^-i]                                  where s_R = s- - s+
        s_ij[p] = s++[p] + s_A[p^-i] + s_A[p^-j] + s_Q[p^-ij]       where s_A = s+- - s++, s_Q = s++ - 2s+- + s--
    and p^-i is the decision pattern p with the bit of feature i set to 0.
    Only the secondary diagonals of the shared matrices are stored: O(2^D) memory instead of
    O(2^D * D) for first-order values and O(2^D * D^) for interaction values.

    Supports ShapleyValues, BanzhafValues, ShapleyInteractionValues and BanzhafInteractionValues (CLOSED_FORM_VALUES).
    create_high_depth_woodelf_p2s falls back to HighDepthWoodelfV1PathToSVectors for the rest.
    """
    COMPUTES_CONSUMER_VALUES = True
    # With more consumers than FULL_S_VECTOR_CONSUMERS_RATIO * 2^D, build each s vector over all the 2^D patterns and
    # index it by the consumer patterns once. Otherwise index s+ and s_R (s++, s_A and s_Q) by the consumer patterns directly.
    # The ratios are the benchmarked crossovers: a pair takes 3 lookups per consumer in the direct way, a feature only 1
    FULL_S_VECTOR_CONSUMERS_RATIO = 1.0
    FULL_PAIR_S_VECTOR_CONSUMERS_RATIO = 0.5

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        if not self.supports_metric(metric):
            raise ValueError(
                f"HighDepthWoodelfV2PathToSVectors does not support {type(metric).__name__} (see supports_metric). "
                f"Use HighDepthWoodelfV1PathToSVectors, or create_high_depth_woodelf_p2s that picks the right one."
            )
        if GPU and not IMPORTED_CP:
            raise ImportError("Couldn't import CuPy. To use GPU, please install Cupy via 'pip install cupy'")
        self.xp = cp if GPU else np
        self.full_s_vector_consumers_ratio = \
            self.FULL_PAIR_S_VECTOR_CONSUMERS_RATIO if metric.INTERACTION_VALUE else self.FULL_S_VECTOR_CONSUMERS_RATIO
        self.s_computation_time = 0
        self.f_prepare_time = 0
        self.consumer_values_time = 0
        self.s_computation_calls = 0
        self.total_f_sizes = 0

        start_time = time.time()
        self.diagonals = {}
        self._build_diagonals()
        self.matrices_init_time = time.time() - start_time

    @staticmethod
    def supports_metric(metric: CubeMetric) -> bool:
        """
        True for ShapleyValues, BanzhafValues, ShapleyInteractionValues and BanzhafInteractionValues (CLOSED_FORM_VALUES).
        Their exact types only, as a subclass may change calc_metric.
        """
        return type(metric) in CLOSED_FORM_VALUES

    @staticmethod
    def metric_diagonals(metric: CubeMetric, depth: int) -> Tuple:
        """
        The secondary diagonals of the shared matrices of the metric for paths with depth unique features:
        (M+, M-) for values, (M++, M--, M+-) for pairwise interaction values.
        """
        if metric.INTERACTION_VALUE:
            return _pair_values_diagonals(depth, *CLOSED_FORM_VALUES[type(metric)])
        return _values_diagonals(depth, *CLOSED_FORM_VALUES[type(metric)])

    def _build_diagonals(self):
        for depth in range(0, self.max_depth + 1):
            # Row k is the secondary diagonal of the k-th shared matrix: M+, M- (M++, M--, M+-). float32 as in v1
            self.diagonals[depth] = self.xp.array(np.stack(self.metric_diagonals(self.metric, depth)), dtype=self.xp.float32)

    @staticmethod
    def _prepare_f(depth, f, GPU):
        f = f.copy() # the updates below are in-place and the caller may reuse f (the neighbor leaf trick does)
        for d in range(depth - 1, -1, -1):
            v = f.reshape(-1, 2, 2 ** d) # view only: split each index into (higher bits, bit d, lower bits)
            v[:, 0] += v[:, 1] # elements with bit d = 0 get the element with bit d = 1 added to them
        return f

    def _get_shared_s_vectors(self, depth: int, f: np.ndarray, w: float) -> Tuple:
        """
        The shared s vectors of a leaf: s+ = M+ * f and s- = M- * f (s++, s-- and s+- for pairwise interaction values),
        all computed with the StrassenLike multiplication and multiplied by the leaf weight w.
        """
        self.s_computation_calls += 1
        self.total_f_sizes += np.sum(f != 0)
        start_time = time.time()

        start_time_f_prepare = time.time()
        f = self._prepare_f(depth, f, self.GPU)
        self.f_prepare_time += time.time() - start_time_f_prepare

        # Row k of s_matrix is the k-th shared s vector. The diagonals are a C-contiguous (rows, 2^depth) array, so is
        # the product: the reshapes below must be views, a reshape that copies would silently lose the in-place updates
        s_matrix = self.diagonals[depth] * f[::-1] # reversed as this is not the (0,0)-(1,1)-..-(n,n) diagonal but the (0,n)-(1,n-1)-..-(n,0) diagonal
        num_rows = s_matrix.shape[0]
        for d in range(0, depth, 1):
            v = s_matrix.reshape(num_rows, -1, 2, 2 ** d) # view only: split each index into (higher bits, bit d, lower bits)
            v[:, :, 1] += v[:, :, 0] # elements with bit d = 1 get the element with bit d = 0 added to them

        s_matrix = s_matrix * w

        self.s_computation_time += time.time() - start_time
        return tuple(s_matrix)

    def _combine_shared_s_vectors(self, shared_s_vectors: Tuple) -> Tuple:
        """
        Turn the shared s vectors (in place) into the vectors that the s vector of every feature (pair) is built from:
        (s+, s-) -> (s+, s_R = s- - s+) and (s++, s--, s+-) -> (s++, s_A = s+- - s++, s_Q = s++ - 2s+- + s--)
        """
        if not self.metric.INTERACTION_VALUE:
            s_plus, s_minus = shared_s_vectors
            S_R = s_minus
            S_R -= s_plus
            return s_plus, S_R
        s_plus_plus, s_minus_minus, s_plus_minus = shared_s_vectors

        s_Q = s_minus_minus
        s_Q += s_plus_plus # s_Q, computed before s+- is overwritten by s_A
        s_Q -= s_plus_minus
        s_Q -= s_plus_minus

        s_P = s_plus_minus
        s_P -= s_plus_plus
        return s_plus_plus, s_P, s_Q

    def _build_s_vector(self, combined_s_vectors: Tuple, bits: Tuple, out: np.ndarray) -> np.ndarray:
        """
        The s vector of a feature (pair) over all the 2^D decision patterns, written to out:
        s_i[p] = s+[p] + s_R[p^-i]  or  s_ij[p] = s++[p] + s_A[p^-i] + s_A[p^-j] + s_Q[p^-ij]
        """
        if len(bits) == 1:
            s_plus, s_r = combined_s_vectors
            shape = (-1, 2, bits[0]) # split each index into (higher bits, the bit of i, lower bits)
            # s_R[:, :1] is s_R with the bit of i set to 0, broadcast to both values of the bit
            self.xp.add(s_plus.reshape(shape), s_r.reshape(shape)[:, :1], out=out.reshape(shape))
            return out
        s_plus_plus, s_a, s_q = combined_s_vectors
        low_bit, high_bit = bits
        shape = (-1, 2, high_bit // (2 * low_bit), 2, low_bit) # split each index into (higher bits, high bit, middle bits, low bit, lower bits)
        out_view = out.reshape(shape)
        s_a_view = s_a.reshape(shape)
        self.xp.add(s_plus_plus.reshape(shape), s_a_view[:, :, :, :1], out=out_view) # s_A with the low bit set to 0
        out_view += s_a_view[:, :1] # s_A with the high bit set to 0
        out_view += s_q.reshape(shape)[:, :1, :, :1] # s_Q with both bits set to 0
        return out

    def _s_vector_at(
            self, combined_s_vectors: Tuple, bits: Tuple, consumer_patterns: np.ndarray, first_values: np.ndarray,
            indices: np.ndarray
    ) -> np.ndarray:
        """
        The s vector of a feature (pair) only at the consumer patterns, as a new array. first_values is s+ (s++) at the
        consumer patterns, shared by all the features. indices is a reused work array.
        Indexing with intp indices is used and not np.take(out=...): it is faster even though it allocates the result.
        """
        all_bits = np.intp(len(combined_s_vectors[0]) - 1) # the consumer patterns are intp (see _consumer_values_given_f)
        if len(bits) == 1:
            s_plus, s_r = combined_s_vectors
            self.xp.bitwise_and(consumer_patterns, all_bits ^ bits[0], out=indices) # p^-i
            values = s_r[indices]
        else:
            s_plus_plus, s_a, s_q = combined_s_vectors
            low_bit, high_bit = bits
            self.xp.bitwise_and(consumer_patterns, all_bits ^ low_bit, out=indices) # p^-i
            values = s_a[indices]
            self.xp.bitwise_and(consumer_patterns, all_bits ^ high_bit, out=indices) # p^-j
            values += s_a[indices]
            self.xp.bitwise_and(consumer_patterns, all_bits ^ low_bit ^ high_bit, out=indices) # p^-ij
            values += s_q[indices]
        values += first_values
        return values

    def _add_consumer_values(
            self, features_in_path: List, consumer_patterns: np.ndarray, f: np.ndarray, w: float, values: Dict
    ):
        depth = len(features_in_path)
        combined_s_vectors = self._combine_shared_s_vectors(self._get_shared_s_vectors(depth, f, w))
        start_time = time.time()

        if len(consumer_patterns) > self.full_s_vector_consumers_ratio * 2 ** depth:
            # Many consumers: build every s vector over all the 2^D patterns (reusing one array) and index it once
            s_vector = self.xp.empty(2 ** depth, dtype=combined_s_vectors[0].dtype)
            for key, bits in get_value_keys_and_decision_pattern_bits(features_in_path, self.metric.INTERACTION_VALUE):
                self._build_s_vector(combined_s_vectors, bits, s_vector)
                feature_values = s_vector[consumer_patterns] # a new array (not the reused s_vector), so it can be stored as is
                if key in values:
                    values[key] += feature_values
                else:
                    values[key] = feature_values
        else:
            # Few consumers: compute every s vector only at the consumer patterns
            first_values = combined_s_vectors[0][consumer_patterns]
            indices = self.xp.empty_like(consumer_patterns)
            for key, bits in get_value_keys_and_decision_pattern_bits(features_in_path, self.metric.INTERACTION_VALUE):
                feature_values = self._s_vector_at(combined_s_vectors, bits, consumer_patterns, first_values, indices) # a new array, so it can be stored as is
                if key in values:
                    values[key] += feature_values
                else:
                    values[key] = feature_values

        self.consumer_values_time += time.time() - start_time

    def _consumer_values_given_f(
            self, features_in_path: List, consumer_patterns: np.ndarray, f: np.ndarray, w: float,
            w_neighbor: Optional[float], values: Optional[Dict]
    ) -> Optional[Dict]:
        return_values = values is None
        if return_values:
            values = {}
        consumer_patterns = consumer_patterns.astype(np.intp, copy=False) # indexing converts other index dtypes to intp on every call, do it once per leaf
        self._add_consumer_values(features_in_path, consumer_patterns, f, w, values)
        if w_neighbor is not None:
            # The right neighbor leaf: its f is the neighbor vector of f and its decision patterns flip the last bit
            self._add_consumer_values(
                features_in_path, consumer_patterns ^ 1, self.neighbor_vector(f, self.GPU), w_neighbor, values
            )
        return values if return_values else None

    def get_background_consumer_values(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: float = None, values: Optional[Dict] = None
    ) -> Optional[Dict]:
        if not features_in_path:
            return {} if values is None else None
        f = compute_f_from_patterns(background_patterns, len(features_in_path), self.GPU)
        return self._consumer_values_given_f(features_in_path, consumer_patterns, f, w, w_neighbor, values)

    def get_path_dependent_consumer_values(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: float = None, values: Optional[Dict] = None
    ) -> Optional[Dict]:
        if not features_in_path:
            return {} if values is None else None
        f = self.compute_f_from_covers(covers)
        if self.GPU:
            f = cp.asarray(f)
        return self._consumer_values_given_f(features_in_path, consumer_patterns, f, w, w_neighbor, values)

    def _get_s_vectors_given_f(self, features_in_path: List, f: np.ndarray, w: float) -> Dict:
        """
        The s vector of every feature (pair) over all the 2^D decision patterns, the output format of
        HighDepthWoodelfV1PathToSVectors. Used by get_background_s_matrix and get_path_dependent_s_matrix.
        Takes O(2^D * D) memory, the get_*_consumer_values functions avoid it.
        """
        depth = len(features_in_path)
        combined_s_vectors = self._combine_shared_s_vectors(self._get_shared_s_vectors(depth, f, w))
        return {
            key: self._build_s_vector(combined_s_vectors, bits, self.xp.empty(2 ** depth, dtype=combined_s_vectors[0].dtype))
            for key, bits in get_value_keys_and_decision_pattern_bits(features_in_path, self.metric.INTERACTION_VALUE)
        }

    def present_statistics(self):
        mean_f_size = self.total_f_sizes / self.s_computation_calls if self.s_computation_calls > 0 else 0
        print(
            f"M time: {round(self.matrices_init_time, 2)} sec, "
            f"s time: {round(self.s_computation_time, 2)} sec ({self.s_computation_calls} shared s vectors computations, "
            f"f prepare time: {self.f_prepare_time}, f mean non zero size: {mean_f_size}), "
            f"consumer values time: {round(self.consumer_values_time, 2)} sec"
        )
