from __future__ import annotations

import time
from bisect import bisect_left
from itertools import combinations
from typing import List, Dict, Optional, Tuple, Callable, Iterator

import numpy as np

from woodelf.core.cube_metric import CubeMetric, ShapleyValues, BanzhafValues, ShapleyInteractionValues, \
    BanzhafInteractionValues, CardinalityInteractionIndicesMetric, nCk
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
        s_i[c] = s+[c] + s_R[c^-i]                                  where s_R = s- - s+
        s_ij[c] = s++[c] + s_A[c^-i] + s_A[c^-j] + s_Q[c^-ij]       where s_A = s+- - s++, s_Q = s++ - 2s+- + s--
    and c^-i is the decision pattern c with the bit of feature i set to 0.
    Only the secondary diagonals of the shared matrices are stored: O(2^D) memory instead of
    O(2^D * D) for first-order values and O(2^D * D^) for interaction values.

    Supports ShapleyValues, BanzhafValues, ShapleyInteractionValues and BanzhafInteractionValues (CLOSED_FORM_VALUES).
    HighDepthWoodelfV2CardinalityInteractionIndicesPathToSVectors extends it to any order. create_high_depth_woodelf_p2s
    falls back to HighDepthWoodelfV1PathToSVectors for the rest.
    """
    COMPUTES_CONSUMER_VALUES = True
    # With more consumers than FULL_S_VECTOR_CONSUMERS_RATIO * 2^D, build each s vector over all the 2^D patterns and
    # index it by the consumer patterns once. Otherwise index s+ and s_R (s++, s_A and s_Q) by the consumer patterns directly.
    # The ratios are the benchmarked crossovers: a pair takes 3 lookups per consumer in the direct way, a feature only 1
    FULL_S_VECTOR_CONSUMERS_RATIO = 1.0
    FULL_PAIR_S_VECTOR_CONSUMERS_RATIO = 0.5
    # The right neighbor leaf is folded into the s vectors (see _fold_neighbor), so the consumer patterns are indexed once
    # for both leaves. Always with many consumers. With few consumers only when the lookups this saves, num_consumers
    # per term of every key (1 for a feature, 3 for a pair), are more than FOLD_NEIGHBOR_LOOKUPS_RATIO * 2^D: folding
    # takes a few vector additions over all the 2^D patterns. The ratio is the benchmarked crossover
    FOLD_NEIGHBOR_LOOKUPS_RATIO = 1.0

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        if not self.supports_metric(metric):
            raise ValueError(
                f"{type(self).__name__} does not support {type(metric).__name__} (see supports_metric). "
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

        s_matrix *= w

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

    @staticmethod
    def _fold_neighbor(combined_s_vectors: Tuple, neighbor_combined_s_vectors: Tuple) -> Tuple[Tuple, Tuple]:
        """
        Folds the right neighbor leaf into the combined s vectors of the leaf, so the consumer patterns are indexed once for
        both leaves, not once per leaf. Where the leaf reads a term of its formula at c^-T (T is the feature, the pair, one
        of its two features, or the empty set for s+ and s++), the neighbor, whose decision patterns flip the last bit,
        reads it at
            (c^1)^-T = (c^-T)^1        if T does not hold the last feature of the path
            (c^1)^-T = c^-T            if it does (its bit is set to 0 anyway)
        So the terms whose T does not hold the last feature read s + swap(s^N), where swap(s^N) is the neighbor's vector with
        the two elements of every pair (2k, 2k+1) swapped, and the terms whose T holds it read s + s^N.
        Returns the combined s vectors, updated in place to s + swap(s^N), and the last bit s vectors: s + s^N of the
        vectors that the terms clearing the last bit read, (s_R,) or (s_A, s_Q). s+ and s++ are never read there.
        """
        # Computed first, the in-place updates below change the combined s vectors
        last_bit_s_vectors = tuple(
            s_vector + neighbor_s_vector
            for s_vector, neighbor_s_vector in zip(combined_s_vectors[1:], neighbor_combined_s_vectors[1:])
        )
        for s_vector, neighbor_s_vector in zip(combined_s_vectors, neighbor_combined_s_vectors):
            s_vector_pairs = s_vector.reshape(-1, 2) # view only
            s_vector_pairs += neighbor_s_vector.reshape(-1, 2)[:, ::-1]
        return combined_s_vectors, last_bit_s_vectors

    def _build_s_vector(
            self, combined_s_vectors: Tuple, bits: Tuple, out: np.ndarray, last_bit_s_vectors: Optional[Tuple] = None
    ) -> np.ndarray:
        """
        The s vector of a feature (pair) over all the 2^D decision patterns, written to out:
        s_i[c] = s+[c] + s_R[c^-i]  or  s_ij[c] = s++[c] + s_A[c^-i] + s_A[c^-j] + s_Q[c^-ij]
        last_bit_s_vectors: None, or with the right neighbor leaf folded in (see _fold_neighbor) the vectors that the terms
        clearing the last bit read. bits[0] is the lowest bit, so it is the only one that can be the last bit.
        """
        reads_last_bit_s_vectors = last_bit_s_vectors is not None and bits[0] == 1
        if len(bits) == 1:
            s_plus, s_r = combined_s_vectors
            if reads_last_bit_s_vectors:
                s_r, = last_bit_s_vectors # c^-i clears the last bit
            shape = (-1, 2, bits[0]) # split each index into (higher bits, the bit of i, lower bits)
            # s_R[:, :1] is s_R with the bit of i set to 0, broadcast to both values of the bit
            self.xp.add(s_plus.reshape(shape), s_r.reshape(shape)[:, :1], out=out.reshape(shape))
            return out
        s_plus_plus, s_a, s_q = combined_s_vectors
        low_bit_s_a = s_a
        if reads_last_bit_s_vectors:
            low_bit_s_a, s_q = last_bit_s_vectors # c^-i and c^-ij clear the last bit, c^-j does not
        low_bit, high_bit = bits
        shape = (-1, 2, high_bit // (2 * low_bit), 2, low_bit) # split each index into (higher bits, high bit, middle bits, low bit, lower bits)
        out_view = out.reshape(shape)
        s_a_view = s_a.reshape(shape)
        self.xp.add(s_plus_plus.reshape(shape), low_bit_s_a.reshape(shape)[:, :, :, :1], out=out_view) # s_A with the low bit set to 0
        out_view += s_a_view[:, :1] # s_A with the high bit set to 0
        out_view += s_q.reshape(shape)[:, :1, :, :1] # s_Q with both bits set to 0
        return out

    def _s_vector_at(
            self, combined_s_vectors: Tuple, last_bit_s_vectors: Optional[Tuple], bits: Tuple,
            consumer_patterns: np.ndarray, first_values: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """
        The s vector of a feature (pair) only at the consumer patterns, as a new array. first_values is s+ (s++) at the
        consumer patterns, shared by all the features. indices is a reused work array.
        last_bit_s_vectors: None, or with the right neighbor leaf folded in (see _fold_neighbor) the vectors that the terms
        clearing the last bit read. bits[0] is the lowest bit, so it is the only one that can be the last bit.
        Indexing with intp indices is used and not np.take(out=...): it is faster even though it allocates the result.
        """
        all_bits = np.intp(len(combined_s_vectors[0]) - 1) # the consumer patterns are intp (see _consumer_values_given_f)
        reads_last_bit_s_vectors = last_bit_s_vectors is not None and bits[0] == 1
        if len(bits) == 1:
            s_plus, s_r = combined_s_vectors
            if reads_last_bit_s_vectors:
                s_r, = last_bit_s_vectors # c^-i clears the last bit
            self.xp.bitwise_and(consumer_patterns, all_bits ^ bits[0], out=indices) # c^-i
            values = s_r[indices]
        else:
            s_plus_plus, s_a, s_q = combined_s_vectors
            low_bit_s_a = s_a
            if reads_last_bit_s_vectors:
                low_bit_s_a, s_q = last_bit_s_vectors # c^-i and c^-ij clear the last bit, c^-j does not
            low_bit, high_bit = bits
            self.xp.bitwise_and(consumer_patterns, all_bits ^ low_bit, out=indices) # c^-i
            values = low_bit_s_a[indices]
            self.xp.bitwise_and(consumer_patterns, all_bits ^ high_bit, out=indices) # c^-j
            values += s_a[indices]
            self.xp.bitwise_and(consumer_patterns, all_bits ^ low_bit ^ high_bit, out=indices) # c^-ij
            values += s_q[indices]
        values += first_values
        return values

    def _uses_full_s_vectors(self, num_consumers: int, depth: int) -> bool:
        return num_consumers > self.full_s_vector_consumers_ratio * 2 ** depth

    def _folds_neighbor(self, num_consumers: int, depth: int, num_keys: int) -> bool:
        """Whether the right neighbor leaf is folded into the s vectors (see FOLD_NEIGHBOR_LOOKUPS_RATIO)."""
        if self._uses_full_s_vectors(num_consumers, depth):
            return True
        lookups_per_key = 3 if self.metric.INTERACTION_VALUE else 1
        return num_consumers * lookups_per_key * num_keys > self.FOLD_NEIGHBOR_LOOKUPS_RATIO * 2 ** depth

    def _add_consumer_values(
            self, features_in_path: List, consumer_patterns: np.ndarray, f: np.ndarray, w: float, values: Dict,
            w_neighbor: Optional[float] = None
    ):
        depth = len(features_in_path)
        keys_and_bits = get_value_keys_and_decision_pattern_bits(features_in_path, self.metric.INTERACTION_VALUE)
        if w_neighbor is not None and not self._folds_neighbor(len(consumer_patterns), depth, len(keys_and_bits)):
            # Few consumers: add the leaf and the right neighbor leaf one after the other. The neighbor's f is the
            # neighbor vector of f and its decision patterns flip the last bit
            self._add_consumer_values(features_in_path, consumer_patterns, f, w, values)
            neighbor_consumer_patterns = consumer_patterns ^ 1
            self._add_consumer_values(
                features_in_path, neighbor_consumer_patterns, self.neighbor_vector(f, self.GPU), w_neighbor, values
            )
            return

        combined_s_vectors = self._combine_shared_s_vectors(self._get_shared_s_vectors(depth, f, w))
        neighbor_combined_s_vectors = None
        if w_neighbor is not None:
            # The right neighbor leaf (its f is the neighbor vector of f) is folded into the s vectors below, so its
            # consumers are not indexed again
            neighbor_combined_s_vectors = self._combine_shared_s_vectors(
                self._get_shared_s_vectors(depth, self.neighbor_vector(f, self.GPU), w_neighbor)
            )
        start_time = time.time()
        last_bit_s_vectors = None
        if neighbor_combined_s_vectors is not None:
            combined_s_vectors, last_bit_s_vectors = self._fold_neighbor(combined_s_vectors, neighbor_combined_s_vectors)

        if self._uses_full_s_vectors(len(consumer_patterns), depth):
            # Many consumers: build every s vector over all the 2^D patterns (reusing one array) and index it once
            s_vector = self.xp.empty(2 ** depth, dtype=combined_s_vectors[0].dtype)
            for key, bits in keys_and_bits:
                self._build_s_vector(combined_s_vectors, bits, s_vector, last_bit_s_vectors)
                feature_values = s_vector[consumer_patterns] # a new array (not the reused s_vector), so it can be stored as is
                if key in values:
                    values[key] += feature_values
                else:
                    values[key] = feature_values
        else:
            # Few consumers: compute every s vector only at the consumer patterns
            first_values = combined_s_vectors[0][consumer_patterns]
            indices = self.xp.empty_like(consumer_patterns)
            for key, bits in keys_and_bits:
                feature_values = self._s_vector_at(
                    combined_s_vectors, last_bit_s_vectors, bits, consumer_patterns, first_values, indices
                ) # a new array, so it can be stored as is
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
        self._add_consumer_values(features_in_path, consumer_patterns, f, w, values, w_neighbor)
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


def _cardinality_diagonals(metric: CardinalityInteractionIndicesMetric, depth: int, order: int) -> np.ndarray:
    """
    An (order + 1, 2^depth) array: the secondary diagonals of the matrices M_0, ..., M_order the order-subsets share.
    With h_u the cardinality_value of a subset with u negated members (0 where the cube can't hold such a subset),
    M_t is the t-th forward difference of h:
        M_t = sum_{u=0}^{t} (-1)^(t-u) C(t, u) h_u
    so that, for every subset S of size order:
        M_S = sum_{T subset of S} M_|T| restricted to the cubes in which all the members of T are negated
    """
    h = np.zeros((order + 1, depth + 1)) # h[u, p]
    for p in range(depth + 1):
        q = depth - p
        for u in range(max(0, order - p), min(order, q) + 1):
            h[u, p] = metric.cardinality_value(q, p, order - u, u)
    c = np.zeros_like(h) # c[t, p]
    for t in range(order + 1):
        for u in range(t + 1):
            c[t] += (-1) ** (t - u) * nCk(t, u) * h[u]
    return c[:, secondary_diagonal_popcounts(depth)]


class PrefixesZ:
    """
    The Z vectors of prefixes of the subsets, the table of the dynamic programming of
    HighDepthWoodelfV2CardinalityInteractionIndicesPathToSVectors:
        Z[r, i] = Z_A^(r) over all the patterns, for the prefix A = As[i]
    A prefix A is a tuple of path indices, increasing, and all the prefixes have the same size |A|. The patterns are the
    2^depth decision patterns, or with the right neighbor leaf folded in (neighbor_folded) 2^(depth + 1) patterns: the Z
    vectors of the rows s~ and then of the rows s^ (see _fold_neighbor_into_rows). The prefixes are sorted by their last
    member, so the ones that a member extends (those that end before it) are the leading ones.
    """

    def __init__(self, As: List[Tuple], Z: np.ndarray, depth: int, xp, neighbor_folded: bool = False):
        self.As, self.Z, self.depth, self.xp, self.neighbor_folded = As, Z, depth, xp, neighbor_folded
        self.last_members = [A[-1] if A else -1 for A in As]
        self.A_index = None # A -> its position in As, filled by the first get

    @classmethod
    def of_prefix(cls, A: Tuple, Z_A: np.ndarray, depth: int, xp, neighbor_folded: bool = False) -> "PrefixesZ":
        """A single prefix A, with Z_A the (num_r, patterns) array of its Z vectors."""
        return cls([A], Z_A[:, None], depth, xp, neighbor_folded) # (r, prefixes, patterns), a single prefix

    @property
    def size(self) -> int:
        """|A|, the size of the prefixes."""
        return len(self.As[0])

    @property
    def num_r(self) -> int:
        """How many Z vectors every prefix has: r = 0, ..., num_r - 1."""
        return self.Z.shape[0]

    def get(self, A: Tuple, r: int) -> np.ndarray:
        """Z_A^(r) over all the patterns."""
        if self.A_index is None:
            self.A_index = {prefix: i for i, prefix in enumerate(self.As)}
        return self.Z[r, self.A_index[A]]

    def next_members(self, order: int) -> range:
        """The members that extend the prefixes and leave room for the rest of a subset of size order after them."""
        return range(self.last_members[0] + 1, self.depth - (order - self.size - 1))

    def count_extended_by(self, member: int) -> int:
        """How many prefixes member extends: the leading ones, that end before it."""
        return bisect_left(self.last_members, member)

    def extend(self, member: int, out: Optional[np.ndarray] = None) -> "PrefixesZ":
        """
        One step of the dynamic programming: the prefixes A that member extends, with member added, and their
            Z_{A + {member}}^(r)[c] = Z_A^(r)[c] + Z_A^(r + 1)[c^-member]        r = 0, ..., num_r - 2
        written to out, an (num_r - 1, count, patterns) array, when given.
        """
        count = self.count_extended_by(member)
        num_r, _, num_patterns = self.Z.shape
        bit = 1 << (self.depth - 1 - member) # the first feature in the path is the most significant bit
        # split each pattern into (higher bits, the bit of member, lower bits). Views only, the reshape of out in
        # particular must not copy, or the result would not reach out
        split = (num_patterns // (2 * bit), 2, bit)
        source = self.Z[:, :count].reshape((num_r, count) + split)
        if out is None:
            out = self.xp.empty((num_r - 1, count, num_patterns), dtype=self.Z.dtype)
        # source[..., :1, :] is Z_A^(r + 1) with the bit of member set to 0, broadcast to both values of the bit
        self.xp.add(source[:-1], source[1:, :, :, :1], out=out.reshape((num_r - 1, count) + split))
        return PrefixesZ([A + (member,) for A in self.As[:count]], out, self.depth, self.xp, self.neighbor_folded)

    def extend_to_subsets(self, member: int) -> "PrefixesZ":
        """
        extend(member) for the last level of the dynamic programming, whose prefixes are the subsets themselves (from
        num_r = 2, so only Z^(0), the s vector of each subset, is computed). With the right neighbor leaf folded in, it
        is computed over the 2^depth decision patterns only: a subset S = A + {member} reads the rows s~ in the terms of
        the T that do not hold the last member of the path, and the rows s^ in the others (see _fold_neighbor_into_rows),
        so with Z~ and Z^ the Z vectors of the rows s~ and s^ (the two halves of the patterns):
            Z_S^(0)[c] = Z~_A^(0)[c] + Z~_A^(1)[c^-member]      when member is not the last member of the path
            Z_S^(0)[c] = Z~_A^(0)[c] + Z^_A^(1)[c^-member]      when it is (A does not hold it)
        """
        if not self.neighbor_folded:
            return self.extend(member)
        num_patterns = 2 ** self.depth
        if member < self.depth - 1:
            # Only the rows s~: the first half of the patterns (a view)
            return PrefixesZ(self.As, self.Z[:, :, :num_patterns], self.depth, self.xp).extend(member)
        count = self.count_extended_by(member)
        out = self.xp.empty((1, count, num_patterns), dtype=self.Z.dtype)
        # The last member is bit 0: split each pattern into (higher bits, bit 0). Views only, see extend
        split = (count, num_patterns // 2, 2)
        # the [..., :1] is Z^_A^(1) with bit 0 set to 0, broadcast to both values of the bit
        self.xp.add(
            self.Z[0, :count, :num_patterns].reshape(split), self.Z[1, :count, num_patterns:].reshape(split)[:, :, :1],
            out=out[0].reshape(split)
        )
        return PrefixesZ([A + (member,) for A in self.As[:count]], out, self.depth, self.xp)

    def next_level(self, order: int) -> "PrefixesZ":
        """
        The next level of the dynamic programming: every prefix A + {m}, with A a prefix of this level and m a later
        member that leaves room for the rest of a subset of size order, with its Z vectors. One extend per member m
        computes all the prefixes it extends at once, so a level takes O(D) calls, not one per prefix. The new level is
        sorted by last member as well.
        """
        # Extend the prefixes of this size by one member, leaving room for the order - size - 1 members after it
        members = self.next_members(order)
        counts = [self.count_extended_by(member) for member in members] # the prefixes that end before member
        num_r, _, num_patterns = self.Z.shape
        Z = self.xp.empty((num_r - 1, sum(counts), num_patterns), dtype=self.Z.dtype)
        As, start = [], 0
        for member, count in zip(members, counts):
            As += self.extend(member, out=Z[:, start:start + count]).As
            start += count
        return PrefixesZ(As, Z, self.depth, self.xp, self.neighbor_folded)

    def memory_usage(self, order: int) -> int:
        """
        The memory the dynamic programming from this single prefix A would need: the number of values in its largest
        level, among the levels of the prefixes of sizes len(A) + 1, ..., order - 1 that extend A (the last level, the
        subsets themselves, is yielded in batches and never held whole). A level holds every r and all the patterns
        of each of its prefixes. 0 when there is no such level.
        """
        A = self.As[0]
        first_member = self.last_members[0] + 1
        # Level j (the prefixes of size j) holds order - j + 1 values of r of the prefixes that extend A by j - len(A)
        # members from first_member on and leave room for the order - j members after them
        largest_level = max([
            (order - j + 1) * nCk(self.depth - first_member - (order - j), j - len(A))
            for j in range(len(A) + 1, order)
        ], default=0)
        num_patterns = self.Z.shape[-1]
        return largest_level * num_patterns

    def s_vectors(self) -> np.ndarray:
        """
        When the prefixes are whole subsets (a single Z vector each, their Z^(0), see extend_to_subsets): their s vectors
        over all the 2^depth patterns, a (len(As), 2^depth) array. With the neighbor leaf, its s vectors are folded in.
        """
        return self.Z[0]


class HighDepthWoodelfV2CardinalityInteractionIndicesPathToSVectors(HighDepthWoodelfV2PathToSVectors):
    """
    HighDepthWoodelfV2PathToSVectors for a CardinalityInteractionIndicesMetric of any orders: GeneralShapleyInteractionValues,
    GeneralBanzhafInteractionValues, MobiusCoefficients (PDIV of any order), the faithful indices, ...
    The symmetry of v2 holds for any order. The value of a subset of size k depends only on how many of its members are
    positive literals, so all the k-subsets share k + 1 matrices M_0, ..., M_k (see _cardinality_diagonals) and with
    s_t = M_t * f:
        s_S[c] = sum over T subset of S of s_|T|[c^-T]               (2^k terms, inclusion-exclusion)
    where c^-T is c with the bits of the members of T set to 0. For k = 1 and k = 2 these are the formulas of
    HighDepthWoodelfV2PathToSVectors: s_0, s_1 are s+, s_R and s_0, s_1, s_2 are s++, s_A, s_Q. A metric of several
    orders gets the (k + 1) shared s vectors of all its orders k from a single StrassenLike multiplication, instead of
    one s vector per subset as in HighDepthWoodelfV1PathToSVectors.

    Dynamic programming shares the work of the 2^k terms along the prefixes of the subsets (the members taken in their
    order in the path):
        Z_A^(r)[c] = sum over T subset of A of s_{|T| + r}[c^-T]
    so Z_{}^(r) = s_r, s_S = Z_S^(0), and adding the member m to the prefix A takes one broadcast addition per r:
        Z_{A + {m}}^(r)[c] = Z_A^(r)[c] + Z_A^(r + 1)[c^-m]         (c^-m is short for c^-{m})
    should_use_inclusion_exclusion picks one of two ways to get the values of the subsets at the consumer patterns:
        1. Inclusion-exclusion: the 2^k terms of s_0, ..., s_k, read at the consumer patterns only (see
           _inclusion_exclusion_s_vectors_at).
        2. Dynamic programming: the s vectors over all the 2^D patterns (see _dynamic_programming_s_vectors, its table
           of Z vectors is PrefixesZ), about one vector addition per subset where v1 takes D of them, and then one
           lookup per consumer.
    Both fold the right neighbor leaf into the shared s vectors (see _fold_neighbor_into_rows), so the consumer patterns
    are read once for both leaves.
    """
    # The dynamic programming works in batches of up to SUBSETS_BATCH_ELEMENTS values
    SUBSETS_BATCH_ELEMENTS = 2 ** 22
    # Inclusion-exclusion reads the subsets at the consumer patterns in batches of about INCLUSION_EXCLUSION_BATCH_ELEMENTS
    # values, small enough for its work arrays to stay in the cache, and of at least INCLUSION_EXCLUSION_MIN_BATCH_SUBSETS
    # subsets. Benchmarked
    INCLUSION_EXCLUSION_BATCH_ELEMENTS = 2 ** 16
    INCLUSION_EXCLUSION_MIN_BATCH_SUBSETS = 4
    # c(depth) of should_use_inclusion_exclusion: how many lookups at the consumer patterns cost about as much as one
    # vector addition over the 2^D patterns. Benchmarked as the crossover between inclusion-exclusion and dynamic
    # programming for orders 2 to 4: about 1.5 on paths deeper than SHORT_PATHS_MAX_DEPTH, where the lookups into the
    # long vectors miss the cache, and about 3 on shorter ones
    FULL_S_VECTORS_CROSSOVER = 1.5
    SHORT_PATHS_FULL_S_VECTORS_CROSSOVER = 3.0
    SHORT_PATHS_MAX_DEPTH = 10

    def __init__(self, metric: CardinalityInteractionIndicesMetric, max_depth: int, GPU: bool = False):
        # depth -> [(order, the slice of the rows of the order's shared matrices M_0, ..., M_order in diagonals[depth])],
        # one item per order the metric reports on paths of that depth. Filled by _build_diagonals, which
        # super().__init__ calls, so it must exist before it
        self.order_rows = {}
        super().__init__(metric, max_depth, GPU)
        self.subset_masks = {} # (depth, order, neighbor_folded) -> _subset_masks, filled on first use

    @staticmethod
    def supports_metric(metric: CubeMetric) -> bool:
        """
        True for a CardinalityInteractionIndicesMetric that keeps the generic calc_metric, so its values are those of
        cardinality_value (functions of the cardinalities only), reported under the sorted tuple of the subset's features.
        """
        return (
            isinstance(metric, CardinalityInteractionIndicesMetric)
            and type(metric).calc_metric is CardinalityInteractionIndicesMetric.calc_metric
            and not metric.INTERACTION_VALUES_ORDER_MATTERS
        )

    @staticmethod
    def metric_diagonals(metric: CardinalityInteractionIndicesMetric, depth: int) -> Dict[int, np.ndarray]:
        """
        For every order k the metric reports on paths with depth unique features: the secondary diagonals of the k + 1
        shared matrices M_0, ..., M_k of the k-subsets (see _cardinality_diagonals).
        """
        return {order: _cardinality_diagonals(metric, depth, order) for order in metric.orders(depth)}

    def _build_diagonals(self):
        """
        diagonals[depth] holds the rows of every order k the metric reports on paths of this depth, one after the other
        (the diagonals of M_0, ..., M_k), and order_rows[depth] the slice of each order.
        float64 and not float32 as in v1: the 2^k terms of the inclusion-exclusion can add up to 3^k times the largest
        value of the metric in magnitude, so float32 would lose about log2(3^k) bits of the result.
        """
        for depth in range(0, self.max_depth + 1):
            self.order_rows[depth] = []
            diagonals = [np.zeros((0, 2 ** depth))]
            start = 0
            for order, order_diagonals in self.metric_diagonals(self.metric, depth).items():
                self.order_rows[depth].append((order, slice(start, start + order + 1)))
                diagonals.append(order_diagonals)
                start += order + 1
            self.diagonals[depth] = self.xp.array(np.concatenate(diagonals), dtype=self.xp.float64)

    def should_use_inclusion_exclusion(self, depth: int, order: int, num_consumers: int) -> bool:
        """
        True to read the 2^order terms of every subset at the consumer patterns only, by inclusion-exclusion (see
        _inclusion_exclusion_s_vectors_at). False to build the s vectors of the subsets over all the 2^depth patterns by
        dynamic programming (see _dynamic_programming_s_vectors) and index them by the consumer patterns. True when it
        costs no more:
            inclusion-exclusion:  C(depth, order) * 2^order * num_consumers / c(depth)
            dynamic programming:  build + C(depth, order) * num_consumers / c(depth)
        where build = 2^depth * sum over i = 1, ..., order of (order - i + 1) * N_i counts the vector additions of the
        dynamic programming (N_i prefixes of size i), and c(depth) is FULL_S_VECTORS_CROSSOVER.
        """
        crossover = self.SHORT_PATHS_FULL_S_VECTORS_CROSSOVER if depth <= self.SHORT_PATHS_MAX_DEPTH else self.FULL_S_VECTORS_CROSSOVER
        build = 0
        for size in range(1, order + 1):
            # the prefixes of this size that leave room for the order - size members after them
            build += (order - size + 1) * nCk(depth - (order - size), size) * 2 ** depth
        subsets_lookups = nCk(depth, order) * num_consumers / crossover
        return 2 ** order * subsets_lookups <= build + subsets_lookups

    def _subset_masks(self, depth: int, order: int, neighbor_folded: bool = False) -> Tuple[np.ndarray, List[int]]:
        """
        The inclusion-exclusion terms of the subsets of size order of a path with depth unique features, for
        _inclusion_exclusion_s_vectors_at. For the i-th subset S_i of combinations(range(depth), order) and every
        0 <= j < 2^order:
            c & masks[j, i] = c^-T_j        for every decision pattern c
            sizes[j] = |T_j|
        where T_j is the subset of S_i that the set bits of j pick (bit m picks the member of S_i with the m-th lowest
        bit). masks is a (2^order, number of subsets) array. Cached per (depth, order, neighbor_folded).
        With neighbor_folded, for the rows of 2^(depth + 1) values of _fold_neighbor_into_rows: masks[j, i] also has bit
        depth set when T_j holds the last member of the path, so that (c with bit depth set) & masks[j, i] is c^-T_j in
        the half of the rows that T_j reads.
        """
        if (depth, order, neighbor_folded) not in self.subset_masks:
            if neighbor_folded:
                masks, sizes = self._subset_masks(depth, order)
                # The last member is bit 0, which the masks of the T_j that hold it clear
                masks = masks | (((~masks) & 1) << depth)
            else:
                # members_bits[i, m] is the bit of the m-th member of the i-th subset, from the lowest bit (the first feature
                # in the path is the most significant bit)
                members_bits = 1 << (depth - 1 - np.array(list(combinations(range(depth), order)), dtype=np.intp)[:, ::-1])
                in_t = (np.arange(2 ** order)[:, None] >> np.arange(order)) & 1 # in_t[j, m]: is the m-th member in T_j
                # the bits of the members are distinct, so their sum is the bits of T_j
                masks = self.xp.asarray((2 ** depth - 1) ^ (in_t @ members_bits.T))
                sizes = in_t.sum(axis=1).tolist()
            self.subset_masks[(depth, order, neighbor_folded)] = (masks, sizes)
        return self.subset_masks[(depth, order, neighbor_folded)]

    def _inclusion_exclusion_s_vectors_at(
            self, s_vectors: Tuple, masks: np.ndarray, sizes: List[int], patterns: np.ndarray,
            folded_patterns: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        The s vectors of many subsets of the same order k at the given patterns, by inclusion-exclusion and all
        together: an (number of subsets, len(patterns)) array of
            s_S[c] = sum over T subset of S of s_|T|[c^-T]
        s_vectors are s_0, ..., s_k, and masks and sizes are those of the subsets (see _subset_masks). The 2^k terms are
        2^k indexing steps over all the subsets and patterns, not a loop over the subsets.
        With the right neighbor leaf folded in, s_vectors are the rows of _fold_neighbor_into_rows, masks are the
        neighbor_folded masks and folded_patterns is patterns with bit depth set.
        """
        masked_patterns = patterns if folded_patterns is None else folded_patterns
        indices = self.xp.empty((masks.shape[1], len(patterns)), dtype=np.intp)
        values = None
        for j in range(1, len(sizes)):
            self.xp.bitwise_and(masked_patterns, masks[j][:, None], out=indices) # c^-T_j of every subset at every pattern
            if values is None:
                values = s_vectors[sizes[j]][indices]
            else:
                values += s_vectors[sizes[j]][indices]
        values += s_vectors[0][patterns] # T is the empty set, the same term for all the subsets (in the first half of folded rows)
        return values

    def _dynamic_programming_s_vectors(self, start: PrefixesZ, order: int) -> Iterator[PrefixesZ]:
        """
        The s vectors s_S = Z_S^(0), over all the 2^D patterns, of the subsets S of size order that extend the single
        prefix of start by later members, by dynamic programming over the prefixes (see PrefixesZ). Yields them in
        batches of subsets (see PrefixesZ.s_vectors).
        Level i holds the prefixes of size i that leave room for the other order - i members (see PrefixesZ.next_level).
        The last level is yielded one last member at a time (see PrefixesZ.extend_to_subsets). If a level before it would
        exceed SUBSETS_BATCH_ELEMENTS values, the prefixes are first split by their next member.
        """
        if start.size == order:
            yield start
            return
        if start.memory_usage(order) > self.SUBSETS_BATCH_ELEMENTS:
            # memory_usage is 0 from size order - 1 on, so these extend to prefixes of size order - 1 at most
            for member in start.next_members(order):
                yield from self._dynamic_programming_s_vectors(start.extend(member), order)
            return

        level = start
        while level.size < order - 1:
            level = level.next_level(order)
        for member in level.next_members(order):
            yield level.extend_to_subsets(member)

    def _fold_neighbor_into_rows(self, s_vectors: np.ndarray, neighbor_s_vectors: np.ndarray) -> np.ndarray:
        """
        The rows s_0, ..., s_k of the leaf with the right neighbor leaf folded in, so the consumer patterns are read once
        for both leaves, not once per leaf: the (k + 1, 2^(depth + 1)) array of the rows
            s~_t = s_t + swap(s^N_t)  and then  s^_t = s_t + s^N_t
        where s^N_t are the rows of the neighbor and swap(s^N_t) is s^N_t with the two elements of every pair (2k, 2k+1)
        swapped. The neighbor's decision patterns flip the last bit, so where the leaf reads the term of T at c^-T, the
        neighbor reads it at (c^1)^-T = (c^-T)^1 if T does not hold the last member of the path, and at c^-T if it does
        (its bit is set to 0 anyway). Hence the values of both leaves are
            s_S[c] + s^N_S[c^1] = sum over T subset of S of (s~_|T|[c^-T] if T does not hold the last member, else s^_|T|[c^-T])
        and bit depth of a pattern of the rows picks the half: s~ when it is 0, s^ when it is 1.
        """
        num_rows, num_patterns = s_vectors.shape
        rows = self.xp.empty((num_rows, 2, num_patterns), dtype=s_vectors.dtype)
        self.xp.add(s_vectors, neighbor_s_vectors, out=rows[:, 1])
        # The pairs (2k, 2k+1) of the s~ half. Views only, rows is a new C-contiguous array
        self.xp.add(
            s_vectors.reshape(num_rows, -1, 2), neighbor_s_vectors.reshape(num_rows, -1, 2)[:, :, ::-1],
            out=rows.reshape(num_rows, 2, -1, 2)[:, 0]
        )
        return rows.reshape(num_rows, 2 * num_patterns)

    def _add_consumer_values(
            self, features_in_path: List, consumer_patterns: np.ndarray, f: np.ndarray, w: float, values: Dict,
            w_neighbor: Optional[float] = None
    ):
        """
        A single StrassenLike multiplication gives the shared s vectors s_0, ..., s_k of every order k of the metric.
        Then for every order, as should_use_inclusion_exclusion picks: inclusion-exclusion, the 2^k terms of every subset
        at the consumer patterns only (see _inclusion_exclusion_s_vectors_at), or dynamic programming, the s vectors of
        the subsets over all the 2^D patterns (see _dynamic_programming_s_vectors) indexed by the consumer patterns.
        The right neighbor leaf, if any, is folded into the shared s vectors (see _fold_neighbor_into_rows).
        """
        depth = len(features_in_path)
        if not self.order_rows[depth]:
            return # the path has fewer features than the lowest order, so it has no subsets

        shared_s_vectors = self._get_shared_s_vectors(depth, f, w)
        neighbor_shared_s_vectors, folded_patterns = None, None
        if w_neighbor is not None:
            # The right neighbor leaf: its f is the neighbor vector of f and its decision patterns flip the last bit
            neighbor_shared_s_vectors = self._get_shared_s_vectors(depth, self.neighbor_vector(f, self.GPU), w_neighbor)
            folded_patterns = consumer_patterns | (1 << depth) # for the neighbor_folded masks of inclusion-exclusion
        inclusion_exclusion_batch_size = max(
            self.INCLUSION_EXCLUSION_MIN_BATCH_SUBSETS,
            self.INCLUSION_EXCLUSION_BATCH_ELEMENTS // max(len(consumer_patterns), 1)
        )
        start_time = time.time()

        for order, rows in self.order_rows[depth]:
            z = self.xp.stack(shared_s_vectors[rows]) # (order + 1, 2^D)
            if neighbor_shared_s_vectors is not None:
                z = self._fold_neighbor_into_rows(z, self.xp.stack(neighbor_shared_s_vectors[rows])) # (order + 1, 2^(D + 1))
            if self.should_use_inclusion_exclusion(depth, order, len(consumer_patterns)):
                masks, sizes = self._subset_masks(depth, order, neighbor_folded=w_neighbor is not None)
                subsets = list(combinations(range(depth), order)) # in the order of the columns of masks
                for start in range(0, len(subsets), inclusion_exclusion_batch_size):
                    batch_masks = masks[:, start:start + inclusion_exclusion_batch_size]
                    batch_values = self._inclusion_exclusion_s_vectors_at(
                        z, batch_masks, sizes, consumer_patterns, folded_patterns
                    )
                    # Each row of batch_values is a view of a new array of this batch only, so it can be stored as is
                    for members, feature_values in zip(subsets[start:start + inclusion_exclusion_batch_size], batch_values):
                        key = tuple(sorted(features_in_path[member] for member in members))
                        if key in values:
                            values[key] += feature_values
                        else:
                            values[key] = feature_values
                continue

            empty_prefix = PrefixesZ.of_prefix((), z, depth, self.xp, neighbor_folded=w_neighbor is not None)
            for subsets in self._dynamic_programming_s_vectors(empty_prefix, order):
                # The prefixes are the subsets, and their Z^(0) are their s vectors over all the 2^D patterns
                for members, s_vector in zip(subsets.As, subsets.s_vectors()):
                    key = tuple(sorted(features_in_path[member] for member in members))
                    # Indexing one s vector at a time: twice as fast as indexing the whole batch (s_vectors[:, consumer_patterns])
                    feature_values = s_vector[consumer_patterns]
                    if key in values:
                        values[key] += feature_values
                    else:
                        values[key] = feature_values

        self.consumer_values_time += time.time() - start_time

    def _get_s_vectors_given_f(self, features_in_path: List, f: np.ndarray, w: float) -> Dict:
        """
        The s vector of every subset over all the 2^D decision patterns, the output format of
        HighDepthWoodelfV1PathToSVectors. Used by get_background_s_matrix and get_path_dependent_s_matrix.
        """
        depth = len(features_in_path)
        if not self.order_rows[depth]:
            return {} # the path has fewer features than the lowest order, so it has no subsets
        shared_s_vectors = self._get_shared_s_vectors(depth, f, w)
        s_vectors = {}
        for order, rows in self.order_rows[depth]:
            z = self.xp.stack(shared_s_vectors[rows])
            for subsets in self._dynamic_programming_s_vectors(PrefixesZ.of_prefix((), z, depth, self.xp), order):
                for members, s_vector in zip(subsets.As, subsets.s_vectors()):
                    s_vectors[tuple(sorted(features_in_path[member] for member in members))] = s_vector
        return s_vectors
