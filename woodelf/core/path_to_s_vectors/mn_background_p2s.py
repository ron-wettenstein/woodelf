from typing import List, Dict, Tuple

import numpy as np

from woodelf.core.cube_metric import CubeMetric
from woodelf.core.path_to_s_vectors.base_p2s import PathToSVectors
from woodelf.core.utils import bits_matrix, safe_bitwise_count


class MNBackgroundPathToSVectors(PathToSVectors):
    """
    The class aims to compute the SVectors using the naive O(nm) complexity where n is the consumer data size and m is the background data size.
    This is the approach used in the shap library. We later use this class for cases of very high depth or very small background or consumer dataset.
    Note: This method assume the provided CubeMetric satisfy the symmetry and linearity axioms.

    Our approach has O(mn) complexity but still have 2 advantages over the shap package:
    1. The m and n are not really the size of the consumer and background datasets but their number of unique decision patterns - which is effectively much smaller.
    2. All operations are vectorized using numpy.
    """

    BATCH_SIZE = 1_000_000
    # True when _compute_s_batch_with_neighbor adds the right neighbor leaf in the same pass over the (consumer, background)
    # pairs. Otherwise get_background_s_matrix computes the neighbor leaf by a second call
    NEIGHBOR_IN_THE_SAME_PASS = False

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        self.pos_contributions, self.neg_contributions = MNBackgroundPathToSVectors._build_contribution_matrices(metric, max_depth)

    @staticmethod
    def _build_contribution_matrices(metric: CubeMetric, D: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute the per-literal contribution tables for a symmetric metric.

        pos_matrix[p][q] = contribution of each positive literal in a cube with p positive and q negative literals.
        neg_matrix[p][q] = contribution of each negative literal in the same cube.

        Uses integer feature proxies (s_plus = {0,..,p-1}, s_minus = {p,..,p+q-1}) so only counts matter.
        Entries where p + q > D are unreachable and remain 0.
        """
        pos_matrix = np.zeros((D + 1, D + 1), dtype=np.float64)
        neg_matrix = np.zeros((D + 1, D + 1), dtype=np.float64)

        for p in range(D + 1):
            for n in range(D + 1 - p):
                if p == 0 and n == 0:
                    continue
                s_plus = set(range(p))
                s_minus = set(range(p, p + n))
                contributions = metric.calc_metric(s_plus, s_minus)
                if p > 0:
                    pos_matrix[p][n] = contributions.get(0, 0.0)       # feature 0 is in s_plus
                if n > 0:
                    neg_matrix[p][n] = contributions.get(p, 0.0)       # feature p is in s_minus

        return pos_matrix, neg_matrix

    def _compute_s_batch(
        self, consumer_batch: np.ndarray, unique_b: np.ndarray,
        b_freqs: np.ndarray, D: int, w: float
    ) -> np.ndarray:
        """
        Core vectorized computation for a batch of consumer patterns against all unique background patterns.

        For each (consumer, background) pair: the bits where they differ tell us which features
        are positive literals (consumer=1, background=0) and which are negative (consumer=0, background=1).
        The counts of each determine the per-literal contribution via the precomputed tables.
        We then spread those contributions to the relevant feature positions and sum over all backgrounds.

        Returns shape (D, batch_size), one s-vector row per feature in the path.
        """
        diff     = consumer_batch[:, None] ^ unique_b[None, :]       # (batch, U_b)
        positive = consumer_batch[:, None] & diff                     # s_plus bits per pair
        negative = unique_b[None, :]       & diff                     # s_minus bits per pair

        # Only pairs where every bit is covered by at least one side going right.
        # Pairs with a "both-diverge" bit (pc_k=0, pb_k=0) have no WDNF rule and contribute 0.
        valid = (consumer_batch[:, None] | unique_b[None, :]) == ((1 << D) - 1)  # (batch, U_b)

        p = safe_bitwise_count(positive)                                # (batch, U_b)
        n = safe_bitwise_count(negative)                                # (batch, U_b)

        pos_weighted = self.pos_contributions[p, n] * b_freqs * valid   # (batch, U_b)
        neg_weighted = self.neg_contributions[p, n] * b_freqs * valid   # (batch, U_b)

        batch_size = len(consumer_batch)
        U_b = len(unique_b)
        pos_bits = bits_matrix(positive.ravel(), D).reshape(D, batch_size, U_b)   # (D, batch, U_b)
        neg_bits = bits_matrix(negative.ravel(), D).reshape(D, batch_size, U_b)   # (D, batch, U_b)

        return (pos_bits * pos_weighted + neg_bits * neg_weighted).sum(axis=2) * w  # (D, batch)

    def _compute_s_batch_with_neighbor(
        self, consumer_batch: np.ndarray, unique_b: np.ndarray,
        b_freqs: np.ndarray, D: int, w: float, w_neighbor: float
    ) -> np.ndarray:
        """_compute_s_batch of the leaf plus the right neighbor leaf. Implemented when NEIGHBOR_IN_THE_SAME_PASS is True."""
        raise NotImplementedError

    def get_background_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: float = None,
    ) -> Dict:
        """
        Compute the background s-matrix for a single leaf.

        Uniquifies background_patterns into (unique_b, b_freqs), then delegates to
        _compute_s_batch. Consumer patterns are processed in chunks of BATCH_SIZE // U_b
        to stay within memory bounds.

        If w_neighbor is provided, also adds the neighbor leaf's contribution: the neighbor
        leaf has patterns where the LSB is flipped (sibling in the tree), so we recurse with
        consumer_patterns ^ 1, background_patterns ^ 1, and w=w_neighbor.
        With NEIGHBOR_IN_THE_SAME_PASS, the neighbor leaf is added by _compute_s_batch_with_neighbor instead.

        Returns Dict[feature -> np.ndarray of shape (U_c,)], one entry per feature in the path.
        """
        if not features_in_path:
            return {}
        D = len(features_in_path)

        unique_b, b_counts = np.unique(background_patterns, return_counts=True)
        b_freqs = b_counts / b_counts.sum()

        U_c = len(consumer_patterns)
        U_b = len(unique_b)

        neighbor_in_the_same_pass = w_neighbor is not None and self.NEIGHBOR_IN_THE_SAME_PASS

        def compute_s_batch(consumer_batch: np.ndarray) -> np.ndarray:
            if neighbor_in_the_same_pass:
                return self._compute_s_batch_with_neighbor(consumer_batch, unique_b, b_freqs, D, w, w_neighbor)
            return self._compute_s_batch(consumer_batch, unique_b, b_freqs, D, w)

        batch_size = max(1, self.BATCH_SIZE // U_b)
        if U_c <= batch_size:
            s = compute_s_batch(consumer_patterns)
        else:
            s = np.empty((D, U_c), dtype=np.float64)
            for start in range(0, U_c, batch_size):
                end = min(start + batch_size, U_c)
                s[:, start:end] = compute_s_batch(consumer_patterns[start:end])

        result = {features_in_path[i]: s[i] for i in range(D)}

        if w_neighbor is not None and not neighbor_in_the_same_pass:
            neighbor = self.get_background_s_matrix(
                features_in_path, consumer_patterns ^ 1, background_patterns ^ 1, w_neighbor,
            )
            for f in result:
                result[f] = result[f] + neighbor[f]

        return result

    def get_path_dependent_s_matrix(self, features_in_path, consumer_patterns, covers, w, w_neighbor=None):
        raise NotImplementedError("MNBackgroundPathToSVectors does not support the path-dependent case.")


class MNBackgroundMatrixMultiplicationPathToSVectors(MNBackgroundPathToSVectors):
    """
    Faster variant of MNBackgroundPathToSVectors.
    Overrides _compute_s_batch to avoid the O(D * batch * U_b) 3D intermediate by
    decomposing the bit spreading into two (D, U_b) @ (U_b, batch) matrix multiplications,
    which carry the same asymptotic cost but benefit from BLAS optimization and better cache use.
    """

    BATCH_SIZE = 10_000_000
    NEIGHBOR_IN_THE_SAME_PASS = True

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        # The case keys of _compute_s_batch_with_neighbor read the flattened (max_depth + 1, max_depth + 1) contribution
        # tables, where the pairs with p positive and n negative literals are at p * key_stride + n
        self.key_stride = max_depth + 1
        # The tables by case key of the leaf and of the right neighbor leaf (see _contributions_by_case_key)
        self.pos_contributions_by_case_key, self.neighbor_pos_contributions_by_case_key = \
            self._contributions_by_case_key(self.pos_contributions)
        self.neg_contributions_by_case_key, self.neighbor_neg_contributions_by_case_key = \
            self._contributions_by_case_key(self.neg_contributions)

    def _contributions_by_case_key(self, contributions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        A contribution table by case key, for _compute_s_batch_with_neighbor, and the same table for the neighbor leaf.
        The case key of a pair is case * S + p' * key_stride + n', with S = key_stride^2, p' and n' the numbers of positive
        and negative literals in all the bits but bit 0, and case = 2 * (bit 0 of the consumer) + (bit 0 of the background):
            case 1, bits (0, 1): bit 0 is a negative literal, the cube has p' positive and n' + 1 negative literals
            case 2, bits (1, 0): bit 0 is a positive literal, p' + 1 and n'
            case 3, bits (1, 1): bit 0 is not a literal, p' and n'
            case 0, bits (0, 0): the pair is not valid, 0
        The key 4 * S, of the pairs that are not valid in the other bits, holds a 0 too.
        The neighbor's pair of (c, b) is (c^1, b^1): the same p' and n', and the case 3 - case.
        """
        S = self.key_stride ** 2
        by_case_key = np.zeros(4 * S + 1, dtype=np.float64)
        for case, extra_p, extra_n in [(1, 0, 1), (2, 1, 0), (3, 0, 0)]:
            # block[p', n'] = contributions[p' + extra_p, n' + extra_n], 0 beyond the table (unreachable counts)
            block = np.zeros((self.key_stride, self.key_stride), dtype=np.float64)
            block[:self.key_stride - extra_p, :self.key_stride - extra_n] = contributions[extra_p:, extra_n:]
            by_case_key[case * S:(case + 1) * S] = block.ravel()
        neighbor_by_case_key = np.zeros_like(by_case_key)
        for case in range(4):
            neighbor_by_case_key[case * S:(case + 1) * S] = by_case_key[(3 - case) * S:(4 - case) * S]
        return by_case_key, neighbor_by_case_key

    @staticmethod
    def _spread_to_features(
        c_bits: np.ndarray, b_bits: np.ndarray, pos_weighted: np.ndarray, neg_weighted: np.ndarray
    ) -> np.ndarray:
        """s = BNW + c_bits * (total_pos - BPW - BNW) of _compute_s_batch, without the leaf weight. Returns (D, batch)."""
        BPW = b_bits @ pos_weighted.T                                  # (D, batch)
        BNW = b_bits @ neg_weighted.T                                  # (D, batch)
        total_pos = pos_weighted.sum(axis=1)                           # (batch,)
        return BNW + c_bits * (total_pos - BPW - BNW)                  # (D, batch)

    def _compute_s_batch(
        self, consumer_batch: np.ndarray, unique_b: np.ndarray,
        b_freqs: np.ndarray, D: int, w: float
    ) -> np.ndarray:
        """
        Same computation as the base class but replaces the 3D intermediate with two matmuls.

        The key identity is that bit d of positive[c,b] factorizes as:
            pos_bits[d,c,b] = c_bits[d,c] * (1 - b_bits[d,b])
        and similarly for negative:
            neg_bits[d,c,b] = (1 - c_bits[d,c]) * b_bits[d,b]

        Substituting into the sum-over-backgrounds formula and pulling c_bits outside:
            s[d,c] =   c_bits[d,c]   * sum_b[ (1 - b_bits[d,b]) * pos_weighted[c,b] ]
                   + (1-c_bits[d,c]) * sum_b[      b_bits[d,b]  * neg_weighted[c,b] ]

        The two sums are matrix multiplications:
            BPW[d,c] = sum_b[ b_bits[d,b] * pos_weighted[c,b] ]  =  b_bits @ pos_weighted.T
            BNW[d,c] = sum_b[ b_bits[d,b] * neg_weighted[c,b] ]  =  b_bits @ neg_weighted.T

        So the full formula collapses to:
            s = BNW + c_bits * (total_pos - BPW - BNW)

        where total_pos[c] = sum_b(pos_weighted[c,b]).
        The (batch, U_b) intermediates from the base class remain, but the (D, batch, U_b) tensor is never allocated.
        """
        diff     = consumer_batch[:, None] ^ unique_b[None, :]       # (batch, U_b)
        positive = consumer_batch[:, None] & diff                     # s_plus bits per pair
        negative = unique_b[None, :]       & diff                     # s_minus bits per pair
        valid = (consumer_batch[:, None] | unique_b[None, :]) == ((1 << D) - 1)  # (batch, U_b)

        p = safe_bitwise_count(positive)                                # (batch, U_b)
        n = safe_bitwise_count(negative)                                # (batch, U_b)

        pos_weighted = self.pos_contributions[p, n] * b_freqs * valid  # (batch, U_b)
        neg_weighted = self.neg_contributions[p, n] * b_freqs * valid  # (batch, U_b)

        c_bits = bits_matrix(consumer_batch, D).astype(np.float64)    # (D, batch)
        b_bits = bits_matrix(unique_b, D).astype(np.float64)          # (D, U_b)

        BPW = b_bits @ pos_weighted.T                                  # (D, batch)
        BNW = b_bits @ neg_weighted.T                                  # (D, batch)
        total_pos = pos_weighted.sum(axis=1)                           # (batch,)
        return (BNW + c_bits * (total_pos - BPW - BNW)) * w           # (D, batch)

    def _compute_s_batch_with_neighbor(
        self, consumer_batch: np.ndarray, unique_b: np.ndarray,
        b_freqs: np.ndarray, D: int, w: float, w_neighbor: float
    ) -> np.ndarray:
        """
        _compute_s_batch of the leaf plus the right neighbor leaf, in the same pass over the pairs. The neighbor's
        patterns flip bit 0 (the last row of the bits matrices), so its pair of (c, b) is (c^1, b^1). Both pairs have the
        same bits but bit 0, so a single case key (see _contributions_by_case_key) reads the contributions of both.
        The background patterns of the neighbor are those of the leaf, so b_freqs are the same too.
        """
        all_bits = consumer_batch.dtype.type((1 << D) - 1)
        bit_0 = consumer_batch.dtype.type(1)
        diff = (consumer_batch[:, None] ^ unique_b[None, :]) & (all_bits ^ bit_0)  # (batch, U_b), all the bits but bit 0
        # Valid in all the bits but bit 0, whose case the key holds
        valid = (consumer_batch[:, None] | unique_b[None, :] | bit_0) == all_bits  # (batch, U_b)

        S = self.key_stride ** 2
        key = safe_bitwise_count(consumer_batch[:, None] & diff).astype(np.intp)   # p'
        key *= self.key_stride
        key += safe_bitwise_count(unique_b[None, :] & diff)                         # + n'
        key += ((consumer_batch & bit_0).astype(np.intp) * (2 * S))[:, None]        # + the case of bit 0: 2 * c0 + b0
        key += ((unique_b & bit_0).astype(np.intp) * S)[None, :]
        key = np.where(valid, key, 4 * S)                                           # (batch, U_b)

        c_bits = bits_matrix(consumer_batch, D).astype(np.float64)    # (D, batch)
        b_bits = bits_matrix(unique_b, D).astype(np.float64)          # (D, U_b)
        s = self._spread_to_features(
            c_bits, b_bits,
            self.pos_contributions_by_case_key[key] * b_freqs, self.neg_contributions_by_case_key[key] * b_freqs
        ) * w
        # The neighbor's patterns flip bit 0, the last row
        c_bits[-1] = 1 - c_bits[-1]
        b_bits[-1] = 1 - b_bits[-1]
        s += self._spread_to_features(
            c_bits, b_bits,
            self.neighbor_pos_contributions_by_case_key[key] * b_freqs,
            self.neighbor_neg_contributions_by_case_key[key] * b_freqs
        ) * w_neighbor
        return s                                                       # (D, batch)


class MNBackgroundMatrixMultiplicationFasterPathToSVectors(MNBackgroundMatrixMultiplicationPathToSVectors):
    """
    The same idea as MNBackgroundMatrixMultiplicationPathToSVectors (matrix multiplication), just faster numpy
    operations, about 2x faster.
    """
    # Two calls of its _compute_s_batch are faster than the same pass of MNBackgroundMatrixMultiplicationPathToSVectors
    # (benchmarked), so the neighbor leaf takes a second call
    NEIGHBOR_IN_THE_SAME_PASS = False

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        # The key of a pair with p positive and n negative literals is p * key_stride + n: its position in the flattened
        # (max_depth + 1, max_depth + 1) contribution tables. The tables by key have a 0 appended at invalid_key, the
        # key of the pairs that are not valid
        self.key_stride = max_depth + 1
        self.invalid_key = self.key_stride ** 2
        self.pos_contributions_by_key = np.append(self.pos_contributions.ravel(), 0.0)
        self.neg_contributions_by_key = np.append(self.neg_contributions.ravel(), 0.0)

    def _compute_s_batch(
        self, consumer_batch: np.ndarray, unique_b: np.ndarray,
        b_freqs: np.ndarray, D: int, w: float
    ) -> np.ndarray:
        """
        The computation of MNBackgroundMatrixMultiplicationPathToSVectors with faster numpy operations:
        1. In a valid pair every bit is 1 in the consumer or in the background, so its positive literals (consumer=1,
           background=0) are exactly the zero bits of the background and its negative literals (consumer=0,
           background=1) exactly the zero bits of the consumer: p = D - popcount(b) and n = D - popcount(c). The table
           key p * key_stride + n is then the sum of a (U_b,) and a (batch,) vector, so no (batch, U_b) bits are counted.
        2. Both tables are read with this single key, and the pairs that are not valid read the 0 at invalid_key instead
           of being multiplied by valid.
        3. pos_weighted[c,b] = pos_contributions[p, n] * b_freqs[b], so b_freqs is multiplied into the (D, U_b)
           background bits instead of the (batch, U_b) arrays:
               BPW = (b_bits * b_freqs) @ pos_contributions.T,    total_pos = pos_contributions @ b_freqs
        """
        valid = (consumer_batch[:, None] | unique_b[None, :]) == ((1 << D) - 1)  # (batch, U_b)

        p_keys = (D - safe_bitwise_count(unique_b).astype(np.intp)) * self.key_stride   # (U_b,), p * key_stride of the valid pairs
        n_keys = D - safe_bitwise_count(consumer_batch).astype(np.intp)                 # (batch,), n of the valid pairs
        key = np.where(valid, p_keys[None, :] + n_keys[:, None], self.invalid_key)     # (batch, U_b)

        pos_contributions = self.pos_contributions_by_key[key]        # (batch, U_b), pos_weighted without b_freqs
        neg_contributions = self.neg_contributions_by_key[key]        # (batch, U_b), neg_weighted without b_freqs

        c_bits = bits_matrix(consumer_batch, D).astype(np.float64)    # (D, batch)
        weighted_b_bits = bits_matrix(unique_b, D) * b_freqs          # (D, U_b)

        BPW = weighted_b_bits @ pos_contributions.T                    # (D, batch)
        BNW = weighted_b_bits @ neg_contributions.T                    # (D, batch)
        total_pos = pos_contributions @ b_freqs                        # (batch,)
        return (BNW + c_bits * (total_pos - BPW - BNW)) * w           # (D, batch)
