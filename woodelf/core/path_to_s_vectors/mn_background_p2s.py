from typing import List, Dict, Tuple

import numpy as np

from woodelf.core.cube_metric import CubeMetric
from woodelf.core.path_to_s_vectors.base_p2s import PathToSVectors
from woodelf.core.utils import bits_matrix


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

        p = np.bitwise_count(positive)                                # (batch, U_b)
        n = np.bitwise_count(negative)                                # (batch, U_b)

        pos_weighted = self.pos_contributions[p, n] * b_freqs * valid   # (batch, U_b)
        neg_weighted = self.neg_contributions[p, n] * b_freqs * valid   # (batch, U_b)

        batch_size = len(consumer_batch)
        U_b = len(unique_b)
        pos_bits = bits_matrix(positive.ravel(), D).reshape(D, batch_size, U_b)   # (D, batch, U_b)
        neg_bits = bits_matrix(negative.ravel(), D).reshape(D, batch_size, U_b)   # (D, batch, U_b)

        return (pos_bits * pos_weighted + neg_bits * neg_weighted).sum(axis=2) * w  # (D, batch)

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

        Returns Dict[feature -> np.ndarray of shape (U_c,)], one entry per feature in the path.
        """
        if not features_in_path:
            return {}
        D = len(features_in_path)

        unique_b, b_counts = np.unique(background_patterns, return_counts=True)
        b_freqs = b_counts / b_counts.sum()

        U_c = len(consumer_patterns)
        U_b = len(unique_b)

        batch_size = max(1, self.BATCH_SIZE // U_b)
        if U_c <= batch_size:
            s = self._compute_s_batch(consumer_patterns, unique_b, b_freqs, D, w)
        else:
            s = np.empty((D, U_c), dtype=np.float64)
            for start in range(0, U_c, batch_size):
                end = min(start + batch_size, U_c)
                s[:, start:end] = self._compute_s_batch(consumer_patterns[start:end], unique_b, b_freqs, D, w)

        result = {features_in_path[i]: s[i] for i in range(D)}

        if w_neighbor is not None:
            neighbor = self.get_background_s_matrix(
                features_in_path, consumer_patterns ^ 1, background_patterns ^ 1, w_neighbor,
            )
            for f in result:
                result[f] = result[f] + neighbor[f]

        return result

    def get_path_dependent_s_matrix(self, features_in_path, consumer_patterns, covers, w, w_neighbor=None):
        raise NotImplementedError("MNBackgroundPathToSVectors does not support the path-dependent case.")


class MNBackgroundFasterPathToSVectors(MNBackgroundPathToSVectors):
    """
    Faster variant of MNBackgroundPathToSVectors.
    Overrides _compute_s_batch to avoid the O(D * batch * U_b) 3D intermediate by
    decomposing the bit spreading into two (D, U_b) @ (U_b, batch) matrix multiplications,
    which carry the same asymptotic cost but benefit from BLAS optimization and better cache use.
    """

    BATCH_SIZE = 10_000_000

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

        p = np.bitwise_count(positive)                                # (batch, U_b)
        n = np.bitwise_count(negative)                                # (batch, U_b)

        pos_weighted = self.pos_contributions[p, n] * b_freqs * valid  # (batch, U_b)
        neg_weighted = self.neg_contributions[p, n] * b_freqs * valid  # (batch, U_b)

        c_bits = bits_matrix(consumer_batch, D).astype(np.float64)    # (D, batch)
        b_bits = bits_matrix(unique_b, D).astype(np.float64)          # (D, U_b)

        BPW = b_bits @ pos_weighted.T                                  # (D, batch)
        BNW = b_bits @ neg_weighted.T                                  # (D, batch)
        total_pos = pos_weighted.sum(axis=1)                           # (batch,)
        return (BNW + c_bits * (total_pos - BPW - BNW)) * w           # (D, batch)
