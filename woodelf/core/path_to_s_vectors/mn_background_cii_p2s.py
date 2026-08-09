from itertools import combinations
from typing import List, Dict

import numpy as np

from woodelf.core.cube_metric import CubeMetric, CardinalityInteractionIndicesMetric
from woodelf.core.path_to_s_vectors.base_p2s import PathToSVectors
from woodelf.core.utils import bits_matrix


class MNBackgroundCIIPathToSVectors(PathToSVectors):
    """
    Extends the MNBackgroundPathToSVectors O(nm) approach to interaction values of any order k, for metrics
    expressible through cube cardinalities (CardinalitySymmetricInteractionMetric).

    For each (consumer, background) pattern pair the cube literals are read off the bit patterns exactly like
    in MNBackgroundPathToSVectors. A k-subset S of the path features contributes only when all its bits are
    literals of the cube, and by symmetry its value depends only on the cube literal counts (p, n) and on how
    many of S's bits are positive literals - so it is read from a precomputed (k+1, D+1, D+1) table.

    The metric reports every order k in metric.orders(D), so the s-matrix holds one row per subset of every
    such size, ordered by order and then as self._subsets(D, order) (see self._all_subsets).

    This base class is the default implementation: it loops over the subsets and reduces the
    (batch, U_b) pair tensors per subset. See MNBackgroundCIIFasterPathToSVectors for the WIP BLAS variant.

    Requires numpy >= 2, as it uses np.bitwise_count (added in numpy 2.0).
    """

    BATCH_SIZE = 1_000_000

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        assert isinstance(metric, CardinalityInteractionIndicesMetric), (
            f"{type(self).__name__} requires a CardinalitySymmetricInteractionMetric metric. Got {type(metric).__name__}."
        )
        super().__init__(metric, max_depth, GPU)
        self.contribution_tables = MNBackgroundCIIPathToSVectors._build_contribution_tables(metric, max_depth)
        self._subsets_cache = {}

    @staticmethod
    def _build_contribution_tables(
        metric: CardinalityInteractionIndicesMetric, D: int
    ) -> Dict[int, np.ndarray]:
        """
        Precompute the per-subset contribution tables of a cardinality-symmetric metric, one per reported
        order k (metric.orders(D), as D bounds the number of variables of any cube along the path).

        tables[k][a][p][n] = value of a k-subset with a positive-literal variables (and k - a negative-literal
        variables) in a cube with p positive and n negative literals.
        Unreachable entries (a > p, k - a > n, p + n > D or p + n < k) remain 0.
        """
        tables = {}
        for k in metric.orders(D):
            tables[k] = np.zeros((k + 1, D + 1, D + 1), dtype=np.float64)

            for p in range(D + 1):
                for n in range(D + 1 - p):
                    if p + n < k:
                        continue
                    for a in range(k + 1):
                        b = k - a
                        if a <= p and b <= n:
                            tables[k][a][p][n] = metric.cardinality_value(n, p, a, b)

        return tables

    def _subsets(self, D: int, order: int) -> np.ndarray:
        """
        All order-subsets of the path positions, as an (n_subsets, order) array of rows of the bits matrices.
        """
        if (D, order) not in self._subsets_cache:
            self._subsets_cache[(D, order)] = np.array(
                list(combinations(range(D), order)), dtype=np.int64
            ).reshape(-1, order)
        return self._subsets_cache[(D, order)]

    def _all_subsets(self, D: int):
        """The subsets of the path positions in s-matrix row order: by order, then as self._subsets."""
        for order in self.metric.orders(D):
            yield from self._subsets(D, order)

    def _order_row_blocks(self, D: int):
        """
        Yields (order, row_start, n_subsets) per reported order, row_start being the s-matrix row its first
        subset occupies. Lets the per-order machinery address its own block of the shared s-matrix.
        """
        row_start = 0
        for order in self.metric.orders(D):
            n_subsets = len(self._subsets(D, order))
            yield order, row_start, n_subsets
            row_start += n_subsets

    def _num_subsets(self, D: int) -> int:
        """The total number of s-matrix rows: the subsets of every reported order."""
        return sum(len(self._subsets(D, order)) for order in self.metric.orders(D))

    def _compute_s_batch(
        self, consumer_batch: np.ndarray, unique_b: np.ndarray,
        b_freqs: np.ndarray, D: int, w: float
    ) -> np.ndarray:
        """
        Core vectorized computation for a batch of consumer patterns against all unique background patterns.

        For each pair, positive/negative literal bits and their counts (p, n) are derived as in
        MNBackgroundPathToSVectors. For each subset S: pairs whose diff bits cover S have all of S as cube
        literals; the count of S's bits that are positive literals selects the contribution table entry.

        Returns shape (n_subsets, batch), one s-vector row per reported subset of the path features
        (rows ordered as self._all_subsets(D)).
        """
        diff     = consumer_batch[:, None] ^ unique_b[None, :]       # (batch, U_b)
        positive = consumer_batch[:, None] & diff                     # s_plus bits per pair

        # Only pairs where every bit is covered by at least one side going right.
        # Pairs with a "both-diverge" bit (pc_k=0, pb_k=0) have no WDNF rule and contribute 0.
        valid = (consumer_batch[:, None] | unique_b[None, :]) == ((1 << D) - 1)  # (batch, U_b)

        p = np.bitwise_count(positive)                                # (batch, U_b)
        n = np.bitwise_count(unique_b[None, :] & diff)                # (batch, U_b)

        base = b_freqs * valid                                        # (batch, U_b)

        s = np.empty((self._num_subsets(D), len(consumer_batch)), dtype=np.float64)
        for s_index, subset in enumerate(self._all_subsets(D)):
            subset_mask = 0
            for d in subset:
                subset_mask |= 1 << (D - 1 - d)                       # row d of bits_matrix is bit D-1-d
            covered = (diff & subset_mask) == subset_mask             # S is fully inside the cube
            a = np.bitwise_count(positive & subset_mask)              # |S ∩ positive literals|
            s[s_index] = (self.contribution_tables[len(subset)][a, p, n] * base * covered).sum(axis=1)
        return s * w

    def get_background_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: float = None,
    ) -> Dict:
        """
        Compute the background s-matrix for a single leaf.

        Uniquifies background_patterns into (unique_b, b_freqs), then delegates to _compute_s_batch.
        Consumer patterns are processed in chunks of BATCH_SIZE // U_b to stay within memory bounds.

        If w_neighbor is provided, also adds the neighbor leaf's contribution: the neighbor leaf has
        patterns where the LSB is flipped (sibling in the tree), so we recurse with consumer_patterns ^ 1,
        background_patterns ^ 1, and w=w_neighbor.

        Returns Dict[feature tuple -> np.ndarray of shape (U_c,)], one entry per subset of the features in
        the path whose size is a reported order. Keys are canonically sorted feature tuples, each holding
        the full metric value of the subset once (no mirrored/permuted duplicate keys).
        """
        D = len(features_in_path)
        if D < self.metric.min_order:
            return {}

        unique_b, b_counts = np.unique(background_patterns, return_counts=True)
        b_freqs = b_counts / b_counts.sum()

        U_c = len(consumer_patterns)
        U_b = len(unique_b)

        batch_size = max(1, self.BATCH_SIZE // U_b)
        if U_c <= batch_size:
            s = self._compute_s_batch(consumer_patterns, unique_b, b_freqs, D, w)
        else:
            s = np.empty((self._num_subsets(D), U_c), dtype=np.float64)
            for start in range(0, U_c, batch_size):
                end = min(start + batch_size, U_c)
                s[:, start:end] = self._compute_s_batch(consumer_patterns[start:end], unique_b, b_freqs, D, w)

        result = {
            tuple(sorted(features_in_path[d] for d in subset)): s[s_index]
            for s_index, subset in enumerate(self._all_subsets(D))
        }

        if w_neighbor is not None:
            neighbor = self.get_background_s_matrix(
                features_in_path, consumer_patterns ^ 1, background_patterns ^ 1, w_neighbor,
            )
            for subset in result:
                result[subset] = result[subset] + neighbor[subset]

        return result

    def get_path_dependent_s_matrix(self, features_in_path, consumer_patterns, covers, w, w_neighbor=None):
        raise NotImplementedError("MNBackgroundCIIPathToSVectors does not support the path-dependent case.")


class MNBackgroundCIIFasterPathToSVectors(MNBackgroundCIIPathToSVectors):
    """
    WIP - not the default: woodelf_sparse uses MNBackgroundCIIPathToSVectors for the CII metrics.

    Faster variant of MNBackgroundCIIPathToSVectors.
    Overrides _compute_s_batch so that the sum over backgrounds becomes dense matrix multiplications,
    the same restructuring MNBackgroundFasterPathToSVectors applies to the order-1 case.

    The key identity: for a k-subset S of the path features, S is fully contained in the cube of a valid
    pair (pc, pb) with S ∩ (positive literals) = A iff
        [pc restricted to S == A] * [pb restricted to S == complement of A within S]
    - a product of a consumer-only indicator and a background-only indicator. Hence, per split level
    a = |A|, the background reduction is a matmul over per-(S, A) indicator rows:
        BM_a[(S,A), c] = sum_b Bmask_a[(S,A), b] * W_a[c, b]
    with W_a[c, b] = b_freqs * valid * tables[a][p, n]. Since for each (S, pc) exactly one A matches,
        s[S, c] = sum_a sum_{A subset of S, |A|=a} Cmask_a[(S,A), c] * BM_a[(S,A), c]
    is a masked sum that selects a single BM entry per (S, c).

    Why this is still WIP - measured on a single leaf at D=22, 30 consumer x 20 background patterns,
    best-of-repeats wall time in seconds:

        k          1        2        3        4        5        6         7         8      9     10
        naive      0.0010   0.0090   0.0636   0.3169   1.1500   2.4206    5.6281   10.94  17.25  22.64
        faster     0.0004   0.0020   0.0251   0.2120   1.2675   8.2320   40.7184       -      -      -
        speedup    2.67x    4.48x    2.53x    1.50x    0.91x    0.29x     0.14x        -      -      -

    The matmul form wins only up to k=4, then loses - and by k=7 it is 7x slower than the class it was
    meant to replace. The crossover is the 2^k in its row count: it trades C(D,k) elementwise passes for
    C(D,k)*2^k BLAS rows, and once that factor outgrows what dgemm buys back, the trade is a loss.
    k>=8 was not measured at all - the cached split-index arrays alone project to ~4.9 GB (k=8), ~17 GB
    (k=9) and ~49 GB (k=10) at this D, the same 2^k blowup surfacing as memory rather than time.

    On full models the effect is milder but the same shape: over a 100-tree IEEE-CIS xgboost ensemble
    this class ran ~2-2.6x faster than the base class at k=2 and k=3 on depth-16/22 trees, but only
    ~1.4x at k=4, and was slower than the base class at k=4 on a depth-6 model (0.78-0.85x), where k is
    large relative to D and the same row-count blowup applies.

    The inclusion-exclusion rewrite noted in _compute_s_batch is the natural fix: it replaces
    C(D,k)*2^k rows by (k+1)*sum_{j<=k} C(D,j) (12.7x fewer at D=16, k=10, and the gap widens with k).
    Until that lands this class is only worth choosing for low orders on deep trees, which is why
    woodelf_sparse defaults to the base class instead.
    """

    BATCH_SIZE = 10_000_000
    # Bounds the (rows, U_b) / (rows, batch) mask and matmul intermediates: subsets are processed in chunks
    # of whole-S row groups such that rows_per_chunk * max(U_b, batch) <= MASK_BATCH.
    MASK_BATCH = 2_000_000

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        self._split_cache = {}

    def _split_index_arrays(self, D: int, k: int, a: int):
        """
        Row layout for split level a of order k: one row for every k-subset S (in self._subsets(D, k) order)
        and every A subset of S with |A| = a. Returns (positions, in_A) of shape (n_subsets * C(k,a), k):
        positions[r, j] = bit-row index of the j-th member of S, in_A[r, j] = whether it belongs to A.
        Rows are grouped by S (r // C(k,a) is the subset index), so reshaping to (n_subsets, C(k,a), ...)
        aligns the per-S splits for the masked sum.
        """
        if (D, k, a) not in self._split_cache:
            subsets = self._subsets(D, k)
            a_combos = list(combinations(range(k), a))
            a_patterns = np.zeros((len(a_combos), k), dtype=bool)
            for combo_index, combo in enumerate(a_combos):
                a_patterns[combo_index, list(combo)] = True
            positions = np.repeat(subsets, len(a_combos), axis=0)
            in_A = np.tile(a_patterns, (len(subsets), 1))
            self._split_cache[(D, k, a)] = (positions, in_A)
        return self._split_cache[(D, k, a)]

    @staticmethod
    def _restriction_masks(bits: np.ndarray, positions: np.ndarray, in_pattern: np.ndarray) -> np.ndarray:
        """
        masks[r, x] = 1 iff column x of `bits`, restricted to positions[r], equals the in_pattern[r] flags:
            prod_j (bits[positions[r,j], x] if in_pattern[r,j] else 1 - bits[positions[r,j], x])
        bits: (D, N) uint8. positions, in_pattern: (rows, k). Returns (rows, N) uint8.
        """
        gathered = bits[positions]                                    # (rows, k, N)
        gathered = np.where(in_pattern[:, :, None], gathered, 1 - gathered)
        return gathered.prod(axis=1, dtype=np.uint8)

    def _compute_s_batch(
        self, consumer_batch: np.ndarray, unique_b: np.ndarray,
        b_freqs: np.ndarray, D: int, w: float
    ) -> np.ndarray:
        """
        Same computation as the base class, restructured into per-split-level matmuls (see class docstring).
        The (S, A) rows are processed in subset-chunks to bound the mask/matmul intermediates; the
        background masks are therefore rebuilt per consumer batch (caching them whole would cost exactly
        the (rows, U_b) memory the chunking avoids, and leaves usually need a single batch anyway).

        A further possible optimization (skipped for now): expanding the (1 - b_bits) mask factors via
        inclusion-exclusion turns all background masks into AND-monomials shared across splits, shrinking
        the matmul rows from C(D,k)*2^k to (k+1)*sum_{j<=k} C(D,j) (~40% fewer at k=3), at the price of a
        signed multi-term recombination instead of the select-like masked sum below.

        Each reported order is an independent instance of this scheme, so the orders are run one after the
        other, each filling its own block of s rows (self._order_row_blocks).
        """
        batch = len(consumer_batch)
        U_b = len(unique_b)

        diff     = consumer_batch[:, None] ^ unique_b[None, :]       # (batch, U_b)
        positive = consumer_batch[:, None] & diff                     # s_plus bits per pair
        valid = (consumer_batch[:, None] | unique_b[None, :]) == ((1 << D) - 1)  # (batch, U_b)

        p = np.bitwise_count(positive)                                # (batch, U_b)
        n = np.bitwise_count(unique_b[None, :] & diff)                # (batch, U_b)

        base = b_freqs * valid                                        # (batch, U_b)

        c_bits = bits_matrix(consumer_batch, D)                       # (D, batch)
        b_bits = bits_matrix(unique_b, D)                             # (D, U_b)

        s = np.zeros((self._num_subsets(D), batch), dtype=np.float64)
        for k, block_start, n_subsets in self._order_row_blocks(D):
            s_k = s[block_start: block_start + n_subsets]             # view into s: += writes through
            for a in range(k + 1):
                W = self.contribution_tables[k][a][p, n] * base       # (batch, U_b)
                positions, in_A = self._split_index_arrays(D, k, a)
                splits_per_subset = len(positions) // n_subsets       # = C(k, a)
                rows_per_chunk = splits_per_subset * max(
                    1, self.MASK_BATCH // (splits_per_subset * max(U_b, batch))
                )
                for row_start in range(0, len(positions), rows_per_chunk):
                    row_end = min(row_start + rows_per_chunk, len(positions))
                    chunk_positions = positions[row_start:row_end]
                    # Background side is inverted vs A: [pb restricted to S == complement of A within S]
                    b_mask = self._restriction_masks(b_bits, chunk_positions, ~in_A[row_start:row_end])
                    BM = b_mask.astype(np.float64) @ W.T              # (chunk_rows, batch)
                    c_mask = self._restriction_masks(c_bits, chunk_positions, in_A[row_start:row_end])
                    s_k[row_start // splits_per_subset: row_end // splits_per_subset] += (
                        (c_mask * BM).reshape(-1, splits_per_subset, batch).sum(axis=1)
                    )
        return s * w
