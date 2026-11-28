from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import scipy

from woodelf.core.cube_metric import CubeMetric

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False


class PathToSVectors:
    """
    Abstract base class for all path-to-s-vectors calculators.
    Subclasses implement get_background_s_matrix and/or get_path_dependent_s_matrix.
    Both methods return Dict[feature_key -> np.array[N_consumers]].

    Subclasses with COMPUTES_CONSUMER_VALUES = True also implement get_background_consumer_values and
    get_path_dependent_consumer_values, which return the leaf's values of every consumer (indexed like
    consumer_patterns) instead of s vectors that the caller indexes by consumer_patterns itself.
    """
    COMPUTES_CONSUMER_VALUES = False

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        self.metric = metric
        self.max_depth = max_depth
        self.GPU = GPU

    def get_background_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: float = None
    ) -> Dict:
        raise NotImplementedError

    def get_path_dependent_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: float = None
    ) -> Dict:
        raise NotImplementedError

    def get_background_consumer_values(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: float = None, values: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Returns the leaf's values of every consumer: Dict[feature_key -> np.array[len(consumer_patterns)]].
        If values is given, the leaf's values are added into it in place and None is returned.
        """
        raise NotImplementedError

    def get_path_dependent_consumer_values(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: float = None, values: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Returns the leaf's values of every consumer: Dict[feature_key -> np.array[len(consumer_patterns)]].
        If values is given, the leaf's values are added into it in place and None is returned.
        """
        raise NotImplementedError

    def present_statistics(self):
        pass


def compute_f_from_patterns(patterns, depth: int, GPU: bool = False) -> np.ndarray:
    """Compute the dense f vector (normalised histogram) from integer decision patterns."""
    if GPU:
        return cp.bincount(patterns, minlength=2 ** depth) / len(patterns)
    return np.bincount(patterns, minlength=2 ** depth) / len(patterns)


class WoodelfPathToSVectors(PathToSVectors):
    """
    Base class for WOODELF-style path-to-s-vectors calculators (dense f vector approach).
    Subclasses implement _get_s_vectors_given_f.
    Provides static helpers for f computation and the neighbor-leaf trick.
    """

    @staticmethod
    def compute_f_from_covers(covers: np.ndarray) -> np.ndarray:
        """
        Build the 2^D f vector from cover ratios (Formula 9 of the article).
        covers[i] is the proceed-cover ratio for the i-th unique feature in the path.
        """
        f = np.ones(1)
        for proceed_cover in covers:
            # Append the bit of the next feature (the first feature is the most significant bit): every pattern so far
            # splits into the one that does not proceed with the feature (bit 0) and the one that does (bit 1)
            # The f vector starts at [1] and doubles in size with every feature, 
            # (for example [c_l1, c_r1], [c_l1 * c_l2, c_l1 * c_r2, c_r1 * c_l2, c_r1 * c_r2], etc),
            # so the final f vector has size 2^D.
            # In each iteration we do f = np.stack([f * (1 - proceed_cover), f * proceed_cover], axis=1).ravel(), but np.multiply.outer is faster.
            f = np.multiply.outer(f, [1 - proceed_cover, proceed_cover]).ravel()
        return f.astype(np.float32)

    @staticmethod
    def neighbor_vector(f, GPU) -> np.ndarray:
        """
        Compute the f (or s-vector) of the neighbor leaf given f of the current leaf.
        Implements improvement 3 of Sec. 9.1: for a pair of sibling leaves, the neighbor's
        frequency array is a specific interleaved swap of the current leaf's array.
        """
        # The neighbor's array is f with the two elements of every pair (2k, 2k+1) swapped: its element i is f[i ^ 1].
        # We do reshape(-1, 2), which views f as a matrix with one row per pair, then [:, ::-1], which reverses every
        # row (still a view, nothing is copied yet), and finally flatten(), which copies the rows back into a new 1D
        # array, so f itself is not changed. Works for NumPy and CuPy arrays alike
        return f.reshape(-1, 2)[:, ::-1].flatten()

    def compose_with_neighbor_trick(
        self, features_in_path: List, f: np.ndarray, w: float, w_neighbor: float = None
    ) -> Dict:
        """
        Given a pre-computed f vector, apply _get_s_vectors_given_f with the optional
        neighbor-leaf trick. Returns Dict[feature_key -> s_vector of size 2^D].
        Caller is responsible for indexing the result by consumer_patterns.
        """
        if w_neighbor is None:
            return self._get_s_vectors_given_f(features_in_path, f, w)
        s_left  = self._get_s_vectors_given_f(features_in_path, f, w)
        s_right = self._get_s_vectors_given_f(
            features_in_path, self.neighbor_vector(f, self.GPU), w_neighbor
        )
        return {k: s_left[k] + self.neighbor_vector(s_right[k], self.GPU) for k in s_left}

    def get_background_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: float = None
    ) -> Dict:
        if not features_in_path:
            return {}
        depth = len(features_in_path)
        f = compute_f_from_patterns(background_patterns, depth, self.GPU)
        return self.compose_with_neighbor_trick(features_in_path, f, w, w_neighbor)

    def get_path_dependent_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: float = None
    ) -> Dict:
        if not features_in_path:
            return {}
        f = self.compute_f_from_covers(covers)
        if self.GPU:
            f = cp.asarray(f)
        return self.compose_with_neighbor_trick(features_in_path, f, w, w_neighbor)

    def _get_s_vectors_given_f(self, features_in_path: List, f: np.ndarray, w: float) -> Dict:
        """
        Core WOODELF computation: given a dense f vector and leaf weight w, return
        Dict[feature_key -> s_vector] where each s_vector has size 2^D.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared classmethods for building M matrices                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def map_patterns_to_cube(cls, features_in_path: List):
        """
        MapPatternsToCube from Sect. 5 of the article.
        Returns wdnf_table[consumer_pattern][background_pattern] = (positive_literals, negative_literals).
        """
        updated_wdnf_table = {0: {0: (set(), set())}}
        current_wdnf_table = None
        for feature in features_in_path:
            current_wdnf_table = updated_wdnf_table
            updated_wdnf_table = {}
            for consumer_pattern in current_wdnf_table:
                updated_wdnf_table[consumer_pattern * 2 + 0] = {}
                updated_wdnf_table[consumer_pattern * 2 + 1] = {}
                for background_pattern in current_wdnf_table[consumer_pattern]:
                    s_plus, s_minus = current_wdnf_table[consumer_pattern][background_pattern]
                    updated_wdnf_table[consumer_pattern * 2 + 1][background_pattern * 2 + 0] = (s_plus | {feature}, s_minus)   # Rule 1
                    updated_wdnf_table[consumer_pattern * 2 + 0][background_pattern * 2 + 1] = (s_plus, s_minus | {feature})   # Rule 2
                    updated_wdnf_table[consumer_pattern * 2 + 1][background_pattern * 2 + 1] = (s_plus, s_minus)               # Rule 3
        return updated_wdnf_table

    @classmethod
    def build_patterns_to_values_sparse_matrix(cls, dl, metric: CubeMetric, path_length):
        """
        Apply the CubeMetric to create sparse M matrices (lines 12-16 of WOODELF pseudocode).
        """
        matrix_details = {}
        for pc in dl:
            for pb in dl[pc]:
                s_plus, s_minus = dl[pc][pb]
                values = metric.calc_metric(s_plus, s_minus)
                for feature in values:
                    if feature not in matrix_details:
                        matrix_details[feature] = {"pcs": [], "pbs": [], "values": []}
                    matrix_details[feature]["pcs"].append(pc)
                    matrix_details[feature]["pbs"].append(pb)
                    matrix_details[feature]["values"].append(values[feature])

        matrixs = {}
        for feature in matrix_details:
            matrix_values = (
                matrix_details[feature]["values"],
                (matrix_details[feature]["pcs"], matrix_details[feature]["pbs"])
            )
            matrixs[feature] = scipy.sparse.coo_matrix(
                matrix_values, shape=(2 ** path_length, 2 ** path_length), dtype=np.float32
            ).tocsc()
        return matrixs