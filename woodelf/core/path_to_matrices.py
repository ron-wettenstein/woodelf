from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import scipy
import time

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
    """
    def __init__(self, max_depth: int, GPU: bool = False):
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

    def present_statistics(self):
        pass


class WoodelfPathToSVectors(PathToSVectors):
    """
    Base class for WOODELF-style path-to-s-vectors calculators (dense f vector approach).
    Subclasses implement _get_s_vectors_given_f.
    Provides static helpers for f computation and the neighbor-leaf trick.
    """
    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(max_depth, GPU)
        self.metric = metric

    @staticmethod
    def compute_f_from_patterns(patterns, depth: int, GPU: bool = False) -> np.ndarray:
        """Compute the dense f vector (normalised histogram) from integer decision patterns."""
        if GPU:
            return cp.bincount(patterns, minlength=2 ** depth) / len(patterns)
        return np.bincount(patterns, minlength=2 ** depth) / len(patterns)

    @staticmethod
    def compute_f_from_covers(covers: np.ndarray) -> np.ndarray:
        """
        Build the 2^D f vector from cover ratios (Formula 9 of the article).
        covers[i] is the proceed-cover ratio for the i-th unique feature in the path.
        """
        D = len(covers)
        if D == 0:
            return np.ones(1, dtype=np.float32)
        f_size = 2 ** D
        f = np.ones(f_size)
        for i in range(D):
            proceed_cover = covers[i]
            f = f * np.tile(
                np.array(
                    [1 - proceed_cover] * (f_size // 2 ** (1 + i)) +
                    [proceed_cover]     * (f_size // 2 ** (1 + i))
                ),
                2 ** i
            )
        return f.astype(np.float32)

    @staticmethod
    def neighbor_vector(f, GPU) -> np.ndarray:
        """
        Compute the f (or s-vector) of the neighbor leaf given f of the current leaf.
        Implements improvement 3 of Sec. 9.1: for a pair of sibling leaves, the neighbor's
        frequency array is a specific interleaved swap of the current leaf's array.
        """
        if GPU:
            idx = cp.arange(len(f))
            neighbor_f = cp.zeros_like(f)
        else:
            idx = np.arange(len(f))
            neighbor_f = np.zeros_like(f)

        neighbor_f_shift_left = neighbor_f.copy()
        neighbor_f_shift_left[1:] = f[:-1]
        neighbor_f_shift_left[(idx & 1) == 0] = 0

        neighbor_f_shift_right = neighbor_f.copy()
        neighbor_f_shift_right[:-1] = f[1:]
        neighbor_f_shift_right[(idx & 1) == 1] = 0
        return neighbor_f_shift_left + neighbor_f_shift_right

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
        self, features_in_path: List,
        background_patterns: np.ndarray, w: float, w_neighbor: float = None
    ) -> Dict:
        if not features_in_path:
            return {}
        depth = len(features_in_path)
        f = self.compute_f_from_patterns(background_patterns, depth, self.GPU)
        return self.compose_with_neighbor_trick(features_in_path, f, w, w_neighbor)

    def get_path_dependent_s_matrix(
        self, features_in_path: List,
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


def get_feature_repetition_sequence(features_in_path: List[str]):
    """
    Replace each feature with its first-occurrence index in the path.
    E.g. ["weight","pulse","age","pulse"] -> [0, 1, 2, 1]
    """
    feature_to_index = {}
    frs = []
    for i, feature in enumerate(features_in_path):
        if feature in feature_to_index:
            frs.append(feature_to_index[feature])
        else:
            feature_to_index[feature] = i
            frs.append(i)
    return frs


class SimpleWoodelfPathToSVectors(WoodelfPathToSVectors):
    """
    An object that in charge of creating the M matrix for every leaf and feature.
    It takes the features along the root-to-leaf path and build the matrix (lines 7-16 in WOODELF pseudo code)
    The class also utilize the fact that the matrix only depends on the repitting sequence of the features along the path.
    For example the feature repetition sequence of ["weight", "pluse", "age", "sex", "pluse", "sex"] is [1, 2, 3, 4, 2, 4].
    All feature lists with this feature repetition sequence have the same set of matrixes.

    This cache mechanism is improvement 2 in Sec. 9.1
    Suitable for shallow trees.
    """
    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        self.cached_used = 0
        self.cache_miss = 0
        self.cache = {}
        self.s_computation_time = 0
        self.m_computation_time = 0

    def get_values_matrices(self, features_in_path: List):
        start_time = time.time()
        frs = get_feature_repetition_sequence(features_in_path)
        frs_tuple = tuple(frs)

        if frs_tuple in self.cache:
            self.cached_used += 1
            matrixes = self.cache[frs_tuple]
        else:
            self.cache_miss += 1
            pc_pb_to_cube = self.map_patterns_to_cube(frs)
            matrixes = self.build_patterns_to_values_sparse_matrix(pc_pb_to_cube, self.metric, len(features_in_path))
            self.cache[frs_tuple] = matrixes

        if not self.metric.INTERACTION_VALUE:
            matrixes_for_features = {features_in_path[index]: matrixes[index] for index in matrixes}
        else:
            matrixes_for_features = {}
            for feature_indexes, current_matrices in matrixes.items():
                if not self.metric.INTERACTION_VALUES_ORDER_MATTERS:
                    feature_tuple = tuple(sorted([features_in_path[i] for i in feature_indexes]))
                else:
                    feature_tuple = tuple([features_in_path[i] for i in feature_indexes])
                matrixes_for_features[feature_tuple] = current_matrices

        self.m_computation_time += time.time() - start_time
        return matrixes_for_features

    def _get_s_vectors_given_f(self, features_in_path: List, f: np.ndarray, w: float) -> Dict:
        matrices = self.get_values_matrices(features_in_path)
        start_time = time.time()
        s_vectors = {}
        for feature in matrices:
            # The matrix multiplication part is implemented in CPU, the matrix is too small for the GPU overhead to be worth it.
            # The sparse matrix multiplication here instead of the naive dense matrix multiplication is improvement 1 in Sec. 9.1
            try:
                s_vectors[feature] = matrices[feature].dot(f) * w
            except Exception as e:
                raise e
        self.s_computation_time += time.time() - start_time
        return s_vectors

    def present_statistics(self):
        print(
            f"cache misses: {self.cache_miss}, cache used: {self.cached_used}, "
            f"M computation time: {round(self.m_computation_time, 2)} sec, "
            f"s computation time: {round(self.s_computation_time, 2)} sec"
        )


class HighDepthWoodelfPathToSVectors(WoodelfPathToSVectors):
    """
    Pre-builds all M-matrix diagonals up to max_depth at construction time.
    Uses the Strassen-like anti-diagonal multiplication.
    Suitable for deep trees with depths up to 18/21 (while SimpleWoodelfPathToSVectors fails on depths bigger than 12).
    """
    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False, use_neighbor_leaf_trick: bool = True):
        super().__init__(metric, max_depth, GPU)
        self.s_computation_time = 0
        self.f_prepare_time = 0
        self.s_computation_calls = 0
        self.total_f_sizes = 0
        self.use_neighbor_leaf_trick = use_neighbor_leaf_trick

        start_time = time.time()
        self.matrices = {}
        self.neighbor_matrices = {}
        self.matrices_frs_subsets = {}
        self._build_matrices()
        self.matrices_init_time = time.time() - start_time

    @classmethod
    def map_patterns_to_cube(cls, features_in_path: List):
        """
        High-depth version: keeps only rules 1 and 2 to compute only the diagonal.
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
                    # Get the current cube (the positive and negated literals) of the consumer and background patterns
                    s_plus, s_minus = current_wdnf_table[consumer_pattern][background_pattern]
                    updated_wdnf_table[consumer_pattern * 2 + 1][background_pattern * 2 + 0] = (s_plus | {feature}, s_minus)  # Rule 1
                    updated_wdnf_table[consumer_pattern * 2 + 0][background_pattern * 2 + 1] = (s_plus, s_minus | {feature})  # Rule 2
                    # Rule 3 dropped: only the diagonal is needed
        return updated_wdnf_table

    @classmethod
    def build_patterns_to_values_sparse_matrix(cls, dl, metric: CubeMetric, path_length=None):
        """
        Apply the CubeMetric object (the v function), to create the matrices M.
        include lines 12-16 in WOODELF pseudocode.
        dl is the returned mapping from the map_patterns_to_cube function
        """
        values_list = []
        all_feature_subsets = set()
        for consumer_pattern in sorted(dl.keys()):
            background_pattern, cube = dl[consumer_pattern].popitem()
            s_plus, s_minus = cube
            values = metric.calc_metric(s_plus, s_minus)
            values_list.append(values)
            all_feature_subsets.update(set(values.keys()))

        matrix_details = {fs: [] for fs in all_feature_subsets}
        for values in values_list:
            for fs in all_feature_subsets:
                matrix_details[fs].append(values.get(fs, 0))

        return {fs: vals for fs, vals in matrix_details.items()}

    def _build_matrices(self):
        for depth in range(0, self.max_depth + 1):
            dl = self.map_patterns_to_cube(list(range(depth)))
            matrices = self.build_patterns_to_values_sparse_matrix(dl, self.metric, path_length=depth)
            self.matrices_frs_subsets[depth] = list(matrices.keys())
            if self.GPU:
                if not IMPORTED_CP:
                    raise ImportError("Couldn't import CuPy. To use GPU, please install Cu{y via 'pip install cupy'")
                self.matrices[depth] = cp.array(
                    [matrices[k] for k in self.matrices_frs_subsets[depth]], dtype=cp.float32
                ).T
            else:
                self.matrices[depth] = np.array(
                    [matrices[k] for k in self.matrices_frs_subsets[depth]], dtype=np.float32
                ).T

    def _get_s_vectors_given_f(self, features_in_path: List, f: np.ndarray, w: float) -> Dict:
        depth = len(features_in_path)
        self.s_computation_calls += 1
        self.total_f_sizes += np.sum(f != 0)
        start_time = time.time()

        if self.GPU:
            idx = cp.arange(len(f))
        else:
            idx = np.arange(len(f))

        start_time_f_prepare = time.time()
        f = self._prepare_f(depth, f, self.GPU)
        self.f_prepare_time += time.time() - start_time_f_prepare

        matrix_diagonals = self.matrices[depth]
        frs2feature_name = self.frs_subsets_to_feature_subsets(features_in_path, depth)
        s_matrix = matrix_diagonals * f[::-1].reshape(-1, 1) # reversed as this is not the (0,0)-(1,1)-..-(n,n) diagonal but the (0,n)-(1,n-1)-..-(n,0) diagonal
        for d in range(0, depth, 1):
            s_matrix_copy = s_matrix.copy()
            s_matrix_copy[2 ** d:, :] = s_matrix[:-2 ** d, :] # shift the array to the left 2**d bits
            s_matrix_copy[(idx & (1 << d)) == 0] = 0 # Zero all elements that are in an even place in the current division
            s_matrix = s_matrix + s_matrix_copy

        s_matrix = s_matrix * w

        s_vectors = {}
        for index, frs_subset in enumerate(self.matrices_frs_subsets[depth]):
            feature_subset = frs2feature_name[frs_subset]
            s_vectors[feature_subset] = s_matrix[:, index]

        self.s_computation_time += time.time() - start_time
        return s_vectors

    def frs_subsets_to_feature_subsets(self, features_in_path: List, depth: int):
        if not self.metric.INTERACTION_VALUE:
            return {index: features_in_path[index] for index in self.matrices_frs_subsets[depth]}
        frs2feature_name = {}
        for frs_subsets in self.matrices_frs_subsets[depth]:
            if not self.metric.INTERACTION_VALUES_ORDER_MATTERS:
                feature_tuple = tuple(sorted([features_in_path[i] for i in frs_subsets]))
            else:
                feature_tuple = tuple([features_in_path[i] for i in frs_subsets])
            frs2feature_name[frs_subsets] = feature_tuple
        return frs2feature_name

    @staticmethod
    def _prepare_f(depth, f, GPU):
        if GPU:
            idx = cp.arange(len(f))
        else:
            idx = np.arange(len(f))
        for d in range(depth - 1, -1, -1):
            f_copy = f.copy()
            f_copy[:-2 ** d] = f[2 ** d:] # shift the array to the left 2**d bits
            f_copy[(idx & (1 << d)) != 0] = 0 # Zero all elements that are in an even place in the current division
            f = f + f_copy
        return f

    def present_statistics(self):
        mean_f_size = self.total_f_sizes / self.s_computation_calls if self.s_computation_calls > 0 else 0
        print(
            f"M time: {round(self.matrices_init_time, 2)} sec, "
            f"s time: {round(self.s_computation_time, 2)} sec ({self.s_computation_calls} _get_s_vectors_given_f calls, "
            f"f prepare time: {self.f_prepare_time}, f mean non zero size: {mean_f_size})"
        )


class HighDepthWoodelfPaperVersionPathToSVectors(HighDepthWoodelfPathToSVectors):
    """
    Paper-version of HighDepthWoodelfPathToSVectors using explicit StrassenLikeMult.
    Slightly slower in practice; used to verify the optimised version in tests.
    """

    @staticmethod
    def StrassenLikeMult(M_diag, f, D):
        idx = np.arange(len(f))
        for d in range(D - 1, -1, -1):
            f_copy = f.copy()
            f_copy[:-2 ** d] = f[2 ** d:]
            f_copy[idx & (1 << d) != 0] = 0
            f = f + f_copy

        f_reverse = f[::-1]
        s = M_diag * f_reverse
        for d in range(D):
            s_copy = s.copy()
            s_copy[2 ** d:] = s[:-2 ** d]
            s_copy[idx & (1 << d) == 0] = 0
            s = s + s_copy
        return s

    def _build_matrices(self):
        for depth in range(0, self.max_depth + 1):
            dl = self.map_patterns_to_cube(list(range(depth)))
            matrices = self.build_patterns_to_values_sparse_matrix(dl, self.metric, path_length=depth)
            self.matrices_frs_subsets[depth] = list(matrices.keys())
            self.matrices[depth] = [np.array(matrices[k]) for k in self.matrices_frs_subsets[depth]]

    def _get_s_vectors_given_f(self, features_in_path: List, f: np.ndarray, w: float) -> Dict:
        depth = len(features_in_path)
        self.s_computation_calls += 1
        self.total_f_sizes += np.sum(f != 0)
        start_time = time.time()

        frs2feature_name = self.frs_subsets_to_feature_subsets(features_in_path, depth)
        s_vectors = {}
        for index, frs_subset in enumerate(self.matrices_frs_subsets[depth]):
            feature_subset = frs2feature_name[frs_subset]
            m_diag = self.matrices[depth][index]
            s_vectors[feature_subset] = self.StrassenLikeMult(m_diag, f, depth) * w

        self.s_computation_time += time.time() - start_time
        return s_vectors
