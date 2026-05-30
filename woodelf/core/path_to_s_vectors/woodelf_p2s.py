from __future__ import annotations

import time
from typing import List, Dict

import numpy as np

from woodelf.core.cube_metric import CubeMetric
from woodelf.core.path_to_s_vectors.base_p2s import WoodelfPathToSVectors

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False

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
                    raise ImportError("Couldn't import CuPy. To use GPU, please install Cupy via 'pip install cupy'")
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
