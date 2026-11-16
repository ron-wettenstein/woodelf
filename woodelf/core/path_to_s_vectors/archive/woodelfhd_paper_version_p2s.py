from __future__ import annotations

import time
from typing import List, Dict

import numpy as np

from woodelf.core.path_to_s_vectors.woodelfhd_v1_p2s import HighDepthWoodelfV1PathToSVectors

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False


class HighDepthWoodelfOriginalVersionPathToSVectors(HighDepthWoodelfV1PathToSVectors):
    """
    Original version of HighDepthWoodelfV1PathToSVectors: every step of the f preparation and of the s computation
    builds a shifted copy of the array, zeros the masked elements and sums the two arrays.
    Replaced by in-place updates on reshaped views; kept to verify and benchmark the newer version.
    """

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


class HighDepthWoodelfPaperVersionPathToSVectors(HighDepthWoodelfV1PathToSVectors):
    """
    Paper-version of HighDepthWoodelfV1PathToSVectors using explicit StrassenLikeMult.
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
