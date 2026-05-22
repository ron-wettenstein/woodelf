from __future__ import annotations

import time
from typing import List, Dict

import numpy as np

from woodelf.core.path_to_s_vectors.woodelf_p2s import HighDepthWoodelfPathToSVectors


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
