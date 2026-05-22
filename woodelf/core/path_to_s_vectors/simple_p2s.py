from __future__ import annotations

import time
from typing import List, Dict

import numpy as np

from woodelf.core.cube_metric import CubeMetric
from woodelf.core.path_to_s_vectors.base_p2s import WoodelfPathToSVectors


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
