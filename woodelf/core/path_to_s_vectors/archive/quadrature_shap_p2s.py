import math
import time
from typing import Optional

import numpy as np

from woodelf.core.cube_metric import ShapleyValues, CubeMetric
from woodelf.core.path_to_s_vectors.archive.lts_polynomial_multiplication import (
    quadrature_tree_shap_batched_approach, quadrature_tree_shap_batched_approach_for_neighbors
)
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import LTSRecursivePathToSVectors


class QuadratureSHAPPathToSVectors(LTSRecursivePathToSVectors):

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        self.quad_nodes, self.quad_weights = self.compute_quads()

    def compute_quads(self):
        max_required_quads = min(max(int(math.ceil(self.max_depth / 2)), 2), 16)
        quad_nodes = {}
        quad_weights = {}
        for n_quad in range(2, max_required_quads + 1):
            _nodes, _weights = np.polynomial.legendre.leggauss(n_quad)
            quad_nodes[n_quad] = np.float64(0.5) * (_nodes.astype(np.float64) + np.float64(1.0))  # (n_quad,)
            quad_weights[n_quad] = np.float64(0.5) * _weights.astype(np.float64)  # (n_quad,)
        return quad_nodes, quad_weights

    def _get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        assert isinstance(self.metric, ShapleyValues), "Banzhaf and interaction values are not supported in this PathToSVectors class"
        start_time = time.time()
        # For D<4 use 2, for D > 32 use 16 for 4<=D<=32 use 0.5*D
        n_quads = min(max(int(math.ceil(len(covers) / 2)), 2), 16)
        if w_neighbor is None:
            s_matrix = quadrature_tree_shap_batched_approach(covers, consumer_patterns, w, self.quad_nodes[n_quads], self.quad_weights[n_quads])
        else:
            s_matrix = quadrature_tree_shap_batched_approach_for_neighbors(covers, consumer_patterns, w, w_neighbor, self.quad_nodes[n_quads], self.quad_weights[n_quads])
        self.computation_time += time.time() - start_time
        return s_matrix
