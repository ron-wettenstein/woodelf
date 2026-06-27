import math
import time
from typing import Optional

import numpy as np

from typing import List

from woodelf.core.cube_metric import ShapleyValues, CubeMetric
from woodelf.core.path_to_s_vectors.archive.lts_polynomial_multiplication import (
    quadrature_tree_shap_batched_approach, quadrature_tree_shap_batched_approach_for_neighbors,
    ABS_STRATEGY_NORMAL, ABS_STRATEGY_BANZHAF_CURVE_ENSEMBLE, ABS_STRATEGY_LEAVES, ABS_STRATEGIES,
)
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import AlwaysParticipatingLTSPathToSVectors, LTSRecursivePathToSVectors


class QuadratureSHAPPathToSVectors(LTSRecursivePathToSVectors):

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False, abs_strategy: str = ABS_STRATEGY_NORMAL):
        super().__init__(metric, max_depth, GPU)
        assert abs_strategy in ABS_STRATEGIES, f"abs_strategy must be one of {ABS_STRATEGIES}, got {abs_strategy!r}"
        assert abs_strategy != ABS_STRATEGY_LEAVES, "abs_leaves is computed on the exact LTS path, not via QuadratureSHAPPathToSVectors"
        self.abs_strategy = abs_strategy
        self.quad_nodes, self.quad_weights = self.compute_quads()
        # The ensemble strategy sums per-node curves across leaves, so every leaf must use the SAME
        # Gauss-Legendre nodes. Fix one node count for the whole ensemble (the max, based on max_depth).
        self._fixed_n_quad = min(max(int(math.ceil(self.max_depth / 2)), 2), 16)

    def compute_quads(self):
        max_required_quads = min(max(int(math.ceil(self.max_depth / 2)), 2), 16)
        quad_nodes = {}
        quad_weights = {}
        for n_quad in range(2, max_required_quads + 1):
            _nodes, _weights = np.polynomial.legendre.leggauss(n_quad)
            quad_nodes[n_quad] = np.float64(0.5) * (_nodes.astype(np.float64) + np.float64(1.0))  # (n_quad,)
            quad_weights[n_quad] = np.float64(0.5) * _weights.astype(np.float64)  # (n_quad,)
        return quad_nodes, quad_weights

    def get_ensemble_quad_weights(self):
        """Gauss-Legendre weights for the fixed node set shared across all leaves (abs_banzhaf_curve)."""
        return self.quad_weights[self._fixed_n_quad]

    def _get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        assert self.abs_strategy != ABS_STRATEGY_NORMAL or isinstance(self.metric, ShapleyValues), "Banzhaf and interaction values are not supported in this PathToSVectors class"
        start_time = time.time()
        if self.abs_strategy == ABS_STRATEGY_BANZHAF_CURVE_ENSEMBLE:
            # All leaves must share the same nodes so their per-node curves can be summed across the ensemble.
            n_quads = self._fixed_n_quad
        else:
            # For D<4 use 2, for D > 32 use 16 for 4<=D<=32 use 0.5*D
            n_quads = min(max(int(math.ceil(len(covers) / 2)), 2), 16)
        if w_neighbor is None:
            s_matrix = quadrature_tree_shap_batched_approach(covers, consumer_patterns, w, self.quad_nodes[n_quads], self.quad_weights[n_quads], self.abs_strategy)
        else:
            s_matrix = quadrature_tree_shap_batched_approach_for_neighbors(covers, consumer_patterns, w, w_neighbor, self.quad_nodes[n_quads], self.quad_weights[n_quads], self.abs_strategy)
        self.computation_time += time.time() - start_time
        return s_matrix


class AlwaysParticipatingQuadratureSHAPPathToSVectors(AlwaysParticipatingLTSPathToSVectors):
    """
    Always-participating logic with Gauss-Legendre quadrature as the underlying computation.
    Delegates _get_s_matrix to a QuadratureSHAPPathToSVectors instance.
    """

    def __init__(self, metric: CubeMetric, max_depth: int, always_participating: List[str],
                 GPU: bool = False, abs_strategy: str = ABS_STRATEGY_NORMAL):
        super().__init__(metric, max_depth, always_participating, GPU)
        self._quadrature = QuadratureSHAPPathToSVectors(metric, max_depth, GPU, abs_strategy)

    def _get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        return self._quadrature._get_s_matrix(covers, consumer_patterns, w, w_neighbor)
