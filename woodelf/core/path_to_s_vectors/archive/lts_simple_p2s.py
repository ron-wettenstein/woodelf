import time
from typing import Optional

import numpy as np

from woodelf.core.cube_metric import ShapleyValues
from woodelf.core.path_to_s_vectors.archive.lts_polynomial_multiplication import (
    linear_tree_shap_magic
)
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import LTSRecursivePathToSVectors


class LTSSimplePathToSVectors(LTSRecursivePathToSVectors):

    def poly_mult_shap_func(self, covers: np.array, consumer_patterns: np.array, f_w: np.array, w: float):
        raise NotImplemented()

    def _get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        start_time = time.time()
        assert isinstance(self.metric, ShapleyValues), "Banzhaf and interaction values are not supported in this PathToSVectors class"
        # assume features in path are unique
        f_w = self.f_ws[len(covers)]
        if w_neighbor is None:
            s_matrix = linear_tree_shap_magic(covers, consumer_patterns, f_w, w)
        else:
            s_matrix_left = linear_tree_shap_magic(covers, consumer_patterns, f_w, w)
            covers_of_right = np.array(list(covers[:-1]) + [1 - covers[-1]])
            consumer_patterns_right = consumer_patterns.copy()
            consumer_patterns_right[consumer_patterns % 2 == 0] += 1
            consumer_patterns_right[consumer_patterns % 2 == 1] -= 1
            s_matrix_right = linear_tree_shap_magic(
                covers_of_right, consumer_patterns_right.astype(np.uint64), f_w, w_neighbor
            )
            s_matrix = s_matrix_left + s_matrix_right
        self.computation_time += time.time() - start_time
        return s_matrix