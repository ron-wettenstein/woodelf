from typing import List
import numpy as np
import scipy

from woodelf.cube_metric import CubeMetric


class PathToMatricesAbstractCls:
    """
    An abstract class for creating M matrix from the features along a root-to-leaf-path.
    Also manage the matrix-vector multiplication of the s vector
    """
    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        self.metric = metric
        self.max_depth = max_depth
        self.GPU = GPU

    def get_values_matrices(self, features_in_path: List):
        raise NotImplemented

    def get_s_matrices(self, features_in_path: List, f: np.array, w: float):
        raise NotImplemented

    def dump(self, file_path: str):
        raise NotImplemented

    @classmethod
    def load(cls, file_path: str):
        raise NotImplemented

    @classmethod
    def map_patterns_to_cube(cls, features_in_path: List):
        """
        The function MapPatternsToCube from Sect. 5 of the article.
        :params tree: The decision tree
        :params current_wdnf_table: The format is: wdnf_table[consumer_decision_pattern][background_decision_pattern] = (cube_positive_literals, cube_negative_literals)
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
                    # Implement the 4 rules
                    updated_wdnf_table[consumer_pattern * 2 + 1][background_pattern * 2 + 0] = (
                    s_plus | {feature}, s_minus)  # Rule 1
                    updated_wdnf_table[consumer_pattern * 2 + 0][background_pattern * 2 + 1] = (
                    s_plus, s_minus | {feature})  # Rule 2
                    updated_wdnf_table[consumer_pattern * 2 + 1][background_pattern * 2 + 1] = (
                    s_plus, s_minus)  # Rule 3

        return updated_wdnf_table


def get_feature_repetition_sequence(features_in_path: List[str]):
    """
    Generate the feature repetition sequence.
    The math is simple, the feature at index i is replaced by i
    unless it appeared before in the sequance, in that case it will be represented by the index it already received.

    Examples:
    ["sex", "pluse", "age", "weight", "heart_rate", "sugar_in_blood"] => [1, 2, 3, 4, 5, 6]
    ["weight", "pluse", "age", "sex", "pluse", "sex"] => [1, 2, 3, 4, 2, 4]
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

class SimplePathToMatrices(PathToMatricesAbstractCls):
    """
    An object that in charge of creating the M matrix for every leaf and feature.
    It takes the features along the root-to-leaf path and build the matrix (lines 7-16 in WOODELF pseudo code)
    The class also utilize the fact that the matrix only depends on the repitting sequence of the features along the path.
    For example the feature repetition sequence of ["weight", "pluse", "age", "sex", "pluse", "sex"] is [1, 2, 3, 4, 2, 4].
    All feature lists with this feature repetition sequence have the same set of matrixes.

    This cache mechanism is improvement 2 in Sec. 9.1
    """

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)

        self.cached_used = 0
        self.cache_miss = 0
        self.cache = {}

    def get_values_matrixes(self, features_in_path: List[str|int]):
        """
        Apply the CubeMetric object (the v function), to create the matrixes M.
        Use the cache when possible, and update the cache with the created matrixes
        """
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
            matrixes_for_the_given_features = {features_in_path[index]: matrixes[index] for index in matrixes}
        else:
            matrixes_for_the_given_features = {}
            for feature_indexes, current_matrices in matrixes.items():
                if not self.metric.INTERACTION_VALUES_ORDER_MATTERS:
                    feature_tuple = tuple(
                        sorted([features_in_path[feature_index] for feature_index in feature_indexes]))
                else:
                    feature_tuple = tuple([features_in_path[feature_index] for feature_index in feature_indexes])
                matrixes_for_the_given_features[feature_tuple] = current_matrices

        return matrixes_for_the_given_features

    def get_s_matrices(self, features_in_path: List, f: np.array, w: float):
        matrices = self.get_values_matrixes(features_in_path)
        s_vectors = {}
        for feature in matrices:
            # The matrix multiplication part is implemented in CPU, the matrix is too small for the GPU overhead to be worth it.
            # The sparse matrix multiplication here instead of the naive dense matrix multiplication is improvement 1 in Sec. 9.1
            s_vectors[feature] = matrices[feature].dot(f) * w
        return s_vectors

    @classmethod
    def build_patterns_to_values_sparse_matrix(cls, dl, metric: CubeMetric, path_length):
        """
        Apply the CubeMetric object (the v function), to create the matrices M.
        include lines 12-16 in WOODELF pseudocode.
        dl is the returned mapping from the map_patterns_to_cube function
        """
        matrix_details = {}
        for pc in dl:
            for pb in dl[pc]:
                s_plus, s_minus = dl[pc][pb]
                values = metric.calc_metric(s_plus, s_minus)
                for feature in values:
                    # Implement the line "M[l][feature][p_c][p_b] = value" in an efficient way that utilize the sparsity of M.
                    if feature not in matrix_details:
                        matrix_details[feature] = {"pcs": [], "pbs": [], "values": []}
                    matrix_details[feature]["pcs"].append(pc)
                    matrix_details[feature]["pbs"].append(pb)
                    matrix_details[feature]["values"].append(values[feature])

        matrixs = {}
        for feature in matrix_details:
            # Save M as a sparse matrix (Improvement 1 in Sec. 9.1)
            matrix_values = (
            matrix_details[feature]["values"], (matrix_details[feature]["pcs"], matrix_details[feature]["pbs"]))
            matrixs[feature] = scipy.sparse.coo_matrix(matrix_values, shape=(2 ** path_length, 2 ** path_length),
                                                       dtype=np.float32).tocsc()
        return matrixs
