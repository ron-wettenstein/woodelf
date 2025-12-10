from typing import List
import numpy as np
import scipy
import time

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


    def present_statistics(self):
        pass


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
        self.s_computation_time = 0
        self.m_computation_time = 0

    def get_values_matrices(self, features_in_path: List[str|int]):
        """
        Apply the CubeMetric object (the v function), to create the matrixes M.
        Use the cache when possible, and update the cache with the created matrixes
        """
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

        self.m_computation_time += time.time() - start_time
        return matrixes_for_the_given_features

    def get_s_matrices(self, features_in_path: List, f: np.array, w: float):
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
            f"cache misses: {self.cache_miss}, cache used: {self.cached_used}, " +
            f"M computation time: {round(self.m_computation_time,2)} sec, " +
            f"s computation time: {round(self.s_computation_time, 2)} sec"
        )



class HighDepthPathToMatrices(PathToMatricesAbstractCls):
    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False, path_dependent: bool = False):
        super().__init__(metric, max_depth, GPU)
        self.path_dependent = path_dependent
        self.s_computation_time = 0
        self.f_prepare_time = 0

        start_time = time.time()
        self.matrices = {}
        self.np_indexing_values = {}
        self.matrices_frs_subsets = {}
        self.build_matrices()
        self.build_numpy_indexing_values()
        self.matrices_init_time = time.time() - start_time

    @classmethod
    def map_patterns_to_cube(cls, features_in_path: List):
        """
        The high depth version of MapPatternsToCube.
        Keep only rule 1 and rule 2 ro compute the diagonal.
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
                    # Drop also Rule 3 as we only want to compute the diagonal

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
            # As dl include only the diagonal, each consumer_pattern have only one matching background pattern with a cube.
            background_pattern, cube = dl[consumer_pattern].popitem()
            s_plus, s_minus = cube
            values = metric.calc_metric(s_plus, s_minus)
            values_list.append(values)
            all_feature_subsets.update(set(values.keys()))

        matrix_details = {feature_subset: [] for feature_subset in all_feature_subsets}
        for values in values_list:
            for feature_subset in all_feature_subsets:
                matrix_details[feature_subset].append(values.get(feature_subset, 0))

        matrices = {feature_subset: values for feature_subset, values in matrix_details.items()}
        return matrices

    def build_matrices(self):
        for depth in range(1, self.max_depth+1):
            dl = self.map_patterns_to_cube(list(range(depth)))
            matrices = self.build_patterns_to_values_sparse_matrix(dl, self.metric, path_length=depth)
            self.matrices_frs_subsets[depth] = list(matrices.keys())
            self.matrices[depth] = np.array([matrices[k] for k in self.matrices_frs_subsets[depth]]).T

    def build_numpy_indexing_values(self):
        for depth in range(1, self.max_depth+1):
            zero_row = 2 ** depth - 1
            idx = np.arange(2 ** depth)
            for d in range(0, depth, 1):
                shifted_indexes = np.full_like(idx, zero_row)
                shifted_indexes[2 ** d:] = idx[:-2 ** d]
                mask = (idx & (1 << d)) == 0
                shifted_indexes[mask] = zero_row
                if depth not in self.np_indexing_values:
                    self.np_indexing_values[depth] = {}
                self.np_indexing_values[depth][d] = shifted_indexes


    def get_s_matrices(self, features_in_path: List, f: np.array, w: float):
        depth = len(features_in_path)
        start_time = time.time()
        if not self.path_dependent:
            start_time_f_prepare = time.time()
            f = self.prepare_f(depth, f)
            self.f_prepare_time += time.time() - start_time_f_prepare

        matrices = self.matrices[depth]
        frs2feature_name = self.frs_subsets_to_feature_subsets(features_in_path, depth)
        s_vectors = {}
        s_matrix = matrices * f[::-1].reshape(-1, 1) # reversed as this is not the (0,0)-(1,1)-..-(n,n) diagonal but the (0,n)-(1,n-1)-..-(n,0) diagonal
        last_row = s_matrix[-1].copy()
        s_matrix[-1] = 0

        for d in range(0, depth, 1):
            s_matrix = s_matrix + s_matrix[self.np_indexing_values[depth][d]]
            last_row = last_row + s_matrix[-1].copy()
            s_matrix[-1] = 0

        s_matrix[-1] = last_row
        s_matrix *= w
        for index, frs_subset in enumerate(self.matrices_frs_subsets[depth]):
            feature_subset = frs2feature_name[frs_subset]
            s_vectors[feature_subset] = s_matrix[:,index]

        self.s_computation_time += time.time() - start_time
        return s_vectors

    def frs_subsets_to_feature_subsets(self, features_in_path: List, depth: int):
        if not self.metric.INTERACTION_VALUE:
            frs2feature_name = {index: features_in_path[index] for index in self.matrices_frs_subsets[depth]}
        else:
            frs2feature_name = {}
            for frs_subsets in self.matrices_frs_subsets[depth]:
                if not self.metric.INTERACTION_VALUES_ORDER_MATTERS:
                    feature_tuple = tuple(
                        sorted([features_in_path[feature_index] for feature_index in frs_subsets]))
                else:
                    feature_tuple = tuple([features_in_path[feature_index] for feature_index in frs_subsets])
                frs2feature_name[frs_subsets] = feature_tuple
        return frs2feature_name

    @staticmethod
    def prepare_f(depth, f):
        idx = np.arange(len(f))
        for d in range(depth - 1, -1, -1):
            f_copy = f.copy()
            f_copy[:-2 ** d] = f[2 ** d:]  # shift the array to the left 2**d bits
            f_copy[(idx & (1 << d)) != 0] = 0  # Zero all elements that are in an even place in the current division
            f = f + f_copy
        return f

    def present_statistics(self):
        print(
            f"M time: {round(self.matrices_init_time, 2)} sec, " +
            f"s time: {round(self.s_computation_time, 2)} sec (f prepare time: {self.f_prepare_time})"
        )
