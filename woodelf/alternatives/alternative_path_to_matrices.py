from typing import List
import numpy as np
import scipy
import time

from woodelf.cube_metric import CubeMetric
from woodelf.path_to_matrices import PathToMatricesAbstractCls


class HighDepthPathToMatricesNumpyIndexingNot2DVectorized(PathToMatricesAbstractCls):
    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False, path_dependent: bool = False):
        super().__init__(metric, max_depth, GPU)
        self.path_dependent = path_dependent
        self.s_computation_time = 0
        self.f_prepare_time = 0

        start_time = time.time()
        self.matrices = {}
        self.np_indexing_values = {}
        self.build_matrices()
        self.build_numpy_indexing_values()
        self.matrices_init_time = time.time() - start_time
        self.timings = {k: 0 for k in ["m*f", "remove last row and set it to zero", "the s line", "return last row", "s * w", "split s to vectors"]}

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
            self.matrices[depth] = self.build_patterns_to_values_sparse_matrix(dl, self.metric, path_length=depth)

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

        frs2feature_name = self.frs_subsets_to_feature_subsets(features_in_path, depth)
        s_vectors = {}
        for frs_subset, matrix in self.matrices[depth].items():
            st = time.time()
            s = matrix * f[::-1] # reversed as this is not the (0,0)-(1,1)-..-(n,n) diagonal but the (0,n)-(1,n-1)-..-(n,0) diagonal
            self.timings["m*f"] += time.time() - st
            st = time.time()
            last_element = s[-1]
            s[-1] = 0
            self.timings["remove last row and set it to zero"] += time.time() - st

            for d in range(0, depth, 1):
                st = time.time()
                s = s + s[self.np_indexing_values[depth][d]]
                self.timings["the s line"] += time.time() - st
                st = time.time()
                last_element = last_element + s[-1]
                s[-1] = 0
                self.timings["remove last row and set it to zero"] += time.time() - st


            st = time.time()
            s[-1] = last_element
            self.timings["return last row"] += time.time() - st

            st = time.time()
            s *= w
            self.timings["s * w"] += time.time() - st
            st = time.time()
            feature_subset = frs2feature_name[frs_subset]
            s_vectors[feature_subset] = s
            self.timings["split s to vectors"] += time.time() - st

        self.s_computation_time += time.time() - start_time
        return s_vectors

    def frs_subsets_to_feature_subsets(self, features_in_path: List, depth: int):
        if not self.metric.INTERACTION_VALUE:
            frs2feature_name = {index: features_in_path[index] for index in self.matrices[depth].keys()}
        else:
            frs2feature_name = {}
            for frs_subsets in self.matrices[depth].keys():
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
        for k in self.timings:
            print(f"{k}: {round(self.timings[k], 2)} sec")


class HighDepthPathToMatricesWithoutNumpyIndexing(PathToMatricesAbstractCls):
    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False, path_dependent: bool = False):
        super().__init__(metric, max_depth, GPU)
        self.path_dependent = path_dependent
        self.s_computation_time = 0
        self.f_prepare_time = 0

        start_time = time.time()
        self.matrices = {}
        self.matrices_frs_subsets = {}
        self.build_matrices()
        self.matrices_init_time = time.time() - start_time
        self.timings = {k: 0 for k in ["m*f", "s line 1", "s line 2", "s line 3", "s line 4", "s * w", "split s to vectors", "prepare mask"]}

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
        idx = np.arange(len(f))
        st = time.time()
        s_matrix = matrices * f[::-1].reshape(-1, 1) # reversed as this is not the (0,0)-(1,1)-..-(n,n) diagonal but the (0,n)-(1,n-1)-..-(n,0) diagonal
        self.timings["m*f"] += time.time() - st
        for d in range(0, depth, 1):
            st = time.time()
            s_matrix_copy = s_matrix.copy()
            self.timings["s line 1"] += time.time() - st
            st = time.time()
            s_matrix_copy[2 ** d:, :] = s_matrix[:-2 ** d, :]  # shift the array to the left 2**d bits
            self.timings["s line 2"] += time.time() - st
            st = time.time()
            mask = (idx & (1 << d)) == 0
            self.timings["prepare mask"] += time.time() - st
            st = time.time()
            s_matrix_copy[mask] = 0  # Zero all elements that are in an even place in the current division
            self.timings["s line 3"] += time.time() - st
            st = time.time()
            s_matrix = s_matrix + s_matrix_copy
            self.timings["s line 4"] += time.time() - st

        st = time.time()
        s_matrix *= w
        self.timings["s * w"] += time.time() - st
        st = time.time()
        for index, frs_subset in enumerate(self.matrices_frs_subsets[depth]):
            feature_subset = frs2feature_name[frs_subset]
            s_vectors[feature_subset] = s_matrix[:,index]
        self.timings["split s to vectors"] += time.time() - st

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
        for k in self.timings:
            print(f"{k}: {round(self.timings[k], 2)} sec")