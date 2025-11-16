from typing import List, Set, Dict

from woodelf.cube_metric import ShapleyValues, CubeMetric
from woodelf.path_to_matrices import SimplePathToMatrices


class PathToValuesMatrixLimitSPlus(SimplePathToMatrices):
    # Ignored all cubes with |S^+| > MAX_S_PLUS_SIZE to reduce the complexity element of TL3**D to TL2**D*(D**MAX_S_PLUS_SIZE)
    MAX_S_PLUS_SIZE = NotImplemented

    @classmethod
    def map_patterns_to_cube(cls, features_in_path: List[str]):
        updated_wdnf_table = {0: {0: (set(), set())}}
        current_wdnf_table = None
        for feature in features_in_path:
            current_wdnf_table = updated_wdnf_table
            updated_wdnf_table = {}
            for consumer_pattern in current_wdnf_table:
                updated_wdnf_table[consumer_pattern * 2 + 0] = {}
                updated_wdnf_table[consumer_pattern * 2 + 1] = {}
                for background_pattern in current_wdnf_table[consumer_pattern]:
                    s_plus, s_minus = current_wdnf_table[consumer_pattern][background_pattern]

                    # The implementation is identical to the PathToValuesMatrix.map_patterns_to_cube implementation, except for this if.
                    if len(s_plus | {feature}) <= cls.MAX_S_PLUS_SIZE:
                        updated_wdnf_table[consumer_pattern * 2 + 1][background_pattern * 2 + 0] = (s_plus | {feature}, s_minus) # Rule 1

                    updated_wdnf_table[consumer_pattern * 2 + 0][background_pattern * 2 + 1] = (s_plus, s_minus | {feature}) # Rule 2
                    updated_wdnf_table[consumer_pattern * 2 + 1][background_pattern * 2 + 1] = (s_plus, s_minus) # Rule 3

        return updated_wdnf_table

class PathToValuesMatrixLimitSPlusTo1(PathToValuesMatrixLimitSPlus):
    # We uses the fact CPDVMetric ignored all cubes with |S^+| > 1 to reduce the complexity element of TL3**D to TL2**D*D
    MAX_S_PLUS_SIZE = 1

class PathToValuesMatrixLimitSPlusTo2(PathToValuesMatrixLimitSPlus):
    # We uses the fact PDIVOrder1Or2 ignored all cubes with |S^+| > 2 to reduce the complexity element of TL3**D to TL(2**D)*(D**2)
    MAX_S_PLUS_SIZE = 2

class CPDVMetric(CubeMetric):
    def calc_metric(self, s_plus: Set, s_minus: Set) -> Dict[str, float]:
        if len(s_plus & s_minus) > 0:
            return {}
        pdp_values = {}
        if len(s_plus) == 1:
            for f in s_plus:
                pdp_values[f] = 1
        if len(s_plus) == 0:
            for f in s_minus:
                pdp_values[f] = -1
        return pdp_values


def test_wdnf_matrix_size():
    p2v = PathToValuesMatrixLimitSPlusTo1(CPDVMetric(), max_depth=16)
    for depth in range(1,17):
        wdnf_table = p2v.map_patterns_to_cube([chr(ord('a') + i) for i in range(depth)])
        table_size = sum([len(wdnf_table[c_patterns]) for c_patterns in wdnf_table])
        assert table_size == (depth+2)*2**(depth-1)
        print(depth, table_size, (depth+2)*2**(depth-1) )

def test_metric_matrix_size():
    p2v = PathToValuesMatrixLimitSPlusTo1(CPDVMetric(), max_depth=16)
    for depth in range(1,17):
        matrices = p2v.get_values_matrices([chr(ord('a') + i) for i in range(depth)])
        size = sum([m.nnz for m in matrices.values()])
        assert size == (2 ** depth) * depth
        print(depth, size, (2 ** depth) * depth )
