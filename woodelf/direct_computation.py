import itertools
from math import factorial
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from woodelf.cube_metric import CubeMetric
from woodelf.decision_trees_ensemble import DecisionTreesEnsemble


class PBFunction:
    def assign(self, assignment):
        raise NotImplemented

    def variables(self):
        raise NotImplemented

class Cube(PBFunction):
    def __init__(self, sp: List[int], sm: List[int]):
        self.sp = sp # S+: positive literals
        self.sm = sm # S-: negative literals

    def assign(self, assignment: List[bool]) -> bool:
        result = True
        for l in self.sp:
            if not assignment[l]:
                result = False
        for l in self.sm:
            if assignment[l]:
                result = False
        return result

    def variables(self) -> List[int]:
        return list(set(self.sp + self.sm))


class WDNF(PBFunction):
    def __init__(self, cubes_and_weights: List[Tuple[float, Cube]]):
        self.cubes_and_weights = cubes_and_weights

    def variables(self) -> List[int]:
        variables = []
        for (weight, cube) in self.cubes_and_weights:
            variables.extend(cube.sp)
            variables.extend(cube.sm)
        return list(set(variables))

    def assign(self, assignment: List[bool]) -> float:
        result = 0
        for weight, cube in self.cubes_and_weights:
            result += weight * cube.assign(assignment)
        return result

    def calc_metric(self, metric: CubeMetric) -> Dict[Any, float]:
        values = {}
        for weight, cube in self.cubes_and_weights:
            cube_values = metric.calc_metric(set(cube.sp), set(cube.sm))
            weighted_cube_values = {k: v*weight for k, v in cube_values.items()}
            for k, v in weighted_cube_values.items():
                if k not in values:
                    values[k] = v
                else:
                    values[k] += v
        return values

    def extend(self, wdnf):
        self.cubes_and_weights.extend(wdnf.cubes_and_weights)


class BackgroundModelCF(PBFunction):
    def __init__(self, model, row: Dict[Any, float], background_data: pd.DataFrame):
        self.model = model
        self.row = row
        self.background_data = background_data
        for col in row:
            assert col in self.background_data.columns

    def variables(self):
        return list(self.row.keys())

    def assign(self, assignment: Dict[Any, bool]) -> float:
        data = self.background_data.copy()
        for feature in assignment:
            if assignment[feature]:
                data[feature] = self.row[feature]

        return self.mean_model_prediction(data)

    def mean_model_prediction(self, data: pd.DataFrame) -> float:
        return self.model.predict(data).mean()


class PathDependentModelCF(PBFunction):
    def __init__(self, model: DecisionTreesEnsemble, row: Dict[Any, float]):
        self.model = model
        self.row = row

    def variables(self):
        return list(self.row.keys())

    def assign(self, assignment: Dict[Any, bool]) -> float:
        score = 0
        for tree in self.model.trees:
            nodes_to_visit = [(1, tree)] # include pairs of (weight, node)
            while len(nodes_to_visit) > 0:
                weight, node = nodes_to_visit.pop(0)
                node_participates = assignment[node.feature_name]
                if node.is_leaf():
                    score += weight * node.value
                else:
                    row_v = self.row[node.feature_name]
                    shall_go_left = (row_v < node.value) or (node.nan_go_left and np.isnan(row_v))
                    if node.left is not None:
                        if node_participates:
                            if shall_go_left:
                                nodes_to_visit.append((weight, node.left))
                        else:
                            nodes_to_visit.append( (weight * (node.left.cover / node.cover), node.left) )
                    if node.right is not None:
                        if node_participates:
                            if not shall_go_left:
                                nodes_to_visit.append((weight, node.right))
                        else:
                            nodes_to_visit.append((weight * (node.right.cover / node.cover), node.right))

        return score


class DirectComputation:
    def compute(self, pb_function: PBFunction) -> Dict[Any, float]:
        raise NotImplemented

    def assignment_weight(self, assignment, variables) -> float:
        """
        Compute the weight of the current subset/assignment
        The given assignment explodes the variable/variables we are currently computing their value
        """
        raise NotImplemented

class GameTheoryMetricDirectComputation(DirectComputation):
    def compute(self, pb_function: PBFunction) -> Dict[Any, float]:
        variables = pb_function.variables()
        values = {}
        for variable in variables:
            values[variable] = 0
            other_vs = [v for v in variables if v != variable]
            for truth_values in itertools.product([True, False], repeat=len(other_vs)):
                assignment = {v: t for v, t in zip(other_vs, truth_values)}
                assignment_participate = assignment.copy()
                assignment_missing = assignment.copy()
                assignment_participate[variable] = True
                assignment_missing[variable] = False
                values[variable] += self.assignment_weight(assignment, variables) * (
                        pb_function.assign(assignment_participate) - pb_function.assign(assignment_missing)
                )
        return values

class BanzhafDirectComputation(GameTheoryMetricDirectComputation):
    def assignment_weight(self, assignment, variables) -> float:
        return 1 / 2 ** len(assignment)

class ShapleyDirectComputation(GameTheoryMetricDirectComputation):
    def assignment_weight(self, assignment, variables) -> float:
        n = len(variables)
        s_size = sum(assignment.values())
        return (factorial(s_size) * factorial(n - s_size - 1)) / factorial(n)

class GameTheoryIVMetricDirectComputation(DirectComputation):
    def compute(self, pb_function: PBFunction) -> Dict[Any, float]:
        variables = pb_function.variables()
        iv_values = {}
        for variable1 in variables:
            for variable2 in variables:
                if variable1 == variable2:
                    continue
                iv_values[(variable1, variable2)] = 0
                other_vs = [v for v in variables if v != variable1 and v != variable2]
                for truth_values in itertools.product([True, False], repeat=len(other_vs)):
                    assignment = {v: t for v, t in zip(other_vs, truth_values)}
                    a1p2p = assignment.copy() # Assignment player1 Participates, player2 Participates: a1p2p
                    a1p2p[variable1] = True
                    a1p2p[variable2] = True
                    a1m2p = assignment.copy() # Assignment player1 is Missing, player2 Participates: a1m2p
                    a1m2p[variable1] = False
                    a1m2p[variable2] = True
                    a1p2m = assignment.copy() # Assignment player1 Participates, player2 is Missing: a1p2m
                    a1p2m[variable1] = True
                    a1p2m[variable2] = False
                    a1m2m = assignment.copy() # Assignment player1 is Missing, player2 is Missing: a1m2m
                    a1m2m[variable1] = False
                    a1m2m[variable2] = False
                    iv_values[(variable1, variable2)] += self.assignment_weight(assignment, variables) * (
                            pb_function.assign(a1p2p) - pb_function.assign(a1m2p) - pb_function.assign(a1p2m)
                            + pb_function.assign(a1m2m)
                    )
        return iv_values

class BanzhafIVDirectComputation(GameTheoryIVMetricDirectComputation):
    def assignment_weight(self, assignment, variables) -> float:
        return 1 / 2 ** len(assignment)

class ShapleyIVDirectComputation(GameTheoryIVMetricDirectComputation):
    def assignment_weight(self, assignment, variables) -> float:
        n = len(variables)
        s_size = sum(assignment.values())
        return (factorial(s_size) * factorial(n - s_size - 2)) / (2 * factorial(n - 1))
