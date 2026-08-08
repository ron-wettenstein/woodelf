from math import factorial
from typing import Set, Dict, Any, Optional, Tuple
from itertools import combinations


def nCk(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

class CubeMetric(object):
    """
    An abstract class that calculate a metric on a cube/clause characteristic function.
    You can implement this class (override the calc_metric function) and then use this class and the WOODELF algorithm
    to calculate your metric efficiently on large background datasets.

    Here, the metrics that inherit this class are: Shapley values, Shapley interaction values, Banzhaf values and Banzhaf interaction values
    """
    INTERACTION_VALUE = False
    INTERACTION_VALUES_ORDER_MATTERS = False
    INTERACTION_VALUES_RETURN_ALL_SUBSET_PERMUTATIONS = False

    def calc_metric(self, s_plus, s_minus) -> Dict[Any, float]:
        raise NotImplemented()

    def should_mirror(self) -> bool:
        """
        In case of pairwise interaction values (and where order does not matter) shap output include
        both (f1, f2) and (f2, f1). So we do mirroring v[(f2,f1)] = v[(f1,f2)].
        """
        return (
            self.INTERACTION_VALUE and not self.INTERACTION_VALUES_ORDER_MATTERS
            and self.INTERACTION_VALUES_RETURN_ALL_SUBSET_PERMUTATIONS
        )

class ShapleyValues(CubeMetric):
    """
    Implement the linear-time formula for Shapley value computation on WDNF/WCNF, see Formula 3 in the paper.
    """
    INTERACTION_VALUE = False

    def calc_metric(self, s_plus, s_minus):
        if len(s_plus & s_minus) > 0:
            return {} # se and sne must be disjoint sets

        s = s_plus | s_minus
        shapley_values = {}

        # The new simple shapley values formula
        if len(s_plus) > 0:
            contribution = (1 / (len(s_plus) * nCk(len(s), len(s_plus))))
            for must_exist_feature in s_plus:
                shapley_values[must_exist_feature] = contribution

        if len(s_minus) > 0:
            contribution = -1 / (len(s_minus) * nCk(len(s), len(s_minus)))
            for must_be_missing_feature in s_minus:
                shapley_values[must_be_missing_feature] = contribution

        return shapley_values

class ShapleyInteractionValues(CubeMetric):
    """
    Implement the formulas for Shapley interaction values computation on WDNF/WCNF, see Table 1 in the paper.
    """
    INTERACTION_VALUE = True
    INTERACTION_VALUES_ORDER_MATTERS = False
    INTERACTION_VALUES_RETURN_ALL_SUBSET_PERMUTATIONS = True

    def calc_metric(self, s_plus, s_minus):
        if len(s_plus & s_minus) > 0:
            return {} # se and sne must be disjoint sets

        shapley_values = {}
        s = s_plus | s_minus
        if len(s_plus) > 0:
            # i,j in S+
            if len(s_plus) > 1:
                # 0.5 because the shapley interaction values in the shap package are actually divided by 2....
                contribution = 0.5 / ((len(s_plus) - 1) * nCk(len(s) - 1, len(s_plus) - 1))
                for must_exists_feature in s_plus:
                    for other_feature in s_plus:
                        if must_exists_feature < other_feature:
                            shapley_values[(must_exists_feature, other_feature)] = contribution

            # i in S+   j in S-
            if len(s_minus) > 0:
                contribution = -0.5 / (len(s_minus) * nCk(len(s) - 1, len(s_minus)))
                for must_exists_feature in s_plus:
                    for other_feature in s_minus:
                        if must_exists_feature < other_feature:
                            shapley_values[(must_exists_feature, other_feature)] = contribution

        if len(s_minus) > 0:
            # i,j in S-
            if len(s_minus) > 1:
                contribution = 0.5 / ((len(s_minus) - 1) * nCk(len(s) - 1, len(s_minus) - 1))
                for must_be_missing_feature in s_minus:
                    for other_feature in s_minus:
                        if must_be_missing_feature < other_feature:
                            shapley_values[(must_be_missing_feature, other_feature)] = contribution
            # i in S-   j in S+
            if len(s_plus) > 0:
                contribution = -0.5 / (len(s_plus) * nCk(len(s) - 1, len(s_plus)))
                for must_be_missing_feature in s_minus:
                    for other_feature in s_plus:
                        if must_be_missing_feature < other_feature:
                            shapley_values[(must_be_missing_feature, other_feature)] = contribution
        return shapley_values


class BanzhafValues(CubeMetric):
    """
    Implement the linear-time formula for Banzhaf value computation on WDNF/WCNF, see Formula 6 in the paper.
    """
    INTERACTION_VALUE = False

    def calc_metric(self, s_plus, s_minus):
        if len(s_plus & s_minus) > 0:
            return {} # se and sne must be disjoint sets

        s = s_plus | s_minus
        banzhaf_values = {}

        s_plus_contribution = 1 / (2 ** (len(s) - 1))
        s_minus_contribution = -s_plus_contribution
        # The new simple shapley values formula
        if len(s_plus) > 0:
            for must_exist_feature in s_plus:
                banzhaf_values[must_exist_feature] = s_plus_contribution

        if len(s_minus) > 0:
            for must_be_missing_feature in s_minus:
                banzhaf_values[must_be_missing_feature] = s_minus_contribution

        return banzhaf_values


class BanzhafInteractionValues(CubeMetric):
    """
    Implement the formulas for Banzhaf interaction values computation on WDNF/WCNF, see Formula 7 in the paper.
    """
    INTERACTION_VALUE = True
    INTERACTION_VALUES_ORDER_MATTERS = False
    INTERACTION_VALUES_RETURN_ALL_SUBSET_PERMUTATIONS = True

    def calc_metric(self, s_plus, s_minus):
        if len(s_plus & s_minus) > 0:
            return {} # se and sne must be disjoint sets
        banzhaf_values = {}

        contribution = (1 / (2 ** (len(s_plus) + len(s_minus) - 2)))

        s = s_plus | s_minus
        if len(s_plus) > 0:
            # i,j in S+
            if len(s_plus) > 1:
                for must_exists_feature in s_plus:
                    for other_feature in s_plus:
                        if must_exists_feature < other_feature:
                            banzhaf_values[(must_exists_feature, other_feature)] = contribution

            # i in S+   j in S-
            if len(s_minus) > 0:
                for must_exists_feature in s_plus:
                    for other_feature in s_minus:
                        if must_exists_feature < other_feature:
                            banzhaf_values[(must_exists_feature, other_feature)] = -contribution

        if len(s_minus) > 0:
            # i,j in S-
            if len(s_minus) > 1:
                for must_be_missing_feature in s_minus:
                    for other_feature in s_minus:
                        if must_be_missing_feature < other_feature:
                            banzhaf_values[(must_be_missing_feature, other_feature)] = contribution
            # i in S-   j in S+
            if len(s_plus) > 0:
                for must_be_missing_feature in s_minus:
                    for other_feature in s_plus:
                        if must_be_missing_feature < other_feature:
                            banzhaf_values[(must_be_missing_feature, other_feature)] = -contribution
        return banzhaf_values


class CardinalityInteractionIndicesMetric(CubeMetric):
    """
    An abstract class for symmetric interaction metrics whose value on a cube depends only on
    cardinalities: the number of negative and positive literals in the cube, and how many of the
    subset's variables appear as positive/negative literals. Subsets containing a variable that does not
    appear in the cube always get 0 (such variables are dummy players of the cube's game).

    The metric reports every subset whose size k satisfies min_order <= k <= max_order, where
    max_order=None means no upper limit (so the subsets are bounded only by the cube's own variables).
    The default min_order=1, max_order=None is therefore the any-order metric.

    Implement cardinality_value to define the metric; calc_metric is derived from it generically.
    """
    INTERACTION_VALUE = True
    INTERACTION_VALUES_ORDER_MATTERS = False

    def __init__(self, min_order: int = 1, max_order: Optional[int] = None):
        assert min_order >= 1
        assert max_order is None or max_order >= min_order
        self.min_order = min_order
        self.max_order = max_order

    def orders(self, num_variables: int) -> range:
        """
        The subset sizes the metric reports over num_variables variables: min_order...max_order, capped by
        num_variables (larger subsets must contain a dummy player and are therefore always 0).
        """
        highest = num_variables if self.max_order is None else min(self.max_order, num_variables)
        return range(self.min_order, highest + 1)

    def cardinality_value(
            self, num_neg_literals: int, num_pos_literals: int, num_subset_pos: int, num_subset_neg: int
    ) -> float:
        """
        The metric value, on a cube with num_pos_literals positive and num_neg_literals negative literals,
        of a variables subset with num_subset_pos variables appearing as positive literals in the cube and
        num_subset_neg appearing as negative literals (the subset's order is num_subset_pos + num_subset_neg).
        For example the subset S = {x1, x2, x5} of the cube (x1 and x4 and not x2 and not x5) is encoded to
        (num_neg_literals=2, num_pos_literals=2, num_subset_pos=1, num_subset_neg=2).
        """
        raise NotImplemented()

    def calc_metric(self, s_plus, s_minus) -> Dict[Tuple, float]:
        if len(s_plus & s_minus) > 0:
            return {}  # se and sne must be disjoint sets

        variables = sorted(s_plus | s_minus)
        values = {}
        for order in self.orders(len(variables)):
            for subset in combinations(variables, order):
                num_subset_pos = len(s_plus.intersection(subset))
                values[subset] = self.cardinality_value(
                    len(s_minus), len(s_plus), num_subset_pos, order - num_subset_pos
                )
        return values


class GeneralShapleyInteractionValues(CardinalityInteractionIndicesMetric):
    """
    Shapley (Grabisch-Roubens) interaction values of any order k. On a cube with p positive and q negative
    literals, a subset with a positive-literal variables and b negative-literal variables gets:
        I = (-1)^b * (p-a)! * (q-b)! / (p+q-k+1)!
    """

    def __init__(self, min_order: int = 1, max_order: Optional[int] = None, shap_convention: bool = False):
        """
        @param shap_convention: If True, divide the values by order! - matching the shap package convention
        of spreading each interaction value across all permutations of the subset (for order=2 the shap
        package reports each pair twice, each holding half the value).
        """
        super().__init__(min_order, max_order)
        self.shap_convention = shap_convention

    def cardinality_value(self, num_neg_literals, num_pos_literals, num_subset_pos, num_subset_neg):
        order = num_subset_pos + num_subset_neg
        value = (
            ((-1) ** num_subset_neg)
            * factorial(num_pos_literals - num_subset_pos) * factorial(num_neg_literals - num_subset_neg)
            / factorial(num_pos_literals + num_neg_literals - order + 1)
        )
        if self.shap_convention:
            value /= factorial(order)
        return value


class GeneralBanzhafInteractionValues(CardinalityInteractionIndicesMetric):
    """
    Banzhaf interaction values of any order k. On a cube with p positive and q negative literals, a subset
    with a positive-literal variables and b negative-literal variables gets:
        I = (-1)^b / 2^(p+q-k)
    For k=1 this reduces to the BanzhafValues formulas and for k=2 to BanzhafInteractionValues.
    """

    def cardinality_value(self, num_neg_literals, num_pos_literals, num_subset_pos, num_subset_neg):
        order = num_subset_pos + num_subset_neg
        return ((-1) ** num_subset_neg) / (2 ** (num_pos_literals + num_neg_literals - order))


############################################################################################################################################################
#
#   PDPs matrices
#
############################################################################################################################################################


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


def all_subsets_of_size_0_1_2(s):
    subsets = [set()]
    for k in [1,2]:
        for subset in combinations(s, k):
            subsets.append(set(subset))
    return subsets

class PDIVOrder1Or2(CubeMetric):
    INTERACTION_VALUE = True
    INTERACTION_VALUES_RETURN_ALL_SUBSET_PERMUTATIONS = True

    def calc_metric(self, s_plus: Set, s_minus: Set) -> Dict[Tuple, float]:
        if len(s_plus & s_minus) > 0:
            return {}

        pdivs = {}
        for sm in all_subsets_of_size_0_1_2(s_minus):
            s = tuple(s_plus | sm)
            if len(s) in [1,2]:
                pdivs[s] = (-1) ** (len(sm))
        return pdivs

