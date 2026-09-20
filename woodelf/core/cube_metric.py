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


class MobiusCoefficients(CardinalityInteractionIndicesMetric):
    """
    Mobius coefficients of order k. Also referred to as Partial Dependence interaction values. 
    On a cube with p positive and q negative literals, a subset
    with a positive-literal variables and b negative-literal variables simply gets:
        I = (-1)^b / 2^(p+q-k)
    For k=1 this reduces to PDV. Equivalent to the class PDIVMetric.
    """

    def cardinality_value(self, num_neg_literals, num_pos_literals, num_subset_pos, num_subset_neg):
        if num_pos_literals != num_subset_pos:
            return 0
        return (-1) ** num_subset_neg

class ArbitraryOrderPDIV(MobiusCoefficients):
    """ These are the same as mathematically as the MobiusCoefficients."""


class FaithfulInteractionIndicesMetric(CardinalityInteractionIndicesMetric):
    """
    Faithful interaction indices (Tsai et al.): unlike the discrete-derivative indices above, these are the
    weighted least-squares fit of a max_order-additive surrogate of the game, under the Shapley kernel
    (FSII) or the Banzhaf kernel (FBII). max_order is therefore the surrogate's order and must be set.

    We use the formulation derived at appendices A.3.4 and A.3.5 in the
    "Proxy-Based Approximation of Shapley and Banzhaf Interactions" paper: https://arxiv.org/pdf/2605.22738

    Propositions A.12 (FBII) and A.13 (FSII) give both indices the same shape on a leaf's interval game,
    the cube here, whose positive literals are the paper's R and whose negative literals are its L:
        lambda(l, r, u, s) = (-1)^u 1[R subset of S]
                             + sum_{i=max(0, k-r-u+1)}^{l-u} (-1)^(u+i+k-s) C(l-u, i) tail_weight(t, s)
    written with
        l = |L|,   r = |R|,   u = |S cap L|,   s = |S|,   k = max_order,   t = r + u + i
    The leading term is the paper's Mobius term and the sum is its faithful tail term, the two parts its
    proofs derive separately. Implement tail_weight to pick the index.
    """

    def __init__(self, min_order: int = 1, max_order: Optional[int] = None):
        assert max_order is not None, (
            f"{type(self).__name__} fits a max_order-additive surrogate of the game, so max_order must be "
            f"set (it is the order of the surrogate, not only a bound on the reported subsets)."
        )
        super().__init__(min_order, max_order)

    def tail_weight(self, t: int, s: int) -> float:
        """
        The index specific factor of the faithful tail term, for a subset of size s and t = r + u + i.
        The (-1)^(u+i+k-s) sign and the C(l-u, i) multiplicity are shared by both indices, so
        cardinality_value applies them and only this factor is left to the subclasses.
        """
        raise NotImplemented()

    def cardinality_value(self, num_neg_literals, num_pos_literals, num_subset_pos, num_subset_neg):
        # The symbols of Propositions A.12 and A.13, see the class docstring
        l, r, u, s, k = num_neg_literals, num_pos_literals, num_subset_neg, num_subset_pos + num_subset_neg, self.max_order

        # The Mobius term. R is a subset of S exactly when the subset holds every positive literal.
        value = float((-1) ** u) if num_subset_pos == r else 0.0

        # The faithful tail term.
        for i in range(max(0, k - r - u + 1), l - u + 1):
            value += ((-1) ** (u + i + k - s)) * nCk(l - u, i) * self.tail_weight(r + u + i, s)
        return value


class FaithfulShapleyInteractionValues(FaithfulInteractionIndicesMetric):
    """
    Faithful Shapley Interaction Index (FSII), the closed form of Proposition A.13. Its factor of the
    faithful tail term is
        s / (k + s) * C(k, s) * C(t - 1, k) / C(t + k - 1, k + s)
    """

    def tail_weight(self, t: int, s: int) -> float:
        k = self.max_order
        return (s / (k + s)) * nCk(k, s) * nCk(t - 1, k) / nCk(t + k - 1, k + s)


class FaithfulBanzhafInteractionValues(FaithfulInteractionIndicesMetric):
    """
    Faithful Banzhaf Interaction Index (FBII), the closed form of Proposition A.12. Its factor of the
    faithful tail term is
        (1/2)^(t - s) * C(t - s - 1, k - s)
    """

    def tail_weight(self, t: int, s: int) -> float:
        return (0.5 ** (t - s)) * nCk(t - s - 1, self.max_order - s)


class ShapleyTaylorInteractionValues(FaithfulInteractionIndicesMetric):
    """
    Shapley-Taylor Interaction Index (STII, Sundararajan et al. 2020), the game theoretic analogue of a
    Taylor expansion truncated at max_order. It is not a least squares fit like the two indices above, but
    it has the same Mobius term plus faithful tail term shape, so it reuses their closed form.

    On the Mobius transform of the game, STII passes every coefficient m_T with |T| <= max_order untouched
    to S = T, and splits every m_T with |T| > max_order uniformly across the C(|T|, max_order) subsets of T
    of the top order. So a subset below the top order keeps the Mobius term alone - which is exactly its
    discrete derivative at the empty coalition - while a top order subset also collects a share
    1 / C(t, max_order) of every higher order coefficient, which is what makes STII efficient:
    its values over the orders 1..max_order sum to v(N) - v(empty set).
    """

    def tail_weight(self, t: int, s: int) -> float:
        if s < self.max_order:
            return 0.0
        return 1.0 / nCk(t, self.max_order)


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

def all_subsets_up_to_k(s, k):
    subsets = []
    for i in range(min(k, len(s)) + 1):
        for subset in combinations(s, i):
            subsets.append(set(subset))
    return subsets

class PDIVOrder1Or2(CubeMetric):
    INTERACTION_VALUE = True
    INTERACTION_VALUES_RETURN_ALL_SUBSET_PERMUTATIONS = True

    def calc_metric(self, s_plus: Set, s_minus: Set) -> Dict[Tuple, float]:
        if len(s_plus & s_minus) > 0:
            return {}

        pdivs = {}
        for sm in all_subsets_up_to_k(s_minus, 2):
            s = tuple(s_plus | sm)
            if len(s) in [1,2]:
                pdivs[s] = (-1) ** (len(sm))
        return pdivs

class PDIVMetric(CubeMetric):
    INTERACTION_VALUE = True
    INTERACTION_VALUES_RETURN_ALL_SUBSET_PERMUTATIONS = False

    def calc_metric(self, s_plus: Set, s_minus: Set) -> Dict[Tuple, float]:
        if len(s_plus & s_minus) > 0:
            return {}

        pdivs = {}
        for sm in all_subsets_up_to_k(s_minus, len(s_minus)):
            s = tuple(s_plus | sm)
            pdivs[s] = (-1) ** (len(sm))
        return pdivs