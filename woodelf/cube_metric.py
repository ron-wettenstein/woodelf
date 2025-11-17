from math import factorial


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

    def calc_metric(self, s_plus, s_minus):
        raise NotImplemented()

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


class BanzahfValues(CubeMetric):
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
