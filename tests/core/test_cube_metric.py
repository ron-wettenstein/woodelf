import pytest

from woodelf.core.cube_metric import (
    BanzhafInteractionValues, BanzhafValues, CubeMetric, ShapleyInteractionValues, ShapleyValues,
    GeneralShapleyInteractionValues, GeneralBanzhafInteractionValues, MobiusCoefficients, CPDVMetric, PDIVMetric
)
from woodelf.core.direct_computation import (
    BanzhafDirectComputation, BanzhafIVDirectComputation, ShapleyIVDirectComputation, ShapleyDirectComputation,
    BanzhafCIIDirectComputation, ShapleyCIIDirectComputation, MobiusCIIDirectComputation, DirectComputation
)
from tests.core.wdnfs import ALL_WDNFs

TOLERANCE = 1e-7

# The any-order metrics and their matching direct computations. Order 1 and 2 overlap with the metrics
# above (see test_general_metric_of_order_1_matches_the_order_1_metrics and its order 2 counterpart),
# order 3 exercises the general formulas beyond the orders the dedicated classes support.
GENERAL_METRICS_AND_DIRECT_COMPUTATIONS = [
    (GeneralShapleyInteractionValues(order, order), ShapleyCIIDirectComputation(order)) for order in [1, 2, 3]
] + [
    (GeneralBanzhafInteractionValues(order, order), BanzhafCIIDirectComputation(order)) for order in [1, 2, 3]
] + [
    (MobiusCoefficients(order, order), MobiusCIIDirectComputation(order)) for order in [1, 2, 3]
]
GENERAL_METRICS_IDS = [
    f"General{name}InteractionValues_order_{order}"
    for name in ["Shapley", "Banzhaf", "Mobius"] for order in [1, 2, 3]
]

ALL_METRICS_AND_DIRECT_COMPUTATIONS = [
    (BanzhafValues(), BanzhafDirectComputation()),
    (BanzhafInteractionValues(), BanzhafIVDirectComputation()),
    (ShapleyInteractionValues(), ShapleyIVDirectComputation()),
    (ShapleyValues(), ShapleyDirectComputation()),
] + GENERAL_METRICS_AND_DIRECT_COMPUTATIONS
ALL_METRICS_IDS = [
    "BanzhafValues", "BanzhafInteractionValues", "ShapleyInteractionValues", "ShapleyValues",
] + GENERAL_METRICS_IDS


def assert_same_values(values, other_values):
    assert set(values) == set(other_values)
    for key in values:
        assert abs(values[key] - other_values[key]) < TOLERANCE

@pytest.mark.parametrize("metric, direct_computation", ALL_METRICS_AND_DIRECT_COMPUTATIONS,
                         ids=ALL_METRICS_IDS)
def test_metric(metric: CubeMetric, direct_computation: DirectComputation):
    for wdnf in ALL_WDNFs:
        values_using_metric = wdnf.calc_metric(metric)
        values_using_direct_computation = direct_computation.compute(wdnf)
        for v in values_using_metric:
            assert abs(values_using_metric[v] - values_using_direct_computation[v]) < TOLERANCE


@pytest.mark.parametrize("order_1_metric, general_metric", [
    (ShapleyValues(), GeneralShapleyInteractionValues(1, 1)),
    (BanzhafValues(), GeneralBanzhafInteractionValues(1, 1)),
], ids=["ShapleyValues", "BanzhafValues"])
def test_general_metric_of_order_1_matches_the_order_1_metrics(order_1_metric, general_metric):
    for wdnf in ALL_WDNFs:
        values = wdnf.calc_metric(order_1_metric)
        general_values = wdnf.calc_metric(general_metric)
        assert_same_values(general_values, {(k,): v for k, v in values.items()})


@pytest.mark.parametrize("order_2_metric, general_metric", [
    (ShapleyInteractionValues(), GeneralShapleyInteractionValues(2, 2, shap_convention=True)),
    (BanzhafInteractionValues(), GeneralBanzhafInteractionValues(2, 2)),
], ids=["ShapleyInteractionValues", "BanzhafInteractionValues"])
def test_general_metric_of_order_2_matches_the_interaction_values_metrics(order_2_metric, general_metric):
    # The shap_convention flag is what makes the general Shapley interaction values comparable to
    # ShapleyInteractionValues, which halves every pair as the shap package spreads it over both orderings.
    for wdnf in ALL_WDNFs:
        assert_same_values(wdnf.calc_metric(general_metric), wdnf.calc_metric(order_2_metric))


def order_k_values(metric, wdnf, order):
    """
    The non zero order k values a metric gives a wdnf, keyed by a sorted tuple. The canonicalization is
    what makes CPDVMetric, PDIVMetric and MobiusCoefficients comparable: CPDVMetric keys a feature by
    itself rather than by a 1-tuple, PDIVMetric keys a subset by the set iteration order rather than by a
    sorted tuple and reports all the orders at once, and MobiusCoefficients reports every subset of the
    cube's variables while CPDVMetric and PDIVMetric report only the subsets they give a non zero value.
    """
    values = {}
    for key, value in wdnf.calc_metric(metric).items():
        key = tuple(sorted(key)) if isinstance(key, tuple) else (key,)
        if len(key) == order:
            values[key] = values.get(key, 0) + value
    return {key: value for key, value in values.items() if abs(value) > TOLERANCE}


def test_cpdv_pdiv_and_the_mobius_coefficients_are_the_same_metric_at_order_1():
    for wdnf in ALL_WDNFs:
        cpdv_values = order_k_values(CPDVMetric(), wdnf, 1)
        assert_same_values(cpdv_values, order_k_values(PDIVMetric(), wdnf, 1))
        assert_same_values(cpdv_values, order_k_values(MobiusCoefficients(1, 1), wdnf, 1))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_pdiv_and_the_mobius_coefficients_are_the_same_metric_at_higher_orders(order):
    # PDIVMetric sums (-1)^|S-| over the subsets of the cube's negative literals, which is exactly the
    # Mobius coefficient of a subset holding all the cube's positive literals (any other subset holds a
    # variable that is a dummy player of the cube and therefore gets 0).
    for wdnf in ALL_WDNFs:
        assert_same_values(
            order_k_values(PDIVMetric(), wdnf, order),
            order_k_values(MobiusCoefficients(order, order), wdnf, order)
        )
