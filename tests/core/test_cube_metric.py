import pytest

from woodelf.core.cube_metric import (
    BanzhafInteractionValues, BanzhafValues, CubeMetric, ShapleyInteractionValues, ShapleyValues,
    GeneralShapleyInteractionValues, GeneralBanzhafInteractionValues, CardinalityInteractionIndicesMetric
)
from woodelf.core.direct_computation import (
    BanzhafDirectComputation, BanzhafIVDirectComputation, ShapleyIVDirectComputation,
    ShapleyDirectComputation, BanzhafCIIDirectComputation, ShapleyCIIDirectComputation,
    DirectComputation, WDNF, Cube
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
]
GENERAL_METRICS_IDS = [
    f"General{name}InteractionValues_order_{order}"
    for name in ["Shapley", "Banzhaf"] for order in [1, 2, 3]
]

ALL_METRICS_AND_DIRECT_COMPUTATIONS = [
    (BanzhafValues(), BanzhafDirectComputation()),
    (BanzhafInteractionValues(), BanzhafIVDirectComputation()),
    (ShapleyInteractionValues(), ShapleyIVDirectComputation()),
    (ShapleyValues(), ShapleyDirectComputation()),
] + GENERAL_METRICS_AND_DIRECT_COMPUTATIONS
ALL_METRICS_IDS = [
    "BanzahfValues", "BanzhafInteractionValues", "ShapleyInteractionValues", "ShapleyValues"
] + GENERAL_METRICS_IDS


@pytest.mark.parametrize("metric, direct_computation", ALL_METRICS_AND_DIRECT_COMPUTATIONS,
                         ids=ALL_METRICS_IDS)
def test_metric(metric: CubeMetric, direct_computation: DirectComputation):
    for wdnf in ALL_WDNFs:
        values_using_metric = wdnf.calc_metric(metric)
        values_using_direct_computation = direct_computation.compute(wdnf)
        for v in values_using_metric:
            assert abs(values_using_metric[v] - values_using_direct_computation[v]) < TOLERANCE


@pytest.mark.parametrize("metric, direct_computation", ALL_METRICS_AND_DIRECT_COMPUTATIONS,
                         ids=ALL_METRICS_IDS)
def test_metric_applies_on_wcnf(metric: CubeMetric, direct_computation: DirectComputation):
    # Swapping the positive and the negative literals of a cube multiplies its order k interaction values
    # by (-1)^k, so treating a wdnf as a wcnf flips the sign of the values only for even orders.
    for wdnf in ALL_WDNFs:
        # Uses the identity w*c_k = w - w*(not(c_k)) to treat the wdnf as a wcnf and encode it back to wdnf
        wdnf_of_the_wdnf_treated_as_wdnf = WDNF(
            [(w, Cube(set(), set())) for w, cube in wdnf.cubes_and_weights] +
            [(-w, Cube(cube.sm, cube.sp)) for w, cube in wdnf.cubes_and_weights]
        )
        values_using_metric = wdnf.calc_metric(metric)
        values_using_direct_computation = direct_computation.compute(wdnf_of_the_wdnf_treated_as_wdnf)
        for v in values_using_metric:
            if not metric.INTERACTION_VALUE:
                sign = 1
            else:
                sign = (-1) ** (len(v) + 1)
            assert abs(sign * values_using_metric[v] - values_using_direct_computation[v]) < TOLERANCE


@pytest.mark.parametrize("order_1_metric, general_metric", [
    (ShapleyValues(), GeneralShapleyInteractionValues(1, 1)),
    (BanzhafValues(), GeneralBanzhafInteractionValues(1, 1)),
], ids=["ShapleyValues", "BanzhafValues"])
def test_general_metric_of_order_1_matches_the_order_1_metrics(order_1_metric, general_metric):
    for wdnf in ALL_WDNFs:
        values = wdnf.calc_metric(order_1_metric)
        general_values = wdnf.calc_metric(general_metric)
        # The general metrics key every subset by a tuple, so a single feature is keyed by a 1-tuple
        assert set(general_values) == {(v,) for v in values}
        for v in values:
            assert abs(values[v] - general_values[(v,)]) < TOLERANCE


@pytest.mark.parametrize("order_2_metric, general_metric", [
    (ShapleyInteractionValues(), GeneralShapleyInteractionValues(2, 2, shap_convention=True)),
    (BanzhafInteractionValues(), GeneralBanzhafInteractionValues(2, 2)),
], ids=["ShapleyInteractionValues", "BanzhafInteractionValues"])
def test_general_metric_of_order_2_matches_the_interaction_values_metrics(order_2_metric, general_metric):
    # The shap_convention flag is what makes the general Shapley interaction values comparable to
    # ShapleyInteractionValues, which halves every pair as the shap package spreads it over both orderings.
    for wdnf in ALL_WDNFs:
        values = wdnf.calc_metric(order_2_metric)
        general_values = wdnf.calc_metric(general_metric)
        assert set(general_values) == set(values)
        for pair in values:
            assert abs(values[pair] - general_values[pair]) < TOLERANCE
