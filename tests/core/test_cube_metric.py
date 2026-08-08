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


def key_order(metric: CubeMetric, key) -> int:
    """
    The interaction order a value reported under `key` belongs to: the CII metrics key every subset by its
    own tuple, while the dedicated classes are either order 1 (values) or order 2 (interaction values).
    """
    if isinstance(metric, CardinalityInteractionIndicesMetric):
        return len(key)
    return 2 if metric.INTERACTION_VALUE else 1


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
            sign = (-1) ** (key_order(metric, v) + 1)
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


GENERAL_METRIC_CLASSES = [GeneralShapleyInteractionValues, GeneralBanzhafInteractionValues]
GENERAL_METRIC_CLASSES_IDS = ["GeneralShapleyInteractionValues", "GeneralBanzhafInteractionValues"]


@pytest.mark.parametrize("metric_class", GENERAL_METRIC_CLASSES, ids=GENERAL_METRIC_CLASSES_IDS)
def test_the_default_any_order_metric_is_the_union_of_all_the_fixed_order_metrics(metric_class):
    # The default (min_order=1, max_order=None) reports every subset of every size at once, and each value
    # must equal the one the matching fixed-order metric gives.
    for wdnf in ALL_WDNFs:
        any_order_values = wdnf.calc_metric(metric_class(1, None))
        fixed_order_values = {}
        for order in range(1, len(wdnf.variables()) + 1):
            fixed_order_values.update(wdnf.calc_metric(metric_class(order, order)))
        assert set(any_order_values) == set(fixed_order_values)
        for subset in any_order_values:
            assert abs(any_order_values[subset] - fixed_order_values[subset]) < TOLERANCE


@pytest.mark.parametrize("metric_class", GENERAL_METRIC_CLASSES, ids=GENERAL_METRIC_CLASSES_IDS)
@pytest.mark.parametrize("min_order, max_order", [(1, 1), (1, 2), (1, None), (2, 3), (3, None), (4, 4)])
def test_an_order_range_reports_exactly_the_subsets_of_the_sizes_in_the_range(metric_class, min_order, max_order):
    # An order range is a filter over the any-order metric: same values, only the subsets whose size falls
    # in [min_order, max_order] (subsets larger than the cube contain a dummy player and are never reported).
    for wdnf in ALL_WDNFs:
        any_order_values = wdnf.calc_metric(metric_class())
        values = wdnf.calc_metric(metric_class(min_order, max_order))
        expected_subsets = {
            subset for subset in any_order_values
            if min_order <= len(subset) and (max_order is None or len(subset) <= max_order)
        }
        assert set(values) == expected_subsets
        for subset in values:
            assert abs(values[subset] - any_order_values[subset]) < TOLERANCE