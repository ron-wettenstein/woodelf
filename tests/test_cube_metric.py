import pytest

from woodelf.cube_metric import (
    BanzhafInteractionValues, BanzhafValues, CubeMetric, ShapleyInteractionValues, ShapleyValues
)
from woodelf.direct_computation import (
    BanzhafDirectComputation, BanzhafIVDirectComputation, ShapleyIVDirectComputation,
    ShapleyDirectComputation, DirectComputation, WDNF, Cube
)
from tests.wdnfs import ALL_WDNFs

TOLERANCE = 1e-7


@pytest.mark.parametrize("metric, direct_computation", [
    (BanzhafValues(), BanzhafDirectComputation()),
    (BanzhafInteractionValues(), BanzhafIVDirectComputation()),
    (ShapleyInteractionValues(), ShapleyIVDirectComputation()),
    (ShapleyValues(), ShapleyDirectComputation()),
], ids=["BanzahfValues", "BanzhafInteractionValues", "ShapleyInteractionValues", "ShapleyValues"])
def test_metric(metric: CubeMetric, direct_computation: DirectComputation):
    for wdnf in ALL_WDNFs:
        values_using_metric = wdnf.calc_metric(metric)
        values_using_direct_computation = direct_computation.compute(wdnf)
        for v in values_using_metric:
            assert abs(values_using_metric[v] - values_using_direct_computation[v]) < TOLERANCE



@pytest.mark.parametrize("metric, direct_computation", [
    (BanzhafValues(), BanzhafDirectComputation()),
    (ShapleyValues(), ShapleyDirectComputation()),
], ids=["BanzahfValues", "ShapleyValues"])
def test_metric_applies_on_wcnf(metric: CubeMetric, direct_computation: DirectComputation):
    for wdnf in ALL_WDNFs:
        # Uses the identity w*c_k = w - w*(not(c_k)) to treat the wdnf as a wcnf and encode it back to wdnf
        wdnf_of_the_wdnf_treated_as_wdnf = WDNF(
            [(w, Cube(set(), set())) for w, cube in wdnf.cubes_and_weights] +
            [(-w, Cube(cube.sm, cube.sp)) for w, cube in wdnf.cubes_and_weights]
        )
        values_using_metric = wdnf.calc_metric(metric)
        values_using_direct_computation = direct_computation.compute(wdnf_of_the_wdnf_treated_as_wdnf)
        for v in values_using_metric:
            assert abs(values_using_metric[v] - values_using_direct_computation[v]) < TOLERANCE


@pytest.mark.parametrize("metric, direct_computation", [
    (BanzhafInteractionValues(), BanzhafIVDirectComputation()),
    (ShapleyInteractionValues(), ShapleyIVDirectComputation()),
], ids=["BanzhafInteractionValues", "ShapleyInteractionValues"])
def test_metric_times_minus_1_applies_on_wcnf(metric: CubeMetric, direct_computation: DirectComputation):
    for wdnf in ALL_WDNFs:
        # Uses the identity w*c_k = w - w*(not(c_k)) to treat the wdnf as a wcnf and encode it back to wdnf
        wdnf_of_the_wdnf_treated_as_wdnf = WDNF(
            [(w, Cube(set(), set())) for w, cube in wdnf.cubes_and_weights] +
            [(-w, Cube(cube.sm, cube.sp)) for w, cube in wdnf.cubes_and_weights]
        )
        values_using_metric = wdnf.calc_metric(metric)
        values_using_direct_computation = direct_computation.compute(wdnf_of_the_wdnf_treated_as_wdnf)
        for v in values_using_metric:
            # This time we should multiply the values computed by the metric by -1
            assert abs(-1*values_using_metric[v] - values_using_direct_computation[v]) < TOLERANCE