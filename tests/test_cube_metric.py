import pytest

from cube_metric import BanzhafInteractionValues, BanzahfValues, CubeMetric, ShapleyInteractionValues, ShapleyValues
from direct_computation import BanzhafDirectComputation, BanzhafIVDirectComputation, ShapleyIVDirectComputation, \
    ShapleyDirectComputation, DirectComputation
from tests.wdnfs import ALL_WDNFs

TOLERANCE = 1e-7


@pytest.mark.parametrize("metric, direct_computation", [
    (BanzahfValues(), BanzhafDirectComputation()),
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
