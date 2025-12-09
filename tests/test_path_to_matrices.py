import numpy as np

from woodelf.cube_metric import ShapleyValues
from woodelf.path_to_matrices import SimplePathToMatrices, HighDepthPathToMatrices, HighDepthPathToMatricesVectorized

TOLERANCE = 0.00001


def test_prepare_f():
    r1=0.3
    r2=0.6
    r3=0.8
    l1=1-r1
    l2=1-r2
    l3=1-r3

    # Path dependent f
    original_f = np.array([l1*l2*l3, l1*l2*r3, l1*r2*l3, l1*r2*r3, r1*l2*l3, r1*l2*r3, r1*r2*l3, r1*r2*r3])
    actual_f = HighDepthPathToMatrices.prepare_f(depth=3, f=original_f)

    # Prepared path dependent f
    l1,l2,l3=1,1,1
    expected_f = np.array([l1*l2*l3, l1*l2*r3, l1*r2*l3, l1*r2*r3, r1*l2*l3, r1*l2*r3, r1*r2*l3, r1*r2*r3])

    np.testing.assert_allclose(actual_f, expected_f)


def test_simple_and_high_depth_are_equivalent():
    high_depth_p2m = HighDepthPathToMatricesVectorized(metric=ShapleyValues(), max_depth=6, GPU=False)
    simple_p2m = SimplePathToMatrices(metric=ShapleyValues(), max_depth=6, GPU=False)

    f = np.arange(2 ** 6)
    expected_s = simple_p2m.get_s_matrices(features_in_path=list(range(1,7)), f=f, w=1)

    actual_s = high_depth_p2m.get_s_matrices(features_in_path=list(range(1,7)), f=f, w=1, path_dependent=False)

    for feature in expected_s:
        np.testing.assert_allclose(actual_s[feature], expected_s[feature], atol=TOLERANCE)

    # weight other than 1
    f = np.arange(2 ** 6) / 2 ** 6
    expected_s = simple_p2m.get_s_matrices(features_in_path=list(range(1, 7)), f=f, w=3)
    actual_s = high_depth_p2m.get_s_matrices(features_in_path=list(range(1, 7)), f=f, w=3, path_dependent=False)
    for feature in expected_s:
        np.testing.assert_allclose(actual_s[feature], expected_s[feature], atol=TOLERANCE)



