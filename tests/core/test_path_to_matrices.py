import numpy as np

from woodelf.core.cube_metric import ShapleyValues
from woodelf.core.path_to_s_vectors.simple_p2s import SimpleWoodelfPathToSVectors
from woodelf.core.path_to_s_vectors.woodelf_p2s import HighDepthWoodelfPathToSVectors
from woodelf.core.path_to_s_vectors.archive.woodelfhd_paper_version_p2s import HighDepthWoodelfPaperVersionPathToSVectors
from woodelf.core.path_to_s_vectors.mn_background_p2s import MNBackgroundPathToSVectors, MNBackgroundFasterPathToSVectors
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import LTSRecursivePathToSVectors

TOLERANCE = 0.00001

# Shared test fixtures for the sparse-vs-dense comparison tests.
# Depth 6 gives 64 unique patterns — small enough to enumerate all of them,
# which is required so the sparse methods (MN, LTS) and the dense HighDepth method
# see identical data and produce directly comparable outputs.
_DEPTH = 6
_ALL_PATTERNS = np.arange(2 ** _DEPTH, dtype=np.int64)
_FEATURES_IN_PATH = list(range(_DEPTH))
_COVERS = np.array([0.7, 0.6, 0.8, 0.5, 0.9, 0.4])


def _assert_s_dicts_close(actual: dict, reference: dict):
    """
    Compare two s-vector dicts.
    reference values are size-2^D arrays indexed by pattern value.
    actual values are size-U_c arrays indexed by position in ALL_PATTERNS.
    Since ALL_PATTERNS = np.arange(2**D), both orderings are identical.
    """
    assert actual.keys() == reference.keys(), f"Feature key mismatch: {actual.keys()} vs {reference.keys()}"
    for feature in reference:
        np.testing.assert_allclose(actual[feature], reference[feature][_ALL_PATTERNS], atol=TOLERANCE)


def test_prepare_f():
    r1=0.3
    r2=0.6
    r3=0.8
    l1=1-r1
    l2=1-r2
    l3=1-r3

    # Path dependent f
    original_f = np.array([l1*l2*l3, l1*l2*r3, l1*r2*l3, l1*r2*r3, r1*l2*l3, r1*l2*r3, r1*r2*l3, r1*r2*r3])
    actual_f = HighDepthWoodelfPathToSVectors._prepare_f(depth=3, f=original_f, GPU=False)

    # Prepared path dependent f
    l1,l2,l3=1,1,1
    expected_f = np.array([l1*l2*l3, l1*l2*r3, l1*r2*l3, l1*r2*r3, r1*l2*l3, r1*l2*r3, r1*r2*l3, r1*r2*r3])

    np.testing.assert_allclose(actual_f, expected_f)


def test_simple_and_high_depth_are_equivalent():
    high_depth_p2m = HighDepthWoodelfPathToSVectors(metric=ShapleyValues(), max_depth=6, GPU=False)
    simple_p2m = SimpleWoodelfPathToSVectors(metric=ShapleyValues(), max_depth=6, GPU=False)

    f = np.arange(2 ** 6)
    expected_s = simple_p2m._get_s_vectors_given_f(features_in_path=list(range(1,7)), f=f, w=1)

    actual_s = high_depth_p2m._get_s_vectors_given_f(features_in_path=list(range(1,7)), f=f, w=1)

    for feature in expected_s:
        np.testing.assert_allclose(actual_s[feature], expected_s[feature], atol=TOLERANCE)

    # weight other than 1
    f = np.arange(2 ** 6) / 2 ** 6
    expected_s = simple_p2m._get_s_vectors_given_f(features_in_path=list(range(1, 7)), f=f, w=3)
    actual_s = high_depth_p2m._get_s_vectors_given_f(features_in_path=list(range(1, 7)), f=f, w=3)
    for feature in expected_s:
        np.testing.assert_allclose(actual_s[feature], expected_s[feature], atol=TOLERANCE)


def test_mn_background_matches_high_depth_woodelf():
    reference_p2s = HighDepthWoodelfPathToSVectors(metric=ShapleyValues(), max_depth=_DEPTH)
    mn_p2s = MNBackgroundPathToSVectors(metric=ShapleyValues(), max_depth=_DEPTH)

    reference = reference_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=1.0)
    actual = mn_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=1.0)
    _assert_s_dicts_close(actual, reference)

    # non-unit weight
    reference = reference_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=3.0)
    actual = mn_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=3.0)
    _assert_s_dicts_close(actual, reference)

    # with neighbor leaf trick
    reference = reference_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=2.0, w_neighbor=0.5)
    actual = mn_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=2.0, w_neighbor=0.5)
    _assert_s_dicts_close(actual, reference)


def test_mn_background_faster_matches_high_depth_woodelf():
    reference_p2s = HighDepthWoodelfPathToSVectors(metric=ShapleyValues(), max_depth=_DEPTH)
    mn_faster_p2s = MNBackgroundFasterPathToSVectors(metric=ShapleyValues(), max_depth=_DEPTH)

    reference = reference_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=1.0)
    actual = mn_faster_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=1.0)
    _assert_s_dicts_close(actual, reference)

    # non-unit weight
    reference = reference_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=3.0)
    actual = mn_faster_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=3.0)
    _assert_s_dicts_close(actual, reference)

    # with neighbor leaf trick
    reference = reference_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=2.0, w_neighbor=0.5)
    actual = mn_faster_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=2.0, w_neighbor=0.5)
    _assert_s_dicts_close(actual, reference)


def test_mn_background_faster_matches_mn_background():
    reference_p2s = MNBackgroundPathToSVectors(metric=ShapleyValues(), max_depth=_DEPTH)
    mn_faster_p2s = MNBackgroundFasterPathToSVectors(metric=ShapleyValues(), max_depth=_DEPTH)

    reference = reference_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=1.0)
    actual = mn_faster_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=1.0)
    _assert_s_dicts_close(actual, reference)

    # non-unit weight
    reference = reference_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=3.0)
    actual = mn_faster_p2s.get_background_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _ALL_PATTERNS, w=3.0)
    _assert_s_dicts_close(actual, reference)


def test_lts_recursive_matches_high_depth_woodelf():
    reference_p2s = HighDepthWoodelfPathToSVectors(metric=ShapleyValues(), max_depth=_DEPTH)
    lts_p2s = LTSRecursivePathToSVectors(metric=ShapleyValues(), max_depth=_DEPTH)

    reference = reference_p2s.get_path_dependent_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _COVERS, w=1.0)
    actual = lts_p2s.get_path_dependent_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _COVERS, w=1.0)
    _assert_s_dicts_close(actual, reference)

    # non-unit weight
    reference = reference_p2s.get_path_dependent_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _COVERS, w=3.0)
    actual = lts_p2s.get_path_dependent_s_matrix(_FEATURES_IN_PATH, _ALL_PATTERNS, _COVERS, w=3.0)
    _assert_s_dicts_close(actual, reference)

def test_simple_and_high_depth_paper_are_equivalent():
    high_depth_p2m = HighDepthWoodelfPaperVersionPathToSVectors(metric=ShapleyValues(), max_depth=6, GPU=False)
    simple_p2m = SimpleWoodelfPathToSVectors(metric=ShapleyValues(), max_depth=6, GPU=False)

    f = np.arange(2 ** 6)
    expected_s = simple_p2m._get_s_vectors_given_f(features_in_path=list(range(1,7)), f=f, w=1)

    actual_s = high_depth_p2m._get_s_vectors_given_f(features_in_path=list(range(1,7)), f=f, w=1)

    for feature in expected_s:
        np.testing.assert_allclose(actual_s[feature], expected_s[feature], atol=TOLERANCE)

    # weight other than 1
    f = np.arange(2 ** 6) / 2 ** 6
    expected_s = simple_p2m._get_s_vectors_given_f(features_in_path=list(range(1, 7)), f=f, w=3)
    actual_s = high_depth_p2m._get_s_vectors_given_f(features_in_path=list(range(1, 7)), f=f, w=3)
    for feature in expected_s:
        np.testing.assert_allclose(actual_s[feature], expected_s[feature], atol=TOLERANCE)

