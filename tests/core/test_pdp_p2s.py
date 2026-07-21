import numpy as np
import pytest

from woodelf.core.cube_metric import PDIVOrder1Or2
from woodelf.core.path_to_s_vectors.pdp_p2s import PDPPathToSVectors, PDPIVPathToSVectors
from woodelf.core.path_to_s_vectors.woodelf_p2s import HighDepthWoodelfPathToSVectors

TOLERANCE = 0.00001

# Depth 5 gives 32 unique patterns — small enough to enumerate all of them as the consumer
# patterns, so the dedicated PDP calculators (which index by position in consumer_patterns)
# and the generic HighDepth calculator (which indexes by pattern value) are directly comparable.
DEPTH = 5
ALL_PATTERNS = np.arange(2 ** DEPTH, dtype=np.int64)
FEATURES_IN_PATH = list(range(DEPTH))
COVERS = np.array([0.7, 0.6, 0.8, 0.5, 0.9])
BACKGROUND = np.array(
    [31, 31, 30, 29, 27, 23, 15, 15, 28, 26, 21, 7, 0, 1, 16, 31], dtype=np.int64
)
LEAF_WEIGHTS = [1.0, 3.0, -2.0]


def _reference_p2s():
    return HighDepthWoodelfPathToSVectors(metric=PDIVOrder1Or2(), max_depth=DEPTH, GPU=False)


def _assert_order_1_close(actual: dict, reference: dict):
    """
    actual keys are bare features, reference keys are 1-tuples of features.
    reference values are size-2^D arrays indexed by pattern value; actual values are
    indexed by position in ALL_PATTERNS — identical orderings since ALL_PATTERNS = arange(2**D).
    """
    assert set(actual.keys()) == {key[0] for key in reference if len(key) == 1}
    for feature in actual:
        np.testing.assert_allclose(
            actual[feature], reference[(feature,)][ALL_PATTERNS], atol=TOLERANCE
        )


def _assert_order_2_close(actual: dict, reference: dict):
    """
    actual holds both (f1, f2) and (f2, f1) for every pair; reference holds only the sorted key.
    """
    expected_pairs = {key for key in reference if len(key) == 2}
    assert set(actual.keys()) == {(f1, f2) for f1, f2 in expected_pairs} | {(f2, f1) for f1, f2 in expected_pairs}
    for (f1, f2), values in actual.items():
        np.testing.assert_allclose(
            values, reference[tuple(sorted((f1, f2)))][ALL_PATTERNS], atol=TOLERANCE
        )


@pytest.mark.parametrize(
    "p2s_class, assert_close",
    [
        (PDPPathToSVectors, _assert_order_1_close),
        (PDPIVPathToSVectors, _assert_order_2_close),
    ],
    ids=["pdp", "pdp_iv"],
)
def test_pdp_background_matches_high_depth_woodelf(p2s_class, assert_close):
    reference_p2s = _reference_p2s()
    pdp_p2s = p2s_class(max_depth=DEPTH)

    for w in LEAF_WEIGHTS:
        reference = reference_p2s.get_background_s_matrix(FEATURES_IN_PATH, ALL_PATTERNS, BACKGROUND, w=w)
        actual = pdp_p2s.get_background_s_matrix(FEATURES_IN_PATH, ALL_PATTERNS, BACKGROUND, w=w)
        assert_close(actual, reference)


def test_pdp_path_dependent_matches_high_depth_woodelf():
    reference_p2s = _reference_p2s()
    pdp_p2s = PDPPathToSVectors(max_depth=DEPTH)

    for w in LEAF_WEIGHTS:
        reference = reference_p2s.get_path_dependent_s_matrix(FEATURES_IN_PATH, ALL_PATTERNS, COVERS, w=w)
        actual = pdp_p2s.get_path_dependent_s_matrix(FEATURES_IN_PATH, ALL_PATTERNS, COVERS, w=w)
        _assert_order_1_close(actual, reference)


def test_pdp_empty_path_returns_empty():
    assert PDPPathToSVectors(max_depth=DEPTH).get_background_s_matrix([], ALL_PATTERNS, BACKGROUND, w=1.0) == {}
    # PDP-IV needs at least 2 features in the path to have any pair
    assert PDPIVPathToSVectors(max_depth=DEPTH).get_background_s_matrix([0], ALL_PATTERNS, BACKGROUND, w=1.0) == {}
