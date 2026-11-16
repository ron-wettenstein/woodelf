import numpy as np
import pytest

from woodelf.core.cube_metric import ShapleyValues, BanzhafValues, ShapleyInteractionValues, BanzhafInteractionValues, \
    GeneralShapleyInteractionValues
from woodelf.core.path_to_s_vectors.simple_p2s import SimpleWoodelfPathToSVectors
from woodelf.core.path_to_s_vectors.woodelfhd_v1_p2s import HighDepthWoodelfV1PathToSVectors
from woodelf.core.path_to_s_vectors.woodelfhd_v2_p2s import HighDepthWoodelfV2PathToSVectors
from woodelf.core.path_to_s_vectors.archive.woodelfhd_paper_version_p2s import HighDepthWoodelfPaperVersionPathToSVectors, \
    HighDepthWoodelfOriginalVersionPathToSVectors
from woodelf.core.path_to_s_vectors.mn_background_p2s import MNBackgroundPathToSVectors, MNBackgroundFasterPathToSVectors
from woodelf.core.path_to_s_vectors.lts_recursive_p2s import LTSRecursivePathToSVectors

TOLERANCE = 0.00001


class HighDepthWoodelfV2FullSVectorsPathToSVectors(HighDepthWoodelfV2PathToSVectors):
    """V2 that always builds every s vector over all the 2^D patterns and indexes it by the consumer patterns."""
    FULL_S_VECTOR_CONSUMERS_RATIO = 0
    FULL_PAIR_S_VECTOR_CONSUMERS_RATIO = 0


class HighDepthWoodelfV2ConsumerPatternsOnlyPathToSVectors(HighDepthWoodelfV2PathToSVectors):
    """V2 that always computes every s vector only at the consumer patterns."""
    FULL_S_VECTOR_CONSUMERS_RATIO = np.inf
    FULL_PAIR_S_VECTOR_CONSUMERS_RATIO = np.inf



# Shared test fixtures for the p2s equivalence tests.
# Depth 6 gives 64 unique patterns — small enough to enumerate all of them,
# which is required so the sparse methods (MN, LTS) and the dense HighDepth method
# see identical data and produce directly comparable outputs.
_DEPTH = 6
_ALL_PATTERNS = np.arange(2 ** _DEPTH, dtype=np.int64)
_FEATURES_IN_PATH = ["f3", "f0", "f5", "f1", "f4", "f2"] # not sorted, so the bit of each feature and the pair keys are tested
_COVERS = np.array([0.7, 0.6, 0.8, 0.5, 0.9, 0.4])
_BACKGROUND_PATTERNS = np.random.default_rng(0).integers(0, 2 ** _DEPTH, size=300)
_CONSUMER_PATTERNS = {
    "all_patterns": np.random.default_rng(1).permutation(_ALL_PATTERNS),
    "few_unique": np.random.default_rng(2).choice(_ALL_PATTERNS, size=20, replace=False),
    "many_repeated": np.random.default_rng(3).integers(0, 2 ** _DEPTH, size=500),
}

_ALL_METRICS = [ShapleyValues, BanzhafValues, ShapleyInteractionValues, BanzhafInteractionValues]
_BOTH_MODES = ["background", "path_dependent"]
# p2s class -> (supported metrics, supported modes). A subclass supports what its base class supports
_P2S_SUPPORT = {
    HighDepthWoodelfV1PathToSVectors: (_ALL_METRICS, _BOTH_MODES),
    HighDepthWoodelfV2PathToSVectors: (_ALL_METRICS, _BOTH_MODES),
    SimpleWoodelfPathToSVectors: (_ALL_METRICS, _BOTH_MODES),
    HighDepthWoodelfOriginalVersionPathToSVectors: (_ALL_METRICS, _BOTH_MODES),
    HighDepthWoodelfPaperVersionPathToSVectors: (_ALL_METRICS, _BOTH_MODES),
    MNBackgroundPathToSVectors: ([ShapleyValues, BanzhafValues], ["background"]),
    MNBackgroundFasterPathToSVectors: ([ShapleyValues, BanzhafValues], ["background"]),
    LTSRecursivePathToSVectors: ([ShapleyValues, BanzhafValues, ShapleyInteractionValues], ["path_dependent"]),
}
# These return the values of the given consumer patterns (by position), the rest return s vectors indexed by pattern value
_POSITIONAL_P2S = (MNBackgroundPathToSVectors, MNBackgroundFasterPathToSVectors, LTSRecursivePathToSVectors)
# (reference, candidate). V2 has two ways to get the consumers values (see FULL_S_VECTOR_CONSUMERS_RATIO), its two
# variants above force each of them
_P2S_PAIRS = [
    (HighDepthWoodelfV1PathToSVectors, HighDepthWoodelfV2PathToSVectors),
    (HighDepthWoodelfV1PathToSVectors, HighDepthWoodelfV2FullSVectorsPathToSVectors),
    (HighDepthWoodelfV1PathToSVectors, HighDepthWoodelfV2ConsumerPatternsOnlyPathToSVectors),
    (HighDepthWoodelfV2PathToSVectors, MNBackgroundPathToSVectors),
    (MNBackgroundPathToSVectors, MNBackgroundFasterPathToSVectors),
    (HighDepthWoodelfV2PathToSVectors, LTSRecursivePathToSVectors),
    (HighDepthWoodelfV1PathToSVectors, SimpleWoodelfPathToSVectors),
    (HighDepthWoodelfV1PathToSVectors, HighDepthWoodelfOriginalVersionPathToSVectors),
    (HighDepthWoodelfV1PathToSVectors, HighDepthWoodelfPaperVersionPathToSVectors),
]


def _is_supported(p2s_cls, metric_cls, mode, w_neighbor, consumers):
    metrics, modes = next(_P2S_SUPPORT[cls] for cls in p2s_cls.__mro__ if cls in _P2S_SUPPORT)
    if metric_cls not in metrics or mode not in modes:
        return False
    if issubclass(p2s_cls, LTSRecursivePathToSVectors):
        # LTS needs unique consumer patterns and has no neighbor leaf trick for interaction values
        return consumers != "many_repeated" and (w_neighbor is None or metric_cls is not ShapleyInteractionValues)
    return True


def _equivalence_cases(reference_cls, candidate_cls):
    """The (metric, mode, w_neighbor, consumers) cases that both p2s classes support."""
    cases = []
    for metric_cls in _ALL_METRICS:
        for mode in _BOTH_MODES:
            for w_neighbor in [None, 0.5]:
                for consumers in _CONSUMER_PATTERNS:
                    if all(_is_supported(cls, metric_cls, mode, w_neighbor, consumers) for cls in (reference_cls, candidate_cls)):
                        cases.append((metric_cls, mode, w_neighbor, consumers))
    return cases


def _leaf_consumer_values(p2s, mode, consumer_patterns, w, w_neighbor):
    """Dict[feature key -> the leaf's value of every consumer], whatever the p2s class returns."""
    if p2s.COMPUTES_CONSUMER_VALUES:
        if mode == "background":
            return p2s.get_background_consumer_values(_FEATURES_IN_PATH, consumer_patterns, _BACKGROUND_PATTERNS, w, w_neighbor)
        return p2s.get_path_dependent_consumer_values(_FEATURES_IN_PATH, consumer_patterns, _COVERS, w, w_neighbor)

    if mode == "background":
        s_matrix = p2s.get_background_s_matrix(_FEATURES_IN_PATH, consumer_patterns, _BACKGROUND_PATTERNS, w, w_neighbor)
    else:
        s_matrix = p2s.get_path_dependent_s_matrix(_FEATURES_IN_PATH, consumer_patterns, _COVERS, w, w_neighbor)
    if isinstance(p2s, LTSRecursivePathToSVectors) and p2s.metric.INTERACTION_VALUE:
        # LTS returns both (i, j) and (j, i), keep the sorted keys like the rest
        s_matrix = {key: s_vector for key, s_vector in s_matrix.items() if key == tuple(sorted(key))}
    if isinstance(p2s, _POSITIONAL_P2S):
        return s_matrix
    return {key: s_vector[consumer_patterns] for key, s_vector in s_matrix.items()}


@pytest.mark.parametrize(
    "reference_cls, candidate_cls", _P2S_PAIRS,
    ids=[f"{reference_cls.__name__}-vs-{candidate_cls.__name__}" for reference_cls, candidate_cls in _P2S_PAIRS]
)
def test_p2s_classes_are_equivalent(reference_cls, candidate_cls):
    cases = _equivalence_cases(reference_cls, candidate_cls)
    assert len(cases) > 0
    for metric_cls, mode, w_neighbor, consumers in cases:
        case = f"{metric_cls.__name__}, {mode}, w_neighbor={w_neighbor}, {consumers}"
        reference_p2s = reference_cls(metric=metric_cls(), max_depth=_DEPTH, GPU=False)
        candidate_p2s = candidate_cls(metric=metric_cls(), max_depth=_DEPTH, GPU=False)
        consumer_patterns = _CONSUMER_PATTERNS[consumers]

        for w in [1.0, 3.0]:
            reference = _leaf_consumer_values(reference_p2s, mode, consumer_patterns, w, w_neighbor)
            actual = _leaf_consumer_values(candidate_p2s, mode, consumer_patterns, w, w_neighbor)
            assert actual.keys() == reference.keys(), f"{case}: feature key mismatch: {actual.keys()} vs {reference.keys()}"
            for feature in reference:
                np.testing.assert_allclose(
                    actual[feature], reference[feature], atol=TOLERANCE, err_msg=f"{case}, w={w}, feature {feature}"
                )


@pytest.mark.parametrize("p2s_cls", [HighDepthWoodelfV1PathToSVectors, HighDepthWoodelfV2PathToSVectors])
def test_prepare_f(p2s_cls):
    r1=0.3
    r2=0.6
    r3=0.8
    l1=1-r1
    l2=1-r2
    l3=1-r3

    # Path dependent f
    original_f = np.array([l1*l2*l3, l1*l2*r3, l1*r2*l3, l1*r2*r3, r1*l2*l3, r1*l2*r3, r1*r2*l3, r1*r2*r3])
    actual_f = p2s_cls._prepare_f(depth=3, f=original_f, GPU=False)

    # Prepared path dependent f
    l1,l2,l3=1,1,1
    expected_f = np.array([l1*l2*l3, l1*l2*r3, l1*r2*l3, l1*r2*r3, r1*l2*l3, r1*l2*r3, r1*r2*l3, r1*r2*r3])

    np.testing.assert_allclose(actual_f, expected_f)

    # _prepare_f must not change the f it receives (the neighbor leaf trick reuses it)
    f = np.arange(2 ** 10, dtype=np.float64)
    p2s_cls._prepare_f(depth=10, f=f, GPU=False)
    np.testing.assert_array_equal(f, np.arange(2 ** 10, dtype=np.float64))


def test_paths_shorter_than_the_interaction_order():
    # paths shorter than the interaction order have no frs subsets
    high_depth_p2m = HighDepthWoodelfV1PathToSVectors(metric=GeneralShapleyInteractionValues(3, 3), max_depth=4, GPU=False)
    original_p2m = HighDepthWoodelfOriginalVersionPathToSVectors(metric=GeneralShapleyInteractionValues(3, 3), max_depth=4, GPU=False)
    for path_depth in [1, 2, 3, 4]:
        f = np.arange(2 ** path_depth) / 2 ** path_depth
        expected_s = original_p2m._get_s_vectors_given_f(list(range(path_depth)), f, w=1.5)
        actual_s = high_depth_p2m._get_s_vectors_given_f(list(range(path_depth)), f, w=1.5)
        assert actual_s.keys() == expected_s.keys()
        for feature in expected_s:
            np.testing.assert_allclose(actual_s[feature], expected_s[feature], atol=TOLERANCE)
