import numpy as np

from woodelf.core.cube_metric import ShapleyValues, BanzhafValues, \
    GeneralShapleyInteractionValues, GeneralBanzhafInteractionValues
from woodelf.core.path_to_s_vectors.mn_background_p2s import MNBackgroundPathToSVectors
from woodelf.core.path_to_s_vectors.mn_background_cii_p2s import MNBackgroundCIIPathToSVectors, \
    MNBackgroundCIIFasterPathToSVectors

# The any-order metrics themselves are tested against the direct computation framework in
# tests/core/test_cube_metric.py. This module tests the path-to-s-vectors classes that compute them.
# The end-to-end runs through woodelf_sparse live in tests/test_woodelf_sparse.py.

EXACT_TOLERANCE = 1e-9

D = 8

def test_cii_contribution_tables_match_order_1_contribution_matrices():
    for order_1_metric, general_metric in [
        (ShapleyValues(), GeneralShapleyInteractionValues(1, 1)),
        (BanzhafValues(), GeneralBanzhafInteractionValues(1, 1)),
    ]:
        pos_matrix, neg_matrix = MNBackgroundPathToSVectors._build_contribution_matrices(order_1_metric, D)
        tables = MNBackgroundCIIPathToSVectors._build_contribution_tables(general_metric, D)
        assert set(tables) == {1}
        np.testing.assert_allclose(tables[1][1], pos_matrix, atol=EXACT_TOLERANCE)
        np.testing.assert_allclose(tables[1][0], neg_matrix, atol=EXACT_TOLERANCE)


def test_cii_faster_matches_naive_on_random_patterns():
    rng = np.random.default_rng(4)
    features_in_path = [f"f{i}" for i in range(D)]
    consumer_patterns = rng.integers(0, 2 ** D, size=40)
    background_patterns = rng.integers(0, 2 ** D, size=60)
    metric = GeneralShapleyInteractionValues(1, 3)  # an order range, so several orders are covered at once

    naive_s_matrix = MNBackgroundCIIPathToSVectors(metric, max_depth=D).get_background_s_matrix(
        features_in_path, consumer_patterns, background_patterns, 1.5, -0.7
    )
    faster_s_matrix = MNBackgroundCIIFasterPathToSVectors(metric, max_depth=D).get_background_s_matrix(
        features_in_path, consumer_patterns, background_patterns, 1.5, -0.7
    )

    assert set(naive_s_matrix) == set(faster_s_matrix)
    for subset in naive_s_matrix:
        np.testing.assert_allclose(
            naive_s_matrix[subset], faster_s_matrix[subset], atol=EXACT_TOLERANCE
        )


def test_cii_order_range_s_matrix_is_the_union_of_the_fixed_order_s_matrices():
    rng = np.random.default_rng(11)
    features_in_path = [f"f{i}" for i in range(D)]
    consumer_patterns = rng.integers(0, 2 ** D, size=30)
    background_patterns = rng.integers(0, 2 ** D, size=40)

    range_p2s = MNBackgroundCIIPathToSVectors(GeneralBanzhafInteractionValues(1, None), max_depth=D)
    range_s_matrix = range_p2s.get_background_s_matrix(
        features_in_path, consumer_patterns, background_patterns, 1.5, -0.7
    )
    fixed_s_matrix = {}
    for order in range(1, D + 1):
        fixed_p2s = MNBackgroundCIIPathToSVectors(GeneralBanzhafInteractionValues(order, order), max_depth=D)
        fixed_s_matrix.update(fixed_p2s.get_background_s_matrix(
            features_in_path, consumer_patterns, background_patterns, 1.5, -0.7
        ))
    assert set(range_s_matrix) == set(fixed_s_matrix)
    for subset in range_s_matrix:
        np.testing.assert_allclose(
            range_s_matrix[subset], fixed_s_matrix[subset], atol=EXACT_TOLERANCE
        )


def test_cii_faster_subset_chunking_does_not_change_results():
    rng = np.random.default_rng(5)
    order = 2
    features_in_path = [f"f{i}" for i in range(D)]
    consumer_patterns = rng.integers(0, 2 ** D, size=30)
    background_patterns = rng.integers(0, 2 ** D, size=50)
    metric = GeneralBanzhafInteractionValues(order, order)
    faster_p2s = MNBackgroundCIIFasterPathToSVectors(metric, max_depth=D)
    chunked_p2s = MNBackgroundCIIFasterPathToSVectors(metric, max_depth=D)
    chunked_p2s.MASK_BATCH = 64  # force many subset-chunks per split level
    s_matrix = faster_p2s.get_background_s_matrix(features_in_path, consumer_patterns, background_patterns, 2.0)
    chunked_s_matrix = chunked_p2s.get_background_s_matrix(features_in_path, consumer_patterns, background_patterns, 2.0)
    for subset in s_matrix:
        np.testing.assert_allclose(s_matrix[subset], chunked_s_matrix[subset], atol=EXACT_TOLERANCE)


