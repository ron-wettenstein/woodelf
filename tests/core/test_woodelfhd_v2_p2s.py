import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from woodelf.core.cube_metric import CubeMetric, ShapleyValues, BanzhafValues, ShapleyInteractionValues,     BanzhafInteractionValues, GeneralShapleyInteractionValues, PDIVOrder1Or2, PDIVMetric, CPDVMetric, nCk
from woodelf.core.path_to_s_vectors.woodelfhd_v1_p2s import HighDepthWoodelfV1PathToSVectors
from woodelf.core.path_to_s_vectors.woodelfhd_v2_p2s import HighDepthWoodelfV2PathToSVectors, CLOSED_FORM_VALUES
from woodelf.high_depth_woodelf import woodelf_for_high_depth, create_high_depth_woodelf_p2s

TOLERANCE = 0.00001


class RenamedShapleyValues(CubeMetric):
    """
    A reimplementation of ShapleyValues under another name. HighDepthWoodelfV2PathToSVectors supports only the closed
    form metrics, so create_high_depth_woodelf_p2s gives it HighDepthWoodelfV1PathToSVectors.
    """
    INTERACTION_VALUE = False

    def calc_metric(self, s_plus, s_minus):
        if len(s_plus & s_minus) > 0:
            return {}

        s = s_plus | s_minus
        shapley_values = {}
        if len(s_plus) > 0:
            contribution = (1 / (len(s_plus) * nCk(len(s), len(s_plus))))
            for must_exist_feature in s_plus:
                shapley_values[must_exist_feature] = contribution

        if len(s_minus) > 0:
            contribution = -1 / (len(s_minus) * nCk(len(s), len(s_minus)))
            for must_be_missing_feature in s_minus:
                shapley_values[must_be_missing_feature] = contribution

        return shapley_values


class RenamedShapleyInteractionValues(ShapleyInteractionValues):
    """ShapleyInteractionValues under another name. A subclass may change calc_metric, so it gets the v1 p2s too."""


@pytest.mark.parametrize("metric_cls", [ShapleyValues, BanzhafValues, ShapleyInteractionValues, BanzhafInteractionValues])
def test_closed_form_diagonals_match_the_v1_diagonals(metric_cls):
    # V1 applies calc_metric to the full cube of every row of the secondary diagonal. Its diagonal of a feature is v+
    # where the feature's bit is 1 and v- where it is 0, of a pair v++, v-- or v+- by how many of its two bits are 1
    assert metric_cls in CLOSED_FORM_VALUES
    for depth in range(1, 9):
        v1 = HighDepthWoodelfV1PathToSVectors(metric=metric_cls(), max_depth=depth)
        diagonals = HighDepthWoodelfV2PathToSVectors.metric_diagonals(metric_cls(), depth)
        assert all(diagonal.shape == (2 ** depth,) for diagonal in diagonals)
        rows = np.arange(2 ** depth)
        for column, frs_subset in enumerate(v1.matrices_frs_subsets[depth]):
            positions = frs_subset if isinstance(frs_subset, tuple) else (frs_subset,)
            set_bits = sum((rows >> (depth - 1 - position)) & 1 for position in positions)
            if metric_cls.INTERACTION_VALUE:
                m_plus_plus, m_minus_minus, m_plus_minus = diagonals
                expected = np.select([set_bits == 2, set_bits == 0], [m_plus_plus, m_minus_minus], m_plus_minus)
            else:
                m_plus, m_minus = diagonals
                expected = np.where(set_bits == 1, m_plus, m_minus)
            np.testing.assert_allclose(v1.matrices[depth][:, column], expected, rtol=1e-6) # v1 stores float32


@pytest.mark.parametrize("metric, expected_cls", [
    (ShapleyValues(), HighDepthWoodelfV2PathToSVectors),
    (BanzhafValues(), HighDepthWoodelfV2PathToSVectors),
    (ShapleyInteractionValues(), HighDepthWoodelfV2PathToSVectors),
    (BanzhafInteractionValues(), HighDepthWoodelfV2PathToSVectors),
    (RenamedShapleyValues(), HighDepthWoodelfV1PathToSVectors),
    (RenamedShapleyInteractionValues(), HighDepthWoodelfV1PathToSVectors),
    (CPDVMetric(), HighDepthWoodelfV1PathToSVectors),
    (GeneralShapleyInteractionValues(2, 2), HighDepthWoodelfV1PathToSVectors), # CardinalityInteractionIndicesMetric, not supported for now
    (GeneralShapleyInteractionValues(3, 3), HighDepthWoodelfV1PathToSVectors),
    (PDIVOrder1Or2(), HighDepthWoodelfV1PathToSVectors), # orders 1 and 2 mixed
    (PDIVMetric(), HighDepthWoodelfV1PathToSVectors),
], ids=lambda value: type(value).__name__ if isinstance(value, CubeMetric) else value.__name__)
def test_create_high_depth_woodelf_p2s_picks_v1_for_unsupported_metrics(metric, expected_cls):
    p2s = create_high_depth_woodelf_p2s(metric, max_depth=6)
    assert type(p2s) is expected_cls
    if expected_cls is HighDepthWoodelfV1PathToSVectors:
        with pytest.raises(ValueError):
            HighDepthWoodelfV2PathToSVectors(metric, max_depth=6)


@pytest.mark.parametrize("background", [True, False], ids=["background", "path_dependent"])
def test_custom_metric_end_to_end(background):
    X, y = make_regression(n_samples=2000, n_features=10, random_state=0)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    # Unit scale values: the path dependent values are float32, whose rounding errors grow with the values (~150 unscaled)
    model = DecisionTreeRegressor(max_depth=8, random_state=0).fit(X, y / np.std(y))
    consumer_data, background_data = X.iloc[:200], (X.iloc[200:500] if background else None)

    # ShapleyValues runs on HighDepthWoodelfV2PathToSVectors, RenamedShapleyValues on HighDepthWoodelfV1PathToSVectors
    expected = woodelf_for_high_depth(model, consumer_data, background_data, ShapleyValues(), verbose=False)
    actual = woodelf_for_high_depth(model, consumer_data, background_data, RenamedShapleyValues(), verbose=False)
    assert actual.keys() == expected.keys()
    for feature in expected:
        np.testing.assert_allclose(actual[feature], expected[feature], atol=TOLERANCE)


@pytest.mark.parametrize("metric", [ShapleyValues(), ShapleyInteractionValues()], ids=["shap", "shap_iv"])
@pytest.mark.parametrize("full_s_vector_consumers_ratio", [0, np.inf], ids=["full_s_vector", "consumer_patterns_only"])
def test_consumer_values_are_added_into_the_given_values(metric, full_s_vector_consumers_ratio):
    depth = 6
    features_in_path = list(range(depth))
    rng = np.random.default_rng(0)
    consumer_patterns = rng.integers(0, 2 ** depth, size=100)
    background_patterns = rng.integers(0, 2 ** depth, size=300)
    p2s = HighDepthWoodelfV2PathToSVectors(metric, max_depth=depth)
    p2s.full_s_vector_consumers_ratio = full_s_vector_consumers_ratio

    leaf_values = p2s.get_background_consumer_values(features_in_path, consumer_patterns, background_patterns, 2.0, 0.5)
    some_key = next(iter(leaf_values))
    values = {some_key: np.ones(len(consumer_patterns))} # a key from a previous leaf, the rest are new
    assert p2s.get_background_consumer_values(features_in_path, consumer_patterns, background_patterns, 2.0, 0.5, values=values) is None
    assert p2s.get_background_consumer_values(features_in_path, consumer_patterns, background_patterns, 2.0, 0.5, values=values) is None

    assert values.keys() == leaf_values.keys()
    assert len({id(feature_values) for feature_values in values.values()}) == len(values), "every key needs its own array"
    for key in leaf_values:
        np.testing.assert_allclose(values[key], 2 * leaf_values[key] + (1 if key == some_key else 0), atol=TOLERANCE)
