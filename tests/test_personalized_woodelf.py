import numpy as np
import pytest
import shap

from shared_fixtures_and_utils import testset, xgb_model, assert_woodelf_dicts_equal
from woodelf.core.cube_metric import BanzhafValues, ShapleyValues
from woodelf.personalized_woodelf import personalized_baseline_delta_update, personalized_baseline_woodelf

TOLERANCE = 0.00001
N = 10


@pytest.mark.parametrize("use_neighbor_leaf_trick", [False, True])
def test_personalized_woodelf_matches_shap_package(testset, xgb_model, use_neighbor_leaf_trick):
    consumer_data   = testset.iloc[:N].reset_index(drop=True)
    background_data = testset.iloc[N:2*N].reset_index(drop=True)
    metric = ShapleyValues()

    personalized_result = personalized_baseline_woodelf(
        xgb_model, consumer_data, background_data, metric, use_neighbor_leaf_trick=use_neighbor_leaf_trick
    )

    for i in range(N):
        explainer = shap.TreeExplainer(xgb_model, background_data.iloc[i:i+1], feature_perturbation="interventional")
        shap_vals = explainer.shap_values(consumer_data.iloc[i:i+1])  # (1, n_features)
        for j, feature in enumerate(consumer_data.columns):
            personalized_val = personalized_result[feature][i] if feature in personalized_result else 0.0
            np.testing.assert_allclose(
                personalized_val, shap_vals[0, j], atol=TOLERANCE,
                err_msg=f"Mismatch at consumer {i}, feature {feature}"
            )


def test_features_subset_matches_full_result(testset, xgb_model):
    """
    Verify that running personalized_baseline_woodelf with features_subset returns the same values
    for the subset features as a full (no-subset) run.

    Why this works: a feature's Banzhaf value is the sum of its contributions across all leaves where
    it appears in the path. The subset generator yields exactly those leaves — it skips leaves whose
    path contains none of the subset features, but those skipped leaves contribute zero to any subset
    feature anyway (a feature cannot contribute from a leaf where it is absent from the path).
    Therefore the subset run accumulates the identical partial sums as the full run, feature by feature.

    Also checks that no feature outside the subset leaks into the result (compute_effect_on_other_features
    defaults to False).
    """
    consumer_data = testset.iloc[:N].reset_index(drop=True)
    background_data = testset.iloc[N:2*N].reset_index(drop=True)
    metric = BanzhafValues()
    subset = list(testset.columns[:2])

    full_result = personalized_baseline_woodelf(xgb_model, consumer_data, background_data, metric)
    subset_result = personalized_baseline_woodelf(
        xgb_model, consumer_data, background_data, metric, features_subset=subset,
    )

    assert set(subset_result.keys()) <= set(subset), "subset result contains features outside the subset"
    zeros = np.zeros(N)
    for f in subset:
        np.testing.assert_allclose(
            subset_result.get(f, zeros), full_result.get(f, zeros),
            atol=TOLERANCE, err_msg=f"Subset result mismatch for feature {f!r}",
        )


def test_delta_trick_is_exact(testset, xgb_model):
    """
    Verify that personalized_baseline_delta_update produces exactly the same result as a full
    recompute after changing the background of a feature subset. See that function's docstring
    for the mathematical argument for exactness.
    """
    consumer_data = testset.iloc[:N].reset_index(drop=True)
    B = testset.iloc[N:2*N].reset_index(drop=True)
    metric = BanzhafValues()
    subset = list(testset.columns[:2])

    full_result_old = personalized_baseline_woodelf(xgb_model, consumer_data, B, metric)

    B_new = B.copy()
    for f in subset:
        B_new[f] = B[f] * 1.5 + 1.0

    full_result_new = personalized_baseline_woodelf(xgb_model, consumer_data, B_new, metric)

    updated = personalized_baseline_delta_update(
        xgb_model, consumer_data, B, B_new, subset, full_result_old, metric,
    )

    assert_woodelf_dicts_equal(updated, full_result_new, testset, TOLERANCE)
