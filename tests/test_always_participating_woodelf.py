import pytest

from shared_fixtures_and_utils import testset, xgb_model, assert_woodelf_dicts_equal
from woodelf.always_participating_woodelf import (
    always_participating_delta_update,
    path_dependent_under_always_participating_features,
)
from woodelf.core.cube_metric import BanzhafValues, ShapleyValues
from woodelf.woodelf_sparse import linear_tree_shap_meets_woodelf

TOLERANCE = 0.00001
N = 10


@pytest.mark.parametrize("metric", [ShapleyValues(), BanzhafValues()])
def test_empty_always_participating_matches_linear_tree_shap(testset, xgb_model, metric):
    """
    With an empty always_participating_features list the restricted game is identical to the
    standard path-dependent game, so results must match linear_tree_shap_meets_woodelf exactly.
    """
    consumer_data = testset.iloc[:N].reset_index(drop=True)

    expected = linear_tree_shap_meets_woodelf(xgb_model, consumer_data, metric)
    actual = path_dependent_under_always_participating_features(
        xgb_model, consumer_data, metric, always_participating_features=[], verbose=False
    )

    assert_woodelf_dicts_equal(actual, expected, testset, TOLERANCE)


def test_delta_trick_is_exact(testset, xgb_model):
    """
    Verify that always_participating_delta_update produces exactly the same result as a full
    recompute after promoting a feature subset from regular to always-participating.
    """
    consumer_data = testset.iloc[:N].reset_index(drop=True)
    metric = BanzhafValues()

    prev_always_participating = []
    new_always_participating = list(testset.columns[:2])
    changed_features = new_always_participating

    full_result_old = path_dependent_under_always_participating_features(
        xgb_model, consumer_data, metric, prev_always_participating, verbose=False
    )
    full_result_new = path_dependent_under_always_participating_features(
        xgb_model, consumer_data, metric, new_always_participating, verbose=False
    )

    updated = always_participating_delta_update(
        xgb_model, consumer_data,
        prev_always_participating, new_always_participating,
        changed_features, full_result_old, metric,
    )

    assert_woodelf_dicts_equal(updated, full_result_new, testset, TOLERANCE)

    # Second delta: add a third feature on top of the existing two
    prev_always_participating = new_always_participating
    new_always_participating = list(testset.columns[:3])
    changed_features = [testset.columns[2]]

    full_result_old = full_result_new
    full_result_new = path_dependent_under_always_participating_features(
        xgb_model, consumer_data, metric, new_always_participating, verbose=False
    )

    updated = always_participating_delta_update(
        xgb_model, consumer_data,
        prev_always_participating, new_always_participating,
        changed_features, full_result_old, metric,
    )

    assert_woodelf_dicts_equal(updated, full_result_new, testset, TOLERANCE)
