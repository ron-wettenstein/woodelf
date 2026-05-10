"""
Tests for WoodelfPartialDependenceDisplay.

Covers:
- _build_2way_grid: standalone unit test
- from_estimator: Bunch structure, method→accurate/centered mapping (mocked)
- Integration: end-to-end on a real sklearn model — grid and values compared
  directly against sklearn's PartialDependenceDisplay.from_estimator for both
  1-way (brute and recursion) and 2-way PDPs; full_pdp smoke test; rendering.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from unittest.mock import patch, MagicMock

from shared_fixtures_and_utils import trainset, hist_gradient_boosting_model

FIXTURES = [trainset, hist_gradient_boosting_model]

from woodelf.pdp_display import WoodelfPartialDependenceDisplay


# ---------------------------------------------------------------------------
# Shared helpers and constants
# ---------------------------------------------------------------------------

DUMMY_X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})

GRID_K = 5
JOINT_PAIRS = [("a", "b"), ("a", "c"), ("b", "c")]
GRID_TOLERANCE = 1e-5
VALUE_TOLERANCE = 1e-5


def _mock_pdp_return():
    return (
        {"f1": np.array([0.1, 0.2, 0.3], dtype=np.float32),
         "f2": np.array([0.4, 0.5, 0.6], dtype=np.float32)},
        pd.DataFrame({"f1": [1., 2., 3.], "f2": [4., 5., 6.]}),
    )


@pytest.fixture
def mock_pdp():
    with patch("woodelf.pdp_display.woodelf_pdp") as m:
        m.return_value = _mock_pdp_return()
        yield m


@pytest.fixture(scope="module")
def small_model_and_data():
    """
    Tiny HistGradientBoostingRegressor on 3 synthetic features.
    Module-scoped so training runs once for all integration tests.
    3 features → 3 joint pairs, k=5 → 75 grid points total: fast.
    """
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "a": rng.randn(300),
        "b": rng.randn(300),
        "c": rng.randn(300),
    })
    y = X["a"] * 2 + X["b"] ** 2
    model = HistGradientBoostingRegressor(max_iter=30, max_depth=4, random_state=42)
    model.fit(X, y)
    return model, X


# ---------------------------------------------------------------------------
# _build_2way_grid
# ---------------------------------------------------------------------------

def test_build_2way_grid():
    """
    Covers f1-fast axis (i1 < i2), f1-slow axis (i1 > i2), shape, values, dtype.
    raw layout for f1-fast: (k_f2, k_f1) C-order → result = reshape.T
    raw layout for f1-slow: (k_f1, k_f2) C-order → result = reshape directly
    """
    build = WoodelfPartialDependenceDisplay._build_2way_grid
    k_f1, k_f2 = 3, 4
    raw = np.arange(k_f1 * k_f2, dtype=np.float64)

    # f1 fast axis: i1=0 < i2=1, n_features=2 → bit_f1=0
    result = build(raw, i1=0, i2=1, k_f1=k_f1, k_f2=k_f2, n_features=2)
    assert result.shape == (k_f1, k_f2)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, raw.reshape(k_f2, k_f1).T.astype(np.float32))

    # f1 slow axis: i1=2 > i2=0, n_features=4 → bit_f1=1
    result2 = build(raw, i1=2, i2=0, k_f1=k_f1, k_f2=k_f2, n_features=4)
    assert result2.shape == (k_f1, k_f2)
    assert result2.dtype == np.float32
    np.testing.assert_array_equal(result2, raw.reshape(k_f1, k_f2).astype(np.float32))


# ---------------------------------------------------------------------------
# from_estimator — structure and method mapping (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("mock_pdp")
def test_pdvs_contains_all_features():
    display = WoodelfPartialDependenceDisplay.from_estimator(
        MagicMock(), DUMMY_X, compute_pdp=True,
    )
    assert set(display._pdvs.keys()) == {"f1", "f2"}


@pytest.mark.usefixtures("mock_pdp")
def test_joint_pdvs_empty_when_not_requested():
    display = WoodelfPartialDependenceDisplay.from_estimator(
        MagicMock(), DUMMY_X, compute_pdp=True, compute_joint_pdp=False,
    )
    assert display._joint_pdvs == {}


@pytest.mark.usefixtures("mock_pdp")
def test_bunch_average_shape_1way():
    display = WoodelfPartialDependenceDisplay.from_estimator(
        MagicMock(), DUMMY_X, compute_pdp=True,
    )
    b = display._pdvs["f1"]
    assert b.average.ndim == 2
    assert b.average.shape[0] == 1
    assert len(b.grid_values) == 1


@pytest.mark.usefixtures("mock_pdp")
def test_bunch_average_dtype_float64():
    display = WoodelfPartialDependenceDisplay.from_estimator(
        MagicMock(), DUMMY_X, compute_pdp=True,
    )
    assert display._pdvs["f1"].average.dtype == np.float64


def test_method_mapping(mock_pdp):
    """recursion → accurate=False, centered=True; brute/auto → accurate=True."""
    WoodelfPartialDependenceDisplay.from_estimator(
        MagicMock(), DUMMY_X, compute_pdp=True, method="recursion",
    )
    _, kw = mock_pdp.call_args
    assert kw["accurate"] is False and kw["centered"] is True

    WoodelfPartialDependenceDisplay.from_estimator(
        MagicMock(), DUMMY_X, compute_pdp=True, method="brute",
    )
    _, kw = mock_pdp.call_args
    assert kw["accurate"] is True and kw["centered"] is False

    WoodelfPartialDependenceDisplay.from_estimator(
        MagicMock(), DUMMY_X, compute_pdp=True, method="auto",
    )
    _, kw = mock_pdp.call_args
    assert kw["accurate"] is True


@pytest.mark.usefixtures("mock_pdp")
def test_deciles_keyed_by_int_index():
    display = WoodelfPartialDependenceDisplay.from_estimator(
        MagicMock(), DUMMY_X, compute_pdp=True,
    )
    assert 0 in display._deciles and 1 in display._deciles
    assert len(display._deciles[0]) == 10


@pytest.mark.usefixtures("mock_pdp")
def test_name_to_idx_correct():
    display = WoodelfPartialDependenceDisplay.from_estimator(
        MagicMock(), DUMMY_X, compute_pdp=True,
    )
    assert display._name_to_idx == {"f1": 0, "f2": 1}


# ---------------------------------------------------------------------------
# Integration tests — real model, compared against sklearn's display
# ---------------------------------------------------------------------------

def test_1way_brute_grid_and_values_match_sklearn(small_model_and_data):
    model, X = small_model_and_data
    w = WoodelfPartialDependenceDisplay.from_estimator(
        model, X, compute_pdp=True, method="brute", grid_resolution=GRID_K,
    )
    sk = PartialDependenceDisplay.from_estimator(
        model, X, features=list(X.columns), kind="average",
        method="brute", grid_resolution=GRID_K,
    )
    for i, feat in enumerate(X.columns):
        w_grid = w._pdvs[feat].grid_values[0]
        np.testing.assert_allclose(
            w_grid, sk.pd_results[i].grid_values[0],
            atol=GRID_TOLERANCE, err_msg=f"brute grid mismatch for {feat!r}",
        )
        sk_vals = partial_dependence(
            model, X, features=[feat], method="brute", kind="average",
            custom_values={feat: w_grid},
        )
        np.testing.assert_allclose(
            w._pdvs[feat].average[0], sk_vals["average"][0],
            atol=VALUE_TOLERANCE, err_msg=f"brute value mismatch for {feat!r}",
        )


def test_1way_recursion_grid_and_values_match_sklearn(small_model_and_data):
    model, X = small_model_and_data
    w = WoodelfPartialDependenceDisplay.from_estimator(
        model, X, compute_pdp=True, method="recursion", grid_resolution=GRID_K,
    )
    sk = PartialDependenceDisplay.from_estimator(
        model, X, features=list(X.columns), kind="average",
        method="recursion", grid_resolution=GRID_K,
    )
    for i, feat in enumerate(X.columns):
        w_grid = w._pdvs[feat].grid_values[0]
        np.testing.assert_allclose(
            w_grid, sk.pd_results[i].grid_values[0],
            atol=GRID_TOLERANCE, err_msg=f"recursion grid mismatch for {feat!r}",
        )
        sk_vals = partial_dependence(
            model, X, features=[feat], method="recursion", kind="average",
            custom_values={feat: w_grid},
        )
        np.testing.assert_allclose(
            w._pdvs[feat].average[0], sk_vals["average"][0],
            atol=VALUE_TOLERANCE, err_msg=f"recursion value mismatch for {feat!r}",
        )


def test_joint_pdp_grid_and_values_match_sklearn(small_model_and_data):
    model, X = small_model_and_data
    w = WoodelfPartialDependenceDisplay.from_estimator(
        model, X, compute_pdp=False, compute_joint_pdp=True,
        grid_resolution=GRID_K,
    )
    sk = PartialDependenceDisplay.from_estimator(
        model, X, features=JOINT_PAIRS, kind="average",
        method="brute", grid_resolution=GRID_K,
    )
    for i, (f1, f2) in enumerate(JOINT_PAIRS):
        w_b = w._joint_pdvs[(f1, f2)]
        sk_b = sk.pd_results[i]
        np.testing.assert_allclose(
            w_b.grid_values[0], sk_b.grid_values[0],
            atol=GRID_TOLERANCE, err_msg=f"f1 grid mismatch for ({f1},{f2})",
        )
        np.testing.assert_allclose(
            w_b.grid_values[1], sk_b.grid_values[1],
            atol=GRID_TOLERANCE, err_msg=f"f2 grid mismatch for ({f1},{f2})",
        )
        sk_vals = partial_dependence(
            model, X, features=[f1, f2], method="brute", kind="average",
            custom_values={f1: w_b.grid_values[0], f2: w_b.grid_values[1]},
        )
        np.testing.assert_allclose(
            w_b.average[0], sk_vals["average"][0],
            atol=VALUE_TOLERANCE, err_msg=f"value mismatch for ({f1},{f2})",
        )


def test_full_pdp_no_crash_and_variable_grid(small_model_and_data):
    """
    full_pdp=True assigns each feature its own grid from tree threshold values.
    Features that appear in fewer splits get fewer unique thresholds → smaller grid.
    """
    model, X = small_model_and_data
    display = WoodelfPartialDependenceDisplay.from_estimator(
        model, X, compute_pdp=True, grid_resolution=GRID_K, full_pdp=True,
    )
    grid_lengths = {f: len(display._pdvs[f].grid_values[0]) for f in X.columns}
    assert len(set(grid_lengths.values())) > 1, (
        f"full_pdp should produce different grid sizes per feature; got {grid_lengths}"
    )


def test_renders(small_model_and_data):
    """plot(), plot_pdp(), plot_joint_pdp() must not raise for any combination."""
    model, X = small_model_and_data
    w1 = WoodelfPartialDependenceDisplay.from_estimator(
        model, X, compute_pdp=True, method="brute", grid_resolution=GRID_K,
    )
    w1.plot()
    plt.close("all")
    w1.plot_pdp(["a", "b"])
    plt.close("all")

    w2 = WoodelfPartialDependenceDisplay.from_estimator(
        model, X, compute_pdp=False, compute_joint_pdp=True, grid_resolution=GRID_K,
    )
    w2.plot()
    plt.close("all")
    w2.plot_joint_pdp([("a", "b")])
    plt.close("all")
