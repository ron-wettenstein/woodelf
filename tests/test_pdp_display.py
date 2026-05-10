"""
Tests for WoodelfPartialDependenceDisplay.

Covers:
- _build_2way_grid: pure unit tests
- from_estimator: Bunch structure, method→accurate/centered mapping (mocked)
- TestIntegration: end-to-end on a real sklearn model — values vs sklearn brute/recursion,
  2-way orientation via direct computation, rendering accepted by PartialDependenceDisplay
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest
import sklearn.inspection
from sklearn.ensemble import HistGradientBoostingRegressor
from unittest.mock import patch, MagicMock

from shared_fixtures_and_utils import trainset, hist_gradient_boosting_model

FIXTURES = [trainset, hist_gradient_boosting_model]

from woodelf.pdp_display import WoodelfPartialDependenceDisplay


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DUMMY_X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})


def _mock_pdp_return():
    return (
        {"f1": np.array([0.1, 0.2, 0.3], dtype=np.float32),
         "f2": np.array([0.4, 0.5, 0.6], dtype=np.float32)},
        pd.DataFrame({"f1": [1., 2., 3.], "f2": [4., 5., 6.]}),
    )


def _mock_joint_return():
    # 2 features, k=2 → k²=4 raw values; f1 tiled (fast), f2 repeated (slow)
    return (
        {("f1", "f2"): np.array([1., 2., 3., 4.], dtype=np.float32)},
        {("f1", "f2"): np.array([1., 2., 1., 2.])},
        {("f1", "f2"): np.array([4., 4., 5., 5.])},
    )


# ---------------------------------------------------------------------------
# _build_2way_grid
# ---------------------------------------------------------------------------

class TestBuild2WayGrid:
    """Pure unit tests — no model or sklearn required."""

    def test_f1_fast_axis_shape(self):
        # n_features=2, i1=0 < i2=1 → bit_f1=0 (f1 fast) → result shape (k_f1, k_f2)
        raw = np.arange(12, dtype=np.float32)
        result = WoodelfPartialDependenceDisplay._build_2way_grid(
            raw, i1=0, i2=1, k_f1=3, k_f2=4, n_features=2
        )
        assert result.shape == (3, 4)

    def test_f1_fast_axis_values(self):
        # bit_f1=0: raw is laid out C-order (k_f2, k_f1), i.e. raw[j*k_f1+i] = PDP(f1[i], f2[j])
        # After reshape(k_f2, k_f1).T → result[i, j] = raw[j*k_f1+i]
        k_f1, k_f2 = 3, 4
        raw = np.arange(k_f1 * k_f2, dtype=np.float32)
        result = WoodelfPartialDependenceDisplay._build_2way_grid(
            raw, i1=0, i2=1, k_f1=k_f1, k_f2=k_f2, n_features=2
        )
        for i in range(k_f1):
            for j in range(k_f2):
                assert result[i, j] == raw[j * k_f1 + i], f"Mismatch at [{i},{j}]"

    def test_f1_slow_axis_shape(self):
        # n_features=4, i1=2([1,0]) > i2=0([0,0]): h=0, bit_f1=1 (f1 slow) → result shape (k_f1, k_f2)
        raw = np.arange(12, dtype=np.float32)
        result = WoodelfPartialDependenceDisplay._build_2way_grid(
            raw, i1=2, i2=0, k_f1=3, k_f2=4, n_features=4
        )
        assert result.shape == (3, 4)

    def test_f1_slow_axis_values(self):
        # bit_f1=1: raw is laid out C-order (k_f1, k_f2), i.e. raw[i*k_f2+j] = PDP(f1[i], f2[j])
        # reshape(k_f1, k_f2) → result[i, j] = raw[i*k_f2+j]
        k_f1, k_f2 = 3, 4
        raw = np.arange(k_f1 * k_f2, dtype=np.float32)
        result = WoodelfPartialDependenceDisplay._build_2way_grid(
            raw, i1=2, i2=0, k_f1=k_f1, k_f2=k_f2, n_features=4
        )
        for i in range(k_f1):
            for j in range(k_f2):
                assert result[i, j] == raw[i * k_f2 + j], f"Mismatch at [{i},{j}]"

    def test_square_grid(self):
        k = 5
        raw = np.arange(k * k, dtype=np.float32)
        result = WoodelfPartialDependenceDisplay._build_2way_grid(
            raw, i1=0, i2=1, k_f1=k, k_f2=k, n_features=2
        )
        assert result.shape == (k, k)

    def test_returns_float32(self):
        raw = np.arange(6, dtype=np.float64)
        result = WoodelfPartialDependenceDisplay._build_2way_grid(
            raw, i1=0, i2=1, k_f1=2, k_f2=3, n_features=2
        )
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# from_estimator — 1-way PDP structure
# ---------------------------------------------------------------------------

class TestFromEstimatorOneway:
    @pytest.fixture(autouse=True)
    def mock_pdp(self):
        with patch("woodelf.pdp_display.woodelf_pdp") as m:
            m.return_value = _mock_pdp_return()
            self._mock = m
            yield m

    def test_pdvs_contains_all_features(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=True,
        )
        assert set(display._pdvs.keys()) == {"f1", "f2"}

    def test_joint_pdvs_empty_when_not_requested(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=True, compute_joint_pdp=False,
        )
        assert display._joint_pdvs == {}

    def test_bunch_average_shape_1way(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=True,
        )
        b = display._pdvs["f1"]
        assert b.average.ndim == 2
        assert b.average.shape[0] == 1
        assert len(b.grid_values) == 1

    def test_bunch_average_dtype_float64(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=True,
        )
        assert display._pdvs["f1"].average.dtype == np.float64

    def test_recursion_method_passes_accurate_false_centered_true(self):
        WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=True, method="recursion",
        )
        _, kw = self._mock.call_args
        assert kw["accurate"] is False
        assert kw["centered"] is True

    def test_brute_method_passes_accurate_true_centered_false(self):
        WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=True, method="brute",
        )
        _, kw = self._mock.call_args
        assert kw["accurate"] is True
        assert kw["centered"] is False

    def test_auto_method_passes_accurate_true(self):
        WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=True, method="auto",
        )
        _, kw = self._mock.call_args
        assert kw["accurate"] is True

    def test_deciles_keyed_by_int_index(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=True,
        )
        assert 0 in display._deciles and 1 in display._deciles
        assert len(display._deciles[0]) == 10

    def test_name_to_idx_correct(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=True,
        )
        assert display._name_to_idx == {"f1": 0, "f2": 1}


# ---------------------------------------------------------------------------
# from_estimator — 2-way joint PDP structure
# ---------------------------------------------------------------------------

class TestFromEstimatorTwoway:
    @pytest.fixture(autouse=True)
    def mock_joint(self):
        with patch("woodelf.pdp_display.woodelf_pdp_joint") as m:
            m.return_value = _mock_joint_return()
            yield m

    def test_joint_pdvs_contains_pair(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=False, compute_joint_pdp=True,
        )
        assert ("f1", "f2") in display._joint_pdvs

    def test_pdvs_empty_when_not_requested(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=False, compute_joint_pdp=True,
        )
        assert display._pdvs == {}

    def test_bunch_average_shape_2way(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=False, compute_joint_pdp=True,
        )
        b = display._joint_pdvs[("f1", "f2")]
        assert b.average.ndim == 3
        assert b.average.shape[0] == 1
        assert len(b.grid_values) == 2

    def test_bunch_average_dtype_float64(self):
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=False, compute_joint_pdp=True,
        )
        assert display._joint_pdvs[("f1", "f2")].average.dtype == np.float64

    def test_bunch_values_2way(self):
        """
        raw=[1,2,3,4], f1 fast (i1=0 < i2=1, n_features=2 → bit_f1=0)
        reshape(k_f2=2, k_f1=2)=[[1,2],[3,4]], .T=[[1,3],[2,4]]
        Verified: raw[j*k_f1+i] = PDP at (f1_unique[i], f2_unique[j]).
        """
        display = WoodelfPartialDependenceDisplay.from_estimator(
            MagicMock(), DUMMY_X, compute_pdp=False, compute_joint_pdp=True,
        )
        expected = np.array([[1., 3.], [2., 4.]], dtype=np.float32)
        np.testing.assert_array_equal(display._joint_pdvs[("f1", "f2")].average[0], expected)


# ---------------------------------------------------------------------------
# Integration tests — real model, no mocking
# ---------------------------------------------------------------------------

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


class TestIntegration:
    """
    End-to-end tests on a real model — no mocking anywhere.

    1-way PDPs (both recursion and brute) are verified against sklearn's
    partial_dependence using WOODELF's own grid as custom_values, so both
    sides evaluate at identical points.

    2-way PDPs are verified by direct computation: for every grid point (i, j),
    average[0][i, j] must equal mean(model.predict(X with f1=x_grid[i], f2=y_grid[j])).
    This simultaneously checks correctness and the axis orientation from _build_2way_grid.

    Rendering tests verify that the Bunch format WOODELF assembles is accepted
    by sklearn's PartialDependenceDisplay without error.
    """

    GRID_K = 5
    TOLERANCE = 1e-5

    def test_1way_values_match_sklearn_recursion(self, small_model_and_data):
        model, X = small_model_and_data
        display = WoodelfPartialDependenceDisplay.from_estimator(
            model, X, compute_pdp=True, method="recursion", grid_resolution=self.GRID_K,
        )
        for feat in X.columns:
            grid = display._pdvs[feat].grid_values[0]
            woodelf_avg = display._pdvs[feat].average[0]
            sk = sklearn.inspection.partial_dependence(
                model, X, features=[feat], method="recursion", kind="average",
                custom_values={feat: grid},
            )
            np.testing.assert_allclose(
                woodelf_avg, sk["average"][0],
                atol=self.TOLERANCE, err_msg=f"recursion mismatch for {feat!r}",
            )

    def test_1way_values_match_sklearn_brute(self, small_model_and_data):
        model, X = small_model_and_data
        display = WoodelfPartialDependenceDisplay.from_estimator(
            model, X, compute_pdp=True, method="brute", grid_resolution=self.GRID_K,
        )
        for feat in X.columns:
            grid = display._pdvs[feat].grid_values[0]
            woodelf_avg = display._pdvs[feat].average[0]
            sk = sklearn.inspection.partial_dependence(
                model, X, features=[feat], method="brute", kind="average",
                custom_values={feat: grid},
            )
            np.testing.assert_allclose(
                woodelf_avg, sk["average"][0],
                atol=self.TOLERANCE, err_msg=f"brute mismatch for {feat!r}",
            )

    def test_2way_values_by_direct_computation(self, small_model_and_data):
        """
        average[0][i, j] == mean prediction with f1=x_grid[i], f2=y_grid[j].
        Checks both value correctness and axis orientation from _build_2way_grid.
        """
        model, X = small_model_and_data
        f1, f2 = "a", "b"
        display = WoodelfPartialDependenceDisplay.from_estimator(
            model, X, compute_pdp=False, compute_joint_pdp=True,
            grid_resolution=self.GRID_K,
        )
        b = display._joint_pdvs[(f1, f2)]
        x_grid, y_grid = b.grid_values   # x_grid: f1 axis, y_grid: f2 axis
        woodelf_avg = b.average[0]        # shape (k_f1, k_f2)
        for i, v1 in enumerate(x_grid):
            for j, v2 in enumerate(y_grid):
                X_mod = X.copy()
                X_mod[f1] = v1
                X_mod[f2] = v2
                direct = model.predict(X_mod).mean()
                assert abs(woodelf_avg[i, j] - direct) < self.TOLERANCE, (
                    f"2-way mismatch at ({f1}={v1:.3f}, {f2}={v2:.3f}): "
                    f"woodelf={woodelf_avg[i, j]:.6f}, direct={direct:.6f}"
                )

    def test_plot_1way_accepted_by_sklearn_display(self, small_model_and_data):
        """1-way Bunch format accepted by PartialDependenceDisplay (brute)."""
        model, X = small_model_and_data
        display = WoodelfPartialDependenceDisplay.from_estimator(
            model, X, compute_pdp=True, method="brute", grid_resolution=self.GRID_K,
        )
        display.plot()
        plt.close("all")

    def test_plot_joint_accepted_by_sklearn_display(self, small_model_and_data):
        """2-way Bunch format accepted by PartialDependenceDisplay."""
        model, X = small_model_and_data
        display = WoodelfPartialDependenceDisplay.from_estimator(
            model, X, compute_pdp=False, compute_joint_pdp=True,
            grid_resolution=self.GRID_K,
        )
        display.plot()
        plt.close("all")

    def test_plot_pdp_subset_renders(self, small_model_and_data):
        model, X = small_model_and_data
        display = WoodelfPartialDependenceDisplay.from_estimator(
            model, X, compute_pdp=True, method="brute", grid_resolution=self.GRID_K,
        )
        display.plot_pdp(["a", "b"])
        plt.close("all")

    def test_plot_joint_pdp_subset_renders(self, small_model_and_data):
        model, X = small_model_and_data
        display = WoodelfPartialDependenceDisplay.from_estimator(
            model, X, compute_pdp=False, compute_joint_pdp=True,
            grid_resolution=self.GRID_K,
        )
        display.plot_joint_pdp([("a", "b")])
        plt.close("all")
