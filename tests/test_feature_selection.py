from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from woodelf.feature_selection import feature_selection_ranking, _monotone_xgboost_impute, _init_background

N = 10
N_TOTAL = 60
N_TRAIN = 50


def _make_base_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'A': rng.standard_normal(N_TOTAL),
        'B': rng.standard_normal(N_TOTAL),
        'C': rng.standard_normal(N_TOTAL),
        'D': rng.standard_normal(N_TOTAL),
        'E': rng.standard_normal(N_TOTAL),
    })


def _train_model(X: pd.DataFrame) -> xgb.Booster:
    rng = np.random.default_rng(42)
    y = 2 * X['A'] - X['B'] + 0.5 * X['C'] + rng.standard_normal(len(X)) * 0.1
    return xgb.train(
        {'max_depth': 3, 'nthread': 1, 'seed': 42},
        xgb.DMatrix(X, label=y),
        num_boost_round=20,
    )


@pytest.fixture
def consumer():
    return _make_base_data().iloc[N_TRAIN:].reset_index(drop=True)


@pytest.fixture
def sim_model():
    return _train_model(_make_base_data().iloc[:N_TRAIN])


# --- structural correctness ---

def test_ranking_structure(consumer, sim_model):
    ranking, values = feature_selection_ranking(sim_model, consumer)
    print(ranking)
    assert set(ranking) == set(consumer.columns)
    assert len(ranking) == len(set(ranking))
    assert len(ranking) == len(consumer.columns)
    for f, v in values.items():
        assert isinstance(v, np.ndarray), f"values[{f!r}] is not an ndarray"
        assert len(v) == N, f"values[{f!r}] has wrong length"


# --- auto initial selection ---

def test_auto_initial_selection_prepended_with_values(consumer, sim_model):
    ranking, values = feature_selection_ranking(sim_model, consumer)
    top_k = 3 if len(consumer.columns) <= 30 else 5
    for f in ranking[:top_k]:
        assert f in values, f"auto-selected feature {f!r} missing from ranking_values"
        assert len(values[f]) == N


# --- provided initial selection ---

def test_provided_initial_selection_order_preserved(consumer, sim_model):
    provided = list(['A'])
    ranking, _ = feature_selection_ranking(sim_model, consumer, initial_selection=provided)
    print(ranking)
    assert ranking[:1] == provided


# --- correlated features are neutralized ---

def corr_consumer_and_model() -> Tuple[pd.DataFrame, xgb.Booster]:
    data = _make_base_data()
    data['A_same']     = data['A']
    data['B_factored'] = 3 * data['B']
    data['A_corr_1']   = 2 - 0.6 * data['A']
    data['B_corr_1']   = 2 + 2 * data['B']
    return data.iloc[N_TRAIN:].reset_index(drop=True), _train_model(data.iloc[:N_TRAIN])


def test_correlated_features_neutralized_with_provided_anchors():
    consumer, model = corr_consumer_and_model()
    ranking, values = feature_selection_ranking(model, consumer, initial_selection=['A', 'B'])
    print(ranking)
    for f in ['A_same', 'B_factored', 'A_corr_1', 'B_corr_1']:
        v = values.get(f, np.zeros(N))
        np.testing.assert_array_equal(v, np.zeros(N), err_msg=f"{f!r} should have zero contribution when A and B are anchors")

    ranking, values = feature_selection_ranking(model, consumer, initial_selection=['A_corr_1', 'B_corr_1'])
    print(ranking)
    for f in ['A_same', 'B_factored', 'A', 'B']:
        v = values.get(f, np.zeros(N))
        np.testing.assert_array_equal(v, np.zeros(N), err_msg=f"{f!r} should have zero contribution when A_corr_1 and B_corr_1 are anchors")


def test_correlated_features_neutralized_with_auto_selection():
    consumer, model = corr_consumer_and_model()
    a_group = {'A', 'A_same', 'A_corr_1'}
    b_group = {'B', 'B_factored', 'B_corr_1'}
    ranking, values = feature_selection_ranking(model, consumer)
    print(ranking)
    initial, after_initial = set(ranking[:3]), ranking[3:]

    for group_name, group in [('A', a_group), ('B', b_group)]:
        nonzero = [f for f in after_initial if f in group and np.any(values.get(f, np.zeros(N)) != 0)]
        if len(initial & group) > 0:
            assert len(nonzero) == 0, f"{group_name}-group anchor in initial selection but {nonzero} have non-zero contribution"
        else:
            assert len(nonzero) <= 1, f"No {group_name}-group anchor selected but {nonzero} have non-zero contribution"


# --- monoton_xgboost baseline scheme ---

_XGB_N_TOTAL = 500
_XGB_N_TRAIN = 400
_XGB_N_CONSUMER = _XGB_N_TOTAL - _XGB_N_TRAIN  # 100


def _monotone_redundancy_data() -> pd.DataFrame:
    # 'x3' is a monotone-but-NON-linear function of the anchor 'x'. 'x' is placed last so that its
    # column index sits beyond the model's features (model is trained on the first five columns),
    # keeping the parser's positional feature mapping aligned while 'x' stays an unused-by-model anchor.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(_XGB_N_TOTAL)
    return pd.DataFrame({
        'x3': x ** 3,
        'B': rng.standard_normal(_XGB_N_TOTAL),
        'C': rng.standard_normal(_XGB_N_TOTAL),
        'D': rng.standard_normal(_XGB_N_TOTAL),
        'E': rng.standard_normal(_XGB_N_TOTAL),
        'x': x,
    })


def _redundancy_model(df: pd.DataFrame) -> xgb.Booster:
    model_cols = ['x3', 'B', 'C', 'D', 'E']  # excludes 'x' so the model genuinely splits on the nonlinear x3
    train = df.iloc[:_XGB_N_TRAIN]
    rng = np.random.default_rng(1)
    y = 2 * train['x3'] - train['B'] + 0.5 * train['C'] + rng.standard_normal(_XGB_N_TRAIN) * 0.1
    return xgb.train(
        {'max_depth': 4, 'nthread': 1, 'seed': 1},
        xgb.DMatrix(train[model_cols], label=y),
        num_boost_round=40,
    )


def test_monoton_xgboost_ranking_structure():
    df = _monotone_redundancy_data()
    model = _redundancy_model(df)
    consumer = df.iloc[_XGB_N_TRAIN:].reset_index(drop=True)
    ranking, values = feature_selection_ranking(
        model, consumer,
        baseline_init_scheme="monoton_xgboost", baseline_updating_scheme="monoton_xgboost",
    )
    assert set(ranking) == set(consumer.columns)
    assert len(ranking) == len(set(ranking)) == len(consumer.columns)
    for f, v in values.items():
        assert isinstance(v, np.ndarray) and len(v) == _XGB_N_CONSUMER


def test_monoton_xgboost_neutralizes_nonlinear_redundancy_better_than_pearson():
    df = _monotone_redundancy_data()
    model = _redundancy_model(df)
    consumer = df.iloc[_XGB_N_TRAIN:].reset_index(drop=True)

    def x3_residual(scheme: str) -> float:
        _, values = feature_selection_ranking(
            model, consumer, initial_selection=['x', 'B', 'C'],
            baseline_init_scheme=scheme, baseline_updating_scheme=scheme,
        )
        return float(np.mean(np.abs(values.get('x3', np.zeros(_XGB_N_CONSUMER)))))

    pearson = x3_residual("pearson_correlation")
    monotone = x3_residual("monoton_xgboost")
    # The OLS line a + b*x cannot cancel the cubic, so x3 keeps a large residual; the monotone XGBoost
    # imputes x3 from x far better, so x3 is recognised as more redundant (smaller residual).
    assert monotone < pearson, f"monoton_xgboost residual {monotone} should be < pearson residual {pearson}"


def test_monotone_xgboost_impute_falls_back_on_degenerate():
    # Constant anchor -> undefined correlation -> constant (mean) baseline, mirroring _linreg_coeffs.
    x = pd.Series(np.ones(100))
    y = pd.Series(np.arange(100, dtype=float))
    out = _monotone_xgboost_impute(x, y, n=100)
    np.testing.assert_allclose(out, float(y.mean()))


# --- min_corr threshold ---

def test_min_corr_forces_mean_baseline_for_weak_anchor():
    # 'strong' is ~perfectly correlated with anchor 'a'; 'weak' is independent of it.
    rng = np.random.default_rng(7)
    a = rng.standard_normal(200)
    consumer = pd.DataFrame({
        'a': a,
        'strong': a + rng.standard_normal(200) * 0.1,
        'weak': rng.standard_normal(200),
    })
    anchor_info = {'strong': ('a', 0.9), 'weak': ('a', 0.02)}
    remaining = ['strong', 'weak']

    # Threshold above the weak anchor's |corr|: 'weak' falls back to a constant mean baseline,
    # while 'strong' is still imputed (non-constant).
    B = _init_background(consumer, remaining, "pearson_correlation", anchor_info, len(consumer), min_corr=0.5)
    np.testing.assert_allclose(B['weak'].to_numpy(), consumer['weak'].mean())
    assert B['strong'].nunique() > 1

    # No threshold: 'weak' is imputed via OLS, not forced to the constant mean.
    B_none = _init_background(consumer, remaining, "pearson_correlation", anchor_info, len(consumer), min_corr=None)
    assert not np.allclose(B_none['weak'].to_numpy(), consumer['weak'].mean())


def test_min_corr_none_matches_default(consumer, sim_model):
    r_default, v_default = feature_selection_ranking(sim_model, consumer)
    r_none, v_none = feature_selection_ranking(sim_model, consumer, min_corr=None)
    assert r_default == r_none
    assert set(v_default) == set(v_none)
    for f in v_default:
        np.testing.assert_array_equal(v_default[f], v_none[f])


def test_min_corr_runs_end_to_end(consumer, sim_model):
    ranking, values = feature_selection_ranking(sim_model, consumer, min_corr=0.3)
    assert set(ranking) == set(consumer.columns)
    assert len(ranking) == len(set(ranking)) == len(consumer.columns)
    for f, v in values.items():
        assert isinstance(v, np.ndarray) and len(v) == N


def test_min_corr_out_of_range_raises(consumer, sim_model):
    with pytest.raises(ValueError):
        feature_selection_ranking(sim_model, consumer, min_corr=1.5)
