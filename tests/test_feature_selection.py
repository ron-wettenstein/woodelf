from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from woodelf.feature_selection import feature_selection_ranking

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
