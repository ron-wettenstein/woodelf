import time

import numpy as np
import pytest
import shap

from shared_fixtures_and_utils import testset, xgb_model
from woodelf.core.cube_metric import ShapleyValues
from woodelf.personalized_woodelf import personalized_baseline_woodelf

TOLERANCE = 0.00001
N = 10


@pytest.mark.parametrize("use_neighbor_leaf_trick", [False, True])
def test_personalized_woodelf_matches_shap_package(testset, xgb_model, use_neighbor_leaf_trick):
    consumer_data   = testset.iloc[:N].reset_index(drop=True)
    background_data = testset.iloc[N:2*N].reset_index(drop=True)
    metric = ShapleyValues()

    t0 = time.perf_counter()
    personalized_result = personalized_baseline_woodelf(
        xgb_model, consumer_data, background_data, metric, use_neighbor_leaf_trick=use_neighbor_leaf_trick
    )
    t_personalized = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(N):
        explainer = shap.TreeExplainer(xgb_model, background_data.iloc[i:i+1], feature_perturbation="interventional")
        shap_vals = explainer.shap_values(consumer_data.iloc[i:i+1])  # (1, n_features)
        for j, feature in enumerate(consumer_data.columns):
            personalized_val = personalized_result[feature][i] if feature in personalized_result else 0.0
            np.testing.assert_allclose(
                personalized_val, shap_vals[0, j], atol=TOLERANCE,
                err_msg=f"Mismatch at consumer {i}, feature {feature}"
            )
    t_shap = time.perf_counter() - t0

    print(f"\n[{metric.__class__.__name__}, neighbor_leaf_trick={use_neighbor_leaf_trick}] "
          f"personalized={t_personalized:.3f}s | {N}x shap package={t_shap:.3f}s")
