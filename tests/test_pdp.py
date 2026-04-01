import numpy as np
import sklearn

from shared_fixtures_and_utils import trainset, testset, hist_gradient_boosting_model
from woodelf.pdp import woodelf_fast_pdp, build_points_for_pdp

FIXTURES = [testset, trainset, hist_gradient_boosting_model]

TOLERANCE = 0.00001

def test_calculate_background_metric_for_high_depth(trainset, hist_gradient_boosting_model):
    points_df = build_points_for_pdp(hist_gradient_boosting_model, trainset, k=5)
    pdp_woodelf_values = woodelf_fast_pdp(
        hist_gradient_boosting_model, consumer_data=points_df, background_data=trainset, GPU = False, model_was_loaded = False, centered = False
    )

    for f in trainset.columns:
        print(f)
        estimated_pdp_result_sklean = sklearn.inspection.partial_dependence(
            estimator=hist_gradient_boosting_model, X=trainset, features=[f], feature_names=[f],
            custom_values={f: points_df[f]},
            method='brute', kind='average'
        )
        if f not in pdp_woodelf_values:
            # pdv should be all the mean prediction, so std should be zero
            assert np.std(np.abs(estimated_pdp_result_sklean['average'][0])) < TOLERANCE
        else:
            assert np.max(np.abs(estimated_pdp_result_sklean['average'][0] - pdp_woodelf_values[f])) < TOLERANCE

        # np.testing.assert_allclose(simple_woodelf_values[feature], high_depth_woodelf_values[feature], atol=TOLERANCE)
