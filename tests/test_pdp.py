import numpy as np
import sklearn

from shared_fixtures_and_utils import trainset, testset, hist_gradient_boosting_model
from woodelf.pdp import woodelf_fast_pdp, build_points_for_pdp, woodelf_pdp_joint

FIXTURES = [testset, trainset, hist_gradient_boosting_model]

TOLERANCE = 0.00001

def test_woodelf_pdp_vs_sklearn(trainset, hist_gradient_boosting_model):
    points_df = build_points_for_pdp(hist_gradient_boosting_model, trainset, k=5)
    pdp_woodelf_values = woodelf_fast_pdp(
        hist_gradient_boosting_model, consumer_data=points_df, background_data=trainset, GPU = False, model_was_loaded = False, centered = False
    )

    for f in trainset.columns:
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


def test_woodelf_pdp_estimation_vs_sklearn(trainset, hist_gradient_boosting_model):
    points_df = build_points_for_pdp(hist_gradient_boosting_model, trainset, k=5)
    pdp_woodelf_values = woodelf_fast_pdp(
        hist_gradient_boosting_model, consumer_data=points_df, background_data=trainset, GPU=False, model_was_loaded=False, centered=True, accurate=False
    )

    for f in trainset.columns:
        estimated_pdp_result_sklean = sklearn.inspection.partial_dependence(
            estimator=hist_gradient_boosting_model, X=trainset, features=[f], feature_names=[f],
            custom_values={f: points_df[f]},
            method='recursion', kind='average'
        )
        if f not in pdp_woodelf_values:
            # pdv should be all the mean prediction, so std should be zero
            assert np.std(np.abs(estimated_pdp_result_sklean['average'][0])) < TOLERANCE
        else:
            assert np.max(np.abs(estimated_pdp_result_sklean['average'][0] - pdp_woodelf_values[f])) < TOLERANCE



def test_woodelfhd_pdp_vs_sklearn(trainset, hist_gradient_boosting_model):
    points_df = build_points_for_pdp(hist_gradient_boosting_model, trainset, k=5)
    pdp_woodelf_values = woodelf_fast_pdp(
        hist_gradient_boosting_model, consumer_data=points_df, background_data=trainset, GPU=False, model_was_loaded=False, centered=False, use_woodelfhd=True
    )

    for f in trainset.columns:
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


def test_woodelfhd_pdp_estimation_vs_sklearn(trainset, hist_gradient_boosting_model):
    points_df = build_points_for_pdp(hist_gradient_boosting_model, trainset, k=5)
    pdp_woodelf_values = woodelf_fast_pdp(
        hist_gradient_boosting_model, consumer_data=points_df, background_data=trainset,
        GPU=False, model_was_loaded=False, centered=True, accurate=False, use_woodelfhd=True
    )

    for f in trainset.columns:
        estimated_pdp_result_sklean = sklearn.inspection.partial_dependence(
            estimator=hist_gradient_boosting_model, X=trainset, features=[f], feature_names=[f],
            custom_values={f: points_df[f]},
            method='recursion', kind='average'
        )
        if f not in pdp_woodelf_values:
            # pdv should be all the mean prediction, so std should be zero
            assert np.std(np.abs(estimated_pdp_result_sklean['average'][0])) < TOLERANCE
        else:
            assert np.max(np.abs(estimated_pdp_result_sklean['average'][0] - pdp_woodelf_values[f])) < TOLERANCE


def test_pdp_iv_vs_naive_algorithm(trainset, hist_gradient_boosting_model):
    estimated_pdp_woodelf, f1_points, f2_points = woodelf_pdp_joint(
        hist_gradient_boosting_model, data=trainset, k=5, accurate=True, GPU=False, seed=42
    )

    pairs_indexes_10 = [(98, 2),
     (364, 185),
     (187, 139),
     (372, 161),
     (67, 253),
     (4, 146),
     (363, 12),
     (280, 278),
     (307, 369),
     (215, 20)]

    for f1, f2 in pairs_indexes_10:
        feature_1 = list(trainset.columns)[f1]
        feature_2 = list(trainset.columns)[f2]

        for i, (v1, v2) in enumerate(zip(f1_points[(feature_1, feature_2)], f2_points[(feature_1, feature_2)])):
            current_train = trainset.copy()
            current_train[feature_1] = v1
            current_train[feature_2] = v2
            direct_computation = hist_gradient_boosting_model.predict(current_train).mean()
            if (feature_1, feature_2) in estimated_pdp_woodelf:
                woodelf_computation = estimated_pdp_woodelf[(feature_1, feature_2)][i]
            else:
                woodelf_computation = 0

            assert abs(direct_computation - woodelf_computation) < TOLERANCE, f"{(feature_1, feature_2)} ({f1, f2}) pdv do not agree: {direct_computation} != {woodelf_computation}  ({v1=}, {v2=})"