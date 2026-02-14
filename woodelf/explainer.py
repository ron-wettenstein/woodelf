import time
from typing import List

import numpy as np
import pandas as pd

from woodelf.cube_metric import ShapleyValues, CubeMetric, ShapleyInteractionValues, BanzhafValues, \
    BanzhafInteractionValues
from woodelf.decision_trees_ensemble import DecisionTreesEnsemble
from woodelf.high_depth_woodelf import woodelf_for_high_depth
from woodelf.parse_models import load_decision_tree_ensemble_model
from woodelf.path_to_matrices import PathToMatricesAbstractCls

AVAILABLE_MODEL_OUTPUTS = ["raw", "probability", "log_loss"]
AVAILABLE_FEATURE_PERTURBATION = ["auto", "interventional", "tree_path_dependent"]
AVAILABLE_CACHE_OPTIONS = ["auto", "yes", "no"]

MAX_SUGGESTED_CACHE_SIZE = 250 * 2 ^ 20 # Use cache if it is predicted to take less than 250MB

class WoodelfExplainer:
    def __init__(
            self, model, data: pd.DataFrame = None,
            model_output : str="raw", feature_perturbation: str="auto",
            # Additional options exists only in Woodelf:
            cache_option: str = "auto", GPU: bool = False
    ):
        self.raw_model = model
        self.cache_option = cache_option
        self.background_data = data

        self.verify_init_input(model_output, feature_perturbation, cache_option)
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation
        self.is_path_dependent = (self.feature_perturbation == "tree_path_dependent") or (
                self.feature_perturbation == "auto" and self.background_data is None
        )
        self.GPU = GPU

        # TODO fix this in the right way. data can be a np.array
        self.model: DecisionTreesEnsemble = None
        if data is not None or isinstance(model, DecisionTreesEnsemble):
            if isinstance(model, DecisionTreesEnsemble):
                self.model = model
            else:
                self.model: DecisionTreesEnsemble = load_decision_tree_ensemble_model(model, list(data.columns))
            self.model_was_loaded = True
            self.cache = [{} for i in range(len(self.model.trees))] if self.use_cache() else None
            self.cache_filled = False
        else:
            self.model_was_loaded = False

    @classmethod
    def verify_init_input(cls, model_output: str, feature_perturbation: str, cache: str):
        assert model_output in AVAILABLE_MODEL_OUTPUTS, f"Available model_outputs are {AVAILABLE_MODEL_OUTPUTS}. Given '{model_output}'"
        assert feature_perturbation in AVAILABLE_FEATURE_PERTURBATION, f"Available feature_perturbations are {AVAILABLE_FEATURE_PERTURBATION}. Given '{feature_perturbation}'"
        assert cache in AVAILABLE_CACHE_OPTIONS, f"Available cache options are {AVAILABLE_CACHE_OPTIONS}. Given '{cache}'"

        assert model_output == "raw", f"Currently supports only model_output='raw'. Given {model_output}"

    def use_cache(self):
        if self.background_data is None:
            return False
        if self.cache_option == "auto":
            # Use cache if it is predicted to take less than 250MB
            return self.predict_cache_size() <= MAX_SUGGESTED_CACHE_SIZE
        return self.cache_option == "yes"

    def predict_cache_size(self):
        total_cache_size = 0
        for tree in self.model.trees:
            for leaf, features_in_path in tree.get_all_leaves_with_paths():
                D = len(set(features_in_path))
                total_cache_size += 2 ** D * 4
        return total_cache_size

    def shap_values(
            self, X, tree_limit: int = None,
            # Additional options exists only in Woodelf:
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        return self.calc_metric(
            X, ShapleyValues(), tree_limit, as_df, exclude_zero_contribution_features,
            path_to_matrices_calculator, verbose
        )

    def shap_interaction_values(
            self, X, tree_limit: int = None, include_interaction_with_itself: bool = True,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        shapley_ivs = self.calc_metric(
            X, ShapleyInteractionValues(), tree_limit, as_df=True,
            exclude_zero_contribution_features=exclude_zero_contribution_features,
            path_to_matrices_calculator=path_to_matrices_calculator, verbose=verbose
        )
        if include_interaction_with_itself:
            print("""Compute also shapley values in order to find the interactions of features with themselves. 
            The interaction of a feature with itself is its shapley value minus all the shapley 
            interaction values it has with other features (when it is the first feature in the pair).
            a.k.a:
            shap_(i,i) = shap_i - \\sum_(j!=i) shap_(i,j) """)
            shapley_values = self.calc_metric(
                X, ShapleyValues(), tree_limit, as_df=True,
                exclude_zero_contribution_features=exclude_zero_contribution_features,
                path_to_matrices_calculator=path_to_matrices_calculator, verbose=verbose
            )
            zeros_series = pd.Series(0, index=X.index)
            for feature in X.columns:
                shap_value = shapley_values[feature] if feature in shapley_values.columns else zeros_series
                interactions_with_feature = [pair for pair in shapley_ivs.columns if feature == pair[0]]
                sum_interactions = shapley_ivs[interactions_with_feature].sum(axis=1)
                shapley_ivs[(feature, feature)] = shap_value - sum_interactions

        if not as_df:
            return shapley_ivs.values.reshape(
                len(X), len(X.columns), len(X.columns)
            )
        return shapley_ivs

    def banzhaf_values(
            self, X, tree_limit: int = None,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        return self.calc_metric(
            X, BanzhafValues(), tree_limit, as_df, exclude_zero_contribution_features,
            path_to_matrices_calculator, verbose
        )

    def banzhaf_interaction_values(
            self, X, tree_limit: int = None,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        return self.calc_metric(
            X, BanzhafInteractionValues(), tree_limit, as_df,
            exclude_zero_contribution_features,
            path_to_matrices_calculator, verbose
        )

    def calc_metric(
            self, consumer_data, metric: CubeMetric, tree_limit: int = None,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False):
        if not self.model_was_loaded:
            self.model = load_decision_tree_ensemble_model(self.raw_model, list(consumer_data.columns))
            self.model_was_loaded = True
            self.cache = [{} for i in range(len(self.model.trees))] if self.use_cache() else None
            self.cache_filled = False

        model = self.model if tree_limit is not None else DecisionTreesEnsemble(self.model.trees[:tree_limit])
        cache_kwargs = {}
        if self.cache is not None:
            if self.cache_filled:
                # use the cache
                cache_kwargs["cache_to_use"] = self.cache
            else:
                # fill the cache
                cache_kwargs["cache_to_fill"] = self.cache
                self.cache_filled = True # will fill the cache now

        woodelf_values = woodelf_for_high_depth(
            model, consumer_data, self.background_data, metric, GPU=self.GPU,
            path_to_matrices_calculator=path_to_matrices_calculator, model_was_loaded=True, **cache_kwargs
        )

        return self._output_formatting(
            woodelf_values, as_df, exclude_zero_contribution_features, list(consumer_data.columns),
            metric.INTERACTION_VALUE, consumer_data
        )

    @staticmethod
    def _output_formatting(
            woodelf_values, as_df: bool, exclude_zero_contribution_features: bool, columns: List[str],
            iv: bool, consumer_data: pd.DataFrame
    ):
        assert not as_df or not iv or not exclude_zero_contribution_features, (
            "Can not exclude zero contributions in interaction values and return them as a 3D np array."
        )
        zeros_array = np.zeros(len(consumer_data), dtype=np.float32)
        if not exclude_zero_contribution_features:
            if iv:
                # TODO interaction with itself
                for f1 in columns:
                    for f2 in columns:
                        if (f1, f2) not in woodelf_values:
                            woodelf_values[(f1, f2)] = zeros_array
            else:
                for feature in columns:
                    if feature not in woodelf_values:
                        woodelf_values[feature] = zeros_array

        if iv:
            df_columns = [(f1, f2) for f1 in columns for f2 in columns if (f1,f2) in woodelf_values]
            woodelf_df = pd.DataFrame(woodelf_values, columns=df_columns, index=consumer_data.index)
        else:
            df_columns = [f for f in columns if f in woodelf_values]
            woodelf_df = pd.DataFrame(woodelf_values, columns=df_columns, index=consumer_data.index)
        if not as_df:
            if iv:
                return woodelf_df.values.reshape(len(consumer_data), len(columns), len(columns))
            return woodelf_df.values
        return woodelf_df


    @property
    def expected_value(self):
        # Use shap expected_value implementation...
        from shap.explainers import TreeExplainer
        explainer = TreeExplainer(
            self.raw_model, self.background_data,
            feature_perturbation=self.feature_perturbation, model_output=self.model_output
        )
        return explainer.expected_value


    def __call__(self, consumer_data, interactions: bool = False):
        from shap import Explanation

        start_time = time.time()
        feature_names = consumer_data.columns
        if interactions:
            v = self.shap_interaction_values(consumer_data)
        else:
            v = self.shap_values(consumer_data)

        # the Explanation object expects an `expected_value` for each row
        if hasattr(self.expected_value, "__len__") and len(self.expected_value) > 1:
            # `expected_value` is a list / array of numbers, length k, e.g. for multi-output scenarios
            # we repeat it N times along the first axis, so ev_tiled.shape == (N, k)
            num_rows = v.shape[0]
            ev_tiled = np.tile(self.expected_value, (num_rows, 1))
        else:
            # `expected_value` is a scalar / array of 1 number, so we simply repeat it for every row in `v`
            # ev_tiled.shape == (N,)
            ev_tiled = np.tile(self.expected_value, v.shape[0])

        return Explanation(
            v,
            base_values=ev_tiled,
            data=consumer_data.values,
            feature_names=feature_names,
            compute_time=time.time() - start_time
        )
