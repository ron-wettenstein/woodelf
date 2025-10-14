from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.cube_metric import ShapleyValues, CubeMetric, ShapleyInteractionValues, BanzahfValues, \
    BanzhafInteractionValues
from woodelf.decision_trees_ensemble import DecisionTreeNode
from woodelf.parse_models import load_decision_tree_ensemble_model
from woodelf.path_to_matrices import PathToMatricesAbstractCls, SimplePathToMatrices
from woodelf.simple_woodelf import get_cupy_data, preprocess_tree_background, \
    calculation_given_preprocessed_tree_ensemble, calculation_given_preprocessed_tree, fast_preprocess_path_dependent, \
    fill_mirror_pairs

AVAILABLE_MODEL_OUTPUTS = ["raw", "probability", "log_loss"]
AVAILABLE_FEATURE_PERTURBATION = ["auto", "interventional", "tree_path_dependent"]
AVAILABLE_CACHE_OPTIONS = ["auto", "yes", "no"]

class WoodelfExplainer:
    def __init__(
            self, model, data: pd.DataFrame = None, model_output : str="raw",
            feature_perturbation: str="auto", cache_option: str = "auto", GPU: bool = False
    ):
        self.model_objs = load_decision_tree_ensemble_model(model, list(data.columns))
        assert len(self.model_objs) > 0, "Did not load the model properly"
        self.depth = self.model_objs[0].depth
        self.background_data = data

        self.verify_init_input(model_output, feature_perturbation, cache_option)
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation
        self.is_path_dependent = (self.feature_perturbation == "tree_path_dependent") or (
                self.feature_perturbation == "auto" and self.background_data is None
        )
        self.GPU = GPU

        self.cache_option = cache_option
        self.cache = {}

    @classmethod
    def verify_init_input(cls, model_output: str, feature_perturbation: str, cache: str):
        assert model_output in AVAILABLE_MODEL_OUTPUTS, f"Available model_outputs are {AVAILABLE_MODEL_OUTPUTS}. Given '{model_output}'"
        assert feature_perturbation in AVAILABLE_FEATURE_PERTURBATION, f"Available feature_perturbations are {AVAILABLE_FEATURE_PERTURBATION}. Given '{feature_perturbation}'"
        assert cache in AVAILABLE_CACHE_OPTIONS, f"Available cache options are {AVAILABLE_CACHE_OPTIONS}. Given '{cache}'"

        assert model_output == "raw", f"Currently supports only model_output='raw'. Given {model_output}"


    def __call__(self, consumer_data, interactions: bool = False):
        if interactions:
            pass
        return 0

    def use_cache(self):
        if self.cache_option == "auto":
            return self.depth <= 8
        return self.cache_option == "yes"

    def shap_values(
            self, consumer_data, tree_limit: int = None,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        metric_name = "path_dependent_shap" if self.is_path_dependent else "background_shap"
        return self.calc_metric(
            consumer_data, ShapleyValues(), metric_name, tree_limit, as_df, exclude_zero_contribution_features,
            path_to_matrices_calculator, verbose
        )

    def shap_interaction_values(
            self, consumer_data, tree_limit: int = None, include_interaction_with_itself: bool = True,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        metric_name = "path_dependent_shap_iv" if self.is_path_dependent else "background_shap_iv"
        shapley_ivs = self.calc_metric(
            consumer_data, ShapleyInteractionValues(), metric_name, tree_limit, as_df=True,
            exclude_zero_contribution_features=exclude_zero_contribution_features,
            path_to_matrices_calculator=path_to_matrices_calculator, verbose=verbose
        )
        if include_interaction_with_itself:
            print("""Compute also shapley values in order to find the interactions of features with themselves. 
            The interaction of a feature with itself is its shapley value minus all the shapley 
            interaction values it has with other features.""")
            shapley_values = self.calc_metric(
                consumer_data, ShapleyValues(), metric_name, tree_limit, as_df=True,
                exclude_zero_contribution_features=exclude_zero_contribution_features,
                path_to_matrices_calculator=path_to_matrices_calculator, verbose=verbose
            )
            zeros_series = pd.Series(0, index=consumer_data.index)
            for feature in consumer_data.columns:
                shap_value = shapley_values[feature] if feature in shapley_values.columns else zeros_series
                interactions_with_feature = [pair for pair in shapley_ivs.columns if feature in pair]
                sum_interactions = shapley_ivs[interactions_with_feature].sum(axis=1)
                shapley_ivs[(feature, feature)] = shap_value - sum_interactions

        if not as_df:
            return shapley_ivs.values.reshape(
                len(consumer_data), len(consumer_data.columns), len(consumer_data.columns)
            )
        return shapley_ivs

    def banzhaf_values(
            self, consumer_data, tree_limit: int = None,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        metric_name = "path_dependent_banzhaf" if self.is_path_dependent else "background_banzahf"
        return self.calc_metric(
            consumer_data, BanzahfValues(), metric_name, tree_limit, as_df, exclude_zero_contribution_features,
            path_to_matrices_calculator, verbose
        )

    def banzhaf_interaction_values(
            self, consumer_data, tree_limit: int = None,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        metric_name = "path_dependent_banzhaf_iv" if self.is_path_dependent else "background_banzahf_iv"
        return self.calc_metric(
            consumer_data, BanzhafInteractionValues(), metric_name, tree_limit, as_df,
            exclude_zero_contribution_features,
            path_to_matrices_calculator, verbose
        )

    def calc_metric(
            self, consumer_data, metric: CubeMetric, metric_name: str, tree_limit: int = None,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False):
        if path_to_matrices_calculator is None:
            path_to_matrices_calculator = SimplePathToMatrices(
                metric=metric, max_depth=self.model_objs[0].depth, GPU=self.GPU
            )
        if self.GPU:
            consumer_data = get_cupy_data(self.model_objs, consumer_data)

        if self.use_cache() and (tree_limit is not None):
            if metric_name in self.cache:
                preprocessed_trees = self.cache[metric_name]
            else:
                preprocessed_trees = []
                for tree in tqdm(self.model_objs, desc="Preprocessing the trees", disable=not verbose):
                    preprocessed_trees.append(self._preprocess_tree(tree, path_to_matrices_calculator))
                self.cache[metric_name] = preprocessed_trees

            woodelf_values = calculation_given_preprocessed_tree_ensemble(
                preprocessed_trees, consumer_data, global_importance=False,
                iv_one_sized=metric.INTERACTION_VALUES_RETURN_ALL_SUBSET_PERMUTATIONS, GPU=self.GPU
            )
        else:
            woodelf_values = {}
            trees = self.model_objs if tree_limit is None else self.model_objs[tree_limit:]
            for tree in tqdm(trees, desc="Running (preprocessing and computation)", disable=not verbose):
                preprocessed_tree = self._preprocess_tree(tree, path_to_matrices_calculator)
                woodelf_values = calculation_given_preprocessed_tree(
                    preprocessed_tree, consumer_data, values=woodelf_values, GPU=self.GPU
                )

            # Improvement 4 of Sec. 9.1
            if metric.INTERACTION_VALUES_RETURN_ALL_SUBSET_PERMUTATIONS:
                fill_mirror_pairs(woodelf_values)

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


    def _preprocess_tree(self, tree: DecisionTreeNode, path_to_matrices_calculator: PathToMatricesAbstractCls):
        if self.is_path_dependent:
            return fast_preprocess_path_dependent(tree, path_to_matrixes_calculator=path_to_matrices_calculator)
        return preprocess_tree_background(
            tree, self.background_data, depth=tree.depth,
            path_to_matrixes_calculator=path_to_matrices_calculator, GPU=self.GPU
        )
