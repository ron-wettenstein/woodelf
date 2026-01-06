from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from woodelf.cube_metric import ShapleyValues, CubeMetric, ShapleyInteractionValues, BanzhafValues, \
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
            self, model, data: pd.DataFrame = None,
            model_output : str="raw", feature_perturbation: str="auto",
            # Additional options exists only in Woodelf:
            cache_option: str = "auto", GPU: bool = False
    ):
        self.raw_model = model
        # TODO fix this in the right way. data can be a np.array
        if data is not None:
            self.model_objs = load_decision_tree_ensemble_model(model, list(data.columns))
            assert len(self.model_objs) > 0, "Did not load the model properly"
            self.depth = self.model_objs[0].depth
            self.model_was_loaded = True
        else:
            self.model_objs = None
            self.depth = None
            self.model_was_loaded = False
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

    def use_cache(self):
        if self.cache_option == "auto":
            return self.depth <= 8
        return self.cache_option == "yes"

    def shap_values(
            self, X, tree_limit: int = None,
            # Additional options exists only in Woodelf:
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        metric_name = "path_dependent_shap" if self.is_path_dependent else "background_shap"
        return self.calc_metric(
            X, ShapleyValues(), metric_name, tree_limit, as_df, exclude_zero_contribution_features,
            path_to_matrices_calculator, verbose
        )

    def shap_interaction_values(
            self, X, tree_limit: int = None, include_interaction_with_itself: bool = True,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        metric_name = "path_dependent_shap_iv" if self.is_path_dependent else "background_shap_iv"
        shapley_ivs = self.calc_metric(
            X, ShapleyInteractionValues(), metric_name, tree_limit, as_df=True,
            exclude_zero_contribution_features=exclude_zero_contribution_features,
            path_to_matrices_calculator=path_to_matrices_calculator, verbose=verbose
        )
        if include_interaction_with_itself:
            print("""Compute also shapley values in order to find the interactions of features with themselves. 
            The interaction of a feature with itself is its shapley value minus all the shapley 
            interaction values it has with other features (when it is the first feature in the pair).
            a.k.a:
            shap_(i,i) = shap_i - \\sum_(j!=i) shap_(i,j) """)
            shap_metric_name = "path_dependent_shap" if self.is_path_dependent else "background_shap"
            shapley_values = self.calc_metric(
                X, ShapleyValues(), shap_metric_name, tree_limit, as_df=True,
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
        metric_name = "path_dependent_banzhaf" if self.is_path_dependent else "background_banzahf"
        return self.calc_metric(
            X, BanzhafValues(), metric_name, tree_limit, as_df, exclude_zero_contribution_features,
            path_to_matrices_calculator, verbose
        )

    def banzhaf_interaction_values(
            self, X, tree_limit: int = None,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False
    ):
        metric_name = "path_dependent_banzhaf_iv" if self.is_path_dependent else "background_banzahf_iv"
        return self.calc_metric(
            X, BanzhafInteractionValues(), metric_name, tree_limit, as_df,
            exclude_zero_contribution_features,
            path_to_matrices_calculator, verbose
        )

    def calc_metric(
            self, consumer_data, metric: CubeMetric, metric_name: str, tree_limit: int = None,
            as_df: bool = False, exclude_zero_contribution_features: bool = False,
            path_to_matrices_calculator: PathToMatricesAbstractCls = None,
            verbose: bool = False):
        if not self.model_was_loaded:
            self.model_objs = load_decision_tree_ensemble_model(self.raw_model, list(consumer_data.columns))
            assert len(self.model_objs) > 0, "Did not load the model properly"
            self.depth = self.model_objs[0].depth
            self.model_was_loaded = True

        if path_to_matrices_calculator is None:
            path_to_matrices_calculator = SimplePathToMatrices(
                metric=metric, max_depth=self.model_objs[0].depth, GPU=self.GPU
            )
        if self.GPU:
            consumer_data = get_cupy_data(self.model_objs, consumer_data)

        if self.use_cache() and tree_limit is None:
            preprocessed_trees = []
            if metric_name in self.cache:
                print("Skipped preprocessing - used cache")
                for tree, replacement_values_lst in zip(self.model_objs, self.cache[metric_name]):
                    for leaf, replacement_values in zip(tree.get_all_leaves(), replacement_values_lst):
                        leaf.feature_contribution_replacement_values = replacement_values
                    preprocessed_trees.append(tree)
            else:
                self.cache[metric_name] = []
                for tree in tqdm(self.model_objs, desc="Preprocessing the trees", disable=not verbose):
                    preprocessed_tree = self._preprocess_tree(tree, path_to_matrices_calculator)
                    preprocessed_trees.append(preprocessed_tree)
                    self.cache[metric_name].append([
                        leaf.feature_contribution_replacement_values.copy() for leaf in preprocessed_tree.get_all_leaves()
                    ])

            woodelf_values = calculation_given_preprocessed_tree_ensemble(
                preprocessed_trees, consumer_data, global_importance=False,
                iv_one_sized=not metric.INTERACTION_VALUES_ORDER_MATTERS and metric.INTERACTION_VALUE, GPU=self.GPU
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
            if not metric.INTERACTION_VALUES_ORDER_MATTERS and metric.INTERACTION_VALUE:
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

        X_data = consumer_data.values
        return Explanation(
            v,
            base_values=ev_tiled,
            data=X_data,
            feature_names=feature_names,
            compute_time=time.time() - start_time
        )