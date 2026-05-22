import math
import time
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy

from woodelf.lts_vectorized import get_covers_vector
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.simple_woodelf import get_int_dtype_from_depth
from woodelf.core.cube_metric import PDIVOrder1Or2, CPDVMetric
from woodelf.core.decision_patterns import decision_patterns_generator
from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode
from woodelf.core.path_to_matrices import PathToSVectors
from woodelf.high_depth_woodelf import woodelf_for_high_depth


def build_sampled_points_df(data: pd.DataFrame, k: int, seed: int = None):
    """
    Sample k points from every column.
    """
    sample_points_data = {}
    for f in data.columns:
        sample_points_data[f] = list(data[f].sample(k, random_state=seed))
        sample_points_data[f].sort()
    return pd.DataFrame(sample_points_data)[data.columns]

def build_equally_distanced_points_df(data: pd.DataFrame, k: int, percentiles: Tuple[float]):
    """
    Take equally distanced points from each column. The min point will be in the precentile percentiles[0]
    and the max point will be in the precentile percentiles[1].
    This is also the default implementation of sklearn
    """
    sample_points_data = {}
    for f in data.columns:
        # np.percentile(data[f].dropna(), [percentiles[0] * 100, percentiles[1] * 100])
        low, high = scipy.stats.mstats.mquantiles(data[f].dropna(), prob=percentiles, axis=0)
        # get k equally spaced points between them
        points = np.linspace(low, high, k)
        sample_points_data[f] = list(points)
        sample_points_data[f].sort()
    return pd.DataFrame(sample_points_data)[data.columns]

def build_points_for_full_pdp(data: pd.DataFrame, model, as_df: bool = True, model_was_loaded: bool = False):
    """
    Provide the points that will create a full PDP - a graph the will provide the PDV for every x value.
    Does this by collecting all the threshold values from the model. See Sect. of the paper.
    """
    if model_was_loaded:
        model_objs = model
    else:
        model_objs = load_decision_tree_ensemble_model(model, list(data.columns))

    # collect all the theshold values for each feature
    th_values = {f: [] for f in list(data.columns)}
    for tree in model_objs.trees:
        for node in tree.bfs(including_myself=True, including_leaves=False):
            th_values[node.feature_name].append(node.value)

    # Make sure the thesholds are unique and sort them
    for f in th_values:
        th_values[f] = sorted(list(set(th_values[f])))

    if not as_df:
        return th_values

    # zfill
    max_th_length = max([len(thersholds) for thersholds in th_values.values()])
    for f in th_values:
        th_values[f].extend([0] * (max_th_length - len(th_values[f])))

    # from the built thershold build the points Data Frame
    return pd.DataFrame(th_values)


def build_points_for_pdp(
        model, data: pd.DataFrame, k: int = 100, percentiles: Tuple[float] = (0.05, 0.95), sampled: bool = False,
        seed: int = 42, full_pdp: bool = False, model_was_loaded: bool = False
):
    if sampled:
        points_df = build_sampled_points_df(data, k, seed)
    elif full_pdp:
        points_df = build_points_for_full_pdp(data, model, model_was_loaded=model_was_loaded)
    else:
        points_df = build_equally_distanced_points_df(data, k, percentiles)
    return points_df


# Joint PDP:

def bits(n, D):
    bs = []
    for i in range(D):
        bs.append(n % 2)
        n = n // 2
    return reversed(bs)

def build_points_for_joint_pdp(points_df: pd.DataFrame):
    D = math.ceil(math.log2(len(points_df.columns)))
    data = {f: [] for f in points_df.columns}
    k = len(points_df)
    for i, f in enumerate(points_df.columns):
        for b in bits(i, D):
            if b == 0:
                data[f].extend(np.tile(points_df[f].values, k))
            elif b == 1:
                data[f].extend(np.repeat(points_df[f].values, k))
    return pd.DataFrame(data)

def first_different_bit(n1, n2, D):
    assert n1 != n2
    i = 0
    for b1, b2 in zip(bits(n1, D), bits(n2, D)):
        if b1 != b2:
            return i
        i += 1

def clip_result(pdvs, features, k):
    D = math.ceil(math.log2(len(features)))
    feature_to_index = {f:i for i,f in enumerate(features)}
    clipped = {}
    for f1, f2 in pdvs:
        i1 = feature_to_index[f1]
        i2 = feature_to_index[f2]
        h = first_different_bit(i1, i2, D)
        clipped[(f1, f2)] = pdvs[(f1, f2)][h*(k**2): (h+1)*(k**2)]
    return clipped


def bits_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """
    x: shape (n,), integers
    returns: shape (k, n), rows are bits (k-1),...,1,0 (2^(k-1) down to LSB)
    """
    # ensure x is unsigned (np.uint) for fast bit ops
    shifts = np.arange(k-1, -1, -1, dtype=np.uint8)[:, None]  # (5,1): 4,3,2,1,0
    return ((x[None, :] >> shifts) & 1).astype(np.uint8)


class PDPPathToSVectors(PathToSVectors):
    """
    PDP path-to-s-vectors calculator.
    get_background_s_matrix  — exact PDP using background data patterns.
    get_path_dependent_s_matrix — estimated PDP using tree cover ratios
                                  (was EstimatedPDPPathToMatrices).
    """

    def __init__(self, max_depth: int, GPU: bool = False):
        super().__init__(None, max_depth, GPU) # As we don't use it, we give metric=None
        self.computation_time = 0
        self.compute_f_time = 0

    def _build_f_from_patterns(self, background_patterns: np.ndarray, D: int):
        """Exact: compute f and only_positive_literals from integer background patterns."""
        x = background_patterns[np.bitwise_count(background_patterns) >= D - 1]
        int_type = get_int_dtype_from_depth(D)
        full = int_type(2 ** D - 1)
        orig_x_len = len(x)
        x = x[x != full]
        only_positive_literals = (orig_x_len - len(x)) / len(background_patterns)
        zero_bit_location = np.bitwise_count(x + 1) - 1  # which is equivalent to: D - 1 - np.log2(full ^ x).astype(np.uint16)

        f = np.bincount(zero_bit_location, minlength=D) / len(background_patterns)
        return f, only_positive_literals

        # Clearer but less optimized equivalent:
        # full = int_type(2 ** D - 1)
        # only_positive_literals = np.sum(background_patterns == full) / len(background_patterns)
        # x = background_patterns[D - np.bitwise_count(background_patterns) == 1]
        # zero_bit_location = np.bitwise_count(x + 1) - 1
        # f = np.bincount(zero_bit_location, minlength=D) / len(background_patterns)
        # return f, only_positive_literals

    def _build_f_from_covers(self, covers: np.ndarray, D: int):
        """Estimated: compute f and only_positive_literals from tree cover ratios."""
        only_positive_literals = float(np.prod(covers))
        f_list = []
        for i in range(D):
            current_covers = covers.copy()
            current_covers[i] = 1 - covers[i]
            f_list.append(np.prod(current_covers))
        return np.array(f_list), only_positive_literals

    def _compute_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        f: np.ndarray, only_positive_literals: float, w: float
    ) -> Dict:
        D = len(features_in_path)
        bm_consumer = bits_matrix(consumer_patterns, D).T
        s_matrix = bm_consumer * f
        if only_positive_literals != 0:
            s_matrix[bm_consumer == 0] = -1 * only_positive_literals
        s_matrix = w * s_matrix
        return {feature: s_matrix[:, i].astype(np.float32) for i, feature in enumerate(features_in_path)}

    def get_background_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        if not features_in_path:
            return {}
        start_time = time.time()
        D = len(features_in_path)
        f_start = time.time()
        f, only_positive_literals = self._build_f_from_patterns(background_patterns, D)
        self.compute_f_time += time.time() - f_start
        result = self._compute_s_matrix(features_in_path, consumer_patterns, f, only_positive_literals, w)
        self.computation_time += time.time() - start_time
        return result

    def get_path_dependent_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        if not features_in_path:
            return {}
        start_time = time.time()
        D = len(features_in_path)
        f_start = time.time()
        f, only_positive_literals = self._build_f_from_covers(covers, D)
        self.compute_f_time += time.time() - f_start
        result = self._compute_s_matrix(features_in_path, consumer_patterns, f, only_positive_literals, w)
        self.computation_time += time.time() - start_time
        return result

    def present_statistics(self):
        print(f"{self.__class__.__name__} took {round(self.computation_time, 2)}, computing f took {round(self.compute_f_time, 2)}")


class PDPIVPathToSVectors(PDPPathToSVectors):
    """
    PDP Interaction Values path-to-s-vectors calculator.
    get_background_s_matrix  — exact PDP-IV using background data patterns.
    get_path_dependent_s_matrix — not yet implemented (raises NotImplementedError).
    """

    def get_needed_b_vectors(self, background_patterns: np.ndarray, D: int):
        x_up_to_2_zero_bits = background_patterns[np.bitwise_count(background_patterns) >= D - 2]
        x_bitwise_count = np.bitwise_count(x_up_to_2_zero_bits)
        x_up_to_1_zero_bit = x_up_to_2_zero_bits[x_bitwise_count >= D - 1]

        int_type = get_int_dtype_from_depth(D)
        full = int_type(2 ** D - 1)
        orig_x_len = len(x_up_to_1_zero_bit)
        x_exactly_1_zero_bit = x_up_to_1_zero_bit[x_up_to_1_zero_bit != full]
        only_positive_literals = (orig_x_len - len(x_exactly_1_zero_bit)) / len(background_patterns)

        zero_bit_location = np.bitwise_count(x_exactly_1_zero_bit + 1) - 1
        f_1_zero_bit = np.bincount(zero_bit_location, minlength=D) / len(background_patterns)

        i_idx, j_idx = np.triu_indices(D, k=1)
        b_v_1 = f_1_zero_bit[j_idx]
        b_v_2 = f_1_zero_bit[i_idx]

        x_exactly_2_zero_bit = x_up_to_2_zero_bits[x_bitwise_count == D - 2]
        if len(x_exactly_2_zero_bit) == 0:
            return only_positive_literals, b_v_1, b_v_2, np.zeros(len(i_idx), dtype=float)

        z = full ^ x_exactly_2_zero_bit # exactly two 1 bits, at the zero positions of x2

        # lower 1 bit
        low = z & -z
        # upper 1 bit
        high = z ^ low

        # actual bit positions: 0 = LSB, ..., D-1 = MSB
        p_low = np.bitwise_count(low - 1)
        p_high = np.bitwise_count(high - 1)

        # convert actual bit positions to bits_matrix row indices
        r1 = D - 1 - p_high
        r2 = D - 1 - p_low

        # ensure r1 < r2
        i = np.minimum(r1, r2)
        j = np.maximum(r1, r2)

        pair_index = i * (2 * D - i - 1) // 2 + (j - i - 1)

        b_v_3 = np.bincount(pair_index, minlength=len(i_idx)) / len(background_patterns)
        return only_positive_literals, b_v_1, b_v_2, b_v_3

    def get_background_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        background_patterns: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        if not features_in_path or len(features_in_path) < 2:
            return {}
        start_time = time.time()
        D = len(features_in_path)
        bm_consumer = bits_matrix(consumer_patterns, D).T
        i_idx, j_idx = np.triu_indices(D, k=1)
        c_v = 2 * bm_consumer[:, i_idx] + bm_consumer[:, j_idx]
        f_start = time.time()
        only_positive_literals, b_v_1, b_v_2, b_v_3 = self.get_needed_b_vectors(background_patterns, D)
        self.compute_f_time += time.time() - f_start
        s_matrix = (
            only_positive_literals * (c_v == 0)
            - b_v_1 * (c_v == 1)
            - b_v_2 * (c_v == 2)
            + b_v_3 * (c_v == 3)
        )
        s_matrix = (w * s_matrix).astype(np.float32)
        self.computation_time += time.time() - start_time

        results = {}
        for index1, feature1 in enumerate(features_in_path):
            for index2, feature2 in enumerate(features_in_path):
                if index1 == index2:
                    continue
                idx = triu_pair_to_index(index1, index2, D)
                if (feature1, feature2) not in results:
                    results[(feature1, feature2)] = s_matrix[:, idx]
                else:
                    results[(feature1, feature2)] += s_matrix[:, idx]
        return results

    def get_path_dependent_s_matrix(
        self, features_in_path: List, consumer_patterns: np.ndarray,
        covers: np.ndarray, w: float, w_neighbor: Optional[float] = None
    ) -> Dict:
        raise NotImplementedError("Estimated PDP-IV is not yet implemented")

def get_unique_features_in_path(path: List[DecisionTreeNode]):
    unique_features_in_path = []
    for n in path:
        if n.feature_name not in unique_features_in_path:
            unique_features_in_path.append(n.feature_name)
    return unique_features_in_path


def triu_pair_to_index(i: int, j: int, D: int) -> int:
    if i == j:
        raise ValueError("Diagonal pair has no index in triu_indices(D, k=1)")
    if i > j:
        i, j = j, i
    return i * (2 * D - i - 1) // 2 + (j - i - 1)

def fast_pdp_for_a_single_tree(
    tree: DecisionTreeNode, consumer_data: pd.DataFrame, background_data: pd.DataFrame,
    values: Dict, p2s: PDPPathToSVectors, GPU: bool, accurate: bool = True
):
    leaf_index_to_unique_features_in_path = {}
    leaf_index_to_weight = {}
    leaf_index_to_covers = {}
    for leaf, path in tree.get_all_leaves_with_paths(only_feature_names=False):
        unique_features_in_path = get_unique_features_in_path(path)
        leaf_index_to_unique_features_in_path[leaf.index] = unique_features_in_path
        leaf_index_to_weight[leaf.index] = leaf.value
        if not accurate:
            leaf_index_to_covers[leaf.index] = np.array(get_covers_vector(path + [leaf], unique_features_in_path))

    background_patterns_generator = None
    if accurate:
        background_patterns_generator = decision_patterns_generator(tree, background_data, GPU, ignore_neighbor_leaf=False)
    for leaf, consumer_patterns in decision_patterns_generator(tree, consumer_data, GPU, ignore_neighbor_leaf=False):
        features_in_path = leaf_index_to_unique_features_in_path[leaf.index]
        w = leaf_index_to_weight[leaf.index]
        if accurate:
            leaf_b, background_patterns = next(background_patterns_generator)
            assert leaf_b.index == leaf.index
            s_matrix = p2s.get_background_s_matrix(features_in_path, consumer_patterns, background_patterns, w)
        else:
            s_matrix = p2s.get_path_dependent_s_matrix(features_in_path, consumer_patterns, leaf_index_to_covers[leaf.index], w)

        for feature, feature_values in s_matrix.items():
            if feature not in values:
                values[feature] = feature_values
            else:
                values[feature] += feature_values


def woodelf_fast_pdp(
        model, consumer_data: pd.DataFrame, background_data: pd.DataFrame,
        GPU: bool = False, model_was_loaded: bool = False, centered: bool = True, accurate: bool = True, 
        use_woodelfhd: bool = False, avg_prediction: float=None
):
    if not model_was_loaded:
        model_obj = load_decision_tree_ensemble_model(model, list(consumer_data.columns))
    else:
        model_obj = model    

    if use_woodelfhd:
        pdvs = woodelf_for_high_depth(
            model, consumer_data=consumer_data, background_data=background_data if accurate else None,
            metric=CPDVMetric(), GPU=GPU, use_neighbor_leaf_trick=True, global_importance=False, model_was_loaded=model_was_loaded
        )
    else:
        p2s = PDPPathToSVectors(model_obj.max_depth, GPU)
        pdvs = {}
        for tree in tqdm(model_obj.trees, desc="Preprocessing the trees and computing PDP"):
            fast_pdp_for_a_single_tree(tree, consumer_data, background_data, pdvs, p2s, GPU, accurate=accurate)
        p2s.present_statistics()

    if centered:
        return pdvs

    if not model_was_loaded and background_data is not None and avg_prediction is None:
        avg_prediction = float(model.predict(background_data).mean())
    assert avg_prediction is not None, "avg_prediction must be provided when model_was_loaded is True or when background_data is None"
    
    avg_prediction = 0 if avg_prediction is None else avg_prediction

    for f in pdvs:
        pdvs[f] += avg_prediction
    return pdvs

def woodelf_fast_pdp_iv(
        model, consumer_data: pd.DataFrame, background_data: pd.DataFrame,
        GPU: bool = False,
):
    p2s = PDPPathToSVectors(model.max_depth, GPU)
    p2s_iv = PDPIVPathToSVectors(model.max_depth, GPU)
    pdvs = {}
    pdivs = {}
    for tree in tqdm(model.trees, desc="Preprocessing the trees and computing PDP"):
        fast_pdp_for_a_single_tree(tree, consumer_data, background_data, pdvs, p2s, GPU, accurate=True)
        fast_pdp_for_a_single_tree(tree, consumer_data, background_data, pdivs, p2s_iv, GPU, accurate=True)
    p2s.present_statistics()
    p2s_iv.present_statistics()

    for feature in pdvs:
        pdivs[(feature,)] = pdvs[feature]
    return pdivs

def woodelf_pdp(model, data: pd.DataFrame, k: int = 100, GPU: bool = False, centered: bool = True, accurate: bool = True,
                percentiles: Tuple[float] = (0.05, 0.95), sampled: bool = False, seed: int = 42, full_pdp: bool = False,
                use_woodelfhd: Optional[bool] = None) -> Tuple[Dict[str, np.array], pd.DataFrame]:
    """
    Compute all the PDVs needed in order to plot the PDP values of all features.

    Uses WOODELF for exact PDP computation. When ``accurate=False``, the PDP is
    approximated using the Path-Dependent approach.

    :params model: A fitted decision-tree ensemble (XGBoost, LightGBM, RandomForest, ...).
    :params data: Background dataset used for the PDP computation. Feature names are taken from ``data.columns``.
    :params k: Number of grid values used for each feature PDP. Similar to scikit-learn's ``grid_resolution``. If is ignored when full_pdp=True.
    :params GPU: If True, use GPU acceleration.
    :params centered: If True, center each PDP curve around the average prediction on the provided data.
    :params accurate: If True, compute exact PDPs. If False, estimate the PDPs using the Path-Dependent approximation.
    :params percentiles: Lower and upper percentiles used to define the feature value grid. Must be values in the range [0, 1].
    :params sampled: If True, instead of feature value grid, sample the values to plot in the graph from the provided data.
    :params seed: Random seed used when ``sampled=True``.
    :params full_pdp: If True, compute the full PDP. This plot displays average prediction for any possible feature value. To achieve this behavior,
    e used the features' threshold values .
    :params use_woodelfhd: Whether to use the WoodelfHD implementation. If None, the implementation is selected automatically.

    :returns: The function return 2 elements:
    1. A dictionary that map each feature to its partial dependence values. The values are saved in the NumPy array.
    2. A DataFrame with the PDP x values - the feature values we measured the average model prediction in.
    """
    model_obj = load_decision_tree_ensemble_model(model, list(data.columns))
    avg_prediction = None
    if not centered and data is not None:
        avg_prediction = float(model.predict(data).mean())
    if use_woodelfhd is None:
        use_woodelfhd = (model_obj.max_depth <= 6) and accurate

    points_df = build_points_for_pdp(model_obj, data, k, percentiles, sampled, seed, full_pdp, model_was_loaded=True)
    return woodelf_fast_pdp(
        model_obj, points_df, data, GPU, model_was_loaded=True, centered=centered, accurate=accurate, 
        use_woodelfhd=use_woodelfhd, avg_prediction=avg_prediction
    ), points_df

def woodelf_pdp_joint(
        model, data: pd.DataFrame, k: int = 100, accurate: bool = True, GPU: bool = False, use_woodelfhd: Optional[bool] = None,
        percentiles: Tuple[float] = (0.05, 0.95), sampled: bool = False, seed: int = 42, full_pdp: bool = False, verbose: bool = True
) -> Tuple[Dict[Tuple[str, str], np.array], Dict[Tuple[str, str], np.array], Dict[Tuple[str, str], np.array]]:
    """
    Compute all the PDVs needed in order to plot the PDP values of all the features. Use WOODELF!
    See the docs of woodelf_pdp for parameters explanation.

    :returns: The function return 3 elements:
    1. A dictionary that map each feature pair to their joint partial dependence values. The values are saved in the NumPy array.
    2. A dictionary that map each feature pair to the values of the first feature in the pair we measured the average model prediction in.
    3. A dictionary that map each feature pair to the values of the second feature in the pair we measured the average model prediction in.
    """
    start_time = time.time()
    original_points_df = build_points_for_pdp(model, data, k, percentiles, sampled, seed, full_pdp)
    if full_pdp:
        k = len(original_points_df)

    points_df = build_points_for_joint_pdp(original_points_df)
    if verbose:
        print(f"Building the points took: {time.time() - start_time} sec. The size of the created df {len(points_df)}")

    model_obj = load_decision_tree_ensemble_model(model, list(data.columns))
    if use_woodelfhd is None:
        use_woodelfhd = (model_obj.max_depth <= 10)

    if not use_woodelfhd and accurate:
        pdivs = woodelf_fast_pdp_iv(model_obj, consumer_data=points_df, background_data=data, GPU=GPU)
    else:
        metric = PDIVOrder1Or2()
        pdivs = woodelf_for_high_depth(
            model_obj, consumer_data=points_df, background_data=data if accurate else None,
            metric=metric, GPU=GPU, model_was_loaded=True
        )
    avg_prediction = float(model.predict(data).mean()) if accurate else 0
    base_pdv = np.array([avg_prediction] * len(points_df))
    zero_array = np.array([0] * len(points_df))
    pdvs = {}

    D = math.ceil(math.log2(len(points_df.columns)))
    points_parts = {f: [points_df[f].values[i:i + k**2] for i in range(0, len(points_df[f]), k**2)] for f in data.columns}
    f1_points = {}
    f2_points = {}
    for i, f1 in enumerate(data.columns):
        for j, f2 in enumerate(data.columns):
            if f1 != f2:
                pdvs[(f1,f2)] = base_pdv + pdivs.get((f1,), zero_array) + pdivs.get((f2,), zero_array) + pdivs.get((f1, f2), zero_array)
                points_part_index = first_different_bit(i,j,D)
                f1_points[(f1,f2)] = points_parts[f1][points_part_index]
                f2_points[(f1,f2)] = points_parts[f2][points_part_index]
    clipped_pdvs = clip_result(pdvs, list(data.columns), k)
    return clipped_pdvs, f1_points, f2_points
