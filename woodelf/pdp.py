import math
import time
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.lts_vectorized import get_covers_vector
from woodelf.parse_models import load_decision_tree_ensemble_model
from woodelf.simple_woodelf import get_int_dtype_from_depth
from woodelf.cube_metric import PDIVOrder1Or2
from woodelf.decision_patterns import decision_patterns_generator
from woodelf.decision_trees_ensemble import DecisionTreeNode
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
        low, high = np.percentile(data[f].dropna(), [percentiles[0] * 100, percentiles[1]*100])
        # get k equally spaced points between them
        points = np.linspace(low, high, k)
        sample_points_data[f] = list(points)
        sample_points_data[f].sort()
    return pd.DataFrame(sample_points_data)[data.columns]

def build_points_for_full_pdp(data: pd.DataFrame, model, as_df: bool=True):
    """
    Provide the points that will create a full PDP - a graph the will provide the PDV for every x value.
    Does this by collecting all the threshold values from the model. See Sect. of the paper.
    """
    # load the model
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
        th_values[f].extend([0] * (max_th_length - len(th_values[f])) )

    # from the built thershold build the points Data Frame
    return pd.DataFrame(th_values)


def build_points_for_pdp(model, data: pd.DataFrame, k: int = 100, percentiles: Tuple[float] = (0.05, 0.95), sampled: bool = False, seed: int = 42, full_pdp: bool = False):
    if sampled:
        points_df = build_sampled_points_df(data, k, seed)
    elif full_pdp:
        points_df = build_points_for_full_pdp(data, model)
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


def woodelf_pdp_joint(model, data: pd.DataFrame, k: int = 100, accurate: bool = True, GPU: bool = False,
                      percentiles: Tuple[float] = (0.05, 0.95), sampled: bool = False, seed: int = 42, full_pdp: bool = False, verbose: bool = True):
    """
    Compute all the PDVs needed in order to plot the PDP values of all the features. Use WOODELF!
    """
    start_time = time.time()
    original_points_df = build_points_for_pdp(model, data, k, percentiles, sampled, seed, full_pdp)
    if full_pdp:
        k = len(original_points_df)

    points_df = build_points_for_joint_pdp(original_points_df)
    if verbose:
        print(f"Building the points took: {time.time() - start_time} sec. The size of the created df {len(points_df)}")

    metric = PDIVOrder1Or2()

    pdivs = woodelf_for_high_depth(
        model, consumer_data=points_df, background_data=data if accurate else None,
        metric=metric, GPU=GPU, model_was_loaded = False
    )
    avg_prediction = float(model.predict(data).mean())
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
                pair = (f1, f2)
                pdvs[(f1,f2)] = base_pdv + pdivs.get((f1,), zero_array) + pdivs.get((f2,), zero_array) + pdivs.get(pair, zero_array)
                points_part_index = first_different_bit(i,j,D)
                f1_points[(f1,f2)] = points_parts[f1][points_part_index]
                f2_points[(f1,f2)] = points_parts[f2][points_part_index]
    clipped_pdvs = clip_result(pdvs, list(data.columns), k)
    return clipped_pdvs, f1_points, f2_points



def bits_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """
    x: shape (n,), integers
    returns: shape (k, n), rows are bits (k-1),...,1,0 (2^(k-1) down to LSB)
    """
    # ensure x is unsigned (np.uint) for fast bit ops
    shifts = np.arange(k-1, -1, -1, dtype=np.uint64)[:, None]  # (5,1): 4,3,2,1,0
    return ((x[None, :] >> shifts) & 1).astype(np.uint8)


class PDPPathToMatrices: # doesn't inherit PathToMatricesAbstractCls as its API is different

    def __init__(self, max_depth: int, GPU: bool = False):
        self.max_depth = max_depth
        self.GPU = GPU
        self.computation_time = 0

    def build_f_vector_and_only_positive_literals_values(self, background_patterns: np.array, D: int):
        int_type = get_int_dtype_from_depth(D)
        full = (int_type(1) << int_type(D)) - int_type(1)
        only_positive_literals =  np.mean(background_patterns == full)

        x = background_patterns[D - np.bitwise_count(background_patterns) == 1]
        zero_bit_location = D - 1 - np.log2(full ^ x).astype(np.uint16)
        f = np.bincount(zero_bit_location, minlength=D) / len(background_patterns)

        return f, only_positive_literals


    def get_s_matrix(self, consumer_patterns: np.array, background_patterns: np.array, w: float, D: int):
        start_time = time.time()
        bm_consumer = bits_matrix(consumer_patterns, D).T
        f, only_positive_literals = self.build_f_vector_and_only_positive_literals_values(background_patterns, D)
        s_matrix = bm_consumer * f
        if only_positive_literals != 0:
            s_matrix[bm_consumer == 0] = -1 * only_positive_literals
        s_matrix = w * s_matrix
        self.computation_time += time.time() - start_time
        return s_matrix

    def present_statistics(self):
        print(f"PDPPathToMatrices took {round(self.computation_time, 2)}")


class EstimatedPDPPathToMatrices(PDPPathToMatrices):

    def build_f_vector_and_only_positive_literals_values(self, background_patterns: np.array, D: int):
        # background_patterns are actually cover values now...
        covers = background_patterns
        only_positive_literals = np.prod(covers)

        f_list = []
        for i in range(D):
            current_covers = covers.copy()
            current_covers[i] = 1 - covers[i]
            f_list.append(np.prod(current_covers))

        return np.array(f_list), only_positive_literals

def get_unique_features_in_path(path: List[DecisionTreeNode]):
    unique_features_in_path = []
    for n in path:
        if n.feature_name not in unique_features_in_path:
            unique_features_in_path.append(n.feature_name)
    return unique_features_in_path


def fast_pdp_for_a_single_tree(
    tree: DecisionTreeNode, consumer_data: pd.DataFrame, background_data: pd.DataFrame, values: Dict, p2m: PDPPathToMatrices, GPU: bool, accurate: bool = True
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
        if accurate:
            leaf_b, background_patterns = next(background_patterns_generator)
            assert leaf_b.index == leaf.index

            s_matrix = p2m.get_s_matrix(
                consumer_patterns=consumer_patterns,
                background_patterns=background_patterns,
                w=leaf_index_to_weight[leaf.index],
                D=len(leaf_index_to_unique_features_in_path[leaf.index])
            )
        else:
            s_matrix = p2m.get_s_matrix(
                consumer_patterns=consumer_patterns,
                background_patterns=leaf_index_to_covers[leaf.index],
                w=leaf_index_to_weight[leaf.index],
                D=len(leaf_index_to_unique_features_in_path[leaf.index])
            )
        s_matrix = s_matrix.astype(np.float32)

        for index, feature in enumerate(leaf_index_to_unique_features_in_path[leaf.index]):
            if feature not in values:
                values[feature] = s_matrix[:, index]
            else:
                values[feature] += s_matrix[:, index]


def woodelf_fast_pdp(
        model, consumer_data: pd.DataFrame, background_data: pd.DataFrame,
        GPU: bool = False, model_was_loaded: bool = False, centered: bool = True, accurate: bool = True
):
    avg_prediction = 0
    if not model_was_loaded:
        avg_prediction = float(model.predict(background_data).mean())
        model = load_decision_tree_ensemble_model(model, list(consumer_data.columns))
    else:
        if not centered:
            raise NotImplemented("Don't support centered=False on a loaded model")

    if accurate:
        p2m = PDPPathToMatrices(model.max_depth, GPU)
    else:
        p2m = EstimatedPDPPathToMatrices(model.max_depth, GPU)
    pdvs = {}
    for tree in tqdm(model.trees, desc="Preprocessing the trees and computing PDP"):
        fast_pdp_for_a_single_tree(tree, consumer_data, background_data, pdvs, p2m, GPU, accurate=accurate)
    p2m.present_statistics()
    if centered:
        return pdvs

    for f in pdvs:
        pdvs[f] += avg_prediction
    return pdvs

def woodelf_pdp(model, data: pd.DataFrame, k: int = 100, GPU: bool = False, centered: bool = True, accurate: bool = True,
                percentiles: Tuple[float] = (0.05, 0.95), sampled: bool = False, seed: int = 42, full_pdp: bool = False):
    """
    Compute all the PDVs needed in order to plot the PDP values of all the features. Use WOODELF!
    when accurate is False estimate the PDP using the Path-Dependent approach
    """
    points_df = build_points_for_pdp(model, data, k, percentiles, sampled, seed, full_pdp)
    return woodelf_fast_pdp(model, points_df, data, GPU, model_was_loaded=False, centered=centered, accurate=accurate), points_df
