from typing import List, Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode, DecisionTreesEnsemble

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False


def bits_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """
    x: shape (n,), integers
    returns: shape (k, n), rows are bits (k-1),...,1,0 (2^(k-1) down to LSB)
    """
    # ensure x is unsigned (np.uint) for fast bit ops
    shifts = np.arange(k-1, -1, -1, dtype=np.uint8)[:, None]  # (5,1): 4,3,2,1,0
    return ((x[None, :] >> shifts) & 1).astype(np.uint8)

def neg_bits_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """
    identical to bits_matrix(x,k).replace({0:1, 1:0})
    """
    # ensure x is unsigned (np.uint) for fast bit ops
    shifts = np.arange(k-1, -1, -1, dtype=np.uint8)[:, None]  # (5,1): 4,3,2,1,0
    return (((x[None, :] >> shifts) + 1) & 1).astype(np.uint8)


def safe_bitwise_count(x: np.ndarray) -> np.ndarray:
    """
    The number of 1-bits in the absolute value of every element of x, as uint8.
    np.bitwise_count on numpy >= 2 (where it was added), bitwise_count_fallback on older numpy versions.
    """
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return bitwise_count_fallback(x)


def bitwise_count_fallback(x: np.ndarray) -> np.ndarray:
    """
    np.bitwise_count for numpy < 2 (where it does not exist): adds up the bits of x one bit position at a time.
    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.signedinteger):
        # np.bitwise_count counts the bits of the absolute value. int64 first, so abs of int8/16/32 minimums does not
        # overflow (abs of the int64 minimum wraps to itself, which is the right value once it is uint64)
        x = np.abs(x.astype(np.int64)).astype(np.uint64)
    counts = np.zeros(x.shape, dtype=np.uint8)
    for bit in range(8 * x.dtype.itemsize):
        counts += ((x >> bit) & 1).astype(np.uint8)
    return counts


def get_unique_features_in_path(path: List[DecisionTreeNode]):
    unique_features_in_path = []
    for n in path:
        if n.feature_name not in unique_features_in_path:
            unique_features_in_path.append(n.feature_name)
    return unique_features_in_path


def get_covers_vector(path: List[DecisionTreeNode], unique_features_in_path: List[Any]):
    if len(unique_features_in_path) == 0:
        # If the leaf is the tree's root, it has cover of 1.
        return [1]

    feature_index = {f: i for i, f in enumerate(unique_features_in_path)}

    proceed_covers = [1] * len(unique_features_in_path)
    for i in range(len(path)-1):
        proceed_covers[ feature_index[path[i].feature_name] ] *= (path[i+1].cover / path[i].cover)
    return proceed_covers


def get_cupy_data(model: DecisionTreesEnsemble, df: pd.DataFrame):
    """
    Cast the dataframe to cupy dict mapping between columns of CuPy arrays.
    We only do this for feature participating in the trees.
    """
    data = {}
    for tree in model.trees:
        for feature in tree.get_all_features():
            if feature not in data:
                data[feature] = cp.asarray(df[feature].to_numpy())
    return data


def get_numpy_data(df: pd.DataFrame):
    """
    Cast the dataframe to a dict mapping between columns and NumPy arrays, the CPU version of get_cupy_data.
    Splitting NumPy arrays instead of the dataframe saves building a pandas Series in every node, which made computing
    the decision patterns slow. The arrays are views of the dataframe data (not copies), so all the columns are kept.
    """
    return {feature: df[feature].to_numpy() for feature in df.columns}


def data_to_dict(
        data: Optional[Union[pd.DataFrame, Dict[Any, np.ndarray]]], GPU: bool,
        model: Optional[DecisionTreesEnsemble] = None
) -> Optional[Dict[Any, np.ndarray]]:
    """
    Cast a dataframe to the dict of columns the decision patterns are computed on: NumPy arrays (get_numpy_data), or
    CuPy arrays when GPU is True (get_cupy_data, which casts only the features of the model, or all the columns when
    no model is given). Data that is already a dict (or None) is returned as is.
    """
    if not isinstance(data, pd.DataFrame):
        return data
    if not GPU:
        return get_numpy_data(data)
    if model is not None:
        return get_cupy_data(model, data)
    return {feature: cp.asarray(data[feature].to_numpy()) for feature in data.columns}


def consumer_and_background_to_dict(
        consumer_data: Union[pd.DataFrame, Dict[Any, np.ndarray]],
        background_data: Optional[Union[pd.DataFrame, Dict[Any, np.ndarray]]], GPU: bool,
        model: Optional[DecisionTreesEnsemble] = None
) -> Tuple[Dict[Any, np.ndarray], Optional[Dict[Any, np.ndarray]]]:
    """
    data_to_dict of the consumer and background data, a None background stays None.
    """
    return data_to_dict(consumer_data, GPU, model), data_to_dict(background_data, GPU, model)