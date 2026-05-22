from typing import List, Any

import numpy as np

from woodelf.core.trees.decision_trees_ensemble import DecisionTreeNode


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