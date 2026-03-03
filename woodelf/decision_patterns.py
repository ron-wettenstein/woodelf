from typing import Tuple, Generator

import numpy as np
import pandas as pd

from woodelf.decision_trees_ensemble import DecisionTreeNode
from woodelf.simple_woodelf import (
    GPU_get_int_dtype_from_depth, get_int_dtype_from_depth
)

try:
    import cupy as cp
    IMPORTED_CP = True
except ModuleNotFoundError as e:
    cp = None
    IMPORTED_CP = False


def init_patterns_dict(tree: DecisionTreeNode, data: pd.DataFrame, GPU: bool):
    # Use a tight uint type for efficiency. This is improvement 5 of Sec. 9.1
    int_dtype = GPU_get_int_dtype_from_depth(tree.depth) if GPU else get_int_dtype_from_depth(tree.depth)
    if GPU:
        data_length = len(data[list(data.keys())[0]])
        root_pattern = cp.zeros(data_length, dtype=int_dtype)
    else:
        root_pattern = pd.Series(0, index=data.index).to_numpy().astype(int_dtype)
    return {tree.index: root_pattern}

def add_children_patterns(patterns_dict, node, path_features, data: pd.DataFrame, GPU: bool, int_dtype=int, ignore_neighbor_leaf: bool = False):
    """
    Compute the pattern of the provided node given the cahced patterns in the patterns_dict.
    We assume its parent pattern is in this patterns_dict.

    Compute unique_features_decision_pattern. i.e:
    - If the feature was not appear in the path, do the regular stuff: shift left and add the condition bit
    - If the feature appear in the path, find its relevant bit in the pattern and update it to be an 'and' operation
    between the bit already there and the new condition bit (implement it using masking, creating numbers that are
    all 1's except for the location of the condition bit and then 'and' them.)
    """
    if node.is_leaf():
        return

    left_condition = node.shall_go_left(data, GPU)
    ignore_right = node.feature_name not in path_features and ignore_neighbor_leaf and node.right.is_leaf() and node.left.is_leaf()
    right_condition = None
    if not ignore_right:
        right_condition = ~left_condition
        if not GPU:
            right_condition = right_condition.to_numpy().astype(int_dtype)

    if not GPU:
        left_condition = left_condition.to_numpy().astype(int_dtype)
    my_pattern = patterns_dict[node.index]

    if node.feature_name not in path_features:
        shifted_my_pattern = (my_pattern << 1)
        patterns_dict[node.left.index] = shifted_my_pattern + left_condition

        if not ignore_right:
            patterns_dict[node.right.index] = shifted_my_pattern + right_condition
    else:
        unique_features = []
        for feature in path_features:
            if feature not in unique_features:
                unique_features.append(feature)

        current_feature_bit = 2 ** (len(unique_features) - 1 - unique_features.index(node.feature_name))
        mask = ((2 ** len(unique_features) - 1) - current_feature_bit)
        patterns_dict[node.left.index] = my_pattern & (mask + left_condition*current_feature_bit)
        patterns_dict[node.right.index] = my_pattern & (mask + right_condition*current_feature_bit)

def clean_old_patterns(patterns_dict, node):
    """
    Delete the node patterns and all its children patterns from the cache
    """
    if node is not None and patterns_dict is not None:
        for n in node.bfs():
            if n.index in patterns_dict:
                patterns_dict.pop(n.index)


def ignore_right_neighbor(left_leaf, path, use_neighbor_leaf_trick):
    path_features = [n.feature_name for n in path]
    # It is an ignored right leaf situation if:
    ignored_neighbor = (
            use_neighbor_leaf_trick and  # 1. We allow this trick.
            left_leaf.parent is not None and left_leaf == left_leaf.parent.left and left_leaf.parent.right.is_leaf() and  # 2. This is left leaf with a right neighbor leaf
            left_leaf.parent.feature_name not in path_features[:-1]  # 3. The feature of the current leaf does not repeat in the path
    )
    return ignored_neighbor


def decision_patterns_generator(
        tree: DecisionTreeNode, data: pd.DataFrame, GPU: bool = False, ignore_neighbor_leaf: bool = False
) -> Generator[Tuple[DecisionTreeNode, np.array], None, None]:
    """
    Compute the decision patterns of the provided data for all the root-to-leaf paths in the provided tree.
    This generator will return one leaf at a time, it will return a tuple with the leaf index as the first item
    and this leaf's decision patterns as the second item.

    This implementation is RAM efficient. In the naive implementation all the pattern of all the nodes of the
    trees are saved in the RAM (its O(L2^D) RAM). In this implementation only the patterns of the nodes
    of the current path (and their imminent children) as same, achieving an O(D2^D) memory complexity.
    """
    nodes_to_visit_left = [tree]
    nodes_to_visit_right = []

    int_dtype = GPU_get_int_dtype_from_depth(tree.depth) if GPU else get_int_dtype_from_depth(tree.depth)
    patterns = init_patterns_dict(tree, data, GPU)
    nodes_to_path = tree.get_nodes_to_path_dict()
    while len(nodes_to_visit_left) > 0 or len(nodes_to_visit_right) > 0: # Implements a DFS
        if len(nodes_to_visit_left) > 0:
            node = nodes_to_visit_left.pop()
        else:
            node = nodes_to_visit_right.pop()
            clean_old_patterns(patterns, node.parent.left)

        path = nodes_to_path[node.index]
        path_features = [n.feature_name for n in path]

        add_children_patterns(patterns, node, path_features, data, GPU, int_dtype, ignore_neighbor_leaf)
        if node.is_leaf():
            if node.index in patterns: # In case of ignored right leaf the node won't be in patterns
                yield node, patterns[node.index]
        else:
            nodes_to_visit_left.append(node.left)
            nodes_to_visit_right.append(node.right)
