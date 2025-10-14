from typing import Optional

import pandas as pd

# For GPU execution
try:
    import cupy as cp
except:
    cp = None


class DecisionTreeNode:
    """
    Represent a decision tree node. Recursively the root node builds the tree structure (the root node knows it children and so on).
    Include several useful tree functions like BFS and n.split(c).
    """

    def __init__(
            self, feature_name: str, value: float, right: Optional["DecisionTreeNode"], left: Optional["DecisionTreeNode"], nan_go_left=True, index: int=None, cover=None,
            feature_contribution_replacement_values=None,
        ):
        """
        See decision tree definition in the paper (Def. 6)
        The tree split function is a bit different as XGBoost also support NaN values.
        The split function is "go left if df[feature_name]<value or (nan_go_left and df[feature_name] == NaN)".
        Right and left parameters can be a DesicionTreeNode if this node is an inner node or None if the node is a leaf. (The leaf weight will be saved as the 'value')
        Cover is an optional parameter, it includes how many rows in the train set reached this node.
        """
        self.index=index
        self.feature_name = feature_name
        self.value = float(value)
        self.right = right
        self.left = left
        self.nan_go_left = nan_go_left
        self.cover = cover
        self.pc_pb_to_cube = None
        self.feature_contribution_replacement_values = feature_contribution_replacement_values
        self.parent = -1
        self.depth=None

    def shall_go_left(self, row, GPU: bool = False):
        """
        This is the n.split(c) defined in Def.6
        """
        if self.nan_go_left:
            # return (row[self.feature_name] < self.value) | row[self.feature_name].isna()
            return ~(row[self.feature_name] >= self.value)
        else:
            return row[self.feature_name] < self.value

    def shall_go_right(self, row, GPU: bool = False):
        return ~self.shall_go_left(row, GPU)

    def is_leaf(self):
        return self.right is None and self.left is None

    def is_almost_leaf(self):
        return not self.is_leaf() and (self.right.is_leaf() or self.left.is_leaf())

    def predict(self, data, GPU: bool = False):
        if self.is_leaf():
            # TODO if GPU use CuPy series
            return pd.Series(self.value, index=data.index)
        return self.shall_go_left(data, GPU) * self.left.predict(data, GPU) + self.shall_go_right(data, GPU) * self.right.predict(data, GPU)

    def bfs(self, including_myself: bool = True, including_leaves: bool = True):
        """
        Return all the node children (and the node itself) in BFS order. The indexes should be in an increasing order.
        """
        if self.is_leaf():
            return [self] if including_myself and including_leaves else []

        children = [self] if including_myself else []
        nodes_to_visit = []
        if self.right is not None:
            nodes_to_visit.append(self.right)
        if self.left is not None:
            nodes_to_visit.append(self.left)

        while len(nodes_to_visit) > 0:
            current_node = nodes_to_visit.pop(0)
            if current_node.right is not None:
                nodes_to_visit.append(current_node.right)
            if current_node.left is not None:
                nodes_to_visit.append(current_node.left)

            if current_node.is_leaf():
                if including_leaves:
                    children.append(current_node)
            else:
                children.append(current_node)

        return children

    def get_all_leaves(self):
        children = self.bfs(including_leaves=True)
        return [node for node in children if node.is_leaf()]

    def get_all_almost_leaves(self):
        children = self.bfs(including_leaves=True)
        return [node for node in children if node.is_almost_leaf()]

    def get_all_features(self):
        inner_nodes = self.bfs(including_leaves=False)
        return set(n.feature_name for n in inner_nodes)

    def get_all_leaves_with_path_to_root(self):
        nodes_to_visit = [(self, [])]
        leaves = []
        while len(nodes_to_visit) > 0:
            current_node, current_path_to_root = nodes_to_visit.pop(0)
            for next_node in [current_node.right, current_node.left]:
                next_node_obj = (next_node, current_path_to_root + [current_node.feature_name])
                if next_node.is_leaf():
                    leaves.append(next_node_obj)
                else:
                    nodes_to_visit.append(next_node_obj)
        return leaves

    def __repr__(self):
        if self.is_leaf():
            return f"{self.index} (cover: {self.cover}): leaf with value {self.value}"
        return f"{self.index} (cover: {self.cover}): {self.feature_name} < {self.value}"


class LeftIsSmallerEqualDecisionTreeNode(DecisionTreeNode):
    """
    In this decision tree we go left if x <= th (instead of if x < th).
    """

    def shall_go_left(self, row, GPU: bool = False):
        """
        This is the n.split(c) defined in Def.6
        """
        if self.nan_go_left:
            # return (row[self.feature_name] <= self.value) | row[self.feature_name].isna()
            return ~(row[self.feature_name] > self.value)
        else:
            return row[self.feature_name] <= self.value
