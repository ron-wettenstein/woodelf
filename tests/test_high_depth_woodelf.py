import time

import pandas as pd

from woodelf.cube_metric import ShapleyValues, ShapleyInteractionValues
from woodelf.decision_trees_ensemble import DecisionTreeNode
from woodelf.high_depth_woodelf import (
    compute_path_dependent_f, compute_patterns_generator, woodelf_for_high_depth, compute_f, compute_f_of_neighbor
)
import numpy as np
import pytest

from tests.test_woodelf_against_shap import trainset, testset, xgb_model
from woodelf.parse_models import load_decision_tree_ensemble_model
from woodelf.simple_woodelf import calculate_background_metric, calculate_path_dependent_metric, \
    path_dependent_frequencies

FIXTURES = [trainset, testset, xgb_model]

TOLERANCE = 0.00001


def leaf(value, cover=None, index=None):
    return DecisionTreeNode(None, value, left=None, right=None, cover=cover, index=index)

@pytest.fixture
def simple_path():
    # (a<5) -cover=0.5-> (b<3) -cover=0.6-> (a<1) -cover=0.2-> (leaf: 4)
    root = DecisionTreeNode("a", 5, index=1, right=leaf(-1, cover=500, index=2), cover=1000,
        left=DecisionTreeNode("b", 3, index=3, right=leaf(-1, cover=200, index=4), cover=500,
              left=DecisionTreeNode("a", 1, index=5,
                                    right=leaf(-1, cover=240, index=6),
                                    left=leaf(4, cover=60, index=7), cover=300)
              )
        )
    path = [root, root.left, root.left.left, root.left.left.left]

    root.left.parent = root
    root.right.parent = root
    root.left.left.parent = root.left
    root.left.right.parent = root.left
    root.left.left.left.parent = root.left.left
    root.left.left.right.parent = root.left.left
    return path



def test_compute_path_dependent_f(simple_path):
    f = compute_path_dependent_f(simple_path, unique_features_in_path=["a", "b"])
    np.testing.assert_allclose(f, np.array([0.36, 0.54, 0.04, 0.06]))


def test_compute_path_dependent_f_on_a_full_model(trainset, xgb_model):
    model_objs = load_decision_tree_ensemble_model(xgb_model, list(trainset.columns))
    for tree in model_objs:
        simple_woodelf_fs = path_dependent_frequencies(tree)
        index_to_node = {n.index: n for n in tree.bfs()}
        for node_index, path in tree.get_nodes_to_path_dict().items():
            # On paths with unique features the simple woodelf impl is identical to the high depth impl
            if len(set([n.feature_name for n in path])) == len(path):
                node = index_to_node[node_index]
                if node.is_leaf():
                    unique_features_in_path = []
                    for n in path:
                        if n.feature_name not in unique_features_in_path:
                            unique_features_in_path.append(n.feature_name)

                    pd_f = compute_path_dependent_f(path + [node], unique_features_in_path)
                    np.testing.assert_allclose(pd_f, simple_woodelf_fs[node.index])


def test_compute_patterns_generator(simple_path):
    data = pd.DataFrame({"a": [7,8,4,3,0,-1], "b": [4,2,5,2,9,1]})
    simple_path[0].depth = 3
    patterns = {index: p for index, p in compute_patterns_generator(simple_path[0], data)}
    np.testing.assert_equal(patterns[7], np.array([0,1,0,1,2,3]))
    np.testing.assert_equal(patterns[6], np.array([0,1,2,3,0,1]))
    np.testing.assert_equal(patterns[4], np.array([1,0,3,2,3,2]))
    np.testing.assert_equal(patterns[2], np.array([1,1,0,0,0,0]))


def test_compute_f(simple_path):
    data = pd.DataFrame({"a": [7, 8, 4, 3, 0, -1], "b": [4, 2, 5, 2, 9, 1]})
    simple_path[0].depth = 3
    patterns = {index: p for index, p in compute_patterns_generator(simple_path[0], data)}
    f_leaf_7 = compute_f(patterns[7], path_depth=2)
    f_leaf_6 = compute_f(patterns[6], path_depth=2)
    np.testing.assert_allclose(f_leaf_7, np.array([2,2,1,1]) / 6)
    np.testing.assert_allclose(f_leaf_6, np.array([2,2,1,1]) / 6)

def test_compute_f_of_neighbor(simple_path):
    data = pd.DataFrame({"a": [7, 8, 4, 3, 0, -1], "b": [4, 2, 5, 2, 9, 1]})

    # (a<5) -cover=0.5-> (b<3) -cover=0.6-> (leaf: 4)
    new_leaf = leaf(4, index=5, cover=simple_path[0].left.left.cover)
    unique_feature_path = [simple_path[0], simple_path[1], new_leaf]
    simple_path[0].depth = 2
    simple_path[1].left = new_leaf
    new_leaf.parent = simple_path[1].left

    patterns = {index: p for index, p in compute_patterns_generator(unique_feature_path[0], data)}
    f_leaf_5 = compute_f(patterns[5], path_depth=2)
    f_leaf_4 = compute_f(patterns[4], path_depth=2)
    np.testing.assert_allclose(f_leaf_4, compute_f_of_neighbor(f_leaf_5))


@pytest.mark.parametrize("use_neighbor_leaf_trick, metric_cls", [
    (False, ShapleyValues),
    (True, ShapleyValues),
    (False, ShapleyInteractionValues),
    (True, ShapleyInteractionValues),
],
    ids=["SHAP use_neighbor_leaf_trick=False", "SHAP use_neighbor_leaf_trick=True",
         "SHAP IV use_neighbor_leaf_trick=False", "SHAP IV use_neighbor_leaf_trick=True"])
def test_calculate_background_metric_for_high_depth(trainset, testset, xgb_model, use_neighbor_leaf_trick, metric_cls):
    start_time = time.time()
    simple_woodelf_values = calculate_background_metric(
        xgb_model, testset, trainset, metric=metric_cls()
    )
    print("simple woodelf took: ", time.time() - start_time)

    start_time = time.time()
    high_depth_woodelf_values = woodelf_for_high_depth(
        xgb_model, testset, trainset, metric=metric_cls(), use_neighbor_leaf_trick=use_neighbor_leaf_trick
    )
    print("high depth woodelf took: ", time.time() - start_time)

    for feature in simple_woodelf_values:
        np.testing.assert_allclose(simple_woodelf_values[feature], high_depth_woodelf_values[feature], atol=TOLERANCE)


@pytest.mark.parametrize("use_neighbor_leaf_trick, metric_cls", [
    (False, ShapleyValues),
    (True, ShapleyValues),
    (False, ShapleyInteractionValues),
    (True, ShapleyInteractionValues),
],
    ids=["SHAP use_neighbor_leaf_trick=False", "SHAP use_neighbor_leaf_trick=True",
         "SHAP IV use_neighbor_leaf_trick=False", "SHAP IV use_neighbor_leaf_trick=True"])
def test_calculate_path_dependent_metric_for_high_depth(
        trainset, testset, xgb_model, use_neighbor_leaf_trick, metric_cls
):
    start_time = time.time()
    simple_woodelf_values = calculate_path_dependent_metric(
        xgb_model, testset, metric=metric_cls()
    )
    print("simple woodelf took: ", time.time() - start_time)

    start_time = time.time()
    high_depth_woodelf_values = woodelf_for_high_depth(
        xgb_model, testset, background_data=None, metric=metric_cls(), use_neighbor_leaf_trick=use_neighbor_leaf_trick
    )
    print("high depth woodelf took: ", time.time() - start_time)

    for feature in simple_woodelf_values:
        np.testing.assert_allclose(simple_woodelf_values[feature], high_depth_woodelf_values[feature], atol=TOLERANCE)
