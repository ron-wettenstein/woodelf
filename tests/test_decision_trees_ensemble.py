from decision_trees_ensemble import DecisionTreeNode, LeftIsSmallerEqualDecisionTreeNode
import pandas as pd
import numpy as np


def leaf(value):
    return DecisionTreeNode(None, value, left=None, right=None)

def test_stump_predict():
    stump = DecisionTreeNode("value", 5, right=leaf(5), left=leaf(1))
    df = pd.DataFrame({"value": [1,2,3,4,5,6,7,8,9,10]})
    pd.testing.assert_series_equal(pd.Series([1,1,1,1,5,5,5,5,5,5]), stump.predict(df), check_dtype=False)

    stump_nan_go_left = DecisionTreeNode("value", 3.2, right=leaf(4.1), left=leaf(1), nan_go_left=False)
    df = pd.DataFrame({"value": [3, 4, np.nan]})
    pd.testing.assert_series_equal(pd.Series([1, 4.1, 4.1]), stump_nan_go_left.predict(df), check_dtype=False)

    stump = LeftIsSmallerEqualDecisionTreeNode("value", 5, right=leaf(5), left=leaf(1))
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    pd.testing.assert_series_equal(pd.Series([1, 1, 1, 1, 1, 5, 5, 5, 5, 5]), stump.predict(df), check_dtype=False)

