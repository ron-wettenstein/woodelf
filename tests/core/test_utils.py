import numpy as np
import pandas as pd
import pytest

from woodelf.core.utils import safe_bitwise_count, bitwise_count_fallback, get_numpy_data, \
    consumer_and_background_to_dict


def _python_bitwise_count(x: np.ndarray) -> np.ndarray:
    """The 1-bits of the absolute value of every element, counted with Python ints (works on every numpy version)."""
    return np.array([bin(abs(int(value))).count("1") for value in x.ravel()], dtype=np.uint8).reshape(x.shape)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64])
def test_bitwise_count_fallback_matches_numpy(dtype):
    info = np.iinfo(dtype)
    random_values = np.random.default_rng(0).integers(info.min, info.max, size=10_000, dtype=dtype, endpoint=True)
    edge_values = np.array([0, 1, info.max, info.min, info.max - 1, info.min + 1], dtype=dtype)
    x = np.concatenate([random_values, edge_values]).reshape(-1, 2) # 2-D, the shape must be kept

    expected = _python_bitwise_count(x)
    actual = bitwise_count_fallback(x)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(safe_bitwise_count(x), expected)
    if hasattr(np, "bitwise_count"): # numpy >= 2
        np.testing.assert_array_equal(np.bitwise_count(x), expected)


def _mixed_dtypes_data() -> pd.DataFrame:
    # float and int columns (stored in two separate pandas blocks), with NaNs
    return pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.5], "b": [4, 5, 6, 7], "c": [np.nan, 0.25, 0.5, 2.0]})


def test_get_numpy_data_keeps_all_the_columns_without_copying():
    data = _mixed_dtypes_data()
    columns = get_numpy_data(data)
    assert list(columns.keys()) == list(data.columns)
    for feature, column in columns.items():
        np.testing.assert_array_equal(column, data[feature].to_numpy())
        assert np.shares_memory(column, data[feature].to_numpy())


def test_consumer_and_background_to_dict():
    consumer, background = _mixed_dtypes_data(), _mixed_dtypes_data().head(2)
    consumer_columns, background_columns = consumer_and_background_to_dict(consumer, background, GPU=False)
    for data, columns in [(consumer, consumer_columns), (background, background_columns)]:
        assert list(columns.keys()) == list(data.columns)
        for feature in data.columns:
            np.testing.assert_array_equal(columns[feature], data[feature].to_numpy())

    # A dict of columns is returned as is and a None background stays None
    same_consumer_columns, no_background = consumer_and_background_to_dict(consumer_columns, None, GPU=False)
    assert same_consumer_columns is consumer_columns and no_background is None
