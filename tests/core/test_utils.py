import numpy as np
import pytest

from woodelf.core.utils import safe_bitwise_count, bitwise_count_fallback


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
