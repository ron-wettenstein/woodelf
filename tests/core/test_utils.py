import numpy as np
import pytest

from woodelf.core.utils import safe_bitwise_count, bitwise_count_fallback


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64])
def test_bitwise_count_fallback_matches_numpy(dtype):
    info = np.iinfo(dtype)
    random_values = np.random.default_rng(0).integers(info.min, info.max, size=10_000, dtype=dtype, endpoint=True)
    edge_values = np.array([0, 1, info.max, info.min, info.max - 1, info.min + 1], dtype=dtype)
    x = np.concatenate([random_values, edge_values]).reshape(-1, 2) # 2-D, the shape must be kept

    expected = np.bitwise_count(x)
    actual = bitwise_count_fallback(x)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(safe_bitwise_count(x), expected)
