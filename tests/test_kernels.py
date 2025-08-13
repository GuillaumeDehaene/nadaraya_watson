import numpy as np
import pytest

from nadaraya_watson.kernels import (
    KERNELS,
)


@pytest.mark.parametrize("kernel_name", KERNELS.keys())
@pytest.mark.parametrize("test_input_shape", [(), (1,), (3,), (10,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kernel_output_shape_and_dtype(kernel_name, test_input_shape, dtype):
    """
    Test that the kernel function preserves input shape and dtype.

    This test ensures that when a kernel is applied to an input array,
    the output has the same shape and data type as the input.
    """
    # Create test input with specified shape and dtype
    if test_input_shape == ():
        test_input = np.array(0.0, dtype=dtype)
    else:
        test_input = np.random.rand(*test_input_shape).astype(dtype)  # ty: ignore

    # Get kernel function
    kernel_func = KERNELS[kernel_name]

    # Apply kernel function with absolute value transformation
    distance = np.abs(test_input)
    result = kernel_func(distance)

    # Check that output shape matches input shape
    assert result.shape == test_input_shape

    # Check that output dtype matches input dtype
    assert result.dtype == dtype

    # Check shape
    assert result.shape == test_input.shape

    # Check dtype
    assert result.dtype == test_input.dtype


# Test bounded kernels for correct -inf return outside bounds
BOUNDED_KERNELS_AND_BOUNDS = {"uniform": 1.0, "epanechnikov": 1.0, "triangular": 1.0, "cosine": 1.0}


@pytest.mark.parametrize("kernel_name_and_bound", BOUNDED_KERNELS_AND_BOUNDS.items())
@pytest.mark.parametrize("test_value", [1.1, 2.0, 5.0, 10.0])
def test_bounded_kernels_outside_bound(kernel_name_and_bound, test_value):
    """
    Test that bounded kernels return -inf for distances outside their support.

    For kernels with finite support (e.g., uniform, Epanechnikov), values
    beyond the kernel's boundary should result in -inf when evaluated.
    """
    kernel_name, bound = kernel_name_and_bound
    kernel_func = KERNELS[kernel_name]

    # Test with single value
    distance = np.abs(np.array([test_value]))
    result = kernel_func(distance)
    assert np.isneginf(result[0])

    # Test with array of values
    distance = np.abs(np.array([test_value] * 5))
    result = kernel_func(distance)
    assert all(np.isneginf(x) for x in result)


@pytest.mark.parametrize("kernel_name", KERNELS.keys())
def test_kernel_mean_is_zero(kernel_name):
    """
    Test that the mean of symmetric kernel density is zero.

    For symmetric kernels, the expected value (mean) of the probability
    distribution should be zero. This test verifies this property numerically.
    """
    # Generate a range of values around 0
    # NB: odd number of points to ensure 0 is included
    x = np.linspace(-3, 3, 1001)

    kernel_func = KERNELS[kernel_name]

    # Get log density with absolute value transformation
    distance = np.abs(x)
    log_density = kernel_func(distance)

    # Compute actual density
    density = np.exp(log_density)

    # Normalize to make it a proper probability distribution
    # (since we're not using the normalization constant, we'll check mean)
    # For symmetric kernels, mean should be 0
    mean = np.sum(density * x) / np.sum(density)

    # The mean should be close to 0 (within tolerance)
    assert abs(mean) < 1e-3


@pytest.mark.parametrize("kernel_name", KERNELS.keys())
def test_warning_still_active(kernel_name, recwarn):
    """
    Test that kernel functions suppress log warnings while other operations do not.

    This ensures that the kernel implementations handle invalid log inputs
    without issuing warnings, but that standard NumPy operations still raise
    appropriate warnings for invalid logarithmic operations like log(-1) or log(0).
    """
    kernel_func = KERNELS[kernel_name]

    # Calling the kernel does not trigger the runtime warning
    assert len(recwarn) == 0
    _ = kernel_func(np.array([1.0, 10.0]))
    assert len(recwarn) == 0

    # Force a log(-1) scenario which triggers the RuntimeWarning
    with pytest.warns(RuntimeWarning, match="invalid value encountered in log"):
        _ = np.log(np.array([-1]))

    # Force a log(0) scenario which triggers the RuntimeWarning
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
        _ = np.log(np.array([0]))
