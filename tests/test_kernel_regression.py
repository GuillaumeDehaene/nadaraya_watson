import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import InvalidParameterError

from src.nadaraya_watson.kernel_regression import KernelRegressionNW

VALID_KERNEL_NAMES = "gaussian", "uniform", "epanechnikov", "exponential", "triangular", "cosine"
VALID_METRIC_NAMES = "l1", "l2", "cosine"


@pytest.mark.parametrize("bandwidth", [1, 1.0, "scott", "silverman", 0.5])
@pytest.mark.parametrize("kernel", [VALID_KERNEL_NAMES])
@pytest.mark.parametrize("metric", [VALID_METRIC_NAMES])
def test_init_valid_parameters(bandwidth, kernel, metric):
    """Test that KernelRegressionNW initializes correctly with valid parameters."""
    model = KernelRegressionNW(bandwidth=bandwidth, kernel=kernel, metric=metric)
    assert model.bandwidth == bandwidth
    assert model.kernel == kernel
    assert model.metric == metric


@pytest.mark.parametrize(
    "bandwidth,kernel,metric",
    [
        (-1.0, "gaussian", "l2"),  # Invalid bandwidth (negative)
        (1.0, "invalid_kernel", "euclidean"),  # Invalid kernel
        (1.0, "gaussian", "invalid_metric"),  # Invalid metric
        (0.0, "gaussian", "euclidean"),  # Invalid bandwidth (zero)
    ],
)
def test_init_invalid_parameters(bandwidth, kernel, metric):
    """Test that KernelRegressionNW raises errors for invalid parameters."""
    with pytest.raises(InvalidParameterError):
        kr = KernelRegressionNW(bandwidth=bandwidth, kernel=kernel, metric=metric)
        X_train = np.array([[1], [2], [3], [4]])
        y_train = np.array([2, 4, 6, 8])
        kr.fit(X_train, y_train)


@pytest.mark.parametrize("kernel", VALID_KERNEL_NAMES)
@pytest.mark.parametrize("metric", VALID_METRIC_NAMES)
@pytest.mark.parametrize("bandwidth", [0.5, 1.0, "scott"])
@pytest.mark.parametrize("n_x_fit", [5, 10])
@pytest.mark.parametrize("n_x_query", [2, 5])
@pytest.mark.parametrize("x_dim", [1, 2])
@pytest.mark.parametrize("y_dim", [1, 3])
def test_fit_and_predict(kernel, metric, bandwidth, n_x_fit, n_x_query, x_dim, y_dim):
    """Test that the model can be fitted and predictions can be made with various dimensions."""
    # Generate dummy data
    X_train = np.random.rand(n_x_fit, x_dim)
    y_train = np.random.rand(n_x_fit, y_dim)
    X_test = np.random.rand(n_x_query, x_dim)
    # Fit the model
    model = KernelRegressionNW(bandwidth=bandwidth, kernel=kernel, metric=metric)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    assert predictions.shape == (n_x_query, y_dim)


def test_predict_before_fit():
    """Test that predict raises NotFittedError if called before fit."""
    model = KernelRegressionNW()
    X_test = np.array([[1.5], [3.5]])
    with pytest.raises(NotFittedError):
        model.predict(X_test)


@pytest.mark.parametrize("n_x_fit", [5, 10])
@pytest.mark.parametrize("x_dim", [1, 2, 10])
def test_bandwidth_calculation_scott(n_x_fit, x_dim):
    """Test that Scott's bandwidth is calculated correctly."""
    x_fit = np.random.rand(n_x_fit, x_dim)
    model = KernelRegressionNW(bandwidth="scott")
    model.fit(x_fit, np.random.rand(n_x_fit))
    expected_bandwidth = n_x_fit ** (-1 / (x_dim + 4))
    assert model.bandwidth_ == expected_bandwidth


@pytest.mark.parametrize("n_x_fit", [5, 10])
@pytest.mark.parametrize("x_dim", [1, 2, 10])
def test_bandwidth_calculation_silverman(n_x_fit, x_dim):
    """Test that Silverman's bandwidth is calculated correctly."""
    x_fit = np.random.rand(n_x_fit, x_dim)
    model = KernelRegressionNW(bandwidth="silverman")
    model.fit(x_fit, np.random.rand(n_x_fit))
    expected_bandwidth = (n_x_fit * (x_dim + 2) / 4) ** (-1 / (x_dim + 4))
    assert model.bandwidth_ == expected_bandwidth


@pytest.mark.parametrize("n_x_fit", [5, 10])
@pytest.mark.parametrize("n_x_query", [2, 5])
@pytest.mark.parametrize("x_dim", [1, 2, 10])
@pytest.mark.parametrize("y_dim", [1, 3])
def test_sample_weight(n_x_fit, n_x_query, x_dim, y_dim):
    """Test that sample weights are handled correctly."""
    X_train = np.random.rand(n_x_fit, x_dim)
    y_train = np.random.rand(n_x_fit, y_dim)
    sample_weight = np.exp(np.random.randn(n_x_fit))
    X_test = np.random.rand(n_x_query, x_dim)
    model = KernelRegressionNW()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    predictions = model.predict(X_test)
    assert predictions.shape == (n_x_query, y_dim)


def test_valid_metrics():
    """Test that valid_metrics classmethod returns correct list."""
    expected_metrics = [
        "cityblock",
        "cosine",
        "euclidean",
        "haversine",
        "l2",
        "l1",
        "manhattan",
        "precomputed",
        "nan_euclidean",
    ]
    actual_metrics = KernelRegressionNW.valid_metrics()
    assert sorted(actual_metrics) == sorted(expected_metrics)


def test_valid_kernels():
    """Test that valid_kernels classmethod returns correct list."""
    expected_kernels = ["gaussian", "uniform", "epanechnikov", "exponential", "triangular", "cosine"]
    actual_kernels = KernelRegressionNW.valid_kernels()
    assert sorted(actual_kernels) == sorted(expected_kernels)
