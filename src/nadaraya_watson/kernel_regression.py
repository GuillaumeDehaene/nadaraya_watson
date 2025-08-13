from numbers import Real
from typing import Literal, Union

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import distance_metrics
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

from .kernels import KERNELS

VALID_KERNELS = KERNELS.keys()
VALID_METRICS = distance_metrics()


class KernelRegressionNW(MultiOutputMixin, RegressorMixin, BaseEstimator):
    # "noqa: RUF012"  : this is the sklearn default approach to parameter constraints
    _parameter_constraints = {  # noqa: RUF012
        "bandwidth": [
            Interval(Real, 0, None, closed="neither"),
            StrOptions({"scott", "silverman"}),
        ],
        "kernel": [StrOptions(set(VALID_KERNELS))],
        "metric": [StrOptions(set(VALID_METRICS))],
    }

    def __init__(
        self,
        *,
        bandwidth: Union[float, Literal["scott", "silverman"]] = 1.0,
        kernel: str = "gaussian",
        metric: str = "euclidean",
    ):
        """Initialize the Nadaraya-Watson Kernel regression estimator.
        Parameters
        ----------
        bandwidth : float or str, default=1.0
            The bandwidth of the kernel. If a string, it must be one of "scott" or "silverman".
        kernel : str, default="gaussian"
            The kernel to use. Must be one of the valid kernels.
        metric : str, default="euclidean"
            The distance metric to use. Must be one of the valid metrics.
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric

    @classmethod
    def valid_metrics(cls) -> list[str]:
        """Return a list of valid metrics.

        Please note that some names are actually synonymous.
        Please do not feed these values directly to sklearn for grid-search cross-validation.

        e.g. "euclidean" and "l2" are identical.
        Returns
        -------
        list of str
            A list of valid metric names.
        """
        return list(VALID_METRICS.keys())

    @classmethod
    def valid_kernels(cls) -> list[str]:
        """Return a list of valid kernels.

        Returns
        -------
        list of str
            A list of valid kernel names.
        """
        return list(VALID_KERNELS)

    @_fit_context(
        # KernelDensity.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, sample_weight=None):
        """Fit the Nadaraya-Watson Kernel regression estimator on the data.

        Parameters
        ----------
        X : array-like
            array of shape (n_samples, n_features)

            Training data.
        y : array-like
            array of shape (n_samples,) or (n_samples, n_targets)

            Target values.
        sample_weight : array-like
            array of shape (n_samples,)
            Individual weights for each sample; ignored if None is passed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if isinstance(self.bandwidth, str):
            if self.bandwidth == "scott":
                self.bandwidth_ = X.shape[0] ** (-1 / (X.shape[1] + 4))
            elif self.bandwidth == "silverman":
                self.bandwidth_ = (X.shape[0] * (X.shape[1] + 2) / 4) ** (-1 / (X.shape[1] + 4))
        else:
            self.bandwidth_ = self.bandwidth
        X, y = validate_data(self, X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True)
        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = _check_sample_weight(sample_weight, X)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64, ensure_non_negative=True)

        self.X_fit = X
        self.y = y
        self.sample_weight = sample_weight

        return self

    def predict(self, X):
        """Predict using Nadaraya-Watson Kernel regression estimator.

        Parameters
        ----------
        X : array-like
            array of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_prediction : ndarray
            array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=("csr", "csc"), reset=False)

        # shape (n_samples_X_fit, n_samples_X_predict)
        distance_matrix = pairwise_distances(self.X_fit, X, metric=self.metric)
        log_density = KERNELS[self.kernel](distance_matrix / self.bandwidth_)

        # Removing the max for numerical stability
        # Removing a warning when subtracting -np.inf from a line of -np.inf : this produces a NaN and is expected behavior
        with np.errstate(invalid="ignore", divide="ignore"):
            log_density -= np.max(log_density, axis=0, keepdims=True)

        # Broadcast self.sample_weight to shape (n_samples_X_fit, n_samples_X_predict) from (n_samples_X_fit, )
        weight = np.exp(log_density)
        if self.sample_weight is not None:
            weight *= self.sample_weight[:, None]
        weight /= np.sum(weight, axis=0, keepdims=True)

        # weighted sum of predictions
        y_prediction = np.einsum("ij,ik->jk", weight, self.y)

        return y_prediction
