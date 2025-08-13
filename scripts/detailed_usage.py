import warnings

import numpy as np
from sklearn.model_selection import GridSearchCV

from nadaraya_watson import KernelRegressionNW

##################################
# Generating the reference data
###############################

n_x_fit = 200
n_x_query = 200

x_dim = 2
y_dim = 3


def conditional_mean(x: np.ndarray):
    return (1 + (x[:, 0] >= 0) * (x[:, 1] >= 0))[:, None]


np.random.seed(42)
x_fit = np.random.randn(n_x_fit, x_dim)
y_fit = conditional_mean(x_fit) + np.random.randn(n_x_fit, y_dim)

x_query = np.random.randn(n_x_query, x_dim)

##########################################################
## A. Standard usage: fit model with a fixed bandwidth
######################################################

kernel_regression__default_bandwidth = KernelRegressionNW()
kernel_regression__scott_bandwidth = KernelRegressionNW(bandwidth="scott")


kernel_regression__default_bandwidth.fit(x_fit, y_fit)
kernel_regression__scott_bandwidth.fit(x_fit, y_fit)

y_pred__default_bandwidth = kernel_regression__default_bandwidth.predict(x_query)
y_pred__scott_bandwidth = kernel_regression__scott_bandwidth.predict(x_query)

error__default_bandwidth = np.mean((y_pred__default_bandwidth - conditional_mean(x_query)) ** 2)
error__scott_bandwidth = np.mean((y_pred__scott_bandwidth - conditional_mean(x_query)) ** 2)

print()
print(
    f"Mean squared error for default bandwidth :\n\t{error__default_bandwidth = :.2f}\t\t{kernel_regression__default_bandwidth.bandwidth_ = :.2f}"
)

print()
print(
    f"Mean squared error for auto-selected bandwidth (scott formula) :\n\t{error__scott_bandwidth = :.2f}\t\t{kernel_regression__scott_bandwidth.bandwidth_ = :.2f}"
)

####################################################################################
## B. Cross-validation bandwidth selection using the sklearn.model_selection tools
##################################################################################

bandwidths = np.logspace(-1, 1, 20)
grid = GridSearchCV(KernelRegressionNW(), {"bandwidth": bandwidths}, cv=5)

grid.fit(x_fit, y_fit)

best_bandwidth = grid.best_estimator_.bandwidth

# Creating an estimator with the correct best values
kernel_regression__cross_validation = KernelRegressionNW(bandwidth=best_bandwidth)
kernel_regression__cross_validation.fit(x_fit, y_fit)
y_pred__cross_validation = kernel_regression__cross_validation.predict(x_query)
error__cross_validation = np.mean((y_pred__cross_validation - conditional_mean(x_query)) ** 2)

print()
print(
    f"Mean squared error for cross-validated bandwidth:\n\t{error__cross_validation = :.2f}\t\t{kernel_regression__cross_validation.bandwidth_ = :.2f}"
)


#############################################################################
## C. Cross-validation selection of all parameters

bandwidths = np.logspace(-1, 1, 20)
kernels = ["gaussian", "epanechnikov", "uniform", "exponential"]
metrics = ["l1", "l2"]

grid = GridSearchCV(KernelRegressionNW(), {"bandwidth": bandwidths, "kernel": kernels, "metric": metrics}, cv=5)

# Filtering warnings produced by sklearn when encounterings NaN in the predictions
# This is expected: please refer to the scientific documentation and the file `scripts/bounded_kernel_can_produce_nan.py`
warnings.filterwarnings("ignore", module="sklearn")
grid.fit(x_fit, y_fit)

print()
print(
    f"Optimal parameters: {grid.best_estimator_.bandwidth = :.2f}, {grid.best_estimator_.kernel = }, {grid.best_estimator_.metric = }"
)

kernel_regression__full_cross_validation = KernelRegressionNW(
    bandwidth=grid.best_estimator_.bandwidth, kernel=grid.best_estimator_.kernel, metric=grid.best_estimator_.metric
)
kernel_regression__full_cross_validation.fit(x_fit, y_fit)
y_pred__full_cross_validation = kernel_regression__full_cross_validation.predict(x_query)
error__full_cross_validation = np.mean((y_pred__full_cross_validation - conditional_mean(x_query)) ** 2)

print(f"Mean squared error for cross-validation of all parameters:\n\t{error__full_cross_validation = :.2f}")
