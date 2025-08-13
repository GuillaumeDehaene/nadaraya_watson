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

#######################################################
## Recommended usage: cross-validate all parameters
###################################################

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

kernel_regression = KernelRegressionNW(
    bandwidth=grid.best_estimator_.bandwidth, kernel=grid.best_estimator_.kernel, metric=grid.best_estimator_.metric
)
kernel_regression.fit(x_fit, y_fit)
y_pred = kernel_regression.predict(x_query)
error = np.mean((y_pred - conditional_mean(x_query)) ** 2)

print(f"Mean squared error for cross-validation of all parameters:\n\t{error = :.2f}")

print(KernelRegressionNW.valid_metrics())
print(KernelRegressionNW.valid_kernels())
