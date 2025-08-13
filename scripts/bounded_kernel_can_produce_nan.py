import numpy as np

from nadaraya_watson import KernelRegressionNW

##################################
# Generating the reference data
###############################

n_x_fit = 200
n_x_query = 20

x_dim = 2
y_dim = 3


def conditional_mean(x: np.ndarray):
    return (1 + (x[:, 0] >= 0) * (x[:, 1] >= 0))[:, None]


np.random.seed(42)
x_fit = np.random.randn(n_x_fit, x_dim)
y_fit = conditional_mean(x_fit) + np.random.randn(n_x_fit, y_dim)

x_query = np.random.randn(n_x_query, x_dim)

#######################################################

kernel_regression = KernelRegressionNW(bandwidth=0.2, kernel="uniform", metric="l1")
kernel_regression.fit(x_fit, y_fit)
y_pred = kernel_regression.predict(x_query)
error = np.mean((y_pred - conditional_mean(x_query)) ** 2)

print(f"{y_pred = }")
