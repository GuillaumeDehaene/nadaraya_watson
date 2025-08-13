[Nadaraya-Watson kernel regression](https://en.wikipedia.org/wiki/Kernel_regression) is a non-parametric technique used for estimating the conditional expectation of a random variable. It works by placing a kernel function at each training data point and computing a weighted average of the target values, where weights are determined by the proximity of the query point to each training point.

This class is a simple implementation of the Nadaraya-Watson kernel regression estimator for usage with scikit-learn.

::: nadaraya_watson.KernelRegressionNW
