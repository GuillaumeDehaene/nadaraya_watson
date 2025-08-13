# Nadaraya-watson kernel regression

[![Release](https://img.shields.io/github/v/release/GuillaumeDehaene/nadaraya-watson)](https://img.shields.io/github/v/release/GuillaumeDehaene/nadaraya-watson)
[![Build status](https://img.shields.io/github/actions/workflow/status/GuillaumeDehaene/nadaraya-watson/main.yml?branch=main)](https://github.com/GuillaumeDehaene/nadaraya-watson/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/GuillaumeDehaene/nadaraya-watson)](https://img.shields.io/github/license/GuillaumeDehaene/nadaraya-watson)

A simple implementation of the Nadaraya-Watson kernel regression estimator for usage with scikit-learn.

Please note that the parameterization is slightly different from [this other library](https://github.com/jmetzen/kernel_regression). In my implementation, bandwidth is in units of distance, instead of being specific to the kernel.


- **Github repository**: <https://github.com/GuillaumeDehaene/nadaraya-watson/>
- **Documentation** <https://GuillaumeDehaene.github.io/nadaraya-watson/>

## Table of Contents

- [Understanding Nadaraya-Watson regression](#understanding-nadaraya-watson-regression)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Development](#development)

## Understanding Nadaraya-Watson regression

[Nadaraya-Watson kernel regression](https://en.wikipedia.org/wiki/Kernel_regression) is a non-parametric technique used for estimating the conditional expectation of a random variable. It works by placing a kernel function[^1] at each training data point and computing a weighted average of the target values, where weights are determined by the proximity of the query point to each training point.

The prediction at a query point $ x $ is computed as:

$$
\hat{y}(x) = \frac{\sum_{i=1}^{n} K\left(\frac{d(x, x_i)}{h}\right) y_i}{\sum_{i=1}^{n} K\left(\frac{d(x, x_i)}{h}\right)}
$$

where:
- $ K(\cdot) $ is a kernel function[^1] (e.g., Gaussian, Epanechnikov)
- $ h $ is the bandwidth parameter, controlling the width of the kernel
- $ d(x, x_i) $ is the distance between the query point $ x $ and the training point $ x_i $
- $ x_i $ are the training points
- $ y_i $ are the corresponding target values
- $ n $ is the number of training samples

In this implementation, the bandwidth $ h $ can be specified directly or computed using rules like Scott's or Silverman's rule. The kernel function used can be selected from a set of predefined kernels (e.g., Gaussian, Epanechnikov, etc.). The distance metric used to compute proximity between points is also configurable, defaulting to Euclidean distance.

The method is flexible and robust, particularly useful when the underlying relationship between features and targets is unknown or complex.

Please note that **bounded kernels can produce `NaN`values**.
This is not a bug and is the expected behavior.
If you are using a bounded kernel with a low bandwidth, then it is possible that some query points might intersect
0 training points.
In this case, the only reasonable prediction is to return `NaN` to represent an impossible prediction.


[^1]: in the context of Nadaraya-Watson regression, kernels refer to a positive function integrating to 1 and typically symmetric. It is important not to confuse this type of kernel, suitable for example in Kernel Density Estimation, with the kernels used in the "Kernel Trick" which respect instead Mercer's condition.
Please refer to https://en.wikipedia.org/wiki/Kernel_(statistics) and https://en.wikipedia.org/wiki/Kernel_method for additional details.

## Installation

Install using `uv`, [the extremely fast Python package and project manager, written in Rust.](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/GuillaumeDehaene/nadaraya-watson.git
```

Or using `pip`:

```bash
pip install git+https://github.com/GuillaumeDehaene/nadaraya-watson.git
```

## Usage

To use this estimator in your code, simply import and use it as you would any sklearn estimator.

```python
from nadaraya_watson import KernelRegressionNW

kernel_regression = KernelRegressionNW()
kernel_regression.fit(x_fit, y_fit)
y_pred = kernel_regression.predict(x_query)
```

My recommendation is to always use cross-validation, or some other hyperparameter selection scheme,
to find the best value of the bandwidth, kernel, metric.

Please note that **bounded kernels can produce `NaN`values**.
This is not a bug and is the expected behavior.
If you are using a bounded kernel with a low bandwidth, then it is possible that some query points might intersect
0 training points.
In this case, the only reasonable prediction is to return `NaN` to represent an impossible prediction.


## License

Released under the MIT license.


## Development

This project is released as is with no guarantee of further development.
I will only update it from time to time if I wish to play around with python tooling.

Please do not create issues or reach out for updates.
Feel free instead to adapt the code in any way you see fit while respecting the license.


---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
