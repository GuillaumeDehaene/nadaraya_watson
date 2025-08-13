import warnings
from typing import Callable

import numpy as np


def log_gaussian_kernel(dist: np.ndarray) -> np.ndarray:
    """log of the gaussian kernel (unnormalized)"""
    return -0.5 * (dist * dist)


def log_uniform_kernel(dist: np.ndarray) -> np.ndarray:
    """log of the uniform kernel (unnormalized)"""
    return np.where(dist < 1.0, 0.0, -np.inf).astype(dist.dtype)


def log_epanechnikov_kernel(dist: np.ndarray) -> np.ndarray:
    """log of the epanechnikov kernel (unnormalized)"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in log")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        return np.where(dist < 1.0, np.log(1.0 - (dist * dist)), -np.inf)


def log_exponential_kernel(dist: np.ndarray) -> np.ndarray:
    """log of the exponential kernel (unnormalized)"""
    return -np.abs(dist)


def log_triangular_kernel(dist: np.ndarray) -> np.ndarray:
    """log of the triangular kernel (unnormalized)"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in log")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        return np.where(dist < 1.0, np.log(1 - dist), -np.inf)


def log_cosine_kernel(dist: np.ndarray) -> np.ndarray:
    """log of the cosine kernel (unnormalized)"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in log")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        return np.where(dist < 1.0, np.log(np.cos(0.5 * np.pi * dist)), -np.inf)


KERNELS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gaussian": log_gaussian_kernel,
    "uniform": log_uniform_kernel,
    "epanechnikov": log_epanechnikov_kernel,
    "exponential": log_exponential_kernel,
    "triangular": log_triangular_kernel,
    "cosine": log_cosine_kernel,
}
