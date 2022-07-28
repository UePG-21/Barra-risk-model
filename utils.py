from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity


def ewa(arr: Sequence, half_life: int | None = None) -> float | np.ndarray:
    """Exponential Weighted Average (EWA)

    Args:
        arr (Sequence): array of numbers or arrays (later one on axis=0 weights more)
        half_life (int | None, optional): time it takes for weight to reduce to half of
            the original value. Defaults to None, meaning that weights are all equal.

    Returns:
        float: exponential weighted average of elements in `arr`
    """
    arr = np.array(arr)
    alpha = 1.0 if half_life is None else 0.5 ** (1 / half_life)
    weights = alpha ** np.arange(len(arr) - 1, -1, -1)
    w_shape = tuple([arr.shape[0]] + [1] * (len(arr.shape) - 1))
    weights = weights.reshape(w_shape)
    sum_weight = len(arr) - 1 if half_life is None else np.sum(weights)
    return (weights * arr).sum(axis=0) / sum_weight


def cov_ewa(data: np.ndarray, half_life: int | None = None, lag: int = 0) -> np.ndarray:
    """Calculate the covariance matrix as an exponential weighted average of range

    Args:
        data (np.ndarray): data matrix (K features * T periods)
        half_life (int | None, optional): argument of ewa()
        lag (int): difference between to terms of fator, cov(t-lag, t), when lag is
            opposite, the result is transposed. Defaults to 0.

    Returns:
        np.ndarray: covariance matrix
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data matrix should be an ndarray")
    if data.shape[0] > data.shape[1]:
        raise ValueError("data matrix should not have less columns than rows")
    if lag >= data.shape[1]:
        raise ValueError("lag must be smaller than the number of columns of matrix")
    data = data.astype("float64")
    f_bar = data.mean(axis=1)
    data = data - f_bar.reshape(data.shape[0], -1)
    t_range = range(lag, data.shape[1]) if lag > 0 else range(data.shape[1] + lag)
    elements = np.array([np.outer(data[:, t - lag], data[:, t]) for t in t_range])
    return ewa(elements, half_life)


def num_eigvals_explain(pct: float, eigvals: np.ndarray) -> int:
    """The number of eigenvalues it takes to explain a percentage of total variance

    Args:
        pct (float): percentage of total variance
        eigvals (np.ndarray): eigenvalues

    Returns:
        int: number of eigenvalues
    """
    eigvals = np.sort(eigvals)[::-1]  # descending order
    eigvals = eigvals / np.sum(eigvals)
    p, num = 0, 0
    for v in eigvals:
        p += v
        num += 1
        if p > pct:
            break
    return num


def draw_eigvals_edf(
    cov: np.ndarray,
    bandwidth: float | None = None,
    x_range: np.ndarray | None = None,
    label: str | None = None,
) -> None:
    """Draw the empirical distribution function of `cov`

    Args:
        cov (np.ndarray): covariance matrix
        bandwidth (float | None, optional): bandwidth of KDE. Defaults to None.
        x_range (np.ndarray | None, optional): range of x displayed. Defaults to None.
        label (str | None, optional): label on plot. Defaults to None.
    """
    eigvals = np.linalg.eigvalsh(cov).reshape(-1, 1)
    bw = np.cbrt(np.median(eigvals)) if bandwidth is None else bandwidth
    kde = KernelDensity(bandwidth=bw).fit(eigvals)
    if x_range is None:
        x = np.linspace(0, eigvals[-1] * 1.1, len(eigvals) * 10).reshape(-1, 1)
    else:
        x = x_range.reshape(-1, 1)
    probs = np.exp(kde.score_samples(x))
    plt.plot(x, probs, label=label)
