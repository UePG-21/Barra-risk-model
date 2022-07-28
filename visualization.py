import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity


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
