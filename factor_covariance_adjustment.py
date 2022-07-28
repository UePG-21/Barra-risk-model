from typing import Sequence

import numpy as np


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


class BiasStatsCalculator:
    """A commonly used measure to assess a risk model's accuracy"""

    def __init__(self, returns: np.ndarray, volatilities: np.ndarray) -> None:
        """
        Args:
            returns (np.ndarray): returns of K assets * T periods
            volatilities (np.ndarray): volatilities (std) of K assets * T periods
        """
        self.K, self.T = returns.shape if returns.ndim == 2 else (1, len(returns))
        self.r = returns.reshape((self.K, self.T))
        self.v = volatilities.reshape((self.K, -1))
        if self.v.shape[1] != 1 and self.v.shape[1] != self.T:
            raise ValueError("wrong shape of volatilities")

    def single_window(self, half_life: int | None = None) -> np.ndarray:
        """Calculate bias statistics, selecting entire sample period as a single window

        Args:
            half_life (int | None, optional): argument in ewa(). Defaults to None.

        Returns:
            np.ndarray: bias statistics value(s) (K * 1)
        """
        b = self.r / self.v
        b_demeaned = b - b.mean(axis=1).reshape((self.K, -1))
        B = np.sqrt(ewa(b_demeaned.T**2, half_life))
        return B.reshape((self.K, -1))

    def rolling_window(self, periods: int, half_life: int | None = None) -> np.ndarray:
        """Calculate bias statistics, specifying number of periods in rolling window

        Args:
            periods (int): number of periods in observation window
            half_life (int | None, optional): argument in ewa(). Defaults to None.

        Returns:
            np.ndarray: bias statistics values (K * (T - periods + 1)
        """
        if periods > self.T or periods < 2:
            raise ValueError("T must be between 2 and the length of returns")
        b = self.r / self.v
        b_demeaned = b - b.mean(axis=1).reshape((self.K, -1))
        B_lst = [
            np.sqrt(ewa(b_demeaned[:, t : t + periods].T ** 2, half_life))
            for t in range(self.T - periods + 1)
        ]
        return np.array(B_lst).T


class FactorCovAdjuster:
    """Adjustments on factor covariance matrix"""

    def __init__(self, FRM: np.ndarray) -> None:
        """
        Args:
            FRM (np.ndarray): factor return matrix (K*T)
        """
        self.K, self.T = FRM.shape
        if self.K > self.T:
            raise ValueError("number of periods must be larger than number of factors")
        self.FRM = FRM.astype("float64")
        self.FCM = None

    def calc_fcm_raw(self, half_life: int) -> np.ndarray:
        """Calculate the factor covariance matrix, FCM (K*K)

        Args:
            half_life (int): time it takes for weight in EWA to reduce to half of
                original value

        Returns:
            np.ndarray: FCM, denoted by `F_Raw`
        """
        self.FCM = cov_ewa(self.FRM, half_life).astype("float64")
        return self.FCM

    def newey_west_adjust(
        self, half_life: int, max_lags: int, multiplier: int
    ) -> np.ndarray:
        """Apply Newey-West adjustment on `F_Raw`

        Args:
            half_life (int): time it takes for weight in EWA to reduce to half of
                original value
            max_lags (int): maximum Newey-West correlation lags
            multiplier (int): number of periods a FCM with new frequence contains

        Returns:
            np.ndarray: Newey-West adjusted FCM, denoted by `F_NW`
        """
        for i in range(1, max_lags + 1):
            C_pos_delta = cov_ewa(self.FRM, half_life, i)
            self.FCM += (1 - i / (1 + max_lags)) * (C_pos_delta + C_pos_delta.T)
        D, U = np.linalg.eigh(self.FCM * multiplier)
        D[D <= 0] = 1e-14  # fix numerical error
        self.FCM = U.dot(np.diag(D)).dot(U.T)
        D, U = np.linalg.eigh(self.FCM)
        return self.FCM

    def eigenfactor_risk_adjust(self, coef: float, M: int = 1000) -> np.ndarray:
        """Apply eigenfactor risk adjustment on `F_NW`

        Args:
            coef (float): adjustment coefficient
            M (int, optional): times of Monte Carlo simulation. Defaults to 10000.

        Returns:
            np.ndarray: eigenfactor risk adjusted FCM, denoted by `F_Eigen`
        """
        D_0, U_0 = np.linalg.eigh(self.FCM)
        D_0[D_0 <= 0] = 1e-14  # fix numerical error
        Lambda = np.zeros((self.K,))
        for _ in range(M):
            b_m = np.array([np.random.normal(0, d**0.5, self.T) for d in D_0])
            f_m = U_0.dot(b_m)
            F_m = f_m.dot(f_m.T) / (self.T - 1)
            D_m, U_m = np.linalg.eigh(F_m)
            D_m[D_m <= 0] = 1e-14  # fix numerical error
            D_m_tilde = U_m.T.dot(self.FCM).dot(U_m)
            Lambda += np.diag(D_m_tilde) / D_m
        Lambda[Lambda <= 0] = 1e-14
        Lambda = np.sqrt(Lambda / M)
        Gamma = coef * (Lambda - 1.0) + 1.0
        D_0_tilde = Gamma**2 * D_0
        self.FCM = U_0.dot(np.diag(D_0_tilde)).dot(U_0.T)
        return self.FCM

    def volatility_regime_adjust(
        self, prev_FCM: np.ndarray, half_life: int
    ) -> np.ndarray:
        """Apply volatility regime adjustment on `F_Eigen`

        Args:
            prev_FCM (np.ndarray): previously estimated factor covariance matrix
                (last `F_Eigen`, since `F_VRA` could lead to huge fluctuations) on only
                one period (not aggregate); the order of factors should remain the same
            half_life (int): time it takes for weight in EWA to reduce to half of
                original value

        Returns:
            np.ndarray:volatility regime dajusted FCM, denoted by `F_VRA`
        """
        sigma = np.sqrt(np.diag(prev_FCM))
        B = BiasStatsCalculator(self.FRM, sigma).single_window(half_life)
        self.FCM = self.FCM * (B**2).mean(axis=0)  # Lambda^2
        return self.FCM
