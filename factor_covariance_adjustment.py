import numpy as np
from bias_statistics import BiasStatsCalculator
from utils import cov_ewa


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
