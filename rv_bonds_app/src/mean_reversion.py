import numpy as np
import pandas as pd
from scipy.integrate import quad
import warnings

def compute_hurst_exponent(time_series: pd.Series, max_lag=20) -> float:
    """
    Calcul du coefficient de Hurst par méthode de la variance des différences.
    H < 0.5 : Mean-reverting (anti-persistant)
    H = 0.5 : Marche aléatoire
    H > 0.5 : Tendance (persistant)
    """
    time_series = time_series.dropna()
    if len(time_series) < max_lag * 2:
        return np.nan
        
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(time_series[lag:].values, time_series[:-lag].values)) for lag in lags]
    
    # Remplacer les 0 pour éviter log(0)
    tau = [t if t > 0 else 1e-8 for t in tau]
    
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

class MeanReversionModel:
    """
    Modèle Ornstein-Uhlenbeck (OU).
    dX_t = theta * (mu - X_t)dt + sigma * dW_t
    Calibration par MLE strict.
    """
    def __init__(self):
        self.theta = None
        self.mu = None
        self.sigma = None
        self.dt = None
        self.is_calibrated = False
        self.status = "not_available"
        self.message = ""

    def calibrate_ou_model(self, time_series: pd.Series, dt: float) -> dict:
        """
        Calibration MLE exacte du processus OU.
        """
        self.dt = dt
        
        # 1. Vérification des données
        if not isinstance(time_series, pd.Series):
            time_series = pd.Series(time_series)
            
        time_series = time_series.dropna()
        if len(time_series) < 2:
            self.status = "failed"
            self.message = "Pas assez de données pour la calibration."
            return self._return_state()

        X = time_series.values
        N = len(X) - 1
        XX = X[:-1]
        YY = X[1:]

        Sx = np.sum(XX)
        Sy = np.sum(YY)
        Sxx = np.dot(XX, XX)
        Sxy = np.dot(XX, YY)
        Syy = np.dot(YY, YY)

        # Fallback OLS (AR(1)) function
        def fallback_ols(X, dt):
            Y = X[1:]
            X_lag = X[:-1]
            X_mat = np.vstack([np.ones(len(X_lag)), X_lag]).T
            try:
                beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ Y
                c, phi = beta[0], beta[1]
                if phi >= 1.0 or phi <= 0.0:
                    return None
                mu_ols = c / (1.0 - phi)
                theta_ols = -np.log(phi) / dt
                res = Y - (c + phi * X_lag)
                var_res = np.var(res, ddof=1)
                sigma_ols = np.sqrt(var_res * 2 * theta_ols / (1.0 - phi**2))
                return mu_ols, theta_ols, sigma_ols
            except Exception:
                return None

        # Calcul de mu_hat
        den_mu = N * (Sxx - Sxy) - (Sx**2 - Sx * Sy)
        use_fallback = False
        
        if abs(den_mu) < 1e-12:
            use_fallback = True
        else:
            mu_hat = (Sy * Sxx - Sx * Sxy) / den_mu
            num_theta = Sxy - mu_hat * Sx - mu_hat * Sy + N * (mu_hat**2)
            den_theta = Sxx - 2 * mu_hat * Sx + N * (mu_hat**2)
            
            if den_theta <= 0 or num_theta / den_theta <= 0:
                use_fallback = True
            else:
                theta_hat = - (1.0 / dt) * np.log(num_theta / den_theta)
                if theta_hat <= 0:
                    use_fallback = True
                else:
                    exp_theta_dt = np.exp(-theta_hat * dt)
                    exp_2theta_dt = np.exp(-2 * theta_hat * dt)
                    sigma_eps_2 = (1.0 / N) * (
                        Syy 
                        - 2 * exp_theta_dt * Sxy 
                        + exp_2theta_dt * Sxx 
                        - 2 * mu_hat * (1 - exp_theta_dt) * (Sy - exp_theta_dt * Sx) 
                        + N * (mu_hat**2) * ((1 - exp_theta_dt)**2)
                    )
                    
                    if sigma_eps_2 <= 0:
                        use_fallback = True
                    else:
                        sigma_2 = sigma_eps_2 * (2 * theta_hat) / (1 - exp_2theta_dt)
                        if sigma_2 <= 0:
                            use_fallback = True
                        else:
                            sigma_hat = np.sqrt(sigma_2)

        if use_fallback:
            fallback_res = fallback_ols(X, dt)
            if fallback_res is None:
                self.status = "failed"
                self.message = "Calibration MLE et fallback OLS ont échoué."
                return self._return_state()
            else:
                mu_hat, theta_hat, sigma_hat = fallback_res
                self.message = "Calibration par fallback OLS réussie."
        else:
            self.message = "Calibration MLE réussie."

        # Sauvegarde de l'état
        self.theta = theta_hat
        self.mu = mu_hat
        self.sigma = sigma_hat
        self.is_calibrated = True
        self.status = "valid"
        
        return self._return_state()
        
    def _return_state(self):
        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "status": self.status,
            "message": self.message
        }

    def compute_future_values(self, current_value: float, horizon_t: float) -> float:
        """E[X_t|X_0] = X_0 exp(-theta t) + mu (1 - exp(-theta t))"""
        if not self.is_calibrated:
            return np.nan
        return current_value * np.exp(-self.theta * horizon_t) + self.mu * (1 - np.exp(-self.theta * horizon_t))

    def compute_future_volatility(self, horizon_t: float) -> float:
        """sqrt[(sigma^2/(2 theta)) * (1 - exp(-2 theta t))]"""
        if not self.is_calibrated:
            return np.nan
        return np.sqrt((self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * horizon_t)))
        
    def compute_half_life(self) -> float:
        """ln(2) / theta"""
        if not self.is_calibrated or self.theta <= 0:
            return np.nan
        return np.log(2) / self.theta

    def compute_first_hitting_time(self, current_value: float) -> dict:
        """
        Temps de premier franchissement de mu par intégration numérique.
        """
        if not self.is_calibrated:
            return {"expected_time": np.nan, "standard_deviation": np.nan, "status": "failed"}
            
        if abs(current_value - self.mu) < 1e-6:
            return {"expected_time": 0.0, "standard_deviation": 0.0, "status": "valid"}
            
        C = (current_value - self.mu) * np.sqrt(2 * self.theta) / self.sigma
        
        def _density_T_to_theta(t, C):
            # Prévention underflow / overflow dans l'exponentielle
            if t < 1e-8:
                return 0.0
            
            exp_neg_t = np.exp(-t)
            exp_neg_2t = np.exp(-2 * t)
            den = 1.0 - exp_neg_2t
            
            if den <= 0:
                return 0.0
                
            term1 = np.sqrt(2.0 / np.pi) * np.abs(C) * exp_neg_t / (den ** 1.5)
            # Clip pour éviter le warning overflow in exp
            exponent = - ((C**2) * exp_neg_2t) / (2 * den)
            exponent = max(exponent, -700.0)
            
            return term1 * np.exp(exponent)
            
        try:
            # Intégrale pour E[T]
            res_T, err_T = quad(
                lambda t: t * self.theta * _density_T_to_theta(self.theta * t, C), 
                0, 
                1000,
                limit=100
            )
            
            # Intégrale pour Var(T)
            res_var, err_var = quad(
                lambda t: ((t - res_T)**2) * self.theta * _density_T_to_theta(self.theta * t, C), 
                0, 
                1000,
                limit=100
            )
            
            if np.isnan(res_T) or np.isnan(res_var) or res_var < 0:
                return {"expected_time": np.nan, "standard_deviation": np.nan, "status": "failed"}
                
            return {
                "expected_time": res_T, 
                "standard_deviation": np.sqrt(res_var), 
                "status": "valid"
            }
        except Exception as e:
            return {"expected_time": np.nan, "standard_deviation": np.nan, "status": f"failed: {e}"}

from statsmodels.tsa.stattools import adfuller

def compute_adf_pvalue(time_series: pd.Series) -> float:
    """
    Calcule la p-value du test de Dickey-Fuller Augmenté (ADF).
    Une p-value < 0.05 indique que le spread est stationnaire (cointégré).
    """
    ts = time_series.dropna()
    if len(ts) < 30:
        return np.nan
        
    try:
        # maxlag par défaut ou AIC
        result = adfuller(ts, autolag='AIC')
        return result[1] # p-value est le 2ème élément
    except Exception:
        return np.nan
