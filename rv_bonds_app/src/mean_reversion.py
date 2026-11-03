import numpy as np
import pandas as pd
from scipy.integrate import quad, trapezoid, cumulative_trapezoid
from scipy.special import erf, erfi
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

        # Correction de Kendall (1954) du biais de l'AR(1) : sur ~1 an, la demi-vie estimée est trop courte.
        # var_eps = variance des innovations, identique dans les deux chemins : sigma² (1 - phi²) / (2 theta)
        phi = np.exp(-theta_hat * dt)
        var_eps = sigma_hat**2 * (1 - phi**2) / (2 * theta_hat)
        phi = phi + (1 + 3 * phi) / N
        if phi >= 1.0:
            self.status = "failed"
            self.message = "Pas de retour à la moyenne après correction du biais (racine unitaire)."
            return self._return_state()
        theta_hat = -np.log(phi) / dt
        sigma_hat = np.sqrt(var_eps * 2 * theta_hat / (1 - phi**2))

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
            
        # Borne en temps réduit theta·t <= 50 (queue en e^-theta·t) : avec [0, 1000] ans, quad ratait le pic des
        # demi-vies courtes (quelques jours) et renvoyait ~0
        t_max = 50.0 / self.theta
        try:
            # Intégrale pour E[T]
            res_T, err_T = quad(
                lambda t: t * self.theta * _density_T_to_theta(self.theta * t, C),
                0,
                t_max,
                limit=100
            )

            # Intégrale pour Var(T)
            res_var, err_var = quad(
                lambda t: ((t - res_T)**2) * self.theta * _density_T_to_theta(self.theta * t, C),
                0,
                t_max,
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

def horizon_risk(ts: pd.Series, h: int, halflife: float, window: int = 252) -> float:
    """
    Écart-type des variations à h jours de `ts` : vol quotidienne EWM × √(h · VR_h), où VR_h = var(Δ_h) / (h · var(Δ_1))
    est le ratio de variance (Lo-MacKinlay) sur `window` jours. La vol suit le régime récent, la forme de retour à la
    moyenne vient de la fenêtre longue (une EWM des Δ_h qui se chevauchent n'aurait que 2-3 fenêtres indépendantes).
    """
    ts = ts.dropna()
    if len(ts) < h + 30:
        return np.nan
    tail = ts.tail(window)
    # Moments non centrés (dérive ~0) : centrer les Δ_h qui se chevauchent sous-estime la variance de ~h/window
    var1 = (tail.diff() ** 2).mean()
    if not var1 > 0:
        return np.nan
    vr = (tail.diff(h) ** 2).mean() / (h * var1)
    vol_ewm = np.sqrt((ts.diff() ** 2).ewm(halflife=halflife).mean().iloc[-1])
    return float(vol_ewm * np.sqrt(h * vr))

# --- Seuils optimaux pour Y, OU de moyenne 0 (dY = -theta Y dt + sigma dW), position gagnante quand Y monte ---
# En unités u = y·√theta/sigma, la fonction d'échelle de l'OU est ∝ erfi(u) : tout est vectorisé, sans quad par trade.

def bertram_entry_band(theta: float, sigma: float, cost: float) -> float:
    """Demi-largeur m* de la bande symétrique autour de la moyenne qui maximise (2m - cost) / E[cycle] (Bertram 2010)."""
    m = np.linspace(0.05, 3.0, 300) * sigma / np.sqrt(2 * theta)
    cycle = 2 * np.pi / theta * erfi(m * np.sqrt(theta) / sigma)
    return float(m[np.argmax((2 * m - cost) / cycle)])

def leung_li_exit(y0: float, theta: float, sigma: float, cost: float, stop_sd: float):
    """
    Sortie optimale b* (take-profit) avec stop-loss L = y0 - stop_sd·s (Leung & Li 2015), s = écart-type stationnaire.
    V(b) = L - cost + (b - L)·P(b avant L), P venant de la fonction d'échelle.
    shortcut: actualisation r = 0 (limite de leurs fonctions F/G) et objectif plafonné à la moyenne : sans coût d'attente,
    b* partait au-delà de μ (sorties à ~2× la demi-vie ou plus) ; passer à F/G avec r > 0 pour arbitrer temps et gain.
    Retourne (b*, L, gain attendu net V(b*) - y0, P(b* avant L)) ; NaN si les erfi débordent (écart extrême).
    """
    s = sigma / np.sqrt(2 * theta)
    k = np.sqrt(theta) / sigma
    L = y0 - stop_sd * s
    if y0 >= 0:
        return np.nan, L, np.nan, np.nan
    b = np.linspace(y0, 0, 301)[1:]
    with np.errstate(over='ignore', invalid='ignore'):
        p = (erfi(k * y0) - erfi(k * L)) / (erfi(k * b) - erfi(k * L))
    value = L - cost + (b - L) * p
    if not np.isfinite(value).any():
        return np.nan, L, np.nan, np.nan
    i = np.nanargmax(value)
    return float(b[i]), float(L), float(value[i] - y0), float(p[i])

def ou_exit_time(y0: float, lo: float, hi: float, theta: float, sigma: float, n: int = 2001) -> float:
    """
    Temps moyen (en années) avant que Y sorte de ]lo, hi[ en partant de y0 : (sigma²/2) f'' - theta y f' = -1,
    f(lo) = f(hi) = 0. En u : f' = e^{u²} (C - √π/theta · erf u), C fixé par f(hi) = 0.
    """
    k = np.sqrt(theta) / sigma
    w = np.linspace(lo * k, hi * k, n)
    with np.errstate(over='ignore'):
        e = np.exp(w**2)
    if not np.isfinite(e).all():
        return np.nan
    c = np.sqrt(np.pi) / theta * trapezoid(e * erf(w), w) / trapezoid(e, w)
    f = cumulative_trapezoid(e * (c - np.sqrt(np.pi) / theta * erf(w)), w, initial=0)
    return float(np.interp(y0 * k, w, f))

def market_share(ts_market: pd.Series, ts_svf: pd.Series, h: int = 21) -> float:
    """Part du mouvement du résidu portée par le spread de marché : β des variations à h jours du SvF sur celles du
    spread. Le reste vient de la courbe fittée, qui ne paie rien (courbes d'agences peu peuplées)."""
    d_svf = ts_svf.diff(h)
    d_mkt = ts_market.reindex(ts_svf.index).diff(h)
    v = d_svf.var()
    return float(d_mkt.cov(d_svf) / v) if v > 0 else np.nan

def quote_glitches(ts: pd.Series, k: float = 4.0) -> int:
    """Jours où le spread saute de plus de k écarts-types robustes puis revient le lendemain : cotation erronée."""
    d = ts.dropna().diff().dropna()
    thr = k * 1.4826 * (d - d.median()).abs().median()
    if not thr > 0:
        return 0
    nxt = d.shift(-1)
    return int(((d.abs() > thr) & (nxt.abs() > thr) & (np.sign(d) != np.sign(nxt))).sum())

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
