import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
import warnings

class PSplineCurveFitter:
    """
    Fit d'une courbe de taux via B-splines pénalisées (P-spline).
    Minimise: ||y - B*theta||^2 + lambda * ||D*theta||^2
    """
    def __init__(self, n_knots=20, degree=3, penalty_order=2, lambda_param=1.0):
        self.n_knots = n_knots
        self.degree = degree
        self.penalty_order = penalty_order
        self.lambda_param = lambda_param
        
        self.knots = None
        self.theta = None
        self.is_calibrated = False
        self.min_ttm = None
        self.max_ttm = None
        self.rmse = None
        self.mae = None
        
    def _create_diff_matrix(self, n, order):
        """Créer la matrice de différence d'ordre donné"""
        D = np.eye(n)
        for _ in range(order):
            D = np.diff(D, axis=0)
        return D
        
    def load_market_data(self, reference_date, maturity_dates, rates):
        """
        Calcule les TTM (Time To Maturity) en années (Actual/365.25)
        et écarte les données invalides.
        """
        reference_date = pd.to_datetime(reference_date)
        # Convertir en Series pour garantir l'accès à .dt
        maturity_dates = pd.Series(maturity_dates)
        maturity_dates = pd.to_datetime(maturity_dates)
        
        ttm_years = (maturity_dates - reference_date).dt.days / 365.25
        
        # Filtre les valeurs valides
        rates = np.array(rates)
        mask = (ttm_years > 0) & np.isfinite(rates)
        
        self.ttm_clean = ttm_years[mask].values
        self.rates_clean = rates[mask]
        
        # Tri croissant
        sort_idx = np.argsort(self.ttm_clean)
        self.ttm_clean = self.ttm_clean[sort_idx]
        self.rates_clean = self.rates_clean[sort_idx]
        
        return len(self.rates_clean)
        
    def fit_curve(self):
        """
        Calibre le P-Spline.
        Retourne True si succès, False sinon.
        """
        if len(self.ttm_clean) < self.degree + 2:
            warnings.warn("Pas assez de points pour fitter un P-spline.")
            return False
            
        self.min_ttm = np.min(self.ttm_clean)
        self.max_ttm = np.max(self.ttm_clean)
        
        # Placement des nœuds uniformément entre min_ttm et max_ttm
        # On ajoute les noeuds extérieurs pour la condition aux bords du B-spline
        inner_knots = np.linspace(self.min_ttm, self.max_ttm, self.n_knots - 2 * self.degree)
        d_knot = inner_knots[1] - inner_knots[0] if len(inner_knots) > 1 else 1.0
        
        left_knots = [inner_knots[0] - (i + 1) * d_knot for i in range(self.degree)][::-1]
        right_knots = [inner_knots[-1] + (i + 1) * d_knot for i in range(self.degree)]
        
        self.knots = np.concatenate([left_knots, inner_knots, right_knots])
        
        # Construction de la matrice B
        n_obs = len(self.ttm_clean)
        n_splines = len(self.knots) - self.degree - 1
        
        B = np.zeros((n_obs, n_splines))
        for i in range(n_splines):
            c = np.zeros(n_splines)
            c[i] = 1.0
            spline = BSpline(self.knots, c, self.degree, extrapolate=False)
            # Extrapolation manuelle si hors des bornes très légèrement due aux flottants
            B[:, i] = np.where((self.ttm_clean >= self.knots[0]) & (self.ttm_clean <= self.knots[-1]), 
                               np.nan_to_num(spline(self.ttm_clean)), 0.0)
                               
        # Matrice de pénalité D
        D = self._create_diff_matrix(n_splines, self.penalty_order)
        
        # Résolution (B'B + lambda * D'D) * theta = B'y
        BtB = B.T @ B
        DtD = D.T @ D
        Bty = B.T @ self.rates_clean
        
        try:
            self.theta = np.linalg.solve(BtB + self.lambda_param * DtD, Bty)
        except np.linalg.LinAlgError:
            try:
                # Fallback sur pseudo-inverse
                self.theta = np.linalg.pinv(BtB + self.lambda_param * DtD) @ Bty
            except Exception as e:
                warnings.warn(f"Échec de calibration: {e}")
                return False
                
        self.is_calibrated = True
        
        # Diagnostics
        fitted_y = B @ self.theta
        residuals = self.rates_clean - fitted_y
        self.rmse = np.sqrt(np.mean(residuals**2))
        self.mae = np.mean(np.abs(residuals))
        
        return True
        
    def compute_fitted_yield(self, year_fraction):
        """
        Calcule les valeurs fittées. 
        Renvoie la valeur et un flag d'extrapolation.
        """
        if not self.is_calibrated:
            raise ValueError("Le modèle n'est pas calibré.")
            
        year_fraction = np.atleast_1d(year_fraction)
        n_splines = len(self.knots) - self.degree - 1
        
        B_new = np.zeros((len(year_fraction), n_splines))
        for i in range(n_splines):
            c = np.zeros(n_splines)
            c[i] = 1.0
            spline = BSpline(self.knots, c, self.degree, extrapolate=True)
            B_new[:, i] = spline(year_fraction)
            
        y_fit = B_new @ self.theta
        
        is_extrapolated = (year_fraction < self.min_ttm) | (year_fraction > self.max_ttm)
        
        if len(y_fit) == 1:
            return y_fit[0], is_extrapolated[0]
        return y_fit, is_extrapolated
