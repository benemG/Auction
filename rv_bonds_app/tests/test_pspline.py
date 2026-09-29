import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pspline_curve import PSplineCurveFitter

def test_pspline_fitter():
    fitter = PSplineCurveFitter(n_knots=10, degree=3, penalty_order=2, lambda_param=0.1)
    
    # Données synthétiques
    ref_date = '2023-01-01'
    maturities = pd.date_range(start='2024-01-01', periods=20, freq='YE')
    ttm_true = np.linspace(1, 20, 20)
    rates = 0.02 + 0.01 * np.log(ttm_true) # Courbe logarithmique
    
    n_obs = fitter.load_market_data(ref_date, maturities, rates)
    assert n_obs == 20
    
    success = fitter.fit_curve()
    assert success is True
    
    # Vérification RMSE (doit être faible sur des données lisses)
    assert fitter.rmse < 0.005
    
    # Test d'extrapolation
    y_fit, is_extra = fitter.compute_fitted_yield(np.array([10.0, 25.0]))
    assert is_extra[0] == False
    assert is_extra[1] == True
