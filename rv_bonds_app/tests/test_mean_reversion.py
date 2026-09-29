import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.mean_reversion import MeanReversionModel

def test_ou_calibration_and_fht():
    np.random.seed(42)
    
    # Simulation OU exacte Euler-Maruyama
    N = 500
    dt = 1.0 / 252.0
    mu_true = 10.0
    theta_true = 5.0
    sigma_true = 2.0
    
    X = np.zeros(N)
    X[0] = 5.0
    for t in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        X[t] = X[t-1] + theta_true * (mu_true - X[t-1]) * dt + sigma_true * dW
        
    ts = pd.Series(X)
    
    model = MeanReversionModel()
    state = model.calibrate_ou_model(ts, dt)
    
    assert state['status'] == 'valid'
    assert abs(state['mu'] - mu_true) < 1.0
    assert state['theta'] > 0
    assert state['sigma'] > 0
    
    # Test FHT
    fht = model.compute_first_hitting_time(5.0)
    assert fht['status'] == 'valid'
    assert fht['expected_time'] > 0
    
    fht_zero = model.compute_first_hitting_time(state['mu'])
    assert fht_zero['expected_time'] == 0.0
