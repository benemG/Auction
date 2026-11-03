import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.mean_reversion import (MeanReversionModel, horizon_risk, bertram_entry_band, leung_li_exit, ou_exit_time,
                                market_share, quote_glitches)

DT = 1.0 / 252

def _ou(rng, hl_days, n=252):
    phi = np.exp(-np.log(2) / hl_days)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return pd.Series(x)

def test_half_life_bias_is_corrected():
    rng = np.random.default_rng(0)
    hl = []
    for _ in range(200):
        m = MeanReversionModel()
        if m.calibrate_ou_model(_ou(rng, 20), DT)['status'] == 'valid':
            hl.append(m.compute_half_life() * 252)
    assert 16 <= np.median(hl) <= 25  # brut, sans correction : ~14 j

def test_random_walk_rarely_looks_fast_after_correction():
    # Sans correction, ~69 % des marches aléatoires ont une demi-vie <= 60 j ; Kendall ne suffit pas pour une vraie
    # racine unitaire (biais de Dickey-Fuller plus fort), d'où l'ADF à côté
    rng = np.random.default_rng(1)
    fast = 0
    for _ in range(100):
        m = MeanReversionModel()
        if m.calibrate_ou_model(pd.Series(np.cumsum(rng.standard_normal(252))), DT)['status'] == 'valid':
            fast += m.compute_half_life() * 252 <= 60
    assert fast < 40

def test_horizon_risk_random_walk_and_white_noise():
    rng = np.random.default_rng(2)
    rw = np.median([horizon_risk(pd.Series(np.cumsum(rng.standard_normal(600))), 63, 21) for _ in range(50)])
    wn = np.median([horizon_risk(pd.Series(rng.standard_normal(600)), 63, 21) for _ in range(50)])
    assert rw == pytest.approx(np.sqrt(63), rel=0.25)
    assert wn == pytest.approx(np.sqrt(2), rel=0.25)

THETA, SIGMA = 12.0, 30.0
S = SIGMA / np.sqrt(2 * THETA)  # écart-type stationnaire

@pytest.mark.parametrize('theta', [THETA, 252 * np.log(2) / 2])  # demi-vies ~15 j et 2 j
def test_exit_time_matches_first_hitting_time_density(theta):
    sigma = SIGMA * np.sqrt(theta / THETA)  # même écart-type stationnaire S
    m = MeanReversionModel()
    m.theta, m.mu, m.sigma, m.is_calibrated = theta, 0.0, sigma, True
    for y0 in (-S, -2 * S):
        # Stop très éloigné et objectif sur la moyenne : même quantité que l'intégrale de densité existante
        fht = m.compute_first_hitting_time(y0)['expected_time']
        assert fht > 0
        assert ou_exit_time(y0, -6 * S, 0.0, theta, sigma) == pytest.approx(fht, rel=0.02)

def test_leung_li_probability_matches_monte_carlo():
    y0 = -2 * S
    b, L, gain, p = leung_li_exit(y0, THETA, SIGMA, 2.0, 1.0)
    assert y0 < b <= 0 and L == pytest.approx(y0 - S)  # objectif plafonné à la moyenne
    rng = np.random.default_rng(3)
    n, dt = 4000, DT / 20
    y, done, hit = np.full(n, y0), np.zeros(n, bool), np.zeros(n, bool)
    while not done.all():
        live = ~done
        y[live] += -THETA * y[live] * dt + SIGMA * np.sqrt(dt) * rng.standard_normal(live.sum())
        up, dn = (y >= b) & live, (y <= L) & live
        hit |= up
        done |= up | dn
    assert hit.mean() == pytest.approx(p, abs=0.03)

def test_bertram_band_widens_with_cost():
    bands = [bertram_entry_band(THETA, SIGMA, c) for c in (0.5, 2.0, 5.0)]
    assert bands[0] < bands[1] < bands[2]

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

def test_market_share_separates_market_and_fit_driven_residuals():
    rng = np.random.default_rng(1)
    mkt = pd.Series(_ou(rng, 10).values + 50)
    curve = pd.Series(_ou(rng, 10).values)
    assert market_share(mkt, mkt - 50) == pytest.approx(1.0)            # courbe figée : tout vient du marché
    assert abs(market_share(mkt * 0 + 50, curve)) < 1e-9                 # spread figé : tout vient de la courbe

def test_quote_glitches_counts_reverting_spikes_only():
    rng = np.random.default_rng(2)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 252)))
    base = quote_glitches(s)
    s.iloc[100] += 30            # cotation erronée isolée
    s.iloc[200:] += 30           # vrai saut de niveau, qui ne revient pas
    assert quote_glitches(s) == base + 1
