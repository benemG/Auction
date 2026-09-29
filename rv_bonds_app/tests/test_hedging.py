import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.hedge_ratio import compute_spread_hedge_ratio, compute_fly_hedge_ratio, compute_residual_dv01

def test_spread_hedge_ratio():
    n1, n2 = compute_spread_hedge_ratio(dv01_a=5.0, dv01_b=10.0)
    assert n1 == 1.0
    assert n2 == -0.5
    
    residual = compute_residual_dv01([n1, n2], [5.0, 10.0])
    assert abs(residual) < 1e-9
    
def test_fly_hedge_ratio():
    # Ailes équidistantes
    w1, w2, w3 = compute_fly_hedge_ratio(dv01_a=2.0, dv01_b=5.0, dv01_c=8.0, ttm_a=2.0, ttm_b=5.0, ttm_c=8.0)
    assert w1 == 0.5
    assert w3 == 0.5
    # DV01 ailes = 0.5*2 + 0.5*8 = 1 + 4 = 5
    # Ratio = - (5 / 5) = -1.0
    assert w2 == -1.0
    
    residual = compute_residual_dv01([w1, w2, w3], [2.0, 5.0, 8.0])
    assert abs(residual) < 1e-9
