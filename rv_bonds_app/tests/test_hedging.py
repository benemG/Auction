import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.hedge_ratio import (compute_spread_hedge_ratio, compute_fly_hedge_ratio, compute_residual_dv01,
                             parse_coupon, approx_modified_duration)

def test_parse_coupon_formats():
    assert parse_coupon("BTPS 2 1/2 11/15/25") == 2.5
    assert parse_coupon("BTPS 2.85 02/01/31") == 2.85
    assert parse_coupon("AGFRNC 0 1/8 04/30/22") == 0.125
    assert parse_coupon("RATB 0 04/19/21") == 0.0
    assert np.isnan(parse_coupon("FRTR Float 12/13/97"))

def test_approx_modified_duration():
    # Zéro-coupon : D_mac = T
    assert approx_modified_duration(3.0, 0.0, 7.0) == pytest.approx(7.0 / 1.03)
    # 5 ans au pair à 3 % : D_mac = 4.717, D_mod = 4.580
    assert approx_modified_duration(3.0, 3.0, 5.0) == pytest.approx(4.580, abs=1e-3)
    assert np.isnan(approx_modified_duration(3.0, np.nan, 5.0))

def test_residual_dv01_nan_if_missing():
    assert np.isnan(compute_residual_dv01([1.0, -1.0], [5.0, np.nan]))

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
