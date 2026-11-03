import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.opportunity_generation import generate_opportunities_for_issuer

def _data():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2025-01-01', periods=300)
    rows, fits = [], []
    for isin, ttm, ytm0, asw0 in [('A', 5.0, 2.50, 30.0), ('B', 3.0, 2.00, 20.0), ('C', 3.0, 2.10, 25.0)]:
        noise = rng.normal(0, 0.02, len(dates))
        for i, d in enumerate(dates):
            rows.append(dict(date=d, isin=isin, ytm=ytm0 + noise[i], asset_swap_spread=asw0 + 100 * noise[i], dv01=np.nan, modified_duration=np.nan))
            fits.append(dict(date=d, isin=isin, issuer='x', ttm_years=ttm, ytm_fitted=ytm0, asset_swap_spread_fitted=asw0,
                             ytm_rmse=0.01, asset_swap_spread_rmse=1.0))
    return pd.DataFrame(rows), pd.DataFrame(fits), dates[-1]

def test_ytm_metric_is_converted_to_bps_with_its_own_trade_id():
    market, fitted, last = _data()
    trade = [dict(trade_id='SPD | A / B', trade_type='spread', bond1_isin='A', bond2_isin='B', bond3_isin=None,
                  bond1_name='A', bond2_name='B', bond3_name=None, ttm_a=5.0, ttm_b=3.0)]
    asw = generate_opportunities_for_issuer(market, fitted, trade, last, 'asset_swap_spread').iloc[0]
    ytm = generate_opportunities_for_issuer(market, fitted, trade, last, 'ytm').iloc[0]
    assert asw['trade_id'] == 'SPD | A / B' and ytm['trade_id'] == 'SPD | A / B [YTM]'
    assert ytm['metric'] == 'ytm' and ytm['fit_rmse_bps'] == 1.0  # 0.01 % x 100 = 1 bp (moyenne des deux jambes)

def test_same_name_different_isin_gives_unique_trade_ids():
    market, fitted, last = _data()
    trades = [dict(trade_id='SPD | A / B', trade_type='spread', bond1_isin='A', bond2_isin=b, bond3_isin=None,
                   bond1_name='A', bond2_name='B', bond3_name=None, ttm_a=5.0, ttm_b=3.0) for b in ('B', 'C')]
    ids = generate_opportunities_for_issuer(market, fitted, trades, last, 'asset_swap_spread')['trade_id']
    assert ids.is_unique and list(ids) == ['SPD | A / B (A/B)', 'SPD | A / B (A/C)']

def test_dv01_is_approximated_and_weights_are_parseable():
    import ast
    market, fitted, last = _data()
    market['maturity'] = last + pd.to_timedelta(market['isin'].map({'A': 5, 'B': 3, 'C': 3}) * 365.25, unit='D')
    market['bond_name'] = market['isin'].map({'A': 'X 2 1/2 01/01/31', 'B': 'X 2 01/01/29', 'C': 'X 0 01/01/29'})
    market['price_clean'], market['issuer'] = 100.0, 'x'
    trade = [dict(trade_id='SPD | A / B', trade_type='spread', bond1_isin='A', bond2_isin='B', bond3_isin=None,
                  bond1_name='A', bond2_name='B', bond3_name=None, ttm_a=5.0, ttm_b=3.0)]
    row = generate_opportunities_for_issuer(market, fitted, trade, last, 'asset_swap_spread').iloc[0]
    nominals = ast.literal_eval(row['hedge_nominals'])
    assert nominals[0] == 1.0 and -1.7 < nominals[1] < -1.5  # DV01 5 ans / DV01 3 ans ≈ 1.6
    assert abs(row['residual_dv01']) < 1e-9
    assert ast.literal_eval(row['leg_weights']) == [1.0, -1.0]

def test_expected_gain_and_ratio_definition():
    market, fitted, last = _data()
    trade = [dict(trade_id='SPD | A / B', trade_type='spread', bond1_isin='A', bond2_isin='B', bond3_isin=None,
                  bond1_name='A', bond2_name='B', bond3_name=None, ttm_a=5.0, ttm_b=3.0)]
    row = generate_opportunities_for_issuer(market, fitted, trade, last, 'asset_swap_spread').iloc[0]
    assert row['expected_gain_bps'] <= abs(row['target_gap_bps']) + 1e-12
    net_edge = row['expected_gain_bps'] + row['net_carry_roll_3m_bps'] - row['bid_ask_cost_bps']
    assert row['net_edge_ratio'] == np.float64(net_edge / row['risk_h_bps'])
