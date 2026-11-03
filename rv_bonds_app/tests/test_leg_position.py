import ast
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.trade_definitions import leg_position
from src.opportunity_generation import generate_opportunities_for_issuer

def _row(weights, gap):
    isins = ['A', 'B', 'C'][:len(weights)] + [None] * (3 - len(weights))
    return dict(bond1_isin=isins[0], bond2_isin=isins[1], bond3_isin=isins[2], leg_weights=str(weights), target_gap_bps=gap)

@pytest.mark.parametrize('weights, gap, expected', [
    ([1.0, -1.0], 3.0, ['Short', 'Long']),             # achat du spread y_A - y_B : il monte si A sous-performe
    ([1.0, -1.0], -3.0, ['Long', 'Short']),
    ([0.6, -1.0, 0.4], 3.0, ['Short', 'Long', 'Short']),  # achat du fly : long du belly, short des ailes
    ([0.6, -1.0, 0.4], -3.0, ['Long', 'Short', 'Long']),
])
def test_side_of_each_leg(weights, gap, expected):
    row = _row(weights, gap)
    assert [leg_position(row, i) for i in ['A', 'B', 'C'][:len(weights)]] == expected

@pytest.mark.parametrize('weights', [[1.0, -1.0], [0.6, -1.0, 0.4]])
@pytest.mark.parametrize('gap', [3.0, -3.0])
def test_positions_earn_when_spread_moves_to_target(weights, gap):
    """Oracle économique : si chaque rendement bouge dans le sens du trade, le P&L des titres (-position × Δy,
    un titre long gagne quand son rendement baisse) est positif."""
    row = _row(weights, gap)
    dy = np.sign(gap) * np.array(weights)  # Δspread = Σ w·Δy = sign(gap)·Σ w² : vers la cible
    pos = [1 if leg_position(row, i) == 'Long' else -1 for i in ['A', 'B', 'C'][:len(weights)]]
    assert -np.dot(np.abs(weights) * pos, dy) > 0

def test_generated_fly_has_belly_at_minus_one_and_positive_wings():
    """La règle suppose la convention du générateur : belly à -1, ailes positives."""
    dates = pd.bdate_range('2025-01-01', periods=300)
    rng = np.random.default_rng(0)
    rows, fits = [], []
    for isin, ttm, y in [('A', 2.0, 2.0), ('B', 5.0, 2.5), ('C', 8.0, 2.8)]:
        noise = rng.normal(0, 0.02, len(dates))
        for i, d in enumerate(dates):
            rows.append(dict(date=d, isin=isin, ytm=y + noise[i], asset_swap_spread=10 + 100 * noise[i], dv01=np.nan,
                             modified_duration=np.nan))
            fits.append(dict(date=d, isin=isin, issuer='x', ttm_years=ttm, ytm_fitted=y, asset_swap_spread_fitted=10.0,
                             ytm_rmse=0.01, asset_swap_spread_rmse=1.0))
    fly = [dict(trade_id='FLY | A / B / C', trade_type='fly', bond1_isin='A', bond2_isin='B', bond3_isin='C',
                bond1_name='A', bond2_name='B', bond3_name='C', ttm_a=2.0, ttm_b=5.0, ttm_c=8.0)]
    row = generate_opportunities_for_issuer(pd.DataFrame(rows), pd.DataFrame(fits), fly, dates[-1], 'asset_swap_spread').iloc[0]
    w = ast.literal_eval(row['leg_weights'])
    assert w[1] == -1.0 and w[0] > 0 and w[2] > 0
    belly = leg_position(row, 'B')
    assert belly == ('Long' if row['target_gap_bps'] > 0 else 'Short')
    assert leg_position(row, 'A') == leg_position(row, 'C') != belly
