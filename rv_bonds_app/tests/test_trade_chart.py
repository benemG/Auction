import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.charts import plot_trade_performance

def _hist():
    dates = pd.bdate_range('2024-01-01', periods=400)
    mkt = 10 + np.sin(np.arange(400) / 10)
    return pd.DataFrame({'date': dates, 'spread_market': mkt, 'spread_fitted': np.full(400, 9.0)}), dates

TRADE = pd.Series(dict(target_gap_bps=-3.0, spread_market_bps=None, spread_target_bps=None, spread_fitted_bps=9.0,
                       stop_loss_bps=2.0, ou_theta=10.0, ou_calibration_status='valid', bid_ask_cost_bps=0.5,
                       net_carry_roll_3m_bps=1.0, carry_3m_bps=0.4, risk_h_bps=3.0))

def test_realized_pnl_starts_at_minus_cost_and_follows_trade_direction():
    df, dates = _hist()
    as_of = dates[300]
    trade = TRADE.copy()
    trade['spread_market_bps'] = df['spread_market'].iloc[300]
    trade['spread_target_bps'] = 9.0 + 1.0
    fig = plot_trade_performance(df, as_of, trade)
    pnl = next(t for t in fig.data if t.name == 'P&L réalisé')
    assert pnl.y[0] == -0.5
    move = df['spread_market'].iloc[310] - df['spread_market'].iloc[300]
    assert np.isclose(pnl.y[10], -move + 0.4 * 10 / 63 - 0.5)  # vente du spread : gagne quand il baisse

def test_last_date_shows_expected_path_without_realized():
    df, dates = _hist()
    trade = TRADE.copy()
    trade['spread_market_bps'] = df['spread_market'].iloc[-1]
    trade['spread_target_bps'] = 10.0
    names = {t.name for t in plot_trade_performance(df, dates[-1], trade).data}
    assert 'Attendu' in names and 'P&L réalisé' not in names and 'Après entrée' not in names
