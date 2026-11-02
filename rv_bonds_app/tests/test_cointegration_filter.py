import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.trade_definitions import filter_cointegrated_spreads

def _market():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range('2025-01-01', periods=300)
    base = np.cumsum(rng.normal(0, 1, len(dates)))
    series = {
        'A': base,
        'B': base + rng.normal(0, 0.5, len(dates)),   # cointégré avec A
        'C': np.cumsum(rng.normal(0, 1, len(dates))),  # marche aléatoire indépendante
    }
    rows = [dict(date=d, isin=k, asset_swap_spread=v[i]) for k, v in series.items() for i, d in enumerate(dates)]
    return pd.DataFrame(rows), dates[-1]

def _trade(a, b):
    return dict(trade_id=f'SPD | {a} / {b}', bond1_isin=a, bond2_isin=b)

def test_keeps_cointegrated_pair_and_drops_independent_walk():
    market, last = _market()
    kept = filter_cointegrated_spreads([_trade('A', 'B'), _trade('A', 'C')], market, 'asset_swap_spread', last)
    assert [t['trade_id'] for t in kept] == ['SPD | A / B']

def test_drops_pair_with_short_history_or_unknown_isin():
    market, last = _market()
    short = market[~((market['isin'] == 'B') & (market['date'] < market['date'].max() - pd.Timedelta(days=60)))]
    assert filter_cointegrated_spreads([_trade('A', 'B')], short, 'asset_swap_spread', last) == []
    assert filter_cointegrated_spreads([_trade('A', 'Z')], market, 'asset_swap_spread', last) == []

def test_select_pvalues_fdr_is_stricter_than_raw_threshold():
    from src.trade_definitions import select_pvalues
    p = np.array([0.001, 0.04, 0.045, 0.049, 0.2, 0.3, np.nan])
    raw = select_pvalues(p, p, 0.05, None, False)
    fdr = select_pvalues(p, p, 0.05, 0.05, False)
    assert raw.tolist() == [True, True, True, True, False, False, False]
    assert fdr.tolist() == [True, False, False, False, False, False, False]

def test_adf_on_spread_rejects_nonstationary_unit_beta_spread():
    from src.trade_definitions import select_pvalues
    assert select_pvalues(np.array([0.001]), np.array([0.5]), 0.05, None, True).tolist() == [False]
    assert select_pvalues(np.array([0.001]), np.array([0.5]), 0.05, None, False).tolist() == [True]

def test_all_windows_must_pass():
    market, last = _market()
    recent_break = market.copy()
    mask = (recent_break['isin'] == 'B') & (recent_break['date'] > recent_break['date'].max() - pd.Timedelta(days=90))
    recent_break.loc[mask, 'asset_swap_spread'] += np.cumsum(np.ones(mask.sum())) * 3  # B décroche de A récemment
    trades = [_trade('A', 'B')]
    assert filter_cointegrated_spreads(trades, market, 'asset_swap_spread', last, windows=(252, 126), fdr_alpha=None)
    assert not filter_cointegrated_spreads(trades, recent_break, 'asset_swap_spread', last, windows=(252, 126), fdr_alpha=None)

def test_ic_table_is_one_when_spread_fully_reverts():
    from scripts.universe_comparison import ic_table
    z = np.tile(np.linspace(-2, 2, 10), 3)
    obs = pd.DataFrame({'date': np.repeat([1, 2, 3], 10), 'z': z, 'd5': -z})
    t = ic_table(obs, {'tous': np.ones(len(obs), dtype=bool)}, (5,))
    assert abs(t['IC'].iloc[0] - 1) < 1e-9 and t['bon sens'].iloc[0] == 1.0
