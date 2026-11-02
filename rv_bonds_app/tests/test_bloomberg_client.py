import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import bloomberg_client

FIELDS = ['PX_LAST', 'YLD_YTM_MID', 'ASW_SPD_MID', 'Z_SPRD_MID']
DATES = pd.to_datetime(['2026-02-09', '2026-02-10'])

class FakeBlp:
    def __init__(self, df):
        self.df = df
    def bdh(self, **kwargs):
        return self.df

def _wide():
    cols = pd.MultiIndex.from_product([['A Corp', 'B Corp'], FIELDS])
    return pd.DataFrame(np.arange(16, dtype=float).reshape(2, 8), index=DATES, columns=cols)

def _long(wide):
    df = wide.stack(level=0, future_stack=True)
    df.index.names = ['date', 'ticker']
    df = df.reset_index().melt(id_vars=['date', 'ticker'], var_name='field', value_name='value')
    df['value'] = df['value'].astype(str)  # xbbg 1.x : valeurs en texte
    return df

def _fetch(monkeypatch, df):
    monkeypatch.setattr(bloomberg_client, 'blp', FakeBlp(df))
    out = bloomberg_client.fetch_bond_market_data(['A Corp', 'B Corp'], '2026-02-09', '2026-02-10')
    return out.sort_values(['bbg_ticker', 'date']).reset_index(drop=True)

def test_wide_and_long_formats_give_same_result(monkeypatch):
    wide = _wide()
    out_wide, out_long = _fetch(monkeypatch, wide), _fetch(monkeypatch, _long(wide))
    assert list(out_wide.columns) == ['date', 'bbg_ticker', 'price_clean', 'ytm', 'asset_swap_spread', 'z_spread']
    assert out_wide.shape == (4, 6)
    pd.testing.assert_frame_equal(out_wide, out_long)
    assert out_wide.loc[0, 'price_clean'] == 0.0 and out_wide.loc[0, 'z_spread'] == 3.0
