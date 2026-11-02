import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import data_io

def _df(ytm):
    return pd.DataFrame({'date': pd.to_datetime(['2026-02-09', '2026-02-10']), 'isin': ['A', 'A'],
                         'issuer': 'x', 'ytm': ytm, 'ytm_fitted': ytm})

def test_rewrite_replaces_instead_of_duplicating(tmp_path, monkeypatch):
    monkeypatch.setattr(data_io, 'MARKET_DATA_DIR', tmp_path / 'm')
    monkeypatch.setattr(data_io, 'FITTED_CURVES_DIR', tmp_path / 'f')
    for write, read, col in [(data_io.write_market_data, data_io.read_market_data, 'ytm'),
                             (data_io.write_fitted_curves, data_io.read_fitted_curves, 'ytm_fitted')]:
        write(_df([1.0, 2.0]))
        write(_df([3.0, 4.0]))
        assert list(read('x')[col]) == [3.0, 4.0]
    stale = tmp_path / 'm' / 'issuer=x' / 'year=1999'
    stale.mkdir(parents=True); (stale / 'old.parquet').write_bytes(b'')
    data_io.write_market_data(_df([5.0, 6.0]))
    assert not stale.exists()  # année absente de la nouvelle écriture : supprimée
    assert len(list((tmp_path / 'm').rglob('*.parquet'))) == 1
    assert len(list((tmp_path / 'f').rglob('*.parquet'))) == 1
