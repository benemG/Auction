import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / 'scripts'))
from fit_daily_curves import exclude_off_curve

def test_excludes_retail_and_temporary_tap_lines_but_keeps_green_twins():
    dates = pd.bdate_range('2024-01-01', periods=100)
    rows = [(d, isin, name) for d in dates for isin, name in [('MAIN', 'AGFRNC 1 3/8 07/05/32'),
                                                              ('GREEN', 'DBR 0 08/15/30'), ('CONV', 'DBR 0 08/15/30'),
                                                              ('RETAIL', 'BGBRT 2.6 12/11/28')]]
    rows += [(d, 'TEMP', 'AGFRNC 1 3/8 07/05/32') for d in dates[:6]]
    df = pd.DataFrame(rows, columns=['date', 'isin', 'bond_name'])
    assert set(exclude_off_curve(df)['isin']) == {'MAIN', 'GREEN', 'CONV'}
