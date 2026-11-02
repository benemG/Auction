import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.opportunity_generation import apply_repo_spreads

def _df():
    return pd.DataFrame([{
        'target_gap_bps': 4.0, 'net_carry_roll_3m_bps': 1.0, 'carry_roll_adjusted_gap': 3.0,
        'net_edge_ratio': 1.5, 'daily_vol_bps': 2.0,
        'leg_issuers': json.dumps(['bund', 'oat']), 'leg_repo_sens': json.dumps([0.05, -0.025]),
    }])

def test_apply_repo_spreads_adjusts_by_leg_sensitivity():
    out = apply_repo_spreads(_df(), {'bund': -10.0, 'default': 4.0})
    adj = 0.05 * -10.0 + -0.025 * 4.0  # -0.6 bp
    assert out['net_carry_roll_3m_bps'].iloc[0] == 1.0 + adj
    assert out['carry_roll_adjusted_gap'].iloc[0] == 3.0 + adj
    assert out['net_edge_ratio'].iloc[0] == 1.5 + adj / 2.0

def test_apply_repo_spreads_noop_without_columns_or_spreads():
    df = _df()
    assert apply_repo_spreads(df, {'bund': 0.0}) is df
    old = df.drop(columns=['leg_repo_sens'])
    assert apply_repo_spreads(old, {'bund': -10.0}) is old
    # Parquet antérieur lu dans un schéma unifié : colonnes présentes mais NaN
    nan_rows = df.assign(leg_repo_sens=np.nan, leg_issuers=np.nan)
    assert apply_repo_spreads(nan_rows, {'bund': -10.0})['net_edge_ratio'].iloc[0] == 1.5
