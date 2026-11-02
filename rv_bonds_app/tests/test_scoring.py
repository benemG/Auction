import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.scoring import calculate_confidence_score
from src.config import SCORE_WEIGHTS

BASE = dict(ou_calibration_status='valid', half_life_days=20, zscore_12m=2.5, fit_rmse_bps=1.0,
            momentum_signal=np.nan, target_gap_bps=1.0)

def test_stationary_mean_reverting_beats_random_walk():
    good = calculate_confidence_score({**BASE, 'adf_pvalue': 0.01, 'hurst': 0.3}, SCORE_WEIGHTS)
    bad = calculate_confidence_score({**BASE, 'adf_pvalue': 0.5, 'hurst': 0.55}, SCORE_WEIGHTS)
    assert good - bad == 100 * (SCORE_WEIGHTS['cointegration'] + SCORE_WEIGHTS['hurst'])

def test_half_life_outside_window_is_penalised():
    m = {**BASE, 'adf_pvalue': 0.01, 'hurst': 0.3}
    assert calculate_confidence_score(m, SCORE_WEIGHTS) > calculate_confidence_score({**m, 'half_life_days': 200}, SCORE_WEIGHTS)
