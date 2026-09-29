import os
from pathlib import Path

# --- DIRECTORIES ---
BASE_DIR = Path(os.getenv("RV_APP_BASE_DIR", Path(__file__).resolve().parent.parent))
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR.parent / "bonds"
PARQUET_DIR = DATA_DIR / "parquet"
MARKET_DATA_DIR = PARQUET_DIR / "market_data"
FITTED_CURVES_DIR = PARQUET_DIR / "fitted_curves"
OPPORTUNITIES_DIR = PARQUET_DIR / "opportunities"
REFERENCE_DIR = PARQUET_DIR / "reference"

# --- CONVENTIONS & UNITS ---
# Tous les spreads (ASW, Z-spread, YTM residual) doivent être en points de base (bps)
BPS_MULTIPLIER = 10000.0

# Convention temporelle pour P-Spline et Mean Reversion (Actual/365.25 ou similaire)
DAYS_IN_YEAR = 365.25
TRADING_DAYS_IN_YEAR = 252

# --- FILTERING THRESHOLDS ---
MIN_HISTORICAL_DAYS = 252  # 1 an de données quotidiennes
MIN_CALIBRATION_OBSERVATIONS = 63  # ~3 mois pour le modèle OU

# --- P-SPLINE PARAMS ---
PSPLINE_DEFAULT_KNOTS = 20
PSPLINE_DEFAULT_DEGREE = 3
PSPLINE_DEFAULT_PENALTY_ORDER = 2
PSPLINE_DEFAULT_LAMBDA = 1.0

# --- OU MODEL PARAMS ---
OU_DEFAULT_DT = 1.0 / TRADING_DAYS_IN_YEAR

# --- SCORING WEIGHTS ---
SCORE_WEIGHTS = {
    'cointegration': 0.2,
    'hurst': 0.15,
    'half_life': 0.15,
    'zscore': 0.2,
    'fit_quality': 0.15,
    'ou_validity': 0.15,
}

# Crée les dossiers si non existants
for directory in [
    RAW_DATA_DIR, 
    MARKET_DATA_DIR, 
    FITTED_CURVES_DIR, 
    OPPORTUNITIES_DIR, 
    REFERENCE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# --- GRAPH-BASED OPPORTUNITY GENERATION PARAMS ---
MIN_Z_SCORE_EDGE = 1.0        # Minimum absolute Z-score to consider a pair as an edge in the graph
MAX_HALF_LIFE_EDGE = 60       # Maximum half-life in days to keep the edge
MIN_HALF_LIFE_EDGE = 5        # Minimum half-life in days (filtre aberrations)
MAX_HURST_EDGE = 0.40         # Maximum Hurst exponent (significativement mean-reverting)
MAX_FLY_WING_SPREAD_YEARS = 5.0 # Max difference in years between wing1 and wing2 for flies
