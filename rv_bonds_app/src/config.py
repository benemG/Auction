import json
import os
from pathlib import Path

# --- DIRECTORIES ---
BASE_DIR = Path(os.getenv("RV_APP_BASE_DIR", Path(__file__).resolve().parent.parent))
DATA_DIR = BASE_DIR / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"  # réglages propres à la machine (non versionné)
DEFAULT_RAW_DATA_DIR = BASE_DIR.parent / "bonds"

def _read_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

def _write_setting(key: str, value) -> None:
    """Met à jour une clé de settings.json sans écraser les autres."""
    settings = _read_settings()
    settings[key] = value
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings), encoding="utf-8")

def get_raw_data_dir() -> Path:
    """Dossier des CSV d'origine : réglage utilisateur si présent, sinon dossier par défaut."""
    path = _read_settings().get("raw_data_dir")
    return Path(path) if path else DEFAULT_RAW_DATA_DIR

def set_raw_data_dir(path) -> None:
    """Enregistre le dossier des CSV d'origine ; None revient au dossier par défaut."""
    _write_setting("raw_data_dir", str(path) if path else None)

def get_repo_spreads() -> dict:
    """Spread de repo vs ESTER en bp par émetteur (clé 'default' pour les autres) ; appliqué à l'affichage du Scanner."""
    return {k: float(v) for k, v in _read_settings().get("repo_spread_bps", {}).items()}

def set_repo_spreads(spreads: dict) -> None:
    _write_setting("repo_spread_bps", {k: float(v) for k, v in spreads.items() if v is not None and v == v})

RAW_DATA_DIR = get_raw_data_dir()
PARQUET_DIR = DATA_DIR / "parquet"
MARKET_DATA_DIR = PARQUET_DIR / "market_data"
FITTED_CURVES_DIR = PARQUET_DIR / "fitted_curves"
OPPORTUNITIES_DIR = PARQUET_DIR / "opportunities"
REFERENCE_DIR = PARQUET_DIR / "reference"

# --- CONVENTIONS & UNITS ---
# Tous les spreads (ASW, Z-spread, YTM residual) doivent être en points de base (bps)
BPS_MULTIPLIER = 10000.0

# Mesures scannées et facteur de conversion vers les bps (le YTM est stocké en %, l'ASW en bps)
METRIC_TO_BPS = {'asset_swap_spread': 1.0, 'ytm': 100.0}

# Convention temporelle pour P-Spline et Mean Reversion (Actual/365.25 ou similaire)
DAYS_IN_YEAR = 365.25
TRADING_DAYS_IN_YEAR = 252

# Types de coupon exclus du périmètre (obligations à taux variable)
FLOATING_COUPON_TYPES = ('FLOATING', 'VARIABLE')

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
# Résultats du banc d'IC (scripts/universe_comparison.py, 2022 et 2025) : seules les paires séparées à la
# 1re division de l'arbre sont faibles ; demi-vie et Hurst n'améliorent pas le signal ; |z| >= 2 le renforce.
MICRO_RV_N_CLUSTERS = 4       # Coupe de l'arbre en N groupes (exclut la 1re division) ; None = coupe à distance 1.0
EDGE_REQUIRE_MR_STATS = False # True : une arête exige aussi demi-vie dans [MIN, MAX] et Hurst < MAX (ancien comportement)
MIN_Z_SCORE_ENTRY = 2.0       # |z| minimum proposé par défaut dans le Scanner

# --- FILTRE DE COINTÉGRATION DE L'UNIVERS (Engle-Granger, dans les deux sens) ---
COINTEGRATION_FILTER_ENABLED = False  # True : n'évalue que les spreads 2 jambes cointégrés
COINTEGRATION_MAX_PVALUE = 0.05       # p-value maximale, exigée dans les deux sens
COINTEGRATION_WINDOWS_DAYS = (252, 126)  # Le test doit être passé sur chacune de ces fenêtres (persistance)
COINTEGRATION_FDR_ALPHA = 0.05        # Benjamini-Hochberg sur les paires testées ; None = seuil brut MAX_PVALUE
COINTEGRATION_TEST_ON_SPREAD = True   # Exige aussi un ADF sur le spread A-B réellement tradé (bêta = 1)
