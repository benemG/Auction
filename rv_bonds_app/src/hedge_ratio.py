import re
import numpy as np

# Coupon dans le nom Bloomberg : "BTPS 2 1/2 11/15/25", "BTPS 2.85 02/01/31", "AGFRNC 0 1/8 04/30/22"
_COUPON_RE = re.compile(r'^\S+\s+(\d+(?:\.\d+)?)(?:\s+(\d+)/(\d+))?\s+\d{1,2}/\d{1,2}/\d{2,4}$')

def parse_coupon(bond_name) -> float:
    """Coupon annuel en % lu dans le nom Bloomberg ; NaN si non lisible (ex. 'Float')."""
    m = _COUPON_RE.match(str(bond_name).strip())
    if not m:
        return np.nan
    whole, num, den = m.groups()
    return float(whole) + (float(num) / float(den) if num else 0.0)

def approx_modified_duration(ytm_pct: float, cpn_pct: float, ttm: float) -> float:
    """
    Duration modifiée d'une obligation in fine à coupon annuel (forme fermée de Macaulay / (1+y)).
    shortcut: coupon annuel pour tous (BTP semestriels), T fractionnaire et coupon couru ignorés ;
    erreur ~1 %, à raffiner si les couvertures inter-courbes BTP le demandent.
    """
    if np.isnan(ytm_pct) or np.isnan(cpn_pct) or np.isnan(ttm) or ttm <= 0:
        return np.nan
    y, c, T = ytm_pct / 100.0, cpn_pct / 100.0, ttm
    if abs(y) < 1e-6:
        y = 1e-6
    growth = (1 + y) ** T
    d_mac = (1 + y) / y - (1 + y + T * (c - y)) / (c * (growth - 1) + y)
    return d_mac / (1 + y)

def compute_spread_hedge_ratio(dv01_a: float, dv01_b: float) -> tuple:
    """
    Calcule le ratio de nominal pour un spread A - B neutre en DV01.
    N_B = -N_A * (DV01_A / DV01_B)
    """
    if dv01_b == 0 or np.isnan(dv01_b) or np.isnan(dv01_a):
        return 1.0, -1.0
        
    n_b = - (dv01_a / dv01_b)
    return 1.0, n_b

def compute_fly_hedge_ratio(dv01_a: float, dv01_b: float, dv01_c: float, ttm_a: float, ttm_b: float, ttm_c: float) -> tuple:
    """
    Calcule les ratios de nominal pour un fly A - 2B + C neutre en DV01 et "duration neutre" 
    (pondération des ailes basée sur les maturités, puis neutralisation de B).
    """
    # 1. Poids relatifs des ailes basés sur les maturités (distance au belly)
    if (ttm_c - ttm_a) == 0:
        weight_a = 0.5
        weight_c = 0.5
    else:
        weight_a = (ttm_c - ttm_b) / (ttm_c - ttm_a)
        weight_c = (ttm_b - ttm_a) / (ttm_c - ttm_a)
        
    # 2. Conversion en exposition DV01 des ailes
    dv01_wings = weight_a * dv01_a + weight_c * dv01_c
    
    if dv01_b == 0 or np.isnan(dv01_b) or np.isnan(dv01_wings):
        return weight_a, -1.0, weight_c
        
    # 3. Ratio pour le belly
    n_b = - (dv01_wings / dv01_b)
    
    return weight_a, n_b, weight_c

def compute_residual_dv01(nominal_ratios: list, dv01s: list) -> float:
    """
    Exposition DV01 résiduelle pour 1M$ de nominal sur la jambe principale.
    NaN dès qu'un DV01 manque : une jambe ignorée afficherait un faux 0.
    """
    if len(nominal_ratios) != len(dv01s) or any(np.isnan(d) for d in dv01s):
        return np.nan
    return float(sum(n * dv01 for n, dv01 in zip(nominal_ratios, dv01s)))
