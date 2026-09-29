import numpy as np

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
    """
    if len(nominal_ratios) != len(dv01s):
        return np.nan
        
    residual = 0.0
    for n, dv01 in zip(nominal_ratios, dv01s):
        if not np.isnan(dv01):
            residual += n * dv01
            
    return residual
