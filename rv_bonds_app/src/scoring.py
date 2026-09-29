import numpy as np

def calculate_confidence_score(metrics: dict, weights: dict) -> float:
    """
    Calcule un score de confiance composite entre 0 et 100.
    """
    score = 0.0
    
    # 1. Qualité OU (0 à 100)
    ou_score = 0
    if metrics.get('ou_calibration_status') == 'valid':
        ou_score = 100
        # Pénalité si half_life est irréaliste (ex: > 500 jours ou < 2 jours)
        hl = metrics.get('half_life_days', np.nan)
        if not np.isnan(hl):
            if hl > 500 or hl < 2:
                ou_score -= 50
    score += ou_score * weights.get('ou_validity', 0.15)
    
    # 2. Z-score (0 à 100, optimal entre 1.5 et 3.0)
    z_score = abs(metrics.get('zscore_12m', 0))
    z_subscore = 0
    if 1.0 <= z_score <= 4.0:
        z_subscore = 100 - abs(2.5 - z_score) * 20
        
    # Pénalité de momentum : si le momentum va à l'encontre de l'opportunité (falling knife)
    momentum_signal = metrics.get('momentum_signal', np.nan)
    neighborhood_momentum = metrics.get('neighborhood_momentum', np.nan)
    target_gap = metrics.get('target_gap_bps', 0) # Positif -> opportunité Long
    
    if not np.isnan(momentum_signal) and target_gap != 0:
        # Si target_gap est positif (Mean reversion = achat), un momentum négatif est dangereux
        expected_dir = np.sign(target_gap)
        actual_mom_dir = np.sign(momentum_signal)
        
        if expected_dir != actual_mom_dir:
            # Pénalité proportionnelle à la force du momentum (ex: Carver MAC peut aller de 0 à ~3 ou 4)
            penalty = min(30, abs(momentum_signal) * 10) # Max -30 points
            
            # Renforcement de la pénalité si le voisinage est aussi dans la mauvaise direction
            if not np.isnan(neighborhood_momentum) and np.sign(neighborhood_momentum) != expected_dir:
                penalty += min(20, abs(neighborhood_momentum) * 10)
                
            z_subscore -= min(50, penalty) # On limite la pénalité sur le z_subscore
            
    score += max(0, z_subscore) * weights.get('zscore', 0.20)
    
    # 3. Qualité Fit RMSE
    rmse = metrics.get('fit_rmse_bps', np.nan)
    rmse_subscore = 100
    if not np.isnan(rmse) and rmse > 0:
        # Pénalité si RMSE > 2 bps
        rmse_subscore = max(0, 100 - max(0, rmse - 2) * 20)
    score += rmse_subscore * weights.get('fit_quality', 0.15)
    
    # Autres métriques par défaut à 50 si non implémentées/manquantes
    score += 50 * weights.get('cointegration', 0.20)
    score += 50 * weights.get('hurst', 0.15)
    score += 50 * weights.get('half_life', 0.15)
    
    return min(100.0, max(0.0, score))
