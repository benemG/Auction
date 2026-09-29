import numpy as np
import pandas as pd

def compute_ewmac(time_series: pd.Series, fast_spans=[8, 16, 32], slow_spans=[32, 64, 128]) -> float:
    """
    Calcule le signal composite EWMAC (Exponentially Weighted Moving Average Crossover)
    selon l'approche de Rob Carver.
    
    Retourne un signal où > 0 = tendance haussière, < 0 = tendance baissière.
    """
    if len(time_series) < max(slow_spans) / 2: # Besoin d'un minimum de points
        return np.nan
        
    # Volatilité quotidienne (standard deviation des différences)
    diff_ts = time_series.diff().dropna()
    if len(diff_ts) < 10:
        return np.nan
        
    vol = diff_ts.std()
    
    if vol == 0 or np.isnan(vol):
        return 0.0
        
    composite_signal = 0.0
    valid_pairs = 0
    
    for fast, slow in zip(fast_spans, slow_spans):
        # Calcul EWMA
        ewma_fast = time_series.ewm(span=fast, adjust=False).mean()
        ewma_slow = time_series.ewm(span=slow, adjust=False).mean()
        
        # Crossover
        crossover = ewma_fast.iloc[-1] - ewma_slow.iloc[-1]
        
        # Normalisation par la volatilité (rend le signal comparable entre les instruments)
        raw_signal = crossover / vol
        
        composite_signal += raw_signal
        valid_pairs += 1
        
    if valid_pairs == 0:
        return np.nan
        
    return composite_signal / valid_pairs

def compute_ewmac_series(time_series: pd.Series, fast_spans=[8, 16, 32], slow_spans=[32, 64, 128]) -> pd.Series:
    """
    Retourne la série temporelle complète du signal composite EWMAC.
    """
    diff_ts = time_series.diff()
    vol = diff_ts.rolling(window=60, min_periods=10).std()
    
    # Remplacer les volatilités nulles pour éviter les divisions par zéro
    vol = vol.replace(0, np.nan)
    
    composite_signal = pd.Series(0.0, index=time_series.index)
    valid_pairs = 0
    
    for fast, slow in zip(fast_spans, slow_spans):
        ewma_fast = time_series.ewm(span=fast, adjust=False).mean()
        ewma_slow = time_series.ewm(span=slow, adjust=False).mean()
        crossover = ewma_fast - ewma_slow
        raw_signal = crossover / vol
        composite_signal += raw_signal
        valid_pairs += 1
        
    if valid_pairs == 0:
        return pd.Series(np.nan, index=time_series.index)
        
    return composite_signal / valid_pairs
