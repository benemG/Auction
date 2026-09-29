import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_validation import validate_and_clean_market_data

def test_validation_deduplication_and_filtering():
    raw_data = {
        'Date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01'],
        'Issuer': ['OAT', 'OAT', 'OAT', 'OAT'],
        'ISIN': ['FR001', 'FR001', 'FR002', 'FR003'],
        'Bond_Name': ['OAT 1', 'OAT 1', 'OAT 2', 'OAT 3'],
        'Maturity': ['2025-01-01', '2025-01-01', '2022-01-01', '2026-01-01'],
        'Price_Clean': [100.5, 100.5, 99.0, -10.0], # Prix négatif pour FR003
        'YTM': [0.02, 0.02, 0.03, 0.04]
    }
    
    df = pd.DataFrame(raw_data)
    
    df_clean = validate_and_clean_market_data(df)
    
    # Doit enlever le duplicata FR001
    assert len(df_clean[df_clean['isin'] == 'FR001']) == 1
    
    # Doit enlever FR002 car maturité passée
    assert len(df_clean[df_clean['isin'] == 'FR002']) == 0
    
    # Doit enlever FR003 car prix invalide
    assert len(df_clean[df_clean['isin'] == 'FR003']) == 0
    
    assert len(df_clean) == 1
