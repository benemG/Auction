import pandas as pd
import numpy as np

def validate_and_clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide, nettoie et normalise un DataFrame brut contenant les données de marché.
    """
    df = df.copy()
    
    # 1. Standardisation des noms de colonnes
    df.columns = df.columns.str.lower().str.strip()
    
    # 2. Vérification des colonnes obligatoires
    required_cols = ['date', 'issuer', 'isin', 'bond_name', 'maturity', 'price_clean', 'ytm']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans les données brutes : {missing}")

    # 3. Typage
    df['date'] = pd.to_datetime(df['date'])
    df['maturity'] = pd.to_datetime(df['maturity'])
    df['issuer'] = df['issuer'].astype(str)
    df['isin'] = df['isin'].astype(str)
    df['bond_name'] = df['bond_name'].astype(str)
    
    numeric_cols = ['price_clean', 'ytm', 'asset_swap_spread', 'z_spread', 
                    'modified_duration', 'dv01', 'convexity', 'outstanding_amount', 'volume']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Créer la colonne avec des NaN si elle est absente mais attendue dans le schéma
            df[col] = np.nan
            
    # 4. Conversion des pourcentages (règle arbitraire de tolérance pour YTM, à adapter)
    # Si le YTM moyen > 50, on assume qu'il est en basis points. 
    # S'il est entre 0.0 et 1.0, il est en décimal. S'il est entre 1 et 50, en %.
    # L'objectif est d'uniformiser le YTM en décimal (ex: 0.03 pour 3%) et les spreads en bps.
    
    if 'asset_swap_spread' in df.columns:
        # On suppose les spreads déjà en bps, mais on applique un correctif s'ils sont décimaux (très petits)
        mask = df['asset_swap_spread'].abs() < 0.5
        if mask.mean() > 0.9:  # 90% des observations sont très petites -> décimal
            df.loc[mask, 'asset_swap_spread'] *= 10000.0

    if 'z_spread' in df.columns:
        mask = df['z_spread'].abs() < 0.5
        if mask.mean() > 0.9:
            df.loc[mask, 'z_spread'] *= 10000.0

    # 5. Déduplication
    df = df.drop_duplicates(subset=['date', 'isin'])
    
    # 6. Filtres de qualité basiques
    # Retirer les observations avec YTM manquant ou prix invalide
    df = df.dropna(subset=['price_clean', 'ytm'])
    df = df[df['price_clean'] > 0]
    
    # Maturité stricte (exclure échu)
    df = df[df['maturity'] > df['date']]
    
    # 7. Tri
    df = df.sort_values(['date', 'isin']).reset_index(drop=True)
    
    return df
