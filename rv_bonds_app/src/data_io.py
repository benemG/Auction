import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from .config import MARKET_DATA_DIR, FITTED_CURVES_DIR, OPPORTUNITIES_DIR

def write_market_data(df: pd.DataFrame):
    """
    Écrit les données de marché validées en format Parquet, partitionné par émetteur et année.
    """
    if df.empty:
        return
        
    df = df.copy()
    df['year'] = df['date'].dt.year
    
    table = pa.Table.from_pandas(df)
    
    pq.write_to_dataset(
        table,
        root_path=str(MARKET_DATA_DIR),
        partition_cols=['issuer', 'year'],
        compression='zstd',
        existing_data_behavior='overwrite_or_ignore'
    )

def read_market_data(issuer: str, min_date: str = None) -> pd.DataFrame:
    """
    Lit les données de marché filtrées par émetteur via PyArrow Dataset.
    """
    dataset = pq.ParquetDataset(
        str(MARKET_DATA_DIR),
        filters=[('issuer', '=', issuer)]
    )
    
    df = dataset.read().to_pandas()
    
    if min_date:
        df = df[df['date'] >= pd.to_datetime(min_date)]
        
    # Nettoyer les colonnes de partition qui pourraient être rajoutées
    if 'year' in df.columns:
        df = df.drop(columns=['year'])
        
    df = df.drop_duplicates(subset=['date', 'isin'], keep='last')
        
    return df.sort_values(['date', 'isin']).reset_index(drop=True)

def write_fitted_curves(df: pd.DataFrame):
    """
    Sauvegarde les résultats du fit P-Spline.
    """
    if df.empty:
        return
        
    df = df.copy()
    df['year'] = df['date'].dt.year
    
    table = pa.Table.from_pandas(df)
    
    pq.write_to_dataset(
        table,
        root_path=str(FITTED_CURVES_DIR),
        partition_cols=['issuer', 'year'],
        compression='zstd'
    )

def read_fitted_curves(issuer: str, min_date: str = None) -> pd.DataFrame:
    """
    Lit les courbes fittées.
    """
    if not FITTED_CURVES_DIR.exists() or not any(FITTED_CURVES_DIR.iterdir()):
        return pd.DataFrame()
        
    dataset = pq.ParquetDataset(
        str(FITTED_CURVES_DIR),
        filters=[('issuer', '=', issuer)]
    )
    
    df = dataset.read().to_pandas()
    
    if min_date:
        df = df[df['date'] >= pd.to_datetime(min_date)]
        
    if 'year' in df.columns:
        df = df.drop(columns=['year'])
        
    df = df.drop_duplicates(subset=['date', 'isin'], keep='last')
        
    return df.sort_values(['date', 'isin']).reset_index(drop=True)

import shutil

def write_opportunities(df: pd.DataFrame, as_of_date: pd.Timestamp):
    """
    Sauvegarde le scan des opportunités, partitionné par date et émetteur.
    """
    if df.empty:
        return
        
    df = df.copy()
    date_str = as_of_date.strftime('%Y-%m-%d')
    df['date_str'] = date_str
    
    if 'issuer' in df.columns and len(df['issuer'].unique()) == 1:
        issuer = df['issuer'].iloc[0]
        part_dir = OPPORTUNITIES_DIR / f"date_str={date_str}" / f"issuer={issuer}"
        if part_dir.exists():
            shutil.rmtree(part_dir)
    
    table = pa.Table.from_pandas(df)
    
    pq.write_to_dataset(
        table,
        root_path=str(OPPORTUNITIES_DIR),
        partition_cols=['date_str', 'issuer'],
        compression='zstd'
    )

def read_opportunities(date_str: str, issuer: str = None) -> pd.DataFrame:
    """
    Lit les opportunités pour une date donnée.
    """
    filters = [('date_str', '=', date_str)]
    if issuer:
        if isinstance(issuer, list):
            filters.append(('issuer', 'in', issuer))
        else:
            filters.append(('issuer', '=', issuer))
        
    if not OPPORTUNITIES_DIR.exists() or not any(OPPORTUNITIES_DIR.iterdir()):
        return pd.DataFrame()
        
    try:
        dataset = pq.ParquetDataset(
            str(OPPORTUNITIES_DIR),
            filters=filters
        )
        df = dataset.read().to_pandas()
        return df
    except Exception:
        return pd.DataFrame()
