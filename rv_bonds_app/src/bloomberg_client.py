import pandas as pd
from typing import List, Optional
import datetime

try:
    from xbbg import blp
except ImportError:
    blp = None

def fetch_bond_market_data(isins: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Récupère les données historiques de marché via Bloomberg pour une liste d'ISIN.
    
    Fields récupérés :
    - PX_LAST
    - YLD_YTM_MID
    - ASW_SPD_MID
    - Z_SPRD_MID
    """
    if blp is None:
        raise ImportError("Le module 'xbbg' n'est pas installé ou l'API Bloomberg n'est pas accessible.")
        
    if not end_date:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
        
    # Ajouter le suffixe ' Corp' ou ' Govt' requis par Bloomberg pour les ISINs
    # Le plus sûr pour les obligations souveraines européennes est souvent de chercher via l'ISIN + " Govt" ou " Corp"
    # Ici, nous allons utiliser le format ISIN ISIN Corp ou ISIN Govt.
    # Pour faire générique, on suppose que l'utilisateur ou la config fournira des tickers valides (ex: "FR001400H7V7 Govt")
    # Si isins contient juste des ISINs bruts, on rajoute " Corp" par défaut (ou " Govt" pour souverain).
    # Dans le référentiel mapping_indiv_bond, il y a la colonne `cusip` qui contient le ticker Bloomberg complet (ex: "AA212876 Corp")
    # Il vaudra mieux utiliser cette colonne `cusip` en entrée.
    
    tickers = isins 
    fields = ['PX_LAST', 'YLD_YTM_MID', 'ASW_SPD_MID', 'Z_SPRD_MID']
    
    print(f"Fetch Bloomberg data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Appel à l'API via xbbg
    df_bbg = blp.bdh(
        tickers=tickers,
        flds=fields,
        start_date=start_date,
        end_date=end_date
    )
    
    if hasattr(df_bbg, 'to_pandas'):  # xbbg 1.x peut renvoyer un autre backend
        df_bbg = df_bbg.to_pandas()
    if df_bbg.empty:
        return pd.DataFrame()

    fields_map = {'PX_LAST': 'price_clean', 'YLD_YTM_MID': 'ytm',
                  'ASW_SPD_MID': 'asset_swap_spread', 'Z_SPRD_MID': 'z_spread'}

    if isinstance(df_bbg.columns, pd.MultiIndex):
        # xbbg 0.7.x : format large, colonnes (ticker, field)
        df_formatted = df_bbg.stack(level=0, future_stack=True)
        df_formatted.index.names = ['date', 'bbg_ticker']
        df_formatted = df_formatted.reset_index()
        df_formatted.columns = [str(c).upper() if str(c).upper() in fields_map else c for c in df_formatted.columns]
    else:
        # xbbg 1.x : format long (ticker, date, field, value), valeurs en texte
        df_bbg.columns = [str(c).lower() for c in df_bbg.columns]
        df_bbg['value'] = pd.to_numeric(df_bbg['value'], errors='coerce')
        df_bbg['field'] = df_bbg['field'].str.upper()
        df_formatted = (df_bbg.pivot_table(index=['date', 'ticker'], columns='field', values='value', aggfunc='last')
                        .reset_index().rename_axis(columns=None).rename(columns={'ticker': 'bbg_ticker'}))

    df_formatted = df_formatted.rename(columns=fields_map)
    df_formatted['date'] = pd.to_datetime(df_formatted['date'])
    for col in fields_map.values():
        if col not in df_formatted.columns:
            df_formatted[col] = float('nan')
    return df_formatted[['date', 'bbg_ticker'] + list(fields_map.values())]
