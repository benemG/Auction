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
    
    if df_bbg.empty:
        return pd.DataFrame()
        
    # Le DataFrame renvoyé a un MultiIndex en colonnes (Ticker, Field)
    # On va le formater (stack) pour avoir les colonnes: date, ticker, px_last, etc.
    df_formatted = df_bbg.stack(level=0, future_stack=True).reset_index()
    # future_stack=True gère correctement le nommage de l'index dans pandas moderne.
    
    # Renommer les colonnes
    rename_map = {
        'index': 'date',
        'level_1': 'bbg_ticker',
        'PX_LAST': 'price_clean',
        'YLD_YTM_MID': 'ytm',
        'ASW_SPD_MID': 'asset_swap_spread',
        'Z_SPRD_MID': 'z_spread'
    }
    
    # Dans les anciennes versions de pandas, l'index de date peut s'appeler différemment
    if 'date' not in df_formatted.columns and df_formatted.columns[0] != 'index':
        rename_map[df_formatted.columns[0]] = 'date'
        
    df_formatted = df_formatted.rename(columns=rename_map)
    
    return df_formatted
