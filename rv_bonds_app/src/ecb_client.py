import pandas as pd
import requests
import io
import time

def fetch_ecb_rate(flowRef: str, key: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Récupère une série temporelle depuis l'API REST de la BCE au format CSV.
    Retourne un DataFrame avec 'date' et 'rate'.
    """
    url = f"https://data-api.ecb.europa.eu/service/data/{flowRef}/{key}?format=csvdata"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            
            if 'TIME_PERIOD' not in df.columns or 'OBS_VALUE' not in df.columns:
                raise ValueError("Colonnes attendues manquantes dans la réponse de la BCE.")
                
            df_clean = df[['TIME_PERIOD', 'OBS_VALUE']].copy()
            df_clean.rename(columns={'TIME_PERIOD': 'date', 'OBS_VALUE': 'rate'}, inplace=True)
            df_clean['date'] = pd.to_datetime(df_clean['date']).dt.date
            df_clean.set_index('date', inplace=True)
            return df_clean
            
        except requests.exceptions.RequestException as e:
            print(f"Tentative {attempt + 1}/{max_retries} échouée pour {key}: {e}")
            if attempt == max_retries - 1:
                return pd.DataFrame(columns=['rate'])
            time.sleep(2)
            
    return pd.DataFrame(columns=['rate'])

def fetch_ester_and_dfr() -> pd.DataFrame:
    """
    Récupère l'€STER et le taux de facilité de dépôt (DFR),
    puis les combine dans un seul DataFrame.
    """
    print("Fetching ECB Deposit Facility Rate (DFR)...")
    df_dfr = fetch_ecb_rate(flowRef="FM", key="B.U2.EUR.4F.KR.DFR.LEV")
    df_dfr.rename(columns={'rate': 'dfr_rate'}, inplace=True)
        
    print("Fetching €STER...")
    df_ester = fetch_ecb_rate(flowRef="EST", key="B.EU000A2X2A25.WT")
    df_ester.rename(columns={'rate': 'ester_rate'}, inplace=True)
        
    # Fusionner (outer join) pour aligner les dates
    if df_dfr.empty and df_ester.empty:
        # Fallback de secours si l'API est HS
        return pd.DataFrame({'date': [pd.Timestamp.today().date()], 'dfr_rate': [3.50], 'ester_rate': [3.40]})
        
    df_combined = pd.merge(df_dfr, df_ester, left_index=True, right_index=True, how='outer')
    
    df_combined['dfr_rate'] = df_combined['dfr_rate'].ffill()
    df_combined['ester_rate'] = df_combined['ester_rate'].ffill()
    
    # Si l'ESTER est vide mais pas le DFR (ex: timeout), on approxime l'ESTER = DFR - 0.10
    if df_combined['ester_rate'].isna().all():
        df_combined['ester_rate'] = df_combined['dfr_rate'] - 0.10
        
    df_combined.reset_index(inplace=True)
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    
    return df_combined

if __name__ == "__main__":
    df = fetch_ester_and_dfr()
    print(df.tail())
