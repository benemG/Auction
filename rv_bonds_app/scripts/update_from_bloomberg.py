import sys
import os
from pathlib import Path
import pandas as pd
import datetime
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR, BASE_DIR
from src.bloomberg_client import fetch_bond_market_data

def update_from_bloomberg(end_date: str = None):
    print("Mise à jour optimisée des données depuis Bloomberg...")
    
    today = datetime.date.today()
    if end_date is None:
        end_date = today.strftime('%Y-%m-%d')
        
    ref_file = BASE_DIR.parent / "mapping_indiv_bond.csv"
    if not ref_file.exists():
        print(f"Erreur: Fichier référentiel {ref_file} introuvable.")
        return
        
    print("Lecture du référentiel...")
    df_ref = pd.read_csv(ref_file, sep=';', dtype=str)
    df_ref = df_ref.drop_duplicates(subset=['id_isin'], keep='last')
    
    # 1. Exclure les obligations arrivées à maturité
    df_ref['maturity_dt'] = pd.to_datetime(df_ref['maturity'], errors='coerce')
    df_active = df_ref[df_ref['maturity_dt'].dt.date >= today].copy()
    print(f"Obligations actives (non échues) dans le référentiel : {len(df_active)}")
    
    # Construire la cartographie isin -> sous-dossier émetteur existant
    isin_to_folder = {}
    for issuer_dir in RAW_DATA_DIR.iterdir():
        if issuer_dir.is_dir():
            for csv_file in issuer_dir.glob("*.csv"):
                isin_to_folder[csv_file.stem] = issuer_dir

    followed_isins = list(isin_to_folder.keys())
    if not followed_isins:
        print("Aucun fichier CSV existant. On télécharge tout le référentiel actif.")
        followed_isins = df_active['id_isin'].tolist()

    df_active = df_active[df_active['id_isin'].isin(followed_isins)]
    
    ticker_to_isin = {}
    isin_to_ticker = {}
    for _, row in df_active.iterrows():
        isin = row['id_isin']
        bbg_ticker = row['cusip'] if pd.notna(row.get('cusip')) else f"{isin} Corp"
        ticker_to_isin[bbg_ticker] = isin
        isin_to_ticker[isin] = bbg_ticker
        
    if not isin_to_ticker:
        print("Aucun ticker actif à mettre à jour.")
        return

    # 2. Déterminer la dernière date disponible pour chaque ISIN
    print("Analyse des dernières dates disponibles dans les fichiers CSV...")
    start_date_groups = defaultdict(list)
    
    default_start = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d') # Par défaut, historique de 1 an
    
    for isin, bbg_ticker in isin_to_ticker.items():
        if isin in isin_to_folder:
            csv_path = isin_to_folder[isin] / f"{isin}.csv"
            try:
                # Lire juste la première colonne (date) pour aller vite
                df_csv_dates = pd.read_csv(csv_path, sep=';', usecols=[0], names=['date'], header=None)
                # Ignorer la ligne d'en-tête éventuelle (si le premier champ est "date" ou vide)
                df_csv_dates['date'] = pd.to_datetime(df_csv_dates['date'], errors='coerce')
                max_date = df_csv_dates['date'].max()
                
                if pd.notna(max_date):
                    # On demande depuis la date max pour over-lapper et rattraper
                    s_date = max_date.strftime('%Y-%m-%d')
                else:
                    s_date = default_start
            except Exception:
                s_date = default_start
        else:
            s_date = default_start
            
        start_date_groups[s_date].append(bbg_ticker)

    # 3 & 4. Requêter en batch par groupe de start_date
    df_results = []
    chunk_size = 100
    
    for s_date, tickers in start_date_groups.items():
        print(f"\n--- Groupe start_date = {s_date} ({len(tickers)} tickers) ---")
        
        # S'il y a trop de tickers dans le groupe, on chunk
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            print(f"Fetch chunk {i//chunk_size + 1}/{(len(tickers)-1)//chunk_size + 1}...")
            try:
                df_chunk = fetch_bond_market_data(chunk, start_date=s_date, end_date=end_date)
                if not df_chunk.empty:
                    df_results.append(df_chunk)
            except Exception as e:
                print(f"Erreur lors du téléchargement BBG: {e}")

    if not df_results:
        print("Aucune nouvelle donnée récupérée.")
        return
        
    df_bbg = pd.concat(df_results, ignore_index=True)
    df_bbg['isin'] = df_bbg['bbg_ticker'].map(ticker_to_isin)
    
    # Sauvegarde des CSV
    print(f"\nÉcriture et fusion dans les CSV...")
    updated_files = 0
    for isin, df_isin in df_bbg.groupby('isin'):
        if isin in isin_to_folder:
            folder = isin_to_folder[isin]
        else:
            # Créer le dossier s'il n'existe pas
            ticker_str = df_ref[df_ref['id_isin'] == isin]['ticker'].iloc[0] if not df_ref[df_ref['id_isin'] == isin].empty else 'unknown'
            folder = RAW_DATA_DIR / str(ticker_str).lower()
            folder.mkdir(parents=True, exist_ok=True)
            isin_to_folder[isin] = folder
            
        csv_path = folder / f"{isin}.csv"
        
        df_export = df_isin[['date', 'price_clean', 'ytm', 'asset_swap_spread', 'z_spread']].copy()
        df_export = df_export.rename(columns={
            'price_clean': 'px_last',
            'ytm': 'yld_ytm_mid',
            'asset_swap_spread': 'asset_swap_spd_mid',
            'z_spread': 'z_sprd_mid'
        })
        
        df_export['date'] = pd.to_datetime(df_export['date']).dt.strftime('%Y-%m-%d')
        
        if csv_path.exists():
            try:
                df_existing = pd.read_csv(csv_path, sep=';', header=None, skiprows=1, names=['date', 'px_last', 'yld_ytm_mid', 'asset_swap_spd_mid', 'z_sprd_mid'])
            except Exception:
                df_existing = pd.DataFrame(columns=['date', 'px_last', 'yld_ytm_mid', 'asset_swap_spd_mid', 'z_sprd_mid'])
                
            df_combined = pd.concat([df_existing, df_export]).drop_duplicates(subset=['date'], keep='last')
            df_combined = df_combined.sort_values('date')
            df_combined.to_csv(csv_path, sep=';', index=False, header=['', 'px_last', 'yld_ytm_mid', 'asset_swap_spd_mid', 'z_sprd_mid'])
        else:
            df_export.to_csv(csv_path, sep=';', index=False, header=['', 'px_last', 'yld_ytm_mid', 'asset_swap_spd_mid', 'z_sprd_mid'])
            
        updated_files += 1
        
    print(f"{updated_files} fichiers CSV mis à jour avec succès.")
    
    print("\nLancement de l'ingestion vers Parquet depuis les CSV mis à jour...")
    from scripts.ingest_csv_to_parquet import run_ingestion
    run_ingestion()

if __name__ == "__main__":
    update_from_bloomberg()
