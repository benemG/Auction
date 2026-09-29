import sys
import os
from pathlib import Path
import pandas as pd

# Ajout du dossier parent au PYTHON_PATH pour importer src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR, BASE_DIR
from src.data_validation import validate_and_clean_market_data
from src.data_io import write_market_data

def run_ingestion():
    print(f"Démarrage de l'ingestion depuis {RAW_DATA_DIR}")
    
    if not RAW_DATA_DIR.exists():
        print(f"Le dossier {RAW_DATA_DIR} n'existe pas. Veuillez le créer et y déposer vos CSV.")
        return
        
    ref_file = BASE_DIR.parent / "mapping_indiv_bond.csv"
    if ref_file.exists():
        print(f"Chargement du référentiel: {ref_file}")
        df_ref = pd.read_csv(ref_file, sep=';', dtype=str)
        # Gestion des duplicatas d'ISIN dans le référentiel au cas où
        df_ref = df_ref.drop_duplicates(subset=['id_isin'], keep='last')
        mapping = df_ref.set_index('id_isin')
    else:
        print("ATTENTION: Fichier référentiel mapping_indiv_bond.csv non trouvé.")
        mapping = None

    issuers_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    
    if not issuers_dirs:
        print(f"Aucun sous-dossier d'émetteur trouvé dans {RAW_DATA_DIR}.")
        return

    total_rows = 0
    for issuer_dir in issuers_dirs:
        issuer_name = issuer_dir.name
        print(f"\nTraitement de l'émetteur : {issuer_name}")
        
        csv_files = list(issuer_dir.glob("*.csv"))
        if not csv_files:
            print(f"Aucun CSV trouvé dans {issuer_dir}")
            continue
            
        dfs = []
        for csv_file in csv_files:
            try:
                # print(f"  Lecture de {csv_file.name}...")
                df_raw = pd.read_csv(csv_file, sep=';')
                
                # S'assurer que le fichier n'est pas vide
                if df_raw.empty:
                    continue
                
                # Renommage des colonnes
                cols = list(df_raw.columns)
                cols[0] = 'date'
                df_raw.columns = cols
                
                df_raw = df_raw.rename(columns={
                    'px_last': 'price_clean',
                    'yld_ytm_mid': 'ytm',
                    'asset_swap_spd_mid': 'asset_swap_spread',
                    'z_sprd_mid': 'z_spread'
                })
                
                isin = csv_file.stem
                df_raw['isin'] = isin
                df_raw['issuer'] = issuer_name
                
                if mapping is not None and isin in mapping.index:
                    df_raw['bond_name'] = mapping.loc[isin, 'security_name']
                    df_raw['maturity'] = mapping.loc[isin, 'maturity']
                    df_raw['inflation_linked_indicator'] = mapping.loc[isin, 'inflation_linked_indicator']
                else:
                    df_raw['bond_name'] = isin
                    df_raw['maturity'] = pd.NaT
                    df_raw['inflation_linked_indicator'] = 'N'

                dfs.append(df_raw)
            except Exception as e:
                print(f"  Erreur lors de la lecture de {csv_file.name}: {e}")
                
        if not dfs:
            continue
            
        df_concat = pd.concat(dfs, ignore_index=True)
        print(f"  Validation de {len(df_concat)} lignes consolidées pour {issuer_name}...")
        
        try:
            df_valid = validate_and_clean_market_data(df_concat)
            write_market_data(df_valid)
            rows = len(df_valid)
            total_rows += rows
            print(f"  Succès : {rows} lignes valides écrites en Parquet.")
        except Exception as e:
            print(f"  Erreur lors de la validation/écriture : {e}")
            
    print(f"\nIngestion terminée. Total de lignes ingérées : {total_rows}")

if __name__ == "__main__":
    run_ingestion()
