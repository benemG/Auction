import sys
from pathlib import Path
import os
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BASE_DIR
from src.ecb_client import fetch_ester_and_dfr

def update_ecb_rates():
    print("Mise à jour des taux BCE (DFR et €STER)...")
    
    # Récupérer les données
    df_ecb = fetch_ester_and_dfr()
    
    if df_ecb.empty:
        print("Erreur: Impossible de récupérer les données BCE.")
        return
        
    # S'assurer que le dossier data existe
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Sauvegarder en parquet
    parquet_path = data_dir / "ecb_rates.parquet"
    df_ecb.to_parquet(parquet_path, index=False)
    
    print(f"Taux BCE sauvegardés avec succès dans {parquet_path}.")
    print(df_ecb.tail())

if __name__ == "__main__":
    update_ecb_rates()
