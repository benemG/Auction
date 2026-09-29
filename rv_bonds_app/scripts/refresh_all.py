import sys
from pathlib import Path

# Ajouter le dossier courant au path pour importer les autres scripts
sys.path.append(str(Path(__file__).resolve().parent))

from ingest_csv_to_parquet import run_ingestion
from update_ecb_rates import update_ecb_rates
from fit_daily_curves import fit_daily_curves
from generate_opportunities import generate_opportunities

def main():
    print("=== DÉBUT DU REFRESH GLOBAL ===")
    
    print("\n--- 1. INGESTION ---")
    run_ingestion()
    
    print("\n--- 2. MISE A JOUR DES TAUX BCE ---")
    update_ecb_rates()
    
    print("\n--- 3. FIT DES COURBES ---")
    fit_daily_curves()
    
    print("\n--- 4. GÉNÉRATION DES OPPORTUNITÉS ---")
    generate_opportunities()
    
    print("\n=== REFRESH GLOBAL TERMINÉ ===")

if __name__ == "__main__":
    main()
