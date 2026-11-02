import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as cfg

st.set_page_config(page_title="Source Data", page_icon="📁", layout="wide")
st.title("Dossier des données de marché d'origine (CSV)")

st.markdown("""
Dossier contenant un sous-dossier par émetteur, avec un fichier CSV par ISIN.
Il est lu par l'ingestion (CSV vers Parquet) et mis à jour par le script Bloomberg.
""")

current = cfg.get_raw_data_dir()
st.write(f"**Dossier actuel :** `{current}`" + (" (par défaut)" if current == cfg.DEFAULT_RAW_DATA_DIR else ""))

new_path = st.text_input("Nouveau dossier", value=str(current))
folder = Path(new_path.strip()).expanduser()

if not folder.is_dir():
    st.error("Ce dossier n'existe pas.")
else:
    issuers = [d for d in folder.iterdir() if d.is_dir() and any(d.glob("*.csv"))]
    n_csv = sum(len(list(d.glob("*.csv"))) for d in issuers)
    if issuers:
        st.success(f"{len(issuers)} émetteurs et {n_csv} fichiers CSV trouvés.")
    else:
        st.warning("Aucun sous-dossier contenant des CSV : l'ingestion n'aurait rien à lire.")

col1, col2 = st.columns(2)
if col1.button("Enregistrer", type="primary", disabled=not folder.is_dir() or folder == current):
    cfg.set_raw_data_dir(folder)
    st.rerun()
if col2.button("Revenir au dossier par défaut", disabled=current == cfg.DEFAULT_RAW_DATA_DIR):
    cfg.set_raw_data_dir(None)
    st.rerun()

st.info("Le changement s'applique aux prochains lancements des scripts. Relancez ensuite l'ingestion "
        "(`scripts/ingest_csv_to_parquet.py`) pour recharger les données.")
