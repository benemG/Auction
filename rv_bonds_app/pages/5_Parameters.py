import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as cfg

st.set_page_config(page_title="Paramètres", page_icon="⚙️", layout="wide")
st.title("Paramètres et Configuration Système")

st.markdown("""
Cette page affiche les paramètres actuellement utilisés par les batchs de traitement (P-Spline, OU, Filtres).
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("P-Spline")
    st.write(f"- **Noeuds par défaut:** {cfg.PSPLINE_DEFAULT_KNOTS}")
    st.write(f"- **Degré:** {cfg.PSPLINE_DEFAULT_DEGREE}")
    st.write(f"- **Ordre de Pénalité:** {cfg.PSPLINE_DEFAULT_PENALTY_ORDER}")
    st.write(f"- **Lambda par défaut:** {cfg.PSPLINE_DEFAULT_LAMBDA}")

with col2:
    st.subheader("Modèle Ornstein-Uhlenbeck")
    st.write(f"- **Observations Minimales:** {cfg.MIN_CALIBRATION_OBSERVATIONS}")
    st.write(f"- **Pas Temporel (dt):** {cfg.OU_DEFAULT_DT:.6f} (1/{cfg.TRADING_DAYS_IN_YEAR} jours)")
    st.write(f"- **Historique Requis:** {cfg.MIN_HISTORICAL_DAYS} jours")

with col3:
    st.subheader("Pondération du Score (sur 100)")
    st.write(f"- **Cointégration:** {cfg.SCORE_WEIGHTS['cointegration']*100}%")
    st.write(f"- **Hurst:** {cfg.SCORE_WEIGHTS['hurst']*100}%")
    st.write(f"- **Half-Life:** {cfg.SCORE_WEIGHTS['half_life']*100}%")
    st.write(f"- **Z-Score:** {cfg.SCORE_WEIGHTS['zscore']*100}%")
    st.write(f"- **Qualité Fit:** {cfg.SCORE_WEIGHTS['fit_quality']*100}%")
    st.write(f"- **Validité OU:** {cfg.SCORE_WEIGHTS['ou_validity']*100}%")

st.markdown("---")
st.subheader("Repo : spread vs ESTER (bp) par émetteur")
st.caption("Coût de financement = ESTER + spread. Appliqué immédiatement dans le Scanner et le Trade Explorer "
           "à la date scannée, sans batch ni recalcul d'historique. 'default' sert aux émetteurs non renseignés.")
issuers = sorted(d.name.split('=')[1] for d in cfg.MARKET_DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('issuer='))
saved = cfg.get_repo_spreads()
df_repo = pd.DataFrame({'issuer': ['default'] + issuers})
df_repo['spread_bps'] = [saved.get(i, 0.0) for i in df_repo['issuer']]
edited = st.data_editor(df_repo, hide_index=True, disabled=['issuer'],
                        column_config={'spread_bps': st.column_config.NumberColumn("Spread (bp)", format="%.1f")})
if st.button("Enregistrer les spreads de repo"):
    cfg.set_repo_spreads(dict(zip(edited['issuer'], edited['spread_bps'])))
    st.success("Spreads enregistrés dans data/settings.json.")

st.markdown("---")
st.subheader("Chemins des Répertoires")
st.code(f"""
Base Dir: {cfg.BASE_DIR}
Raw Data: {cfg.get_raw_data_dir()}
Market Data Parquet: {cfg.MARKET_DATA_DIR}
Fitted Curves Parquet: {cfg.FITTED_CURVES_DIR}
Opportunities Parquet: {cfg.OPPORTUNITIES_DIR}
""")
