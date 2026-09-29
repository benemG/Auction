import streamlit as st
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
st.subheader("Chemins des Répertoires")
st.code(f"""
Base Dir: {cfg.BASE_DIR}
Raw Data: {cfg.RAW_DATA_DIR}
Market Data Parquet: {cfg.MARKET_DATA_DIR}
Fitted Curves Parquet: {cfg.FITTED_CURVES_DIR}
Opportunities Parquet: {cfg.OPPORTUNITIES_DIR}
""")
