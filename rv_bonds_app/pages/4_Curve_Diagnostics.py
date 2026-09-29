import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_io import read_fitted_curves, FITTED_CURVES_DIR
from src.charts import plot_pspline_curve

st.set_page_config(page_title="Curve Diagnostics", page_icon="📉", layout="wide")
st.title("Diagnostics de Courbe (P-Spline)")

@st.cache_data(ttl=3600)
def get_issuers():
    if not FITTED_CURVES_DIR.exists():
        return []
    issuers = [d.name.split('=')[1] for d in FITTED_CURVES_DIR.iterdir() if d.is_dir() and d.name.startswith('issuer=')]
    return sorted(issuers)

issuers = get_issuers()
if not issuers:
    st.warning("Aucune courbe fittée disponible.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    issuer = st.selectbox("Émetteur", issuers)

@st.cache_data
def load_curves(issuer):
    return read_fitted_curves(issuer)

df_curves = load_curves(issuer)

if df_curves.empty:
    st.info("Aucune donnée pour cet émetteur.")
    st.stop()

dates = sorted(df_curves['date'].dt.strftime('%Y-%m-%d').unique().tolist(), reverse=True)

with col2:
    selected_date = st.selectbox("Date", dates)

with col3:
    metric = st.selectbox("Métrique", ['ytm', 'asset_swap_spread', 'z_spread'])

df_day = df_curves[df_curves['date'] == pd.to_datetime(selected_date)]

st.markdown("---")

col_chart, col_data = st.columns([2, 1])

with col_chart:
    st.plotly_chart(plot_pspline_curve(df_day, metric), use_container_width=True)

with col_data:
    st.write("### Statistiques du Fit")
    
    rmse_col = f"{metric}_rmse"
    mae_col = f"{metric}_mae"
    
    if rmse_col in df_day.columns and not df_day[rmse_col].isna().all():
        rmse = df_day[rmse_col].iloc[0]
        mae = df_day[mae_col].iloc[0]
        st.metric("RMSE", f"{rmse:.2f} bps")
        st.metric("MAE", f"{mae:.2f} bps")
    else:
        st.warning("Diagnostics non disponibles pour cette métrique.")
        
    st.write("### Obligations utilisées")
    display_cols = ['isin', 'bond_name', 'ttm_years', f"{metric}_market", f"{metric}_residual", 'is_extrapolated']
    
    # Garde seulement les colonnes existantes
    display_cols = [c for c in display_cols if c in df_day.columns]
    
    st.dataframe(df_day[display_cols].sort_values('ttm_years'), hide_index=True)
