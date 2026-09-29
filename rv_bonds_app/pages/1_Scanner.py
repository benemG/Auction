import streamlit as st
import pandas as pd
import sys
import json
from pathlib import Path
import itertools

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_io import read_opportunities, OPPORTUNITIES_DIR
from src.config import MARKET_DATA_DIR

st.set_page_config(page_title="Scanner", page_icon="🔍", layout="wide")
st.title("Scanner d'Opportunités RV")

@st.cache_data(ttl=3600)
def load_available_dates():
    if not OPPORTUNITIES_DIR.exists():
        return []
    dates = set()
    for p in OPPORTUNITIES_DIR.rglob('*.parquet'):
        parts = p.parts
        for part in parts:
            if part.startswith('date_str='):
                dates.add(part.split('=')[1])
    return sorted(list(dates), reverse=True)

dates = load_available_dates()

if not dates:
    st.warning("Aucune donnée d'opportunité trouvée. Veuillez exécuter le batch `generate_opportunities.py`.")
    st.stop()

# Cache de la date
if 'selected_date' not in st.session_state or st.session_state['selected_date'] not in dates:
    st.session_state['selected_date'] = dates[0]

col1, col2 = st.columns(2)
with col1:
    selected_date = st.selectbox("Date d'analyse", dates, index=dates.index(st.session_state['selected_date']))
    if selected_date != st.session_state['selected_date']:
        st.session_state['selected_date'] = selected_date
        st.rerun()

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(date_str):
    return read_opportunities(date_str)

df = load_data(selected_date)

if df.empty:
    st.info("Aucune opportunité pour cette date.")
    st.stop()

from src.clustering import get_cluster_issuers

with col2:
    # On récupère tous les émetteurs uniques générés, mais on filtre pour l'affichage (pas de '_')
    raw_issuers = df['issuer'].unique().tolist()
    base_issuers = sorted([iss for iss in raw_issuers if '_' not in iss])
    
    # Options d'affichage
    issuers = ['Tous'] + base_issuers
    
    # Cache de l'émetteur
    if 'selected_issuer' not in st.session_state or st.session_state['selected_issuer'] not in issuers:
        st.session_state['selected_issuer'] = issuers[0]
        
    selected_issuer = st.selectbox("Émetteur", issuers, index=issuers.index(st.session_state['selected_issuer']))
    if selected_issuer != st.session_state['selected_issuer']:
        st.session_state['selected_issuer'] = selected_issuer
        st.rerun()

# Filtrage par cluster entier
df_filtered = df.copy()
if selected_issuer != 'Tous':
    allowed_issuers = get_cluster_issuers(selected_issuer)
    df_filtered = df_filtered[df_filtered['issuer'].isin(allowed_issuers)]

st.sidebar.header("Filtres")
trade_types = st.sidebar.multiselect("Type de Trade", df['trade_type'].unique(), default=df['trade_type'].unique())
min_score = st.sidebar.slider("Score de Confiance Minimum", 0, 100, 50)
only_valid_ou = st.sidebar.checkbox("Uniquement OU valide", value=False)

df_filtered = df_filtered[df_filtered['trade_type'].isin(trade_types)]
df_filtered = df_filtered[df_filtered['confidence_score'] >= min_score]
if only_valid_ou:
    df_filtered = df_filtered[df_filtered['ou_calibration_status'] == 'valid']

# Tri par défaut par score
df_filtered = df_filtered.sort_values('confidence_score', ascending=False)

from src.charts import plot_parallel_coordinates, plot_opportunities_pca

@st.cache_data(show_spinner=False)
def get_parallel_coords(df):
    return plot_parallel_coordinates(df)

@st.cache_data(show_spinner=False)
def get_pca(df):
    return plot_opportunities_pca(df)

st.write('### Exploration Visuelle')
if not df_filtered.empty:
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        fig_pc = get_parallel_coords(df_filtered)
        st.plotly_chart(fig_pc, use_container_width=True)
    with viz_col2:
        fig_pca = get_pca(df_filtered)
        st.plotly_chart(fig_pca, use_container_width=True)

st.write(f'### Resultats : {len(df_filtered)} opportunites')

# Configuration des colonnes pour un affichage compact
display_cols = [
    'trade_id', 'issuer', 'trade_type', 'recommended_direction', 'confidence_score', 
    'target_gap_bps', 'carry_roll_adjusted_gap', 'zscore_12m', 'zscore_3m', 
    'momentum_signal', 'neighborhood_momentum', 'half_life_days', 'adf_pvalue', 'ou_calibration_status'
]

# We might not have 'zscore_3m' or 'carry_roll_adjusted_gap' in older datasets, so we check
available_cols = [c for c in display_cols if c in df_filtered.columns]

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

gb = GridOptionsBuilder.from_dataframe(df_filtered[available_cols])
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
gb.configure_selection('single', use_checkbox=True)
gb.configure_column("confidence_score", type=["numericColumn"], precision=0)
gb.configure_column("target_gap_bps", type=["numericColumn"], precision=2)
if "carry_roll_adjusted_gap" in available_cols:
    gb.configure_column("carry_roll_adjusted_gap", type=["numericColumn"], precision=2)
gb.configure_column("zscore_12m", type=["numericColumn"], precision=2)
if "zscore_3m" in available_cols:
    gb.configure_column("zscore_3m", type=["numericColumn"], precision=2)
if "momentum_signal" in available_cols:
    gb.configure_column("momentum_signal", type=["numericColumn"], precision=2)
if "neighborhood_momentum" in available_cols:
    gb.configure_column("neighborhood_momentum", type=["numericColumn"], precision=2)
gb.configure_column("half_life_days", type=["numericColumn"], precision=0)
if "adf_pvalue" in available_cols:
    gb.configure_column("adf_pvalue", type=["numericColumn"], precision=3)

gridOptions = gb.build()

st.write("Sélectionnez une ligne pour l'analyser dans le Trade Explorer.")
grid_response = AgGrid(
    df_filtered[available_cols],
    gridOptions=gridOptions,
    data_return_mode=DataReturnMode.AS_INPUT,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    fit_columns_on_grid_load=False,
    theme='streamlit',
    enable_enterprise_modules=False,
    height=500,
    width='100%'
)

selected = grid_response.get('selected_rows')
if selected is not None:
    selected_trade_id = None
    if isinstance(selected, pd.DataFrame) and not selected.empty:
        selected_trade_id = selected.iloc[0]['trade_id']
    elif isinstance(selected, list) and len(selected) > 0:
        selected_trade_id = selected[0]['trade_id']
        
    if selected_trade_id:
        st.session_state['selected_trade_id'] = selected_trade_id
        st.switch_page("pages/2_Trade_Explorer.py")

