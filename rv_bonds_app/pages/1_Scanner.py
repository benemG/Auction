import streamlit as st
import pandas as pd
import sys
import json
from pathlib import Path
import itertools

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_io import read_opportunities, OPPORTUNITIES_DIR
from src.config import MARKET_DATA_DIR, MIN_Z_SCORE_ENTRY, get_repo_spreads
from src.opportunity_generation import apply_repo_spreads

METRIC_LABELS = {'asset_swap_spread': 'ASW', 'ytm': 'YTM'}

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
def _read(date_str):
    return read_opportunities(date_str)

def load_data(date_str):
    # Spread de repo appliqué hors cache, à l'affichage seulement : suit la page Paramètres sans batch
    return apply_repo_spreads(_read(date_str), get_repo_spreads())

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
metrics = st.sidebar.multiselect("Mesure", df['metric'].unique(), default=df['metric'].unique(), format_func=lambda m: METRIC_LABELS.get(m, m))
min_score = st.sidebar.slider("Score de Confiance Minimum", 0, 100, 50)
min_z = st.sidebar.slider("|Z-Score 12M| Minimum", 0.0, 4.0, float(MIN_Z_SCORE_ENTRY), 0.5)
only_valid_ou = st.sidebar.checkbox("Uniquement OU valide", value=False)
only_entry_zone = st.sidebar.checkbox("Au-delà du seuil d'entrée optimal (Bertram)", value=False)
max_per_bond = st.sidebar.slider("Trades max par titre", 1, 10, 2,
                                 help="Évite que le même titre cher/riche remplisse le tableau : sélection par gain net / risque décroissant")

df_filtered = df_filtered[df_filtered['trade_type'].isin(trade_types) & df_filtered['metric'].isin(metrics)]
df_filtered = df_filtered[(df_filtered['confidence_score'] >= min_score) & (df_filtered['zscore_12m'].abs() >= min_z)]
if only_valid_ou:
    df_filtered = df_filtered[df_filtered['ou_calibration_status'] == 'valid']
if only_entry_zone and 'in_entry_zone' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['in_entry_zone'].fillna(False).astype(bool)]

# Tri par défaut par gain net / risque
df_filtered = df_filtered.sort_values('net_edge_ratio', ascending=False)

# Redondance : un trade est gardé si aucune de ses jambes n'a déjà atteint le maximum
counts, keep = {}, []
for legs in df_filtered[['bond1_isin', 'bond2_isin', 'bond3_isin']].itertuples(index=False):
    legs = [l for l in legs if isinstance(l, str)]
    keep.append(all(counts.get(l, 0) < max_per_bond for l in legs))
    if keep[-1]:
        for l in legs:
            counts[l] = counts.get(l, 0) + 1
df_filtered = df_filtered[keep]

# --- Changements vs date précédente (top 20 par gain net, émetteurs communs aux deux dates) ---
i = dates.index(selected_date)
with st.expander("Changements vs date précédente", expanded=True):
    df_prev = load_data(dates[i + 1]) if i + 1 < len(dates) else df.iloc[0:0]
    common = set(df['issuer']) & set(df_prev['issuer']) if not df_prev.empty else set()
    if not common:
        st.info("Pas de date précédente comparable.")
    else:
        def top20(d):
            d = d[d['issuer'].isin(common) & (d['confidence_score'] >= min_score)]
            return d.drop_duplicates('trade_id').nlargest(20, 'net_edge_ratio').set_index('trade_id')
        cur_top, prev_top = top20(df), top20(df_prev)
        cols = ['issuer', 'recommended_direction', 'net_edge_ratio', 'confidence_score']
        st.caption(f"{dates[i]} vs {dates[i + 1]} - {len(common)} émetteurs comparés")
        c1, c2 = st.columns(2)
        c1.write("**Nouveaux dans le top 20**")
        c1.dataframe(cur_top.loc[cur_top.index.difference(prev_top.index), cols], use_container_width=True)
        c2.write("**Sortis du top 20**")
        c2.dataframe(prev_top.loc[prev_top.index.difference(cur_top.index), cols], use_container_width=True)

st.write(f'### Resultats : {len(df_filtered)} opportunites')

# Configuration des colonnes pour un affichage compact
display_cols = [
    'trade_id', 'metric', 'issuer', 'recommended_direction', 'net_edge_ratio', 'confidence_score', 'zscore_12m',
    'expected_gain_bps', 'risk_h_bps', 'net_carry_roll_3m_bps', 'half_life_days',
    'exit_target_bps', 'stop_loss_bps', 'exit_time_days', 'gain_per_day_bps', 'residual_dv01'
]

# We might not have 'zscore_3m' or 'carry_roll_adjusted_gap' in older datasets, so we check
available_cols = [c for c in display_cols if c in df_filtered.columns]

df_display = df_filtered[available_cols].reset_index(drop=True)

column_config = {
    "trade_id": st.column_config.TextColumn("Trade ID", help="Identifiant unique de l'opportunité"),
    "metric": st.column_config.TextColumn("Mesure"),
    "issuer": st.column_config.TextColumn("Émetteur"),
    "trade_type": st.column_config.TextColumn("Type"),
    "recommended_direction": st.column_config.TextColumn("Direction"),
    "net_edge_ratio": st.column_config.NumberColumn("Gain net / risque 3M", help="(gain attendu 3M + carry/roll - coûts) / risque 3M du spread de marché ; trié par défaut", format="%.2f"),
    "expected_gain_bps": st.column_config.NumberColumn("Gain attendu 3M (bps)", help="|écart| × (1 - e^(-θ·3M)), demi-vie corrigée du biais", format="%.2f"),
    "risk_h_bps": st.column_config.NumberColumn("Risque 3M (bps)", help="Écart-type des variations à 3M du spread de marché (vol EWM × ratio de variance)", format="%.2f"),
    "exit_target_bps": st.column_config.NumberColumn("Objectif (bps)", help="Mouvement visé jusqu'à la sortie optimale Leung-Li", format="%.2f"),
    "stop_loss_bps": st.column_config.NumberColumn("Stop (bps)", help="Mouvement adverse avant stop-loss", format="%.2f"),
    "exit_time_days": st.column_config.NumberColumn("Durée attendue (j)", help="Temps moyen avant objectif ou stop (premier passage OU)", format="%.0f"),
    "gain_per_day_bps": st.column_config.NumberColumn("Gain / jour (bps)", help="Gain attendu Leung-Li / durée attendue", format="%.3f"),
    "confidence_score": st.column_config.ProgressColumn(
        "Score RV",
        help="Score de confiance composite (0 à 100)",
        format="%d",
        min_value=0,
        max_value=100,
    ),
    "target_gap_bps": st.column_config.NumberColumn(
        "Target Gap (bps)",
        help="Écart entre spread de marché et cible P-Spline",
        format="%.2f",
    ),
    "carry_roll_adjusted_gap": st.column_config.NumberColumn("Gap Ajusté (bps)", format="%.2f"),
    "zscore_12m": st.column_config.NumberColumn("Z-Score 12M", format="%.2f"),
    "net_carry_roll_3m_bps": st.column_config.NumberColumn("Carry+Roll 3M (bps)", format="%.2f"),
    "residual_dv01": st.column_config.NumberColumn("DV01 résiduel", format="%.2f"),
    "zscore_3m": st.column_config.NumberColumn("Z-Score 3M", format="%.2f"),
    "momentum_signal": st.column_config.NumberColumn("Momentum", format="%.2f"),
    "neighborhood_momentum": st.column_config.NumberColumn("Neigh. Mom.", format="%.2f"),
    "half_life_days": st.column_config.NumberColumn("Demi-vie (j)", format="%d"),
    "adf_pvalue": st.column_config.NumberColumn("ADF p-val", format="%.3f"),
    "ou_calibration_status": st.column_config.TextColumn("Statut OU"),
}

st.markdown("💡 *Sélectionnez une ligne puis cliquez sur **Ouvrir dans Trade Explorer**.*")

event = st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
    height=480,
    on_select="rerun",
    selection_mode="single-row",
    column_config=column_config,
    key="scanner_results_grid"
)

# Extraction de la ligne sélectionnée
selected_trade_id = None
selected_rows = []
if event and hasattr(event, "selection") and event.selection:
    selected_rows = getattr(event.selection, "rows", [])
elif event and isinstance(event, dict) and "selection" in event:
    selected_rows = event["selection"].get("rows", [])

if selected_rows and len(selected_rows) > 0:
    row_idx = selected_rows[0]
    if 0 <= row_idx < len(df_display):
        selected_trade_id = df_display.iloc[row_idx]['trade_id']

# Action et redirection
col_info, col_btn = st.columns([3, 1])
with col_info:
    if selected_trade_id:
        trade_data = df_display[df_display['trade_id'] == selected_trade_id].iloc[0]
        st.success(
            f"🎯 **Opportunité sélectionnée :** `{selected_trade_id}` | "
            f"**{trade_data.get('issuer', '')}** ({trade_data.get('trade_type', '')}) | "
            f"Score: **{trade_data.get('confidence_score', 0):.0f}/100**"
        )
    else:
        st.info("Sélectionnez une opportunité dans le tableau ci-dessus.")

with col_btn:
    if st.button("🚀 Ouvrir dans Trade Explorer", type="primary", disabled=(selected_trade_id is None), use_container_width=True):
        st.session_state['selected_trade_id'] = selected_trade_id
        st.switch_page("pages/2_Trade_Explorer.py")

from src.charts import plot_parallel_coordinates, plot_opportunities_pca

@st.cache_data(show_spinner=False)
def get_parallel_coords(df):
    return plot_parallel_coordinates(df)

@st.cache_data(show_spinner=False)
def get_pca(df):
    return plot_opportunities_pca(df)

with st.expander('Exploration visuelle'):
  if not df_filtered.empty:
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        fig_pc = get_parallel_coords(df_filtered)
        st.plotly_chart(fig_pc, use_container_width=True)
    with viz_col2:
        fig_pca = get_pca(df_filtered)
        st.plotly_chart(fig_pca, use_container_width=True)
