import streamlit as st
import pandas as pd
import numpy as np
import ast
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_io import read_opportunities, read_market_data, read_fitted_curves, OPPORTUNITIES_DIR
from src.charts import plot_combined_trade_analysis, plot_rv_network, plot_backtest_analysis
from src.trade_definitions import build_rv_graph
from src.exports import generate_markdown_report, convert_df_to_csv

st.set_page_config(page_title="Trade Explorer", page_icon="🕵️", layout="wide")
st.title("Trade Explorer")

@st.cache_data(ttl=3600)
def get_dates():
    if not OPPORTUNITIES_DIR.exists():
        return []
    dates = set()
    for p in OPPORTUNITIES_DIR.rglob('*.parquet'):
        for part in p.parts:
            if part.startswith('date_str='):
                dates.add(part.split('=')[1])
    return sorted(list(dates), reverse=True)

dates = get_dates()
if not dates:
    st.warning("Aucune donnée disponible.")
    st.stop()

# Cache de la date partagé avec le Scanner
if 'selected_date' not in st.session_state or st.session_state['selected_date'] not in dates:
    st.session_state['selected_date'] = dates[0]

col1, col2 = st.columns(2)
with col1:
    selected_date = st.selectbox("Date d'analyse", dates, index=dates.index(st.session_state['selected_date']))
    if selected_date != st.session_state['selected_date']:
        st.session_state['selected_date'] = selected_date
        st.rerun()

@st.cache_data(ttl=3600, show_spinner=False)
def get_opps(date_str):
    return read_opportunities(date_str)

df_opps = get_opps(selected_date)

if df_opps.empty:
    st.info("Pas d'opportunités à cette date.")
    st.stop()

with col2:
    trade_ids = df_opps['trade_id'].unique().tolist()
    
    # Pré-sélection depuis le Scanner via session_state
    default_idx = 0
    if 'selected_trade_id' in st.session_state:
        saved_trade = st.session_state['selected_trade_id']
        if saved_trade in trade_ids:
            default_idx = trade_ids.index(saved_trade)
            
    trade_id = st.selectbox("Sélectionner un Trade", trade_ids, index=default_idx)
    
    if trade_id != st.session_state.get('selected_trade_id'):
        st.session_state['selected_trade_id'] = trade_id

trade = df_opps[df_opps['trade_id'] == trade_id].iloc[0]

st.markdown("---")
# 1. Fiche de décision
st.subheader("Fiche de Décision")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recommandation", str(trade.get('recommended_direction', 'N/A')).replace('_', ' ').title())
m2.metric("Spread Actuel", f"{trade.get('spread_market_bps', 0):.1f} bps")
m3.metric("Cible P-Spline", f"{trade.get('spread_target_bps', 0):.1f} bps", f"{trade.get('target_gap_bps', 0):.1f} bps")
m4.metric("Score RV", f"{trade.get('confidence_score', 0):.0f} / 100")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Z-Score (12m)", f"{trade.get('zscore_12m', 0):.2f}")
m6.metric("Momentum (EWMAC)", f"{trade.get('momentum_signal', np.nan):.2f}")
m7.metric("Neighborhood Mom.", f"{trade.get('neighborhood_momentum', np.nan):.2f}")
m8.metric("Cible OU (Mu)", f"{trade.get('ou_long_run_mean_bps', 0):.1f} bps")

m9, m10, m11, m12, m13 = st.columns(5)
m9.metric("Statut OU", str(trade.get('ou_calibration_status', 'N/A')))
m10.metric("Demi-vie", f"{trade.get('half_life_days', 0):.0f} j")
m11.metric("Exp. First Hitting", f"{trade.get('first_hitting_time_days', 0):.0f} j")
m12.metric("DV01 Résiduelle", f"{trade.get('residual_dv01', 0):.4f}")
m13.metric("Cointégration (p-val)", f"{trade.get('adf_pvalue', np.nan):.3f}")

# Exporter
st.download_button(
    label="Exporter la fiche (Markdown)",
    data=generate_markdown_report(trade),
    file_name=f"{trade_id}_{selected_date}.md",
    mime="text/markdown"
)

st.markdown("---")
st.subheader("Visualisations")

# Reconstruction de l'historique
@st.cache_data(ttl=3600, show_spinner=False)
def get_trade_history(issuer, isins, weights, as_of_date, metric):
    df_m = read_market_data(issuer)
    df_f = read_fitted_curves(issuer)
    
    df_m = df_m[df_m['date'] <= pd.to_datetime(as_of_date)]
    df_f = df_f[df_f['date'] <= pd.to_datetime(as_of_date)]
    
    dates = df_m['date'].unique()
    history = []
    
    for dt in dates:
        d_m = df_m[df_m['date'] == dt].set_index('isin')
        d_f = df_f[df_f['date'] == dt].set_index('isin')
        
        try:
            m_val = sum(w * d_m.loc[isin, metric] for w, isin in zip(weights, isins))
            f_val = sum(w * d_f.loc[isin, f"{metric}_fitted"] for w, isin in zip(weights, isins))
            history.append({'date': dt, 'spread_market': float(m_val), 'spread_fitted': float(f_val)})
        except Exception:
            pass
            
    df_res = pd.DataFrame(history)
    if df_res.empty:
        return df_res
        
    # --- Calcul du Neighborhood Momentum Historique ---
    try:
        pivot_m = df_m.pivot(index='date', columns='isin', values=metric)
        from src.momentum import compute_ewmac_series
        all_neighbor_mom = []
        trade_isins = set(isins)
        for b in isins:
            for x in pivot_m.columns:
                if x not in trade_isins:
                    ts_spread = pivot_m[b] - pivot_m[x]
                    if ts_spread.count() > 60:
                        mom = compute_ewmac_series(ts_spread.dropna())
                        all_neighbor_mom.append(mom)
        if all_neighbor_mom:
            nh_mom_df = pd.concat(all_neighbor_mom, axis=1)
            nh_mom_series = nh_mom_df.mean(axis=1)
            df_res = df_res.set_index('date')
            df_res['neighborhood_momentum'] = nh_mom_series
            df_res = df_res.reset_index()
        else:
            df_res['neighborhood_momentum'] = np.nan
    except Exception as e:
        print("Erreur neighborhood mom:", e)
        df_res['neighborhood_momentum'] = np.nan
        
    return df_res

@st.cache_data(show_spinner=False)
def get_trade_analysis_fig(df_hist, market_val, ou_mu, ou_theta, ou_sigma):
    return plot_combined_trade_analysis(df_hist, 'spread_market', 'spread_fitted', market_val, ou_mu, ou_theta, ou_sigma)

@st.cache_data(show_spinner=False)
def get_network_fig(df_spreads):
    g = build_rv_graph(df_spreads)
    if g.number_of_nodes() > 0:
        return plot_rv_network(g)
    return None

# Récupération des infos du trade
isins = [trade.get(f'bond{i}_isin') for i in range(1, 5) if trade.get(f'bond{i}_isin') is not None]
weights = ast.literal_eval(trade['leg_weights']) if isinstance(trade['leg_weights'], str) else trade['leg_weights']

with st.spinner("Reconstruction de l'historique..."):
    df_hist = get_trade_history(trade['issuer'], isins, weights, selected_date, trade['metric'])

if not df_hist.empty:
    fig_trade = get_trade_analysis_fig(
        df_hist,
        trade.get('spread_market_bps', 0),
        trade.get('ou_long_run_mean_bps', np.nan),
        trade.get('ou_theta', np.nan),
        trade.get('ou_sigma', np.nan)
    )
    st.plotly_chart(fig_trade, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Simulation Backtest (Filtre Roulant : Z-Score + ADF + Hurst)")
    with st.spinner("Génération du backtest..."):
        try:
            import ast
            mom_str = trade.get('momentum_series', '[]')
            try:
                if pd.notna(mom_str):
                    # Remplacement de 'nan' par 'None' pour que literal_eval le parse correctement
                    mom_str = str(mom_str).replace('nan', 'None')
                    mom_series = ast.literal_eval(mom_str)
                else:
                    mom_series = []
            except Exception as e:
                st.warning(f"Warning parsing momentum: {e}")
                mom_series = []
                
            fig_backtest = plot_backtest_analysis(df_hist, 'spread_market', mom_series)
            st.plotly_chart(fig_backtest, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de générer le backtest : {e}")
            
    st.download_button(
        label="Télécharger l'historique (CSV)",
        data=convert_df_to_csv(df_hist),
        file_name=f"history_{trade_id}.csv",
        mime="text/csv"
    )
else:
    st.warning("Impossible de reconstruire l'historique pour ce trade.")


from src.clustering import get_cluster_issuers

st.markdown('---')
st.subheader('Voisinage RV (Network Graph)')
st.write('Ce graphe montre les relations de co-integration fortes dans la meme famille de trades.')
with st.spinner('Construction du graphe...'):
    trade_issuer = trade.get('issuer', '')
    
    # Si c'est un trade composite (ex: cades_oat), on le split pour chopper un des émetteurs de base
    # et récupérer tout son cluster.
    base_iss = trade_issuer.split('_')[0] if '_' in trade_issuer else trade_issuer
    allowed_issuers = get_cluster_issuers(base_iss)
    
    df_spreads = df_opps[(df_opps['trade_type'] == 'spread') & (df_opps['issuer'].isin(allowed_issuers))]
    
    if not df_spreads.empty:
        fig_net = get_network_fig(df_spreads)
        if fig_net is not None:
            st.plotly_chart(fig_net, use_container_width=True)
        else:
            st.info('Pas assez de relations statistiques pour construire le graphe.')
    else:
        st.info('Pas de donnees de spreads pour cet emetteur.')

