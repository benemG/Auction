import streamlit as st
import pandas as pd
import numpy as np
import ast
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_io import read_opportunities, read_market_data, read_fitted_curves, OPPORTUNITIES_DIR
from src.charts import plot_trade_performance, plot_rv_network, plot_backtest_analysis
from src.trade_definitions import build_rv_graph
from src.config import METRIC_TO_BPS, METRIC_LABELS, get_repo_spreads
from src.opportunity_generation import apply_repo_spreads
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

df_opps = apply_repo_spreads(get_opps(selected_date), get_repo_spreads())  # hors cache : suit la page Paramètres

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

st.divider()

# --- Fiche de décision ; NaN (parquets antérieurs aux nouvelles colonnes) affichés "—"
g = lambda k: trade.get(k, np.nan)
def fmt(v, spec='.1f', unit=' bp'):
    return '—' if pd.isna(v) else f"{v:{spec}}{unit}"

def row(label, value, warn=False, help=None):
    """Ligne libellé / valeur d'une carte ; orange quand la valeur doit alerter."""
    # wrap=False : en colonne étroite la valeur reste en face du libellé au lieu de passer dessous
    with st.container(horizontal=True, horizontal_alignment='distribute', vertical_alignment='center', gap='small', wrap=False):
        st.markdown(label, help=help, width='content')
        st.markdown(f":orange[**{value}**]" if warn else f"**{value}**", width='content')

long_spread = trade.get('recommended_direction') == 'long_spread'
with st.container(horizontal=True, vertical_alignment='center'):
    st.subheader("Fiche de décision", anchor=False)
    st.badge("Achat du spread" if long_spread else "Vente du spread", color='green' if long_spread else 'red',
             icon=':material/trending_up:' if long_spread else ':material/trending_down:')
    st.badge(str(trade.get('issuer', '')).upper(), color='gray')
    st.badge(METRIC_LABELS.get(trade['metric'], trade['metric']), color='gray')
    st.space('stretch')
    st.download_button("Exporter la fiche", data=generate_markdown_report(trade), file_name=f"{trade_id}_{selected_date}.md",
                       mime="text/markdown", icon=':material/download:')

# Ce qui a prédit le P&L 3M (univers du 02/01/2025) : le Sharpe attendu et le carry/roll, en tête de fiche
with st.container(horizontal=True):
    st.metric("Gain net / risque 3M", fmt(g('net_edge_ratio'), '.2f', ''), border=True,
              help="(gain attendu + carry/roll − coûts) / risque 3M : critère de classement du Scanner")
    st.metric("Gain attendu 3M", fmt(g('expected_gain_bps')), border=True,
              help="Convergence OU à 3 mois, décotée de moitié (calibrée sur le réalisé)")
    st.metric("Carry/roll 3M", fmt(g('net_carry_roll_3m_bps'), '+.1f'), border=True,
              help="Meilleur prédicteur du P&L 3M sur l'univers testé")
    st.metric("Risque 3M", fmt(g('risk_h_bps')), border=True, help="Écart-type des variations à 3 mois du spread de marché")

in_zone = g('in_entry_zone') == True  # np.bool_ depuis le Parquet, NaN sur les anciens fichiers
c1, c2, c3 = st.columns(3, border=True)
with c1:
    st.markdown("**:material/insights: Signal**")
    row("Spread actuel", fmt(g('spread_market_bps')))
    row("Cible (courbe + μ)", fmt(g('spread_target_bps')))
    row("Écart à la cible", fmt(g('target_gap_bps'), '+.1f'))
    row("Z-score 12M", fmt(g('zscore_12m'), '.2f', ''))
    row("Seuil Bertram", f"{fmt(g('entry_band_bps'))} · {'atteint' if in_zone else 'non atteint'}", warn=not in_zone,
        help="Écart minimal à la moyenne qui maximise le gain par unité de temps, coûts inclus (Bertram 2010)")
    row("Score RV", fmt(g('confidence_score'), '.0f', ' / 100'))
with c2:
    st.markdown("**:material/flag: Sortie**")
    row("Objectif (retour à μ)", fmt(g('exit_target_bps'), '+.1f'))
    row("Stop", fmt(-g('stop_loss_bps'), '+.1f'))
    row("P(objectif avant stop)", fmt(g('prob_target'), '.0%', ''))
    row("Durée attendue", fmt(g('exit_time_days'), '.0f', ' j'))
    row("Gain attendu à la sortie", fmt(g('ll_expected_gain_bps')))
    row("Gain par jour", fmt(g('gain_per_day_bps'), '.2f', ' bp/j'))
with c3:
    st.markdown("**:material/verified: Qualité**")
    share, glitches = g('market_share'), g('quote_glitches_12m')
    row("Part marché du résidu", fmt(share, '.0%', ''), warn=share < 0.5,
        help="Part du mouvement du résidu portée par le spread de marché ; sous 50 %, il bouge surtout avec la courbe fittée, qui ne paie rien")
    row("Cotations erronées 12M", fmt(glitches, '.0f', ''), warn=glitches > 0,
        help="Jours où le spread saute de plus de 4 écarts-types robustes puis revient le lendemain")
    row("Demi-vie", fmt(g('half_life_days'), '.0f', ' j'))
    row("Statut OU", str(trade.get('ou_calibration_status', '—')), warn=trade.get('ou_calibration_status') != 'valid')
    row("ADF (p-value)", fmt(g('adf_pvalue'), '.3f', ''))
    row("DV01 résiduelle", fmt(g('residual_dv01'), '.4f', ''))

with st.expander("Diagnostics secondaires (peu prédictifs du P&L 3M)"):
    d1, d2, d3 = st.columns(3)
    with d1:
        row("Momentum (EWMAC)", fmt(g('momentum_signal'), '.2f', ''))
        row("Momentum du voisinage", fmt(g('neighborhood_momentum'), '.2f', ''))
    with d2:
        row("Z-score 3M", fmt(g('zscore_3m'), '.2f', ''))
        row("Hurst", fmt(g('hurst'), '.2f', ''))
    with d3:
        row("Moyenne OU (μ du modèle)", fmt(g('ou_long_run_mean_bps')))
        row("Premier passage à μ", fmt(g('first_hitting_time_days'), '.0f', ' j'))

st.divider()
st.subheader("Spread et performance", anchor=False)

# Reconstruction de l'historique
@st.cache_data(ttl=3600, show_spinner=False)
def get_trade_history(issuer, isins, weights, as_of_date, metric):
    # Trade inter-courbes : l'émetteur est de la forme "cades_oat"
    issuers = issuer.split('_')
    df_m = pd.concat([read_market_data(i) for i in issuers], ignore_index=True)
    df_f = pd.concat([read_fitted_curves(i) for i in issuers], ignore_index=True)
    
    # Pas de coupure à la date d'analyse : la suite sert au suivi de performance (EWMAC causal, inchangé avant)
    scale = METRIC_TO_BPS[metric]  # le YTM est en %, les graphiques sont en bps

    dates = df_m['date'].unique()
    history = []
    
    for dt in dates:
        d_m = df_m[df_m['date'] == dt].set_index('isin')
        d_f = df_f[df_f['date'] == dt].set_index('isin')
        
        try:
            m_val = scale * sum(w * d_m.loc[isin, metric] for w, isin in zip(weights, isins))
            f_val = scale * sum(w * d_f.loc[isin, f"{metric}_fitted"] for w, isin in zip(weights, isins))
            history.append({'date': dt, 'spread_market': float(m_val), 'spread_fitted': float(f_val)})
        except Exception:
            pass
            
    df_res = pd.DataFrame(history)
    if df_res.empty:
        return df_res
        
    # --- Calcul du Neighborhood Momentum Historique ---
    try:
        pivot_m = scale * df_m.pivot(index='date', columns='isin', values=metric)
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
def get_network_fig(df_spreads):
    g = build_rv_graph(df_spreads)
    if g.number_of_nodes() > 0:
        return plot_rv_network(g)
    return None

# Récupération des infos du trade
isins = [trade.get(f'bond{i}_isin') for i in range(1, 5) if trade.get(f'bond{i}_isin') is not None]
weights = ast.literal_eval(trade['leg_weights']) if isinstance(trade['leg_weights'], str) else trade['leg_weights']

with st.spinner("Reconstruction de l'historique..."):
    df_full = get_trade_history(trade['issuer'], isins, weights, selected_date, trade['metric'])
df_hist = df_full[df_full['date'] <= pd.to_datetime(selected_date)] if not df_full.empty else df_full

if not df_hist.empty:
    st.caption("Gris : 12 mois de calibration · trait noir : entrée à la date d'analyse · bleu foncé : performance "
               "consécutive (si la date est passée) · pointillé : horizon 3M. Attendu = convergence décotée + carry/roll − coûts.")
    # theme=None : fond blanc du template, le bleu foncé et le trait noir d'entrée restent lisibles en thème sombre
    st.plotly_chart(plot_trade_performance(df_full, selected_date, trade), width="stretch", theme=None)
    
    st.divider()
    st.subheader("Simulation de backtest (filtre roulant : z-score, ADF, Hurst)", anchor=False)
    with st.spinner("Génération du backtest..."):
        try:
            import ast
            mom_raw = trade.get('momentum_series', '[]')
            mom_series = []
            try:
                if isinstance(mom_raw, (list, np.ndarray)):
                    mom_series = list(mom_raw)
                elif pd.notna(mom_raw) and str(mom_raw).strip():
                    # Remplacement de 'nan' par 'None' pour que literal_eval le parse correctement
                    mom_str = str(mom_raw).replace('nan', 'None').replace('NaN', 'None')
                    mom_series = ast.literal_eval(mom_str)
                else:
                    mom_series = []
            except Exception as e:
                mom_series = []
                
            fig_backtest = plot_backtest_analysis(df_hist, 'spread_market', mom_series)
            st.plotly_chart(fig_backtest, width="stretch")
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

st.divider()
st.subheader("Voisinage RV", anchor=False)
st.write('Ce graphe montre les relations de co-integration fortes dans la meme famille de trades.')
with st.spinner('Construction du graphe...'):
    trade_issuer = trade.get('issuer', '')
    
    # Si c'est un trade composite (ex: cades_oat), on le split pour chopper un des émetteurs de base
    # et récupérer tout son cluster.
    base_iss = trade_issuer.split('_')[0] if '_' in trade_issuer else trade_issuer
    allowed_issuers = get_cluster_issuers(base_iss)
    
    df_spreads = df_opps[(df_opps['trade_type'] == 'spread') & (df_opps['issuer'].isin(allowed_issuers)) & (df_opps['metric'] == trade['metric'])]
    
    if not df_spreads.empty:
        fig_net = get_network_fig(df_spreads)
        if fig_net is not None:
            st.plotly_chart(fig_net, width="stretch")
        else:
            st.info('Pas assez de relations statistiques pour construire le graphe.')
    else:
        st.info('Pas de donnees de spreads pour cet emetteur.')

