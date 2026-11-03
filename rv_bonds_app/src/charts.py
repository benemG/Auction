import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_spread_history(df: pd.DataFrame, spread_col: str, fitted_col: str, mu_ou: float = None):
    """
    Tracé du spread historique vs la cible P-spline, avec la moyenne OU en option.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df[spread_col], 
        mode='lines', name='Market Spread (bps)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df[fitted_col], 
        mode='lines', name='P-Spline Target (bps)',
        line=dict(color='red', dash='dash')
    ))
    
    if mu_ou is not None and not np.isnan(mu_ou):
        fig.add_hline(
            y=mu_ou, line_dash="dot", 
            annotation_text="OU Mean", 
            annotation_position="bottom right",
            line_color="green"
        )
        
    fig.update_layout(
        title="Historique du Spread vs Cible",
        xaxis_title="Date",
        yaxis_title="Spread (bps)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def plot_ou_projection(current_val: float, mu: float, theta: float, sigma: float):
    """
    Projection de l'espérance et des intervalles de confiance du modèle OU.
    """
    if np.isnan(mu) or np.isnan(theta) or np.isnan(sigma) or theta <= 0:
        return go.Figure().add_annotation(text="Modèle OU non disponible", showarrow=False)
        
    # Horizons en années
    horizons = np.linspace(0, 1.0, 100) # Jusqu'à 1 an
    
    # E[X_t]
    exp_t = current_val * np.exp(-theta * horizons) + mu * (1 - np.exp(-theta * horizons))
    
    # Volatilité
    vol_t = np.sqrt((sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * horizons)))
    
    upper_1 = exp_t + vol_t
    lower_1 = exp_t - vol_t
    
    upper_2 = exp_t + 2 * vol_t
    lower_2 = exp_t - 2 * vol_t
    
    # Conversion horizons en jours (approximatif pour affichage)
    horizons_days = horizons * 252
    
    fig = go.Figure()
    
    # Bandes 2 sigmas
    fig.add_trace(go.Scatter(
        x=np.concatenate([horizons_days, horizons_days[::-1]]),
        y=np.concatenate([upper_2, lower_2[::-1]]),
        fill='toself', fillcolor='rgba(200,200,200,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='± 2 std'
    ))
    
    # Bandes 1 sigma
    fig.add_trace(go.Scatter(
        x=np.concatenate([horizons_days, horizons_days[::-1]]),
        y=np.concatenate([upper_1, lower_1[::-1]]),
        fill='toself', fillcolor='rgba(150,150,150,0.4)',
        line=dict(color='rgba(255,255,255,0)'),
        name='± 1 std'
    ))
    
    # Espérance
    fig.add_trace(go.Scatter(
        x=horizons_days, y=exp_t, 
        mode='lines', name='Expected Trajectory',
        line=dict(color='green', width=2)
    ))
    
    # Mu
    fig.add_hline(y=mu, line_dash="dot", annotation_text="Long Run Mean", line_color="black")
    
    fig.update_layout(
        title="Projection Ornstein-Uhlenbeck",
        xaxis_title="Horizon (Trading Days)",
        yaxis_title="Spread Expected (bps)",
        template="plotly_white"
    )
    
    return fig

def plot_pspline_curve(df_day: pd.DataFrame, metric: str = 'ytm'):
    """
    Tracé du fit P-spline du jour (Rich / Cheap)
    """
    fig = go.Figure()
    
    m_col = f"{metric}_market"
    f_col = f"{metric}_fitted"
    
    if m_col not in df_day.columns or f_col not in df_day.columns:
        return go.Figure().add_annotation(text="Données de fit manquantes", showarrow=False)
        
    df_sorted = df_day.sort_values('ttm_years').copy()
    df_sorted['residual'] = df_sorted[m_col] - df_sorted[f_col]
    
    # Valeurs ajustées leave-one-out : des points, une ligne les relierait en dents de scie
    fig.add_trace(go.Scatter(
        x=df_sorted['ttm_years'], y=df_sorted[f_col],
        mode='markers', name='Fair value (LOO)',
        marker=dict(color='red', symbol='x', size=7)
    ))
    
    # Points marché colorés selon residual
    fig.add_trace(go.Scatter(
        x=df_sorted['ttm_years'], y=df_sorted[m_col],
        mode='markers', name='Market',
        text=df_sorted.get('isin', '') + '<br>Res: ' + df_sorted['residual'].round(2).astype(str),
        hoverinfo='text',
        marker=dict(
            size=10, 
            color=df_sorted['residual'], 
            colorscale='RdBu', 
            cmin=-df_sorted['residual'].abs().max(),
            cmax=df_sorted['residual'].abs().max(),
            colorbar=dict(title="Residual (bps)"),
            showscale=True,
            line=dict(width=1, color='black')
        )
    ))
    
    fig.update_layout(
        title=f"Rich/Cheap Curve: {metric.upper()}",
        xaxis_title="Maturity (Years)",
        yaxis_title=metric.upper(),
        template="plotly_white"
    )
    
    return fig
from plotly.subplots import make_subplots

def plot_trade_performance(df: pd.DataFrame, as_of, trade: pd.Series, pre: int = 252, post: int = 126):
    """
    Trois panneaux autour de l'entrée (date d'analyse, trait noir ; horizon 3M en pointillé) :
    - spread de marché : gris sur les 12 mois de calibration, bleu foncé après l'entrée, cible = courbe du jour + μ, stop ;
    - P&L dans le sens du trade (carry inclus, coûts déduits) face à l'attendu (convergence décotée + carry/roll) ±1σ ;
    - résidu vs courbe (SvF) avec μ et ±2σ : montre si le spread converge ou si c'est la courbe qui bouge.
    Sans données postérieures à l'entrée (dernière date), seuls l'attendu et le cône sont tracés.
    """
    from .config import TRADING_DAYS_IN_YEAR, HOLDING_HORIZON_DAYS as H, EXPECTED_GAIN_HAIRCUT
    GREY, BLUE, GREEN, RED = '#9e9e9e', '#0b2e59', '#2e7d32', '#c62828'
    g = lambda k, d=np.nan: d if pd.isna(trade.get(k, np.nan)) else trade[k]
    as_of = pd.Timestamp(as_of)
    df = df.set_index('date').sort_index()
    s, svf = df['spread_market'], df['spread_market'] - df['spread_fitted']
    s_pre, s_post = s[s.index <= as_of].tail(pre), s[s.index >= as_of].head(post + 1)
    v_pre, v_post = svf[svf.index <= as_of].tail(pre), svf[svf.index >= as_of].head(post + 1)
    d, s0 = np.sign(g('target_gap_bps', 0.0)), g('spread_market_bps')
    mu = g('spread_target_bps') - g('spread_fitted_bps')
    x_fwd = pd.bdate_range(as_of, periods=post + 1)
    end = max(x_fwd[-1], s_post.index[-1]) if len(s_post) else x_fwd[-1]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.45, 0.3, 0.25],
                        subplot_titles=('Spread de marché (bp)', 'P&L dans le sens du trade (bp) : réalisé vs attendu',
                                        'Résidu vs courbe (SvF, bp)'))
    # 1. Spread
    fig.add_trace(go.Scatter(x=s_pre.index, y=s_pre, line=dict(color=GREY, width=1.3), name='Historique 12M'), 1, 1)
    if len(s_post) > 1:
        fig.add_trace(go.Scatter(x=s_post.index, y=s_post, line=dict(color=BLUE, width=2.2), name='Après entrée'), 1, 1)
    tgt = (df['spread_fitted'] + mu)[s_pre.index[0]:] if len(s_pre) else df['spread_fitted'].iloc[:0]
    fig.add_trace(go.Scatter(x=tgt.index, y=tgt, line=dict(color=GREEN, width=1), opacity=0.7,
                             name='Cible (courbe du jour + μ)'), 1, 1)
    # Stop projeté à courbe inchangée ; l'objectif (μ) est la cible verte
    stop = s0 - d * g('stop_loss_bps')
    if np.isfinite(stop):
        fig.add_trace(go.Scatter(x=[as_of, end], y=[stop, stop], mode='lines', name='Stop',
                                 line=dict(color=RED, dash='dot', width=1.2)), 1, 1)

    # 2. P&L attendu et réalisé
    t = np.arange(post + 1)
    theta = g('ou_theta', 0.0) if trade.get('ou_calibration_status') == 'valid' else 0.0
    cost = g('bid_ask_cost_bps', 0.0)
    exp = (EXPECTED_GAIN_HAIRCUT * abs(g('target_gap_bps', 0.0)) * (1 - np.exp(-theta * t / TRADING_DAYS_IN_YEAR))
           + g('net_carry_roll_3m_bps', 0.0) * t / H - cost)
    sd = g('risk_h_bps', 0.0) * np.sqrt(t / H)  # shortcut: diffusion en √t, surestime le risque d'un spread qui revient vite
    fig.add_trace(go.Scatter(x=list(x_fwd) + list(x_fwd[::-1]), y=list(exp + sd) + list((exp - sd)[::-1]),
                             fill='toself', fillcolor='rgba(11,46,89,0.10)', line=dict(width=0), name='Attendu ±1σ'), 2, 1)
    fig.add_trace(go.Scatter(x=x_fwd, y=exp, line=dict(color=BLUE, dash='dash', width=1), name='Attendu'), 2, 1)
    if len(s_post) > 1:
        pnl = d * (s_post - s0) + g('carry_3m_bps', 0.0) * np.arange(len(s_post)) / H - cost
        fig.add_trace(go.Scatter(x=pnl.index, y=pnl, line=dict(color=BLUE, width=2.2), name='P&L réalisé'), 2, 1)
    fig.add_hline(y=0, line=dict(color='#bbb', width=1), row=2, col=1)

    # 3. Résidu
    fig.add_trace(go.Scatter(x=v_pre.index, y=v_pre, line=dict(color=GREY, width=1.3), showlegend=False), 3, 1)
    if len(v_post) > 1:
        fig.add_trace(go.Scatter(x=v_post.index, y=v_post, line=dict(color=BLUE, width=2.2), showlegend=False), 3, 1)
    sd_svf = v_pre.std()
    x0 = v_pre.index[0] if len(v_pre) else as_of
    for y, col in [(mu, GREEN), (mu + 2 * sd_svf, '#bbb'), (mu - 2 * sd_svf, '#bbb')]:
        fig.add_trace(go.Scatter(x=[x0, end], y=[y, y], mode='lines', showlegend=False, hoverinfo='skip',
                                 line=dict(color=col, dash='dash', width=1)), 3, 1)

    for row in (1, 2, 3):
        fig.add_vline(x=as_of.timestamp() * 1000, line=dict(color='black', width=1.5), row=row, col=1)
        fig.add_vline(x=x_fwd[H].timestamp() * 1000, line=dict(color='#999', width=1, dash='dot'), row=row, col=1)
    # Fond blanc explicite : le bleu foncé et le trait noir d'entrée resteraient illisibles sur le thème sombre
    fig.update_layout(height=760, template='plotly_white', hovermode='x unified', paper_bgcolor='white',
                      plot_bgcolor='white', font=dict(color='#222'),
                      legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='right', x=1))
    return fig

import networkx as nx

def plot_parallel_coordinates(df_opps: pd.DataFrame):
    if df_opps is None or df_opps.empty:
        return go.Figure().add_annotation(text="Pas de donnees pour le graphe", showarrow=False)

    df_plot = df_opps.dropna(subset=['zscore_12m', 'half_life_days', 'target_gap_bps', 'confidence_score']).copy()
    
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = df_plot['confidence_score'],
                       colorscale = 'Viridis',
                       showscale = True,
                       cmin = 0,
                       cmax = 100),
            dimensions = list([
                dict(range = [df_plot['zscore_12m'].min(), df_plot['zscore_12m'].max()],
                     label = 'Z-Score (12m)', values = df_plot['zscore_12m']),
                dict(range = [df_plot['half_life_days'].min(), min(df_plot['half_life_days'].max(), 120)],
                     label = 'Half-Life (Days)', values = df_plot['half_life_days']),
                dict(range = [df_plot['target_gap_bps'].min(), df_plot['target_gap_bps'].max()],
                     label = 'Target Gap (bps)', values = df_plot['target_gap_bps']),
                dict(range = [0, 100],
                     label = 'Confidence Score', values = df_plot['confidence_score'])
            ])
        )
    )

    fig.update_layout(
        title="Filtrage Multi-Dimensionnel (Coordonnees Paralleles)",
        template="plotly_white"
    )
    return fig

import plotly.colors as pcolors
import datetime

def plot_rv_network(graph: nx.Graph):
    if not graph or graph.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="Graphe vide", showarrow=False)

    issue_dates = {}
    try:
        from src.config import BASE_DIR
        mapping_file = BASE_DIR.parent / "mapping_indiv_bond.csv"
        if mapping_file.exists():
            df_ref = pd.read_csv(mapping_file, sep=';', dtype=str, usecols=['id_isin', 'issue_dt'])
            df_ref['issue_dt'] = pd.to_datetime(df_ref['issue_dt'], errors='coerce')
            df_ref = df_ref.dropna(subset=['issue_dt'])
            issue_dates = dict(zip(df_ref['id_isin'], df_ref['issue_dt']))
    except Exception as e:
        print("Erreur chargement issue_dt:", e)

    today = pd.Timestamp(datetime.date.today())

    try:
        pos = nx.kamada_kawai_layout(graph)
    except:
        pos = nx.spring_layout(graph, seed=42)

    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False)

    # Grouper les noeuds par issuer pour créer des traces séparées (pour la légende)
    issuers = {}
    for node in graph.nodes():
        issuer = graph.nodes[node].get('issuer', 'Unknown')
        issuers.setdefault(issuer, []).append(node)
        
    colors = pcolors.qualitative.Plotly
    
    traces = [edge_trace]
    for i, (issuer, nodes) in enumerate(issuers.items()):
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        for node in nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(graph.nodes[node].get('name', str(node)))
            
            size = 12
            if node in issue_dates:
                issue_dt = issue_dates[node]
                duration_years = (today - issue_dt).days / 365.25
                if duration_years > 0:
                    calc_size = 30.0 / max(duration_years, 0.5)
                    size = max(6, min(40, calc_size))
            node_size.append(size)
            
        color = colors[i % len(colors)]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            textposition="bottom center",
            hoverinfo='text',
            text=node_text,
            name=issuer, # Legend entry
            marker=dict(
                color=color,
                size=node_size,
                line_width=2,
                line_color='white'
            ))
        traces.append(node_trace)

    fig = go.Figure(data=traces,
             layout=go.Layout(
                title='Reseau de Co-integration (Voisinage RV)',
                title_font_size=16,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_white"
             ))
    return fig


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_opportunities_pca(df_opps: pd.DataFrame):
    if df_opps is None or df_opps.empty:
        return go.Figure().add_annotation(text="Pas de donnees pour PCA", showarrow=False)

    features = ['zscore_12m', 'half_life_days', 'target_gap_bps', 'confidence_score']
    df_plot = df_opps.dropna(subset=features).copy()
    
    if len(df_plot) < 3:
        return go.Figure().add_annotation(text="Pas assez de donnees pour PCA", showarrow=False)

    X = df_plot[features].values
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_plot['PCA_1'] = X_pca[:, 0]
    df_plot['PCA_2'] = X_pca[:, 1]
    
    fig = go.Figure(data=go.Scatter(
        x=df_plot['PCA_1'],
        y=df_plot['PCA_2'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_plot['confidence_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Score")
        ),
        text=df_plot['trade_id'] + '<br>Gap: ' + df_plot['target_gap_bps'].round(2).astype(str) + ' bps',
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Clustering PCA des Opportunites (Outliers = Forte dislocation)",
        xaxis_title="Composante Principale 1",
        yaxis_title="Composante Principale 2",
        template="plotly_white"
    )
    return fig

from statsmodels.tsa.stattools import adfuller

def fast_hurst(ts):
    if len(ts) < 20: return np.nan
    lags = range(2, 20)
    try:
        var = [np.var(ts[lag:] - ts[:-lag]) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(var), 1)
        return poly[0] / 2.0
    except:
        return np.nan

def fast_adf(ts):
    try:
        res = adfuller(ts, maxlag=1, autolag=None)
        return res[1]
    except:
        return np.nan

def plot_backtest_analysis(df: pd.DataFrame, spread_col: str, precalc_momentum: list = None):
    """
    Simule la stratégie et génère 4 graphiques (subplots) :
    1. Spread + MA + Signaux
    2. Momentum EWMAC (du spread) - issu de la base
    3. Z-Score dynamique
    4. PnL Cumulé
    """
    df = df.copy().sort_values('date').reset_index(drop=True)
    
    # Remplir les petits trous éventuels pour éviter les gaps dans la MA
    df[spread_col] = pd.to_numeric(df[spread_col], errors='coerce').ffill().bfill()
    spread = df[spread_col]
    
    window_ma = 60
    window_filter = 252
    
    # min_periods=10 pour avoir une MA rapidement même après un gap
    MA = spread.rolling(window=window_ma, min_periods=10).mean()
    Std = spread.rolling(window=window_ma, min_periods=10).std().replace(0, np.nan)
    Z_Score = (spread - MA) / Std
    
    df['MA'] = MA
    df['Z_Score'] = Z_Score
    
    # --- Utilisation du Momentum pré-calculé ---
    momentum = pd.Series(np.nan, index=df.index, dtype=float)
    if precalc_momentum is not None and len(precalc_momentum) > 0:
        # Align from the end in case history is longer
        # Conversion sécurisée en float64 avec conversion de None / NaN / chaînes invalides vers np.nan
        clean_mom = pd.to_numeric(pd.Series(precalc_momentum), errors='coerce').to_numpy()
        usable_len = min(len(clean_mom), len(momentum))
        if usable_len > 0:
            momentum.iloc[-usable_len:] = clean_mom[-usable_len:]
        
    df['Momentum'] = momentum
    
    position = 0
    positions = np.zeros(len(df))
    entries = []
    exits = []
    filtered = []
    
    for i in range(window_filter, len(df)):
        z = Z_Score.iloc[i]
        if np.isnan(z):
            positions[i] = position
            continue
            
        if position == 0:
            if z < -2.0 or z > 2.0:
                ts = spread.iloc[i-window_filter:i].values
                h = fast_hurst(ts)
                p = fast_adf(ts)
                direction = 'Long' if z < -2.0 else 'Short'
                
                if h < 0.5 and p < 0.05:
                    position = 1 if z < -2.0 else -1
                    entries.append({
                        'idx': i,
                        'date': df['date'].iloc[i],
                        'spread': spread.iloc[i],
                        'zscore': z,
                        'hurst': h,
                        'adf': p,
                        'direction': direction
                    })
                else:
                    # Filtre non passé
                    filtered.append({
                        'date': df['date'].iloc[i],
                        'spread': spread.iloc[i],
                        'zscore': z,
                        'hurst': h,
                        'adf': p,
                        'direction': direction
                    })
        elif position == 1:
            if z >= 0:
                position = 0
                exits.append({'idx': i, 'date': df['date'].iloc[i], 'spread': spread.iloc[i]})
        elif position == -1:
            if z <= 0:
                position = 0
                exits.append({'idx': i, 'date': df['date'].iloc[i], 'spread': spread.iloc[i]})
                
        positions[i] = position
        
    df['Position'] = positions
    df['Daily_PnL'] = df['Position'].shift(1) * spread.diff()
    df['Cum_PnL'] = df['Daily_PnL'].fillna(0).cumsum()
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.06, 
                        subplot_titles=("Spread & Signaux de Trading", "Momentum EWMAC (du Spread)", "Z-Score Dynamique", "P&L Cumulé (bps)"))
    
    # 1. Spread + MA
    fig.add_trace(go.Scatter(x=df['date'], y=spread, mode='lines', name='Spread (bps)', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=MA, mode='lines', name=f'MA {window_ma}j', line=dict(color='orange', dash='dash')), row=1, col=1)
    
    # Points gris pour les trades filtrés
    if filtered:
        f_dates = [f['date'] for f in filtered]
        f_spreads = [f['spread'] for f in filtered]
        f_texts = []
        for f in filtered:
            z_s = f"{f['zscore']:.2f}" if pd.notna(f.get('zscore')) else "N/A"
            h_s = f"{f['hurst']:.2f}" if pd.notna(f.get('hurst')) else "N/A"
            a_s = f"{f['adf']:.3f}" if pd.notna(f.get('adf')) else "N/A"
            f_texts.append(f"<b>Filtered {f.get('direction', '')}</b><br>Z-Score: {z_s}<br>Hurst: {h_s}<br>ADF: {a_s}")
        fig.add_trace(go.Scatter(
            x=f_dates, y=f_spreads,
            mode='markers', marker=dict(color='gray', symbol='circle', size=6, opacity=0.6),
            name="Filtered Trade", text=f_texts, hoverinfo="text", showlegend=False
        ), row=1, col=1)
    
    # Triangles d'entrée
    for e in entries:
        color = 'green' if e['direction'] == 'Long' else 'red'
        symbol = 'triangle-up' if e['direction'] == 'Long' else 'triangle-down'
        date_str = e['date'].strftime('%Y-%m-%d') if hasattr(e['date'], 'strftime') else str(e['date'])
        sp_str = f"{e['spread']:.2f}" if pd.notna(e.get('spread')) else "N/A"
        z_str = f"{e['zscore']:.2f}" if pd.notna(e.get('zscore')) else "N/A"
        h_str = f"{e['hurst']:.2f}" if pd.notna(e.get('hurst')) else "N/A"
        a_str = f"{e['adf']:.3f}" if pd.notna(e.get('adf')) else "N/A"
        hover_text = (f"<b>{e['direction']} Spread</b><br>"
                      f"Date: {date_str}<br>"
                      f"Spread: {sp_str} bps<br>"
                      f"Z-Score: {z_str}<br>"
                      f"Hurst (252j): {h_str}<br>"
                      f"ADF p-val: {a_str}")
                      
        fig.add_trace(go.Scatter(
            x=[e['date']], y=[e['spread']],
            mode='markers', marker=dict(color=color, symbol=symbol, size=14, line=dict(width=1, color='black')),
            name=f"Entry {e['direction']}", text=[hover_text], hoverinfo="text", showlegend=False
        ), row=1, col=1)
        
    # Croix de sortie
    for ex in exits:
        sp_str = f"{ex['spread']:.2f}" if pd.notna(ex.get('spread')) else "N/A"
        fig.add_trace(go.Scatter(
            x=[ex['date']], y=[ex['spread']],
            mode='markers', marker=dict(color='black', symbol='x', size=10),
            name="Exit", text=[f"Exit at {sp_str} bps"], hoverinfo="text", showlegend=False
        ), row=1, col=1)
        
    # 2. Momentum
    fig.add_trace(go.Scatter(x=df['date'], y=momentum, mode='lines', name='Momentum (Spread)', line=dict(color='brown')), row=2, col=1)
    if 'neighborhood_momentum' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['neighborhood_momentum'], 
            mode='lines', name='Momentum (Voisins)', 
            line=dict(color='gray', dash='dot')
        ), row=2, col=1)
        # Enable legend for the momentum subplot so users can distinguish the lines
        fig.update_layout(showlegend=True)
    fig.add_hline(y=0, line_dash='solid', line_color='black', row=2, col=1)
    
    # 3. Z-Score
    fig.add_trace(go.Scatter(x=df['date'], y=Z_Score, mode='lines', name='Z-Score', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=2, line_dash='dot', line_color='red', row=3, col=1)
    fig.add_hline(y=-2, line_dash='dot', line_color='green', row=3, col=1)
    fig.add_hline(y=0, line_dash='solid', line_color='black', row=3, col=1)
    
    # 4. PnL
    fig.add_trace(go.Scatter(x=df['date'], y=df['Cum_PnL'], mode='lines', name='Cum. PnL (bps)', line=dict(color='teal'), fill='tozeroy'), row=4, col=1)
    
    fig.update_layout(
        height=1000, 
        template="plotly_white", 
        hovermode="x unified", 
        title="Simulation de Stratégie Mean Reversion",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


