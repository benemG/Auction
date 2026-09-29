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
    
    # Courbe fittée
    fig.add_trace(go.Scatter(
        x=df_sorted['ttm_years'], y=df_sorted[f_col],
        mode='lines', name='P-Spline Fit',
        line=dict(color='red')
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

def plot_combined_trade_analysis(df: pd.DataFrame, spread_col: str, fitted_col: str, current_val: float, mu: float, theta: float, sigma: float):
    """
    Graphique unique combinant :
    - Spread historique (Axe Gauche)
    - Spread P-Spline Cible (Axe Gauche)
    - Forecast OU avec bandes de confiance (Axe Gauche, dans le futur)
    - Résidus (Market - Cible) (Axe Droit)
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # --- 1. HISTORIQUE ---
    # Market Spread
    fig.add_trace(go.Scatter(
        x=df['date'], y=df[spread_col], 
        mode='lines', name='Market Spread (bps)',
        line=dict(color='blue', width=2)
    ), secondary_y=False)
    
    # P-Spline Target
    fig.add_trace(go.Scatter(
        x=df['date'], y=df[fitted_col], 
        mode='lines', name='P-Spline Target (bps)',
        line=dict(color='red', dash='dash')
    ), secondary_y=False)
    
    # Résidus
    residuals = df[spread_col] - df[fitted_col]
    fig.add_trace(go.Bar(
        x=df['date'], y=residuals,
        name='Residuals (bps)',
        marker_color='rgba(128, 128, 128, 0.4)'
    ), secondary_y=True)

    # --- 2. FORECAST OU ---
    if not (np.isnan(mu) or np.isnan(theta) or np.isnan(sigma) or theta <= 0):
        last_date = df['date'].max()
        
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
        
        # Conversion horizons en jours calendaires pour l'axe X
        # (horizons * 365) jours calendaires
        future_dates = [last_date + pd.Timedelta(days=int(h * 365)) for h in horizons]
        
        # Bandes 2 sigmas
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=np.concatenate([upper_2, lower_2[::-1]]),
            fill='toself', fillcolor='rgba(200,200,200,0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='± 2 std',
            showlegend=True
        ), secondary_y=False)
        
        # Bandes 1 sigma
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=np.concatenate([upper_1, lower_1[::-1]]),
            fill='toself', fillcolor='rgba(150,150,150,0.4)',
            line=dict(color='rgba(255,255,255,0)'),
            name='± 1 std',
            showlegend=True
        ), secondary_y=False)
        
        # Espérance
        fig.add_trace(go.Scatter(
            x=future_dates, y=exp_t, 
            mode='lines', name='Expected OU Trajectory',
            line=dict(color='green', width=2)
        ), secondary_y=False)
        
        # Ligne Mu
        fig.add_hline(y=mu, line_dash="dot", annotation_text="OU Long Run Mean", line_color="green", secondary_y=False)

    fig.update_layout(
        title="Analyse du Spread & Projection Ornstein-Uhlenbeck",
        xaxis_title="Date",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Spread (bps)", secondary_y=False)
    fig.update_yaxes(title_text="Residuals (bps)", secondary_y=True, showgrid=False)
    
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
    df[spread_col] = df[spread_col].astype(float).ffill()
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
    if precalc_momentum and len(precalc_momentum) > 0:
        # Align from the end in case history is longer
        momentum = pd.Series(np.nan, index=df.index)
        usable_len = min(len(precalc_momentum), len(momentum))
        momentum.iloc[-usable_len:] = precalc_momentum[-usable_len:]
    else:
        momentum = pd.Series(np.nan, index=df.index)
        
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
        f_texts = [f"<b>Filtered {f['direction']}</b><br>Z-Score: {f['zscore']:.2f}<br>Hurst: {f['hurst']:.2f}<br>ADF: {f['adf']:.3f}" for f in filtered]
        fig.add_trace(go.Scatter(
            x=f_dates, y=f_spreads,
            mode='markers', marker=dict(color='gray', symbol='circle', size=6, opacity=0.6),
            name="Filtered Trade", text=f_texts, hoverinfo="text", showlegend=False
        ), row=1, col=1)
    
    # Triangles d'entrée
    for e in entries:
        color = 'green' if e['direction'] == 'Long' else 'red'
        symbol = 'triangle-up' if e['direction'] == 'Long' else 'triangle-down'
        hover_text = (f"<b>{e['direction']} Spread</b><br>"
                      f"Date: {e['date'].strftime('%Y-%m-%d')}<br>"
                      f"Spread: {e['spread']:.2f} bps<br>"
                      f"Z-Score: {e['zscore']:.2f}<br>"
                      f"Hurst (252j): {e['hurst']:.2f}<br>"
                      f"ADF p-val: {e['adf']:.3f}")
                      
        fig.add_trace(go.Scatter(
            x=[e['date']], y=[e['spread']],
            mode='markers', marker=dict(color=color, symbol=symbol, size=14, line=dict(width=1, color='black')),
            name=f"Entry {e['direction']}", text=[hover_text], hoverinfo="text", showlegend=False
        ), row=1, col=1)
        
    # Croix de sortie
    for ex in exits:
        fig.add_trace(go.Scatter(
            x=[ex['date']], y=[ex['spread']],
            mode='markers', marker=dict(color='black', symbol='x', size=10),
            name="Exit", text=[f"Exit at {ex['spread']:.2f} bps"], hoverinfo="text", showlegend=False
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


