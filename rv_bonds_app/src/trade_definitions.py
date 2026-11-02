import itertools
import pandas as pd
import networkx as nx
import sys
from pathlib import Path
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.multitest import multipletests

# Need to import config params
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (MIN_Z_SCORE_EDGE, MAX_HALF_LIFE_EDGE, MIN_HALF_LIFE_EDGE, MAX_HURST_EDGE, MAX_FLY_WING_SPREAD_YEARS,
                        COINTEGRATION_MAX_PVALUE, COINTEGRATION_WINDOWS_DAYS, COINTEGRATION_FDR_ALPHA,
                        COINTEGRATION_TEST_ON_SPREAD, EDGE_REQUIRE_MR_STATS)

def generate_spread_combinations(df_current: pd.DataFrame, max_ttm_diff: float = 2.0) -> list:
    """
    Génère les combinaisons de spreads admissibles (A - B).
    """
    trades = []
    bonds = df_current.to_dict('records')
    
    for b1, b2 in itertools.combinations(bonds, 2):
        # Pour l'intra-courbe : maturité bond 1 > maturité bond 2
        if b1['ttm_years'] < b2['ttm_years']:
            b1, b2 = b2, b1
            
        if (b1['ttm_years'] - b2['ttm_years']) <= max_ttm_diff:
            trade_id = f"SPD | {b1['bond_name']} / {b2['bond_name']}"
            trades.append({
                'trade_id': trade_id,
                'trade_type': 'spread',
                'bond1_isin': b1['isin'],
                'bond2_isin': b2['isin'],
                'bond3_isin': None,
                'bond4_isin': None,
                'bond1_name': b1['bond_name'],
                'bond2_name': b2['bond_name'],
                'bond3_name': None,
                'bond4_name': None,
                'ttm_a': b1['ttm_years'],
                'ttm_b': b2['ttm_years']
            })
    return trades

def generate_inter_curve_spread_combinations(df_current_1: pd.DataFrame, df_current_2: pd.DataFrame, max_ttm_diff: float = 2.0, metric: str = 'asset_swap_spread') -> list:
    """
    Génère les spreads admissibles inter-courbes (bond1 issu de df_current_1, bond2 issu de df_current_2).
    Règle : market_value (metric_fitted) bond 1 > market_value bond 2.
    """
    trades = []
    bonds1 = df_current_1.to_dict('records')
    bonds2 = df_current_2.to_dict('records')
    
    metric = f"{metric}_fitted"
    
    for b1 in bonds1:
        for b2 in bonds2:
            if abs(b1['ttm_years'] - b2['ttm_years']) <= max_ttm_diff:
                
                # Règle d'ordre : market_value bond 1 > market_value bond 2
                mv1 = b1.get(metric, 0)
                mv2 = b2.get(metric, 0)
                
                if mv1 > mv2:
                    bond1, bond2 = b1, b2
                else:
                    bond1, bond2 = b2, b1
                    
                # Pour éviter les doublons (A/B et B/A auront la même orientation grâce à la règle ci-dessus,
                # mais puisqu'on itère sur deux listes distinctes, on n'a qu'un seul sens généré anyway).
                # Vérifions juste si on veut toujours le formater dans ce sens.
                # Oui, spread = bond1 - bond2 sera toujours > 0 sur le spread target actuel.
                
                trade_id = f"SPD | {bond1['bond_name']} / {bond2['bond_name']}"
                trades.append({
                    'trade_id': trade_id,
                    'trade_type': 'spread',
                    'bond1_isin': bond1['isin'],
                    'bond2_isin': bond2['isin'],
                    'bond3_isin': None,
                    'bond4_isin': None,
                    'bond1_name': bond1['bond_name'],
                    'bond2_name': bond2['bond_name'],
                    'bond3_name': None,
                    'bond4_name': None,
                    'ttm_a': bond1['ttm_years'],
                    'ttm_b': bond2['ttm_years']
                })
                
    # Déduplication
    unique_trades = {t['trade_id']: t for t in trades}
    return list(unique_trades.values())
def pair_cointegration_pvalues(px: pd.DataFrame, a: str, b: str, min_obs: int):
    """
    p-values d'une paire sur la fenêtre `px` (niveaux bruts) :
    - p_eg : Engle-Granger, le moins bon des deux sens ;
    - p_adf : ADF sur le spread A-B réellement tradé (bêta = 1).
    Retourne (nan, nan) si l'historique est insuffisant ou si un test échoue.
    """
    if a not in px.columns or b not in px.columns:
        return np.nan, np.nan
    pair = px[[a, b]].dropna()
    if len(pair) < min_obs:
        return np.nan, np.nan
    try:
        p_eg = max(coint(pair[a], pair[b])[1], coint(pair[b], pair[a])[1])
        p_adf = adfuller(pair[a] - pair[b], autolag='AIC')[1]
    except Exception:
        return np.nan, np.nan
    return p_eg, p_adf

def select_pvalues(p_eg: np.ndarray, p_adf: np.ndarray, max_pvalue: float = COINTEGRATION_MAX_PVALUE,
                   fdr_alpha=COINTEGRATION_FDR_ALPHA, test_on_spread: bool = COINTEGRATION_TEST_ON_SPREAD) -> np.ndarray:
    """
    Masque booléen des paires retenues sur une fenêtre. Avec fdr_alpha, le seuil est celui de
    Benjamini-Hochberg sur l'ensemble des paires testées ; sinon un seuil brut max_pvalue par paire.
    """
    p = np.asarray(p_eg, dtype=float)
    ok = ~np.isnan(p)
    keep = np.zeros(len(p), dtype=bool)
    if ok.any():
        if fdr_alpha is None:
            keep[ok] = p[ok] < max_pvalue
        else:
            keep[ok] = multipletests(p[ok], alpha=fdr_alpha, method='fdr_bh')[0]
    if test_on_spread:
        pa = np.asarray(p_adf, dtype=float)
        keep &= (~np.isnan(pa)) & (pa < max_pvalue)
    return keep

def filter_cointegrated_spreads(trades: list, df_market: pd.DataFrame, metric: str, as_of_date: pd.Timestamp,
                                max_pvalue: float = COINTEGRATION_MAX_PVALUE,
                                windows=COINTEGRATION_WINDOWS_DAYS,
                                fdr_alpha=COINTEGRATION_FDR_ALPHA,
                                test_on_spread: bool = COINTEGRATION_TEST_ON_SPREAD) -> list:
    """
    Ne garde que les spreads 2 jambes cointégrés (Engle-Granger dans les deux sens, niveaux bruts de
    `metric`) sur chacune des fenêtres, avec contrôle du taux de fausses découvertes (BH) et, en option,
    un ADF sur le spread A-B tradé. Les paires sans 90 % d'historique sur une fenêtre sont écartées.
    """
    if not trades:
        return trades

    windows = (windows,) if isinstance(windows, int) else tuple(windows)
    px_all = (df_market[df_market['date'] <= as_of_date]
              .drop_duplicates(['date', 'isin'], keep='last')
              .pivot(index='date', columns='isin', values=metric)
              .sort_index())

    keep = np.ones(len(trades), dtype=bool)
    for w in windows:
        px = px_all.tail(w)
        pv = np.array([pair_cointegration_pvalues(px, t['bond1_isin'], t['bond2_isin'], int(w * 0.9)) for t in trades])
        keep &= select_pvalues(pv[:, 0], pv[:, 1], max_pvalue, fdr_alpha, test_on_spread)
    return [t for t, k in zip(trades, keep) if k]

def build_rv_graph(df_spreads: pd.DataFrame, require_mr_stats: bool = EDGE_REQUIRE_MR_STATS) -> nx.Graph:
    """
    Construit un graphe où les noeuds sont des obligations et les arêtes
    sont des spreads RV statistiquement significatifs.
    Une arête exige |z| >= MIN_Z_SCORE_EDGE ; avec require_mr_stats, aussi demi-vie et Hurst dans les bornes.
    """
    G = nx.Graph()
    if df_spreads is None or df_spreads.empty:
        return G
        
    for _, row in df_spreads.iterrows():
        # Filtrage pour ne garder que les edges "pertinents"
        valid_zscore = pd.notna(row.get('zscore_12m')) and abs(row['zscore_12m']) >= MIN_Z_SCORE_EDGE
        valid_mr = pd.notna(row.get('half_life_days')) and MIN_HALF_LIFE_EDGE < row['half_life_days'] <= MAX_HALF_LIFE_EDGE
        valid_hurst = pd.notna(row.get('hurst')) and row['hurst'] < MAX_HURST_EDGE
        
        if valid_zscore and (not require_mr_stats or (valid_mr and valid_hurst)):
            b1 = row['bond1_isin']
            b2 = row['bond2_isin']
            n1 = row['bond1_name']
            n2 = row['bond2_name']
            
            G.add_edge(b1, b2, zscore=row['zscore_12m'], half_life=row['half_life_days'])
            
            # On stocke le nom et l'émetteur (premier mot du ticker)
            G.nodes[b1]['name'] = n1
            G.nodes[b1]['issuer'] = str(n1).split()[0] if pd.notna(n1) else 'Unknown'
            
            G.nodes[b2]['name'] = n2
            G.nodes[b2]['issuer'] = str(n2).split()[0] if pd.notna(n2) else 'Unknown'
            
    return G

def generate_fly_combinations_from_graph(graph: nx.Graph, df_current: pd.DataFrame, max_wing_diff: float = MAX_FLY_WING_SPREAD_YEARS) -> list:
    """
    Génère des flys en cherchant des chemins de longueur 2 dans le graphe des spreads.
    """
    trades = []
    # Créer un dictionnaire lookup pour la maturité et les noms
    bond_info = df_current.set_index('isin').to_dict('index')
    
    # Un fly est un chemin de longueur 2 : Wing1 - Body - Wing2
    # On itère sur chaque noeud qui servira de Body
    for body_isin in graph.nodes():
        if body_isin not in bond_info:
            continue
            
        neighbors = list(graph.neighbors(body_isin))
        if len(neighbors) < 2:
            continue
            
        # Séparer les voisins en "ailes courtes" et "ailes longues" par rapport au corps
        body_ttm = bond_info[body_isin]['ttm_years']
        short_wings = [n for n in neighbors if n in bond_info and bond_info[n]['ttm_years'] < body_ttm]
        long_wings = [n for n in neighbors if n in bond_info and bond_info[n]['ttm_years'] > body_ttm]
        
        for w1 in short_wings:
            for w2 in long_wings:
                ttm_1 = bond_info[w1]['ttm_years']
                ttm_2 = bond_info[w2]['ttm_years']
                
                # Vérifier la contrainte structurelle (max distance entre les ailes)
                if (ttm_2 - ttm_1) <= max_wing_diff:
                    name1 = bond_info[w1]['bond_name']
                    name_body = bond_info[body_isin]['bond_name']
                    name2 = bond_info[w2]['bond_name']
                    trade_id = f"FLY | {name1} / {name_body} / {name2}"
                    trades.append({
                        'trade_id': trade_id,
                        'trade_type': 'fly',
                        'bond1_isin': w1,
                        'bond2_isin': body_isin,
                        'bond3_isin': w2,
                        'bond4_isin': None,
                        'bond1_name': bond_info[w1]['bond_name'],
                        'bond2_name': bond_info[body_isin]['bond_name'],
                        'bond3_name': bond_info[w2]['bond_name'],
                        'bond4_name': None,
                        'ttm_a': ttm_1,
                        'ttm_b': body_ttm,
                        'ttm_c': ttm_2
                    })
    
    # Dé-dupliquer les trades au cas où
    unique_trades = {t['trade_id']: t for t in trades}
    return list(unique_trades.values())

# On garde l'ancienne pour compatibilité si nécessaire, mais on l'a désactivée par défaut
def generate_fly_combinations(df_current: pd.DataFrame, max_wing_diff: float = 3.0) -> list:
    pass # replaced by generate_fly_combinations_from_graph
