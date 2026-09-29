import itertools
import pandas as pd
import networkx as nx
import sys
from pathlib import Path

# Need to import config params
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MIN_Z_SCORE_EDGE, MAX_HALF_LIFE_EDGE, MIN_HALF_LIFE_EDGE, MAX_HURST_EDGE, MAX_FLY_WING_SPREAD_YEARS

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

def generate_inter_curve_spread_combinations(df_current_1: pd.DataFrame, df_current_2: pd.DataFrame, max_ttm_diff: float = 2.0) -> list:
    """
    Génère les spreads admissibles inter-courbes (bond1 issu de df_current_1, bond2 issu de df_current_2).
    Règle : market_value (ici asset_swap_spread_fitted) bond 1 > market_value bond 2.
    """
    trades = []
    bonds1 = df_current_1.to_dict('records')
    bonds2 = df_current_2.to_dict('records')
    
    # On choisit asset_swap_spread_fitted par défaut pour comparer la market value
    metric = 'asset_swap_spread_fitted'
    
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
def build_rv_graph(df_spreads: pd.DataFrame) -> nx.Graph:
    """
    Construit un graphe où les noeuds sont des obligations et les arêtes
    sont des spreads RV statistiquement significatifs.
    """
    G = nx.Graph()
    if df_spreads is None or df_spreads.empty:
        return G
        
    for _, row in df_spreads.iterrows():
        # Filtrage pour ne garder que les edges "pertinents"
        valid_zscore = pd.notna(row.get('zscore_12m')) and abs(row['zscore_12m']) >= MIN_Z_SCORE_EDGE
        valid_mr = pd.notna(row.get('half_life_days')) and MIN_HALF_LIFE_EDGE < row['half_life_days'] <= MAX_HALF_LIFE_EDGE
        valid_hurst = pd.notna(row.get('hurst')) and row['hurst'] < MAX_HURST_EDGE
        
        if valid_zscore and valid_mr and valid_hurst:
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
