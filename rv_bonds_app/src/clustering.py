import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import pyarrow.parquet as pq
from .config import FITTED_CURVES_DIR

def issuer_distance(df_pivot: pd.DataFrame) -> np.ndarray:
    """Distance Micro-RV entre émetteurs (colonnes) : (1 - corrélation) + |écart de niveau moyen| / 15."""
    dist_corr = np.maximum(1 - df_pivot.corr(method='pearson').values, 0)
    mean_spreads = df_pivot.mean().values.reshape(-1, 1)
    return dist_corr + squareform(pdist(mean_spreads, metric='euclidean')) / 15.0

def compute_micro_rv_clusters(threshold=1.0, n_clusters=None):
    """
    Calcule les clusters Micro-RV dynamiques basés sur l'ASW historique 5-10Y.
    Retourne une liste de listes (ex: [['oat', 'cades'], ['bund', 'nether'], ...])
    Avec n_clusters, l'arbre est coupé en n_clusters groupes au lieu d'une coupe à distance threshold.
    """
    if not FITTED_CURVES_DIR.exists():
        return []
        
    try:
        dataset = pq.ParquetDataset(str(FITTED_CURVES_DIR))
        df = dataset.read().to_pandas()
    except Exception:
        return []
        
    df_bucket = df[(df['ttm_years'] >= 5.0) & (df['ttm_years'] <= 10.0)].copy()
    if df_bucket.empty:
        return []
        
    daily_issuer_spread = df_bucket.groupby(['date', 'issuer'])['asset_swap_spread_fitted'].mean().reset_index()
    df_pivot = daily_issuer_spread.pivot(index='date', columns='issuer', values='asset_swap_spread_fitted')
    df_pivot = df_pivot.sort_index().tail(252)
    df_pivot = df_pivot.ffill().dropna(axis=1)
    
    issuers = df_pivot.columns.tolist()
    if len(issuers) < 2:
        return []
        
    condensed_dist = squareform(issuer_distance(df_pivot), checks=False)
    
    Z = linkage(condensed_dist, method='ward')
    if n_clusters:
        cluster_ids = fcluster(Z, t=n_clusters, criterion='maxclust')
    else:
        cluster_ids = fcluster(Z, t=threshold, criterion='distance')
    
    groups = {}
    for issuer, cid in zip(issuers, cluster_ids):
        groups.setdefault(cid, []).append(issuer)
        
    return list(groups.values())

import json
import itertools

def get_cluster_issuers(selected_issuer: str) -> list:
    """
    Retourne la liste de tous les sous-émetteurs et paires inter-courbes 
    associés à l'émetteur sélectionné, en lisant le clusters.json.
    """
    if selected_issuer == 'Tous':
        return []
        
    cluster_file = FITTED_CURVES_DIR.parent / "clusters.json"
    allowed = [selected_issuer]
    if cluster_file.exists():
        with open(cluster_file, "r") as f:
            clusters = json.load(f)
        for c in clusters:
            if selected_issuer in c:
                allowed.extend(c)
                for iss1, iss2 in itertools.combinations(c, 2):
                    allowed.append(f"{iss1}_{iss2}")
                    allowed.append(f"{iss2}_{iss1}")
                break
                
    return list(set(allowed))
