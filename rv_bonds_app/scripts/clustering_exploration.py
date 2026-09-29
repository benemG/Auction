import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import FITTED_CURVES_DIR
import pyarrow.parquet as pq

def run_clustering():
    print("Chargement des courbes fittées...")
    dataset = pq.ParquetDataset(str(FITTED_CURVES_DIR))
    df = dataset.read().to_pandas()
    
    # On se concentre sur le bucket 5-10 ans qui est le plus liquide
    # et représentatif du risque macro de l'émetteur
    df_bucket = df[(df['ttm_years'] >= 5.0) & (df['ttm_years'] <= 10.0)].copy()
    
    # Calcul du niveau d'ASW moyen par jour et par émetteur
    daily_issuer_spread = df_bucket.groupby(['date', 'issuer'])['asset_swap_spread_fitted'].mean().reset_index()
    
    # Pivot pour avoir les émetteurs en colonnes
    df_pivot = daily_issuer_spread.pivot(index='date', columns='issuer', values='asset_swap_spread_fitted')
    
    # Garder la dernière année (env 252 jours)
    df_pivot = df_pivot.sort_index().tail(252)
    
    # Remplir les petits trous (forward fill) et supprimer les émetteurs avec trop de données manquantes
    df_pivot = df_pivot.ffill().dropna(axis=1)
    
    issuers = df_pivot.columns.tolist()
    print(f"\nAnalyse sur {len(issuers)} émetteurs : {issuers}")
    
    # 1. Matrice de distance basée sur la CORRÉLATION (1 - corr)
    # Plus ils sont corrélés, plus la distance s'approche de 0
    corr_matrix = df_pivot.corr(method='pearson')
    dist_corr = 1 - corr_matrix.values
    # S'assurer qu'il n'y a pas de valeurs négatives dues aux arrondis
    dist_corr = np.maximum(dist_corr, 0)
    
    # 2. Matrice de distance basée sur l'ÉCART ABSOLU de niveau (Euclidienne)
    # Pour pénaliser les émetteurs très corrélés mais qui traitent à 100 bps d'écart
    # On prend la moyenne des spreads sur la période pour chaque émetteur
    mean_spreads = df_pivot.mean().values.reshape(-1, 1)
    dist_abs = squareform(pdist(mean_spreads, metric='euclidean'))
    
    # Normalisation de dist_abs pour qu'elle ait un poids comparable à dist_corr (qui est entre 0 et 2)
    # dist_abs est en bps (ex: OAT vs BTP = 80 bps). 
    # On divise par exemple par 15 bps, donc 15 bps d'écart = distance de 1.
    dist_abs_norm = dist_abs / 15.0 
    
    # 3. Distance Combinée
    # Distance totale = Distance de corrélation + Pénalité d'écart de niveau
    dist_combined = dist_corr + dist_abs_norm
    
    # Conversion en condensed distance matrix pour SciPy
    condensed_dist = squareform(dist_combined, checks=False)
    
    # 4. Clustering Hiérarchique (méthode de Ward)
    Z = linkage(condensed_dist, method='ward')
    
    # 5. Découpage de l'arbre
    # On définit un seuil (t) qui détermine la tolérance du cluster.
    # Plus t est grand, plus les clusters sont larges.
    threshold = 1.0
    clusters = fcluster(Z, t=threshold, criterion='distance')
    
    # 6. Affichage des Groupes
    groups = {}
    for issuer, cluster_id in zip(issuers, clusters):
        groups.setdefault(cluster_id, []).append(issuer)
        
    print("\n" + "="*50)
    print("RÉSULTATS DU CLUSTERING STATISTIQUE (MICRO-RV)")
    print("="*50)
    for cid, members in sorted(groups.items()):
        # On calcule le spread moyen du cluster pour info
        cluster_mean_spread = df_pivot[members].mean().mean()
        print(f"Cluster {cid} (Spread ASW moyen: {cluster_mean_spread:.1f} bps):")
        print(f"  -> {', '.join(members)}")
    print("="*50)

if __name__ == '__main__':
    run_clustering()
