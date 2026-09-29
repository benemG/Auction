import sys
import os
import itertools
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import MARKET_DATA_DIR, FITTED_CURVES_DIR, MAX_FLY_WING_SPREAD_YEARS
from src.data_io import read_market_data, read_fitted_curves, write_opportunities
from src.trade_definitions import generate_spread_combinations, build_rv_graph, generate_fly_combinations_from_graph, generate_inter_curve_spread_combinations
from src.opportunity_generation import generate_opportunities_for_issuer
from src.clustering import compute_micro_rv_clusters
import json

def filter_current(df):
    if 'ttm_years' in df.columns:
        df = df[df['ttm_years'] >= 1.0]
    if 'inflation_linked_indicator' in df.columns:
        df = df[df['inflation_linked_indicator'].astype(str).str.upper() != 'Y']
    return df

def generate_opportunities():
    print("Démarrage de la génération d'opportunités (Graph-based + Clustering Inter-courbes)...")
    
    if not MARKET_DATA_DIR.exists() or not FITTED_CURVES_DIR.exists():
        print("Données de marché ou courbes manquantes.")
        return
        
    issuers = [d.name.split('=')[1] for d in MARKET_DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('issuer=')]
    
    market_dict = {}
    fitted_dict = {}
    current_dict = {}
    global_last_date = None
    
    # --- INTRA-COURBES ---
    print("\n" + "="*40 + "\nÉTAPES INTRA-COURBES\n" + "="*40)
    for issuer in issuers:
        print(f"\nÉmetteur : {issuer}")
        df_market = read_market_data(issuer)
        df_fitted = read_fitted_curves(issuer)
        
        if df_market.empty or df_fitted.empty:
            continue
            
        dates = df_market['date'].unique()
        last_date = pd.to_datetime(dates[-1])
        global_last_date = last_date
        print(f"  Date cible : {last_date.strftime('%Y-%m-%d')}")
        
        df_current = df_fitted[df_fitted['date'] == last_date].copy()
        if df_current.empty:
            continue
            
        df_current = filter_current(df_current)
        df_current = df_current.drop_duplicates(subset=['isin'])
        
        market_dict[issuer] = df_market
        fitted_dict[issuer] = df_fitted
        current_dict[issuer] = df_current
            
        if len(df_current) < 2:
            print("  Pas assez d'obligations éligibles.")
            continue
            
        spread_defs = generate_spread_combinations(df_current)
        df_spread_opps = generate_opportunities_for_issuer(df_market, df_fitted, spread_defs, last_date)
        graph = build_rv_graph(df_spread_opps)
        fly_defs = generate_fly_combinations_from_graph(graph, df_current, MAX_FLY_WING_SPREAD_YEARS)
        df_fly_opps = generate_opportunities_for_issuer(df_market, df_fitted, fly_defs, last_date) if len(fly_defs) > 0 else pd.DataFrame()
        
        dfs_to_concat = []
        if df_spread_opps is not None and not df_spread_opps.empty:
            dfs_to_concat.append(df_spread_opps)
        if df_fly_opps is not None and not df_fly_opps.empty:
            dfs_to_concat.append(df_fly_opps)
            
        if dfs_to_concat:
            df_all_opps = pd.concat(dfs_to_concat, ignore_index=True)
            df_all_opps['issuer'] = issuer
            write_opportunities(df_all_opps, last_date)
            print(f"  Succès : {len(df_all_opps)} opportunités écrites.")

    # --- INTER-COURBES ---
    if global_last_date is None:
        return
        
    print("\n" + "="*40 + "\nÉTAPES INTER-COURBES (MICRO-RV SEULEMENT)\n" + "="*40)
    print("Calcul dynamique des clusters Micro-RV (dendrogramme d'ASW sur 1 an)...")
    clusters = compute_micro_rv_clusters(threshold=1.0)
    
    # Sauvegarde des clusters
    cluster_file = MARKET_DATA_DIR.parent / "clusters.json"
    with open(cluster_file, "w") as f:
        json.dump(clusters, f)
        
    for cluster in clusters:
        # Garder seulement les émetteurs qu'on a pu charger
        cluster = [iss for iss in cluster if iss in current_dict]
        if len(cluster) < 2:
            continue
            
        print(f"\nTraitement du cluster Micro-RV : {cluster}")
        
        for iss1, iss2 in itertools.combinations(cluster, 2):
            combo_name = f"{iss1}_{iss2}"
            print(f"  Génération inter-courbes : {iss1} vs {iss2}...")
            
            df_current_1 = current_dict[iss1]
            df_current_2 = current_dict[iss2]
            
            inter_spread_defs = generate_inter_curve_spread_combinations(df_current_1, df_current_2)
            
            if not inter_spread_defs:
                continue
                
            df_market_combined = pd.concat([market_dict[iss1], market_dict[iss2]], ignore_index=True)
            df_fitted_combined = pd.concat([fitted_dict[iss1], fitted_dict[iss2]], ignore_index=True)
            
            df_inter_opps = generate_opportunities_for_issuer(df_market_combined, df_fitted_combined, inter_spread_defs, global_last_date)
            
            if df_inter_opps is not None and not df_inter_opps.empty:
                # Graph / Fly inter-courbe (Optionnel, mais intéressant)
                # On peut construire un graphe RV bipartite entre les deux courbes et chercher des flys "papillons inter-courbes"
                graph = build_rv_graph(df_inter_opps)
                df_current_combined = pd.concat([df_current_1, df_current_2], ignore_index=True).drop_duplicates(subset=['isin'])
                fly_defs = generate_fly_combinations_from_graph(graph, df_current_combined, MAX_FLY_WING_SPREAD_YEARS)
                df_inter_fly = generate_opportunities_for_issuer(df_market_combined, df_fitted_combined, fly_defs, global_last_date) if len(fly_defs) > 0 else pd.DataFrame()
                
                dfs_to_concat = [df_inter_opps]
                if not df_inter_fly.empty:
                    dfs_to_concat.append(df_inter_fly)
                    
                df_all_inter = pd.concat(dfs_to_concat, ignore_index=True)
                df_all_inter['issuer'] = combo_name
                write_opportunities(df_all_inter, global_last_date)
                print(f"    -> Succès : {len(df_all_inter)} opportunités ({len(df_inter_opps)} spd, {len(df_inter_fly)} flys).")

    print("\nGénération terminée avec succès.")

if __name__ == "__main__":
    generate_opportunities()
