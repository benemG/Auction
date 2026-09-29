import streamlit as st
import pandas as pd
import networkx as nx
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_io import read_opportunities, OPPORTUNITIES_DIR
from src.charts import plot_rv_network
from src.trade_definitions import build_rv_graph

st.set_page_config(page_title="Analyse des Clusters", page_icon="🕸️", layout="wide")
st.title("Analyse des Clusters RV")

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

col_date, col_issuer = st.columns(2)
with col_date:
    selected_date = st.selectbox("Date d'analyse", dates)

@st.cache_data
def get_opps(date_str):
    return read_opportunities(date_str)

df_opps = get_opps(selected_date)

if df_opps.empty:
    st.info("Pas d'opportunités à cette date.")
    st.stop()

from src.clustering import get_cluster_issuers

with col_issuer:
    raw_issuers = df_opps['issuer'].unique().tolist()
    base_issuers = sorted([iss for iss in raw_issuers if '_' not in iss])
    selected_issuer = st.selectbox("Émetteur", base_issuers)

st.markdown("---")
st.subheader("Moteur de Recherche des Clusters")

# On récupère tous les émetteurs liés au cluster (intra et inter)
allowed_issuers = get_cluster_issuers(selected_issuer)

# Construction du graphe global pour tout le cluster
df_spreads = df_opps[(df_opps['trade_type'] == 'spread') & (df_opps['issuer'].isin(allowed_issuers))]

if df_spreads.empty:
    st.warning("Pas de données de spread RV pour construire le graphe.")
    st.stop()

G = build_rv_graph(df_spreads)

if G.number_of_nodes() == 0:
    st.warning("Aucune relation statiquement significative trouvée pour cet émetteur (Graphe vide).")
    st.stop()

# Extraction des clusters (composantes connexes)
connected_components = list(nx.connected_components(G))
# Trier les clusters par taille (du plus grand au plus petit)
connected_components.sort(key=len, reverse=True)

# Préparation des données du cluster pour la recherche
bond_to_cluster = {}
cluster_info = []

for i, comp in enumerate(connected_components):
    cluster_id = f"Cluster {i+1}"
    bonds_in_cluster = []
    
    for node in comp:
        bond_name = G.nodes[node].get('name', str(node))
        bond_to_cluster[node] = cluster_id
        bond_to_cluster[bond_name] = cluster_id
        bonds_in_cluster.append({'ISIN': node, 'Nom': bond_name})
        
    cluster_info.append({
        'Cluster': cluster_id,
        'Taille': len(comp),
        'Membres': bonds_in_cluster,
        'Graphe': G.subgraph(comp)
    })

search_query = st.text_input("🔍 Rechercher une obligation (ISIN ou fragment du Nom) pour trouver son cluster :", "")

if search_query:
    query = search_query.lower()
    found_clusters = set()
    for bond_key, c_id in bond_to_cluster.items():
        if query in str(bond_key).lower():
            found_clusters.add(c_id)
            
    if not found_clusters:
        st.info("Aucune obligation trouvée correspondant à cette recherche dans les clusters actuels.")
    else:
        st.success(f"Obligation trouvée dans {len(found_clusters)} cluster(s).")
        cluster_info = [c for c in cluster_info if c['Cluster'] in found_clusters]

# Affichage des clusters
st.markdown("---")
st.write(f"### Visualisation ({len(cluster_info)} clusters affichés)")

for c in cluster_info:
    if c['Taille'] < 2:
        continue # Ignore les noeuds isolés s'il y en a
        
    with st.expander(f"{c['Cluster']} - {c['Taille']} obligations fortement co-intégrées", expanded=(len(cluster_info) == 1)):
        col_graph, col_table = st.columns([2, 1])
        
        with col_graph:
            st.plotly_chart(plot_rv_network(c['Graphe']), use_container_width=True)
            
        with col_table:
            st.dataframe(pd.DataFrame(c['Membres']), hide_index=True, use_container_width=True)
