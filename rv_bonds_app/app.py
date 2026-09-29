import streamlit as st
import sys
from pathlib import Path

# Configurer le sys.path pour les pages
sys.path.append(str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="RV Bonds Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Scanner de Relative Value Obligataire")

st.markdown("""
### Bienvenue dans le RV Bonds Scanner

Cette application vous permet d'explorer les opportunités de trading de type Relative Value 
(Spreads, Flys) pré-calculées par les batchs nocturnes.

👈 **Sélectionnez une vue dans la barre latérale :**

1. **Scanner** : Vue tabulaire filtrable des meilleures opportunités du jour.
2. **Trade Explorer** : Analyse détaillée d'un trade (Historique, Modèle OU).
3. **Curve Diagnostics** : Visualisation du fit P-Spline.
4. **Parameters** : Paramètres de configuration du système.
""")

st.sidebar.info("Modèles : P-Spline pour la courbe, OU (Ornstein-Uhlenbeck) MLE pour le signal.")
