# Scanner de Relative Value Obligataire

Application Streamlit légère pour l'exploration d'opportunités de Relative Value sur obligations souveraines.

## Architecture

L'application repose sur des fichiers Parquet locaux générés par des batchs de calcul (ingestion, P-spline, et Mean Reversion OU).
Le dossier racine attendu est `rv_bonds_app/`.

### Arborescence requise

```text
rv_bonds_app/
├── data/
│   ├── raw/                  # Déposer vos fichiers CSV ici par émetteur (ex: data/raw/OAT/)
│   └── parquet/
│       ├── market_data/      # Données ingérées
│       ├── fitted_curves/    # Fits P-spline quotidiens
│       └── opportunities/    # Scan précalculé
├── src/                      # Modèles métier
├── scripts/                  # Batchs
├── pages/                    # Vues Streamlit
├── app.py                    # Point d'entrée de l'UI
├── requirements.txt
└── README.md
```

## Installation

1. S'assurer d'avoir Python >= 3.10
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Exécution

### 1. Alimenter les données brutes
Déposer les fichiers CSV dans `data/raw/<Issuer>/` (ex: `data/raw/OAT/`).

### 2. Lancer la pipeline Batch
Exécuter le script de rafraîchissement global pour ingérer les données, calibrer les courbes et générer les opportunités :
```bash
python scripts/refresh_all.py
```

*Note: Vous pouvez aussi exécuter chaque script indépendamment :*
- `python scripts/ingest_csv_to_parquet.py`
- `python scripts/fit_daily_curves.py`
- `python scripts/generate_opportunities.py`

### 3. Démarrer l'Application Streamlit
Une fois les Parquets générés dans `data/parquet/opportunities/`, lancer :
```bash
streamlit run app.py
```

## Tests

Pour valider l'intégrité des modèles :
```bash
pytest tests/
```
