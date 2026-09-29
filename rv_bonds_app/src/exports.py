import pandas as pd

def generate_markdown_report(trade: pd.Series) -> str:
    """
    Génère un rapport textuel Markdown pour le trade.
    """
    md = f"""# Fiche de Trade RV : {trade.get('trade_id', 'N/A')}

**Date d'analyse :** {trade.get('as_of_date', 'N/A')}
**Type de Trade :** {trade.get('trade_type', 'N/A')}
**Recommandation :** {trade.get('recommended_direction', 'N/A')}
**Score de Confiance :** {trade.get('confidence_score', 0):.1f}/100

## Détails du Spread ({trade.get('metric', 'N/A')})
* **Niveau Marché :** {trade.get('spread_market_bps', 0):.2f} bps
* **Cible (P-Spline) :** {trade.get('spread_target_bps', 0):.2f} bps
* **Écart à la cible :** {trade.get('target_gap_bps', 0):.2f} bps
* **Z-Score (12m) :** {trade.get('zscore_12m', 0):.2f}
* **Volatilité Quotidienne :** {trade.get('daily_vol_bps', 0):.2f} bps

## Analyse Ornstein-Uhlenbeck (Mean Reversion)
* **Statut de Calibration :** {trade.get('ou_calibration_status', 'N/A')}
* **Moyenne Long Terme (Mu) :** {trade.get('ou_long_run_mean_bps', 0):.2f} bps
* **Demi-vie (Half-Life) :** {trade.get('half_life_days', 0):.1f} jours
* **Espérance First Hitting Time :** {trade.get('first_hitting_time_days', 0):.1f} jours

## Structure
* **Poids des jambes :** {trade.get('leg_weights', 'N/A')}
* **Exposition DV01 résiduelle estimée :** {trade.get('residual_dv01', 0):.4f}

---
*Généré par RV Bonds Scanner*
"""
    return md

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convertit un dataframe en bytes CSV pour le téléchargement.
    """
    return df.to_csv(index=False).encode('utf-8')
