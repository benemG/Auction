import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import MARKET_DATA_DIR, PSPLINE_DEFAULT_KNOTS, PSPLINE_DEFAULT_DEGREE, PSPLINE_DEFAULT_PENALTY_ORDER, PSPLINE_DEFAULT_LAMBDA
from src.data_io import read_market_data, write_fitted_curves
from src.pspline_curve import PSplineCurveFitter

def fit_daily_curves():
    print(f"Démarrage du fit P-Spline depuis {MARKET_DATA_DIR}")
    
    if not MARKET_DATA_DIR.exists():
        print(f"Le dossier {MARKET_DATA_DIR} n'existe pas. Veuillez ingérer les données d'abord.")
        return
        
    issuers = [d.name.split('=')[1] for d in MARKET_DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('issuer=')]
    
    if not issuers:
        print("Aucun émetteur trouvé.")
        return

    for issuer in issuers:
        print(f"\nTraitement de l'émetteur : {issuer}")
        df = read_market_data(issuer)
        
        if df.empty:
            continue
            
        results = []
        dates = df['date'].unique()
        
        for dt in dates:
            df_day = df[df['date'] == dt].copy()
            
            # Filtre : exclure maturité < 1 an et indexées inflation avant le fit
            df_day['ttm_years'] = (df_day['maturity'] - pd.to_datetime(dt)).dt.days / 365.25
            df_day = df_day[df_day['ttm_years'] >= 1.0]
            if 'inflation_linked_indicator' in df_day.columns:
                df_day = df_day[df_day['inflation_linked_indicator'].astype(str).str.upper() != 'Y']
            
            # On veut au moins 5 obligations pour fitter une courbe
            if len(df_day) < 5:
                continue
                
            metrics_to_fit = ['ytm', 'asset_swap_spread', 'z_spread']
            
            cols_to_keep = ['date', 'issuer', 'isin', 'bond_name', 'maturity', 'ttm_years']
            if 'inflation_linked_indicator' in df_day.columns:
                cols_to_keep.append('inflation_linked_indicator')
            day_results = df_day[cols_to_keep].copy()
            ttm_years = df_day['ttm_years']
            day_results['ttm_years'] = ttm_years
            day_results['n_observations'] = len(df_day)
            
            for metric in metrics_to_fit:
                if metric not in df_day.columns or df_day[metric].isna().all():
                    day_results[f"{metric}_market"] = np.nan
                    day_results[f"{metric}_fitted"] = np.nan
                    day_results[f"{metric}_residual"] = np.nan
                    continue
                    
                rates = df_day[metric].values
                
                fitter = PSplineCurveFitter(
                    n_knots=PSPLINE_DEFAULT_KNOTS, 
                    degree=PSPLINE_DEFAULT_DEGREE, 
                    penalty_order=PSPLINE_DEFAULT_PENALTY_ORDER, 
                    lambda_param=PSPLINE_DEFAULT_LAMBDA
                )
                
                n_obs = fitter.load_market_data(dt, df_day['maturity'], rates)
                if n_obs >= 5 and fitter.fit_curve():
                    fitted_vals, is_extrapolated = fitter.compute_fitted_yield(ttm_years.values)
                    residuals = rates - fitted_vals
                    
                    day_results[f"{metric}_market"] = rates
                    day_results[f"{metric}_fitted"] = fitted_vals
                    day_results[f"{metric}_residual"] = residuals
                    day_results[f"{metric}_rmse"] = fitter.rmse
                    day_results[f"{metric}_mae"] = fitter.mae
                    day_results['is_extrapolated'] = is_extrapolated
                else:
                    day_results[f"{metric}_market"] = rates
                    day_results[f"{metric}_fitted"] = np.nan
                    day_results[f"{metric}_residual"] = np.nan
                    day_results[f"{metric}_rmse"] = np.nan
                    day_results[f"{metric}_mae"] = np.nan
                    day_results['is_extrapolated'] = True
                    
            results.append(day_results)
            
        if results:
            df_results = pd.concat(results, ignore_index=True)
            
            # --- CALCUL DES SÉRIES DE MOMENTUM ---
            print("  Calcul des séries de momentum...")
            from src.momentum import compute_ewmac_series
            df_results = df_results.sort_values(['isin', 'date'])
            
            for metric in ['ytm', 'asset_swap_spread', 'z_spread']:
                m_col = f"{metric}_market"
                if m_col in df_results.columns:
                    # Calcul par ISIN (pour ne pas mélanger les séries temporelles de bonds différents)
                    df_results[f"{metric}_momentum"] = df_results.groupby('isin')[m_col].transform(
                        lambda x: compute_ewmac_series(x)
                    )
            
            write_fitted_curves(df_results)
            print(f"  Succès : {len(df_results)} lignes de fit écrites.")
            
    print("\nFit des courbes terminé.")

if __name__ == "__main__":
    fit_daily_curves()
