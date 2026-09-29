import pandas as pd
import numpy as np
from .hedge_ratio import compute_spread_hedge_ratio, compute_fly_hedge_ratio, compute_residual_dv01
from .mean_reversion import MeanReversionModel
from .momentum import compute_ewmac
from .scoring import calculate_confidence_score
from .config import TRADING_DAYS_IN_YEAR, SCORE_WEIGHTS

def generate_opportunities_for_issuer(df_market: pd.DataFrame, df_fitted: pd.DataFrame, trades_def: list, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Évalue une liste de définitions de trades et produit le Parquet final.
    """
    import scipy.interpolate as interp
    import os
    from .config import BASE_DIR
    
    results = []
    
    # Restreindre aux données historiques et dédupliquer
    df_hist = df_market[df_market['date'] <= as_of_date].drop_duplicates(subset=['date', 'isin']).copy()
    df_hist_fit = df_fitted[df_fitted['date'] <= as_of_date].drop_duplicates(subset=['date', 'isin']).copy()
    
    # Pivoter pour un accès rapide
    df_hist_pivot = df_hist.pivot(index='date', columns='isin')
    df_fit_pivot = df_hist_fit.pivot(index='date', columns='isin')
    
    current_market = df_hist[df_hist['date'] == as_of_date].set_index('isin')
    current_fit = df_hist_fit[df_hist_fit['date'] == as_of_date].set_index('isin')
    
    # --- Chargement du Taux ESTER ---
    ester_rate = 3.40 # fallback
    try:
        ecb_path = BASE_DIR / "data" / "ecb_rates.parquet"
        if ecb_path.exists():
            df_ecb = pd.read_parquet(ecb_path)
            # Find closest date <= as_of_date
            df_ecb = df_ecb[df_ecb['date'] <= pd.to_datetime(as_of_date)]
            if not df_ecb.empty:
                ester_rate = df_ecb.iloc[-1]['ester_rate']
    except Exception as e:
        print(f"Attention: Impossible de charger ecb_rates.parquet ({e}), fallback {ester_rate}%")
        
    for trade in trades_def:
        metric = 'asset_swap_spread' # On force ASW pour l'exemple, extensible à YTM/Z-spread
        
        # Helper pour construire l'interpolateur de yield (pour le rolldown)
        # On suppose que le yield de la courbe est stocké dans current_fit.
        # S'il n'y a pas 'ytm_fitted', on fallback sur 'asset_swap_spread_fitted' ou 0
        fit_metric = 'ytm_fitted' if 'ytm_fitted' in current_fit.columns else f"{metric}_fitted"
        ttms = current_fit['ttm_years'].values
        yields = current_fit[fit_metric].values
        valid_mask = ~np.isnan(ttms) & ~np.isnan(yields)
        
        if valid_mask.sum() > 2:
            sort_idx = np.argsort(ttms[valid_mask])
            curve_interp = interp.interp1d(ttms[valid_mask][sort_idx], yields[valid_mask][sort_idx], kind='linear', fill_value="extrapolate")
        else:
            curve_interp = None

        if trade['trade_type'] == 'spread':
            isin1, isin2 = trade['bond1_isin'], trade['bond2_isin']
            
            if isin1 not in current_market.index or isin2 not in current_market.index:
                continue
                
            dv01_1 = current_market.loc[isin1, 'dv01'] if 'dv01' in current_market.columns else 1.0
            dv01_2 = current_market.loc[isin2, 'dv01'] if 'dv01' in current_market.columns else 1.0
            
            n1, n2 = compute_spread_hedge_ratio(dv01_1, dv01_2)
            
            # Series historiques
            try:
                ts_market = n1 * df_hist_pivot[metric][isin1] + n2 * df_hist_pivot[metric][isin2]
                ts_fitted = n1 * df_fit_pivot[f"{metric}_fitted"][isin1] + n2 * df_fit_pivot[f"{metric}_fitted"][isin2]
            except KeyError:
                continue
                
            current_spread = ts_market.iloc[-1]
            current_target = ts_fitted.iloc[-1]
            
            rmse_1 = current_fit.loc[isin1, f"{metric}_rmse"] if f"{metric}_rmse" in current_fit.columns else np.nan
            rmse_2 = current_fit.loc[isin2, f"{metric}_rmse"] if f"{metric}_rmse" in current_fit.columns else np.nan
            rmse = np.nanmean([rmse_1, rmse_2])
            
            legs = [n1, n2]
            dv01s = [dv01_1, dv01_2]
            isins = [isin1, isin2]
            ttm_list = [trade['ttm_a'], trade['ttm_b']]
            
        elif trade['trade_type'] == 'fly':
            isin1, isin2, isin3 = trade['bond1_isin'], trade['bond2_isin'], trade['bond3_isin']
            
            if isin1 not in current_market.index or isin2 not in current_market.index or isin3 not in current_market.index:
                continue
                
            dv01_1 = current_market.loc[isin1, 'dv01'] if 'dv01' in current_market.columns else 1.0
            dv01_2 = current_market.loc[isin2, 'dv01'] if 'dv01' in current_market.columns else 1.0
            dv01_3 = current_market.loc[isin3, 'dv01'] if 'dv01' in current_market.columns else 1.0
            
            n1, n2, n3 = compute_fly_hedge_ratio(dv01_1, dv01_2, dv01_3, trade['ttm_a'], trade['ttm_b'], trade['ttm_c'])
            
            try:
                ts_market = n1 * df_hist_pivot[metric][isin1] + n2 * df_hist_pivot[metric][isin2] + n3 * df_hist_pivot[metric][isin3]
                ts_fitted = n1 * df_fit_pivot[f"{metric}_fitted"][isin1] + n2 * df_fit_pivot[f"{metric}_fitted"][isin2] + n3 * df_fit_pivot[f"{metric}_fitted"][isin3]
            except KeyError:
                continue
                
            current_spread = ts_market.iloc[-1]
            current_target = ts_fitted.iloc[-1]
            
            rmses = [
                current_fit.loc[isin1, f"{metric}_rmse"] if f"{metric}_rmse" in current_fit.columns else np.nan,
                current_fit.loc[isin2, f"{metric}_rmse"] if f"{metric}_rmse" in current_fit.columns else np.nan,
                current_fit.loc[isin3, f"{metric}_rmse"] if f"{metric}_rmse" in current_fit.columns else np.nan
            ]
            rmse = np.nanmean(rmses)
            
            legs = [n1, n2, n3]
            dv01s = [dv01_1, dv01_2, dv01_3]
            isins = [isin1, isin2, isin3]
            ttm_list = [trade['ttm_a'], trade['ttm_b'], trade['ttm_c']]
            
        else:
            continue
            
        # Filtre : Minimum 12 mois d'historique
        if len(ts_market.dropna()) < 252:
            continue
            
        # Calcul du spread "Spread vs Fitted" (SvF) pour l'historique
        ts_svf = ts_market - ts_fitted
        current_svf = current_spread - current_target
            
        # Z-Score sur 12 mois (252 jours)
        ts_svf_12m = ts_svf.dropna().tail(252)
        if len(ts_svf_12m) > 10:
            zscore = (current_svf - ts_svf_12m.mean()) / ts_svf_12m.std()
            daily_vol = ts_svf_12m.std()
        else:
            zscore = np.nan
            daily_vol = np.nan
            
        # Z-Score sur 3 mois (63 jours)
        ts_svf_3m = ts_svf.dropna().tail(63)
        if len(ts_svf_3m) > 10:
            zscore_3m = (current_svf - ts_svf_3m.mean()) / ts_svf_3m.std()
        else:
            zscore_3m = np.nan
            
        # Momentum EWMAC
        ts_clean = ts_svf.dropna()
        momentum_signal = compute_ewmac(ts_clean)
        try:
            from .momentum import compute_ewmac_series
            momentum_series = compute_ewmac_series(ts_clean).tolist()
        except Exception:
            momentum_series = []
            
        # Calibration OU
        ou = MeanReversionModel()
        state = ou.calibrate_ou_model(ts_svf_12m, dt=1.0/TRADING_DAYS_IN_YEAR)
        
        fht = ou.compute_first_hitting_time(current_svf)
        
        from .mean_reversion import compute_hurst_exponent, compute_adf_pvalue
        hurst = compute_hurst_exponent(ts_svf_12m)
        adf_pvalue = compute_adf_pvalue(ts_svf_12m)
        
        # Consolidation des métriques pour le score
        metrics = {
            'ou_calibration_status': state['status'],
            'half_life_days': ou.compute_half_life() * TRADING_DAYS_IN_YEAR if state['status'] == 'valid' else np.nan,
            'zscore_12m': zscore,
            'zscore_3m': zscore_3m,
            'fit_rmse_bps': rmse,
            'hurst': hurst,
            'adf_pvalue': adf_pvalue,
            'momentum_signal': momentum_signal
        }
        
        target_gap = current_target - current_spread
        direction = 'long_spread' if target_gap > 0 else 'short_spread'
        
        # --- CALCUL CARRY & ROLLDOWN TRAP AVOIDANCE ---
        net_carry_roll_3m_bps = 0.0
        total_bid_ask_cost_bps = 0.0
        horizon_years = 0.25 # 3 mois
        for w, isin, ttm in zip(legs, isins, ttm_list):
            # Bid/Ask spread proxy (si non disponible, on assume 1 bps par jambe)
            bid_ask_spread = current_market.loc[isin, 'bid_ask_spread'] if 'bid_ask_spread' in current_market.columns else 1.0
            total_bid_ask_cost_bps += abs(w) * (bid_ask_spread / 2.0)
            
            # Repo rate spécifique si disponible
            repo_rate = current_market.loc[isin, 'repo_rate'] if 'repo_rate' in current_market.columns else ester_rate
            
            # Carry
            ytm_val = current_market.loc[isin, 'ytm'] if 'ytm' in current_market.columns else current_market.loc[isin, metric]
            carry_bps_annual = (ytm_val - repo_rate) * 100 
            carry_bps_3m = carry_bps_annual * horizon_years
            
            # Rolldown
            roll_bps_3m = 0.0
            if curve_interp is not None and ttm > horizon_years:
                yield_now = curve_interp(ttm)
                yield_future = curve_interp(ttm - horizon_years)
                roll_yield_diff = (yield_now - yield_future) * 100
                dv01 = current_market.loc[isin, 'dv01'] if 'dv01' in current_market.columns else 1.0
                roll_bps_3m = roll_yield_diff * dv01
                
            net_carry_roll_3m_bps += w * (carry_bps_3m + roll_bps_3m)
            
        # Adjusted gap = Target - friction costs + net carry roll
        expected_pnl_bps = target_gap + net_carry_roll_3m_bps
        carry_roll_adjusted_gap = expected_pnl_bps - np.sign(expected_pnl_bps) * total_bid_ask_cost_bps

        res = {
            'as_of_date': as_of_date,
            'trade_id': trade['trade_id'],
            'trade_type': trade['trade_type'],
            'metric': metric,
            'bond1_isin': trade['bond1_isin'],
            'bond2_isin': trade['bond2_isin'],
            'bond3_isin': trade['bond3_isin'],
            'bond1_name': trade['bond1_name'],
            'bond2_name': trade['bond2_name'],
            'bond3_name': trade['bond3_name'],
            'leg_weights': str(legs),
            'spread_market_bps': current_spread,
            'spread_fitted_bps': current_target,
            'spread_target_bps': current_target,
            'target_gap_bps': target_gap,
            'net_carry_roll_3m_bps': net_carry_roll_3m_bps,
            'bid_ask_cost_bps': total_bid_ask_cost_bps,
            'carry_roll_adjusted_gap': carry_roll_adjusted_gap,
            'ou_long_run_mean_bps': state['mu'],
            'ou_calibration_status': state['status'],
            'ou_theta': state['theta'],
            'ou_sigma': state['sigma'],
            'half_life_days': metrics['half_life_days'],
            'first_hitting_time_days': fht['expected_time'] * TRADING_DAYS_IN_YEAR if fht['status'] == 'valid' else np.nan,
            'zscore_12m': zscore,
            'zscore_3m': zscore_3m,
            'momentum_signal': momentum_signal,
            'momentum_series': str(momentum_series), # Stored as string to easily parse from Parquet if needed
            'daily_vol_bps': daily_vol,
            'residual_dv01': compute_residual_dv01(legs, dv01s),
            'fit_rmse_bps': rmse,
            'hurst': hurst,
            'adf_pvalue': adf_pvalue,
            'recommended_direction': direction,
            'metrics_dict': metrics # store temporarily for scoring later
        }
        
        results.append(res)
        
    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res
        
    # --- POST-PROCESSING: Neighborhood Momentum & Scoring ---
    neighborhood_momentums = []
    confidence_scores = []
    
    for idx, row in df_res.iterrows():
        # Trouver les voisins (trades partageant au moins 1 ISIN)
        my_isins = set([row['bond1_isin'], row['bond2_isin'], row['bond3_isin']])
        my_isins.discard(None)
        
        neighbors_mom = []
        for jdx, other_row in df_res.iterrows():
            if idx == jdx:
                continue
            other_isins = set([other_row['bond1_isin'], other_row['bond2_isin'], other_row['bond3_isin']])
            other_isins.discard(None)
            
            if len(my_isins.intersection(other_isins)) > 0:
                if pd.notna(other_row['momentum_signal']):
                    neighbors_mom.append(other_row['momentum_signal'])
                    
        nh_mom = np.mean(neighbors_mom) if len(neighbors_mom) > 0 else np.nan
        neighborhood_momentums.append(nh_mom)
        
        # Calculate final score
        metrics = row['metrics_dict'].copy()
        metrics['neighborhood_momentum'] = nh_mom
        metrics['target_gap_bps'] = row['target_gap_bps'] # Needed for penalty direction
        score = calculate_confidence_score(metrics, SCORE_WEIGHTS)
        confidence_scores.append(score)
        
    df_res['neighborhood_momentum'] = neighborhood_momentums
    df_res['confidence_score'] = confidence_scores
    df_res = df_res.drop(columns=['metrics_dict'])
    
    return df_res
