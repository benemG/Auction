import json
import pandas as pd
import numpy as np
from .hedge_ratio import (compute_spread_hedge_ratio, compute_fly_hedge_ratio, compute_residual_dv01,
                          parse_coupon, approx_modified_duration)
from .momentum import compute_ewmac, compute_ewmac_series
from .mean_reversion import MeanReversionModel, compute_hurst_exponent, compute_adf_pvalue
from .scoring import calculate_confidence_score
from .config import TRADING_DAYS_IN_YEAR, SCORE_WEIGHTS, METRIC_TO_BPS

def generate_opportunities_for_issuer(df_market: pd.DataFrame, df_fitted: pd.DataFrame, trades_def: list, as_of_date: pd.Timestamp, metric: str = 'asset_swap_spread') -> pd.DataFrame:
    """
    Évalue une liste de définitions de trades et produit le Parquet final.
    """
    import os
    from .config import BASE_DIR
    
    results = []
    scale = METRIC_TO_BPS[metric]
    id_suffix = '' if metric == 'asset_swap_spread' else f" [{metric.upper()}]"  # garde les trade_id uniques par mesure
    
    # Restreindre aux données historiques et dédupliquer
    df_hist = df_market[df_market['date'] <= as_of_date].drop_duplicates(subset=['date', 'isin']).copy()
    df_hist_fit = df_fitted[df_fitted['date'] <= as_of_date].drop_duplicates(subset=['date', 'isin']).copy()
    
    # Pivoter pour un accès rapide
    df_hist_pivot = df_hist.pivot(index='date', columns='isin')
    df_fit_pivot = df_hist_fit.pivot(index='date', columns='isin')
    
    current_market = df_hist[df_hist['date'] == as_of_date].set_index('isin')
    current_fit = df_hist_fit[df_hist_fit['date'] == as_of_date].set_index('isin')

    # Duration et DV01 absents des données : approximation en forme fermée (coupon lu dans le nom)
    if 'modified_duration' in current_market.columns and 'maturity' in current_market.columns:
        ttm_mkt = (pd.to_datetime(current_market['maturity']) - pd.to_datetime(as_of_date)).dt.days / 365.25
        approx = [approx_modified_duration(y, parse_coupon(n), t)
                  for y, n, t in zip(current_market['ytm'], current_market['bond_name'], ttm_mkt)]
        current_market['modified_duration'] = current_market['modified_duration'].fillna(pd.Series(approx, index=current_market.index))
        if 'dv01' in current_market.columns:
            # €/bp pour 1M de nominal = prix × duration modifiée
            current_market['dv01'] = current_market['dv01'].fillna(current_market['modified_duration'] * current_market['price_clean'])
    
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
        
    # Courbe de yield fittée par émetteur (le roll ne doit pas mélanger deux courbes)
    curves = {iss: (g['ttm_years'].values, g['ytm_fitted'].values)
              for iss, g in current_fit.dropna(subset=['ttm_years', 'ytm_fitted']).sort_values('ttm_years').groupby('issuer')}

    for trade in trades_def:
        
        isins = [trade[k] for k in ('bond1_isin', 'bond2_isin', 'bond3_isin') if trade.get(k)]
        ttm_list = [trade.get(k) for k in ('ttm_a', 'ttm_b', 'ttm_c')][:len(isins)]
        if trade['trade_type'] not in ('spread', 'fly') or any(i not in current_market.index for i in isins):
            continue

        dv01s = [current_market.loc[i, 'dv01'] if 'dv01' in current_market.columns else 1.0 for i in isins]
        if trade['trade_type'] == 'spread':
            legs = list(compute_spread_hedge_ratio(*dv01s))
            sw = [1.0, -1.0]
        else:
            legs = list(compute_fly_hedge_ratio(*dv01s, *ttm_list))
            if any(np.isnan(d) for d in dv01s):
                sw = [legs[0], -1.0, legs[2]]  # poids de maturité des ailes, belly à -1
            else:
                # Ailes pondérées en DV01 (belly à -1) : le spread suivi correspond au P&L de la position couverte
                sw = [l * d / (-legs[1] * dv01s[1]) for l, d in zip(legs, dv01s)]
        # legs = nominaux (DV01-neutres) pour le sizing ; sw = poids qui construisent le spread en bps

        try:
            ts_market = scale * sum(w * df_hist_pivot[metric][i] for w, i in zip(sw, isins))
            ts_fitted = scale * sum(w * df_fit_pivot[f"{metric}_fitted"][i] for w, i in zip(sw, isins))
        except KeyError:
            continue

        current_spread = ts_market.iloc[-1]
        current_target = ts_fitted.iloc[-1]
        rmse = scale * current_fit.reindex(isins)[f"{metric}_rmse"].mean() if f"{metric}_rmse" in current_fit.columns else np.nan

        # Filtre : Minimum 12 mois d'historique
        if len(ts_market.dropna()) < 252:
            continue

        # Calcul du spread "Spread vs Fitted" (SvF) pour l'historique
        ts_svf = ts_market - ts_fitted
        current_svf = current_spread - current_target
        ts_svf_12m = ts_svf.dropna().tail(252)

        # Calibration OU : une seule cible (moyenne OU, sinon moyenne 12M) pour z-score, gap et direction
        ou = MeanReversionModel()
        state = ou.calibrate_ou_model(ts_svf_12m, dt=1.0/TRADING_DAYS_IN_YEAR)
        mu = state['mu'] if state['status'] == 'valid' else ts_svf_12m.mean()
        if np.isnan(current_svf) or np.isnan(mu):
            continue
        fht = ou.compute_first_hitting_time(current_svf)

        if len(ts_svf_12m) > 10:
            daily_vol = ts_svf_12m.std()
            zscore = (current_svf - mu) / daily_vol
        else:
            zscore = daily_vol = np.nan

        ts_svf_3m = ts_svf.dropna().tail(63)
        zscore_3m = (current_svf - ts_svf_3m.mean()) / ts_svf_3m.std() if len(ts_svf_3m) > 10 else np.nan

        # Momentum EWMAC
        ts_clean = ts_svf.dropna()
        momentum_signal = compute_ewmac(ts_clean)
        try:
            momentum_series = compute_ewmac_series(ts_clean).tolist()
        except Exception:
            momentum_series = []

        hurst = compute_hurst_exponent(ts_svf_12m)
        adf_pvalue = compute_adf_pvalue(ts_svf_12m)

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

        # Cible = courbe P-spline + écart moyen (mu) ; gap > 0 : on attend un élargissement du spread
        target = current_target + mu
        target_gap = target - current_spread
        direction = 'long_spread' if target_gap > 0 else 'short_spread'
        dir_sign = np.sign(target_gap)

        # --- CARRY & ROLLDOWN (3M), tout en bps de yield-équivalent comme target_gap ---
        # carry% / duration = bps de yield équivalent ; le roll (Δ yield courbe) est déjà en bps.
        # Position en bonds = -dir_sign * sw (long_spread = short bond1 / long bond2).
        horizon_years = 0.25
        net_carry_roll_3m_bps = 0.0
        total_bid_ask_cost_bps = 0.0
        leg_issuers, leg_repo_sens = [], []  # pour apply_repo_spreads : spread de repo appliqué à l'affichage seulement
        for w, isin, ttm in zip(sw, isins, ttm_list):
            bid_ask_spread = current_market.loc[isin, 'bid_ask_spread'] if 'bid_ask_spread' in current_market.columns else 1.0
            total_bid_ask_cost_bps += abs(w) * (bid_ask_spread / 2.0)

            repo_rate = current_market.loc[isin, 'repo_rate'] if 'repo_rate' in current_market.columns else ester_rate
            dur = current_market.loc[isin, 'modified_duration'] if 'modified_duration' in current_market.columns else np.nan
            if np.isnan(dur):
                dur = ttm  # shortcut: duration ≈ maturité (zéro-coupon), surestime la duration des coupons
            carry_bps = (current_market.loc[isin, 'ytm'] - repo_rate) * horizon_years * 100 / dur

            roll_bps = 0.0
            x, y = curves.get(current_fit.loc[isin, 'issuer'], (None, None))
            if x is not None and ttm > horizon_years:
                # shortcut: np.interp extrapole à plat sous la 1re maturité (roll sous-estimé), à raffiner si besoin
                roll_bps = (np.interp(ttm, x, y) - np.interp(ttm - horizon_years, x, y)) * 100

            net_carry_roll_3m_bps += -dir_sign * w * (carry_bps + roll_bps)
            leg_issuers.append(str(current_market.loc[isin, 'issuer']) if 'issuer' in current_market.columns else '')
            # d(gain net)/d(spread repo en bp) : carry_bps = (ytm - repo)·h·100/dur, donc -h/dur par bp, signé par la position
            leg_repo_sens.append(float(dir_sign * w * horizon_years / dur))

        net_edge_bps = abs(target_gap) + net_carry_roll_3m_bps - total_bid_ask_cost_bps  # >0 : gain net attendu
        carry_roll_adjusted_gap = dir_sign * net_edge_bps
        net_edge_ratio = net_edge_bps / daily_vol if daily_vol > 0 else np.nan  # gain net en écarts-types du spread

        res = {
            'as_of_date': as_of_date,
            'trade_id': trade['trade_id'] + id_suffix,
            'trade_type': trade['trade_type'],
            'metric': metric,
            'bond1_isin': trade['bond1_isin'],
            'bond2_isin': trade['bond2_isin'],
            'bond3_isin': trade['bond3_isin'],
            'bond1_name': trade['bond1_name'],
            'bond2_name': trade['bond2_name'],
            'bond3_name': trade['bond3_name'],
            'leg_weights': str([float(w) for w in sw]),  # floats Python : relu par ast.literal_eval dans Trade Explorer
            'hedge_nominals': str([float(n) for n in legs]),
            'spread_market_bps': current_spread,
            'spread_fitted_bps': current_target,
            'spread_target_bps': target,
            'target_gap_bps': target_gap,
            'net_carry_roll_3m_bps': net_carry_roll_3m_bps,
            'leg_issuers': json.dumps(leg_issuers),
            'leg_repo_sens': json.dumps(leg_repo_sens),
            'bid_ask_cost_bps': total_bid_ask_cost_bps,
            'carry_roll_adjusted_gap': carry_roll_adjusted_gap,
            'net_edge_ratio': net_edge_ratio,
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
        
    # Deux ISIN peuvent partager le même nom : on ajoute les ISIN aux identifiants en double pour les garder uniques
    dup = df_res['trade_id'].duplicated(keep=False)
    if dup.any():
        tags = df_res.loc[dup, ['bond1_isin', 'bond2_isin', 'bond3_isin']].apply(lambda r: '/'.join(i for i in r if pd.notna(i)), axis=1)
        df_res.loc[dup, 'trade_id'] += ' (' + tags + ')'

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


def apply_repo_spreads(df: pd.DataFrame, spreads: dict) -> pd.DataFrame:
    """
    Ajoute à l'affichage l'effet d'un spread de repo (bp vs ESTER) par émetteur sur le carry et le gain net,
    à partir des sensibilités stockées par trade. Aucun recalcul d'historique : la faisabilité financière
    ne compte qu'à la date scannée. Sans effet sur les parquets antérieurs (colonnes absentes) ou si tout est nul.
    """
    if df.empty or 'leg_repo_sens' not in df.columns or not any(spreads.values()):
        return df
    default = spreads.get('default', 0.0)
    # Dates générées avant cette colonne : NaN dans un schéma Parquet unifié, pas d'ajustement
    adj = np.array([sum(s * spreads.get(i, default) for s, i in zip(json.loads(sens), json.loads(iss)))
                    if isinstance(sens, str) else 0.0
                    for sens, iss in zip(df['leg_repo_sens'], df['leg_issuers'])])
    df = df.copy()
    df['net_carry_roll_3m_bps'] += adj
    df['carry_roll_adjusted_gap'] += np.sign(df['target_gap_bps']) * adj
    df['net_edge_ratio'] += adj / df['daily_vol_bps']
    return df
