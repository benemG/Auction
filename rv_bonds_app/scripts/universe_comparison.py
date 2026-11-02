"""
Compare l'univers de spreads 2 jambes de l'app (intra-émetteur + inter-émetteurs d'un même cluster)
avec celui d'un filtre « corrélation + cointégration dans les deux sens » (article pairs trading).
Lecture seule : n'écrit rien dans data/.

Usage : python scripts/universe_comparison.py [--metric asset_swap_spread|ytm] [--window 252] [--max-ttm-diff 2.0]
"""
import sys
import json
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (MARKET_DATA_DIR, FITTED_CURVES_DIR, TRADING_DAYS_IN_YEAR, MIN_Z_SCORE_EDGE,
                        MIN_HALF_LIFE_EDGE, MAX_HALF_LIFE_EDGE, MAX_HURST_EDGE)
from src.data_io import read_fitted_curves
from src.mean_reversion import MeanReversionModel, compute_hurst_exponent, compute_adf_pvalue
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import coint
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from src.clustering import issuer_distance
from src.trade_definitions import pair_cointegration_pvalues, select_pvalues

warnings.filterwarnings("ignore")
CORR_MIN = 0.70
P_MAX = 0.05
HL_MIN, HL_MAX = 5, 60


def load_universe(metric, window, suffix='market', end=None):
    issuers = [d.name.split('=')[1] for d in MARKET_DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('issuer=')]
    frames = []
    for iss in issuers:
        df = read_fitted_curves(iss)
        if df.empty:
            continue
        df['issuer'] = iss
        frames.append(df[['date', 'isin', 'issuer', 'bond_name', 'ttm_years', f'{metric}_{suffix}'] +
                         (['inflation_linked_indicator'] if 'inflation_linked_indicator' in df.columns else [])])
    # Les émetteurs n'ont pas tous la même dernière date : on s'aligne sur la dernière date commune
    last = min(f['date'].max() for f in frames)
    if end is not None:
        last = min(last, pd.Timestamp(end))
    df = pd.concat(frames, ignore_index=True)
    df = df[df['date'] <= last]
    cur = df[df['date'] == last].drop_duplicates('isin')
    cur = cur[cur['ttm_years'] >= 1.0]
    if 'inflation_linked_indicator' in cur.columns:
        cur = cur[cur['inflation_linked_indicator'].astype(str).str.upper() != 'Y']
    px = df.drop_duplicates(['date', 'isin']).pivot(index='date', columns='isin', values=f'{metric}_{suffix}').sort_index()
    px = px.tail(window)
    px = px.dropna(axis=1, thresh=window)  # historique complet sur la fenêtre
    px = px.ffill().dropna(axis=1)
    cur = cur[cur['isin'].isin(px.columns)].set_index('isin')
    return px[cur.index], cur, last


def candidate_pairs(cur, max_ttm_diff):
    isins = cur.index.to_numpy()
    ttm = cur['ttm_years'].to_numpy()
    iss = cur['issuer'].to_numpy()
    i, j = np.triu_indices(len(isins), k=1)
    keep = np.abs(ttm[i] - ttm[j]) <= max_ttm_diff
    i, j = i[keep], j[keep]
    return pd.DataFrame({'a': isins[i], 'b': isins[j], 'iss_a': iss[i], 'iss_b': iss[j]})


def app_universe_flag(pairs, clusters):
    cl = {}
    for k, c in enumerate(clusters):
        for iss in c:
            cl[iss] = k
    same_iss = pairs['iss_a'] == pairs['iss_b']
    same_cl = pairs['iss_a'].map(cl).notna() & (pairs['iss_a'].map(cl) == pairs['iss_b'].map(cl))
    return same_iss | same_cl, same_iss


def ou_hl_days(spread):
    m = MeanReversionModel()
    res = m.calibrate_ou_model(spread, 1.0 / TRADING_DAYS_IN_YEAR)
    if res['status'] != 'valid':
        return np.nan
    hl = m.compute_half_life()
    return hl * TRADING_DAYS_IN_YEAR if np.isfinite(hl) else np.nan


def pair_stats(px, a, b):
    ya, yb = px[a], px[b]
    p_ab = coint(ya, yb)[1]
    p_ba = coint(yb, ya)[1]
    s = ya - yb
    return p_ab, p_ba, ou_hl_days(s), compute_hurst_exponent(s)


def evaluate_persistence(args):
    """
    Walk-forward : sélection sur `window` jours se terminant `oos` jours avant la fin, puis mesure de la
    stabilité du spread A-B sur les `oos` derniers jours (dérive de la moyenne < 1 écart-type de formation
    et |z| maximal < 3). Compare les variantes du filtre et la base de l'app.
    """
    px, cur, last = load_universe(args.metric, args.window + args.oos)
    form_full, test = px.iloc[:-args.oos], px.iloc[-args.oos:]
    print(f"Walk-forward : formation jusqu'au {form_full.index[-1].date()} ({args.window} j), test {args.oos} j jusqu'au {last.date()} | titres : {px.shape[1]}")
    clusters = json.load(open(MARKET_DATA_DIR.parent / 'clusters.json'))
    pairs = candidate_pairs(cur, args.max_ttm_diff)
    in_a, _ = app_universe_flag(pairs, clusters)
    pairs['A'] = in_a.to_numpy()

    out = {}
    for w in (args.window, args.window // 2):
        f = form_full.tail(w)
        out[w] = np.array([pair_cointegration_pvalues(f, a, b, int(w * 0.9)) for a, b in zip(pairs['a'], pairs['b'])])
        print(f"  p-values calculées sur {w} j")
    w1, w2 = args.window, args.window // 2

    f1 = form_full.tail(w1)
    mu = (f1[pairs['a']].to_numpy() - f1[pairs['b']].to_numpy()).mean(axis=0)
    sd = (f1[pairs['a']].to_numpy() - f1[pairs['b']].to_numpy()).std(axis=0)
    st = test[pairs['a']].to_numpy() - test[pairs['b']].to_numpy()
    drift = np.abs(st.mean(axis=0) - mu) / sd
    maxz = np.abs(st - mu).max(axis=0) / sd
    pairs['stable'] = (drift < 1) & (maxz < 3)
    pairs['drift'] = drift

    P = 0.05
    v = {}
    v['toutes paires candidates'] = np.ones(len(pairs), dtype=bool)
    v['A (app, sans filtre)'] = pairs['A'].to_numpy()
    v['EG brut p<0.05, 1 fenêtre'] = select_pvalues(out[w1][:, 0], out[w1][:, 1], P, None, False)
    v['+ 2 fenêtres'] = v['EG brut p<0.05, 1 fenêtre'] & select_pvalues(out[w2][:, 0], out[w2][:, 1], P, None, False)
    v['FDR BH 5 %, 1 fenêtre'] = select_pvalues(out[w1][:, 0], out[w1][:, 1], P, 0.05, False)
    v['FDR + 2 fenêtres'] = v['FDR BH 5 %, 1 fenêtre'] & select_pvalues(out[w2][:, 0], out[w2][:, 1], P, 0.05, False)
    v['FDR + 2 fenêtres + ADF spread'] = (select_pvalues(out[w1][:, 0], out[w1][:, 1], P, 0.05, True)
                                          & select_pvalues(out[w2][:, 0], out[w2][:, 1], P, 0.05, True))
    v['A ∩ FDR + 2 fenêtres + ADF'] = v['FDR + 2 fenêtres + ADF spread'] & pairs['A'].to_numpy()

    rows = []
    for name, m in v.items():
        sub = pairs[m]
        rows.append((name, int(m.sum()), sub['stable'].mean() if len(sub) else np.nan, sub['drift'].median() if len(sub) else np.nan))
    print(f"\n{'variante':36s} {'n':>7s} {'stables %':>10s} {'dérive méd.':>12s}")
    for name, n, st_, dr in rows:
        print(f"{name:36s} {n:7d} {st_ * 100:10.1f} {dr:12.2f}")

def _nw_t(x, lag):
    """t-stat de la moyenne avec correction de Newey-West (dates qui se chevauchent)."""
    x = np.asarray(x, dtype=float)
    n, e = len(x), x - x.mean()
    var = (e @ e) / n + 2 * sum((1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n for l in range(1, min(lag, n - 1) + 1))
    return x.mean() / np.sqrt(var / n)


def ic_table(obs, masks, horizons, step=21):
    """IC de rang moyen par date entre -z et la variation future du spread, avec t-stat (Newey-West) et taux de bon sens."""
    rows = []
    for name, m in masks.items():
        sub = obs[m]
        for h in horizons:
            ic = sub.groupby('date').apply(lambda g: g['z'].rank().corr(-g[f'd{h}'].rank()) if len(g) > 2 else np.nan).dropna()
            hit = (np.sign(sub[f'd{h}']) == -np.sign(sub['z'])).mean()
            rows.append((name, h, len(sub) / max(1, sub['date'].nunique()), ic.mean(), _nw_t(ic, -(-h // step) - 1), hit))
    return pd.DataFrame(rows, columns=['filtre', 'h', 'paires/date', 'IC', 't', 'bon sens'])


def _ic_obs(t, S, A, B, window, horizons, coint_on, light=False):
    """Observations à la date t : z, stats et variations futures des spreads SvF (S) ; A, B = niveaux marché des jambes."""
    keep, p = np.arange(S.shape[1]), None
    if coint_on:  # Engle-Granger dans les deux sens sur les niveaux marché, à date (pas de regard vers le futur)
        a, b = A[t - window + 1:t + 1], B[t - window + 1:t + 1]
        p = np.array([max(coint(a[:, k], b[:, k])[1], coint(b[:, k], a[:, k])[1]) for k in keep])
        if not light:  # toutes les paires : on garde la p-value au lieu de filtrer
            keep = keep[p < P_MAX]
    hist = pd.DataFrame(S[t - window + 1:t + 1, keep])
    sd = hist.std().to_numpy()
    d = pd.DataFrame([(np.nan, np.nan, compute_adf_pvalue(hist[k])) if light else
                      (ou_hl_days(hist[k]), compute_hurst_exponent(hist[k]), compute_adf_pvalue(hist[k])) for k in hist.columns],
                     columns=['hl', 'hurst', 'adf'])
    d['pair'], d['t'], d['z'], d['sd'] = keep, t, (S[t, keep] - hist.mean().to_numpy()) / sd, sd
    for h in horizons:
        d[f'd{h}'] = S[t + h, keep] - S[t, keep]
    if p is not None and light:
        d['p_eg'] = p
    return d


def issuer_cophenetic(dates, window, threshold=1.0):
    """
    À chaque date : arbre de Ward de l'app (ASW ajusté 5-10 ans par émetteur, `window` jours passés, sans regard
    vers le futur) et distance cophénétique entre émetteurs. Même cluster que l'app <=> distance <= threshold.
    """
    df = pq.ParquetDataset(str(FITTED_CURVES_DIR)).read(columns=['date', 'issuer', 'ttm_years', 'asset_swap_spread_fitted']).to_pandas()
    df = df[(df['ttm_years'] >= 5.0) & (df['ttm_years'] <= 10.0)].assign(issuer=lambda x: x['issuer'].astype(str))
    piv = df.groupby(['date', 'issuer'])['asset_swap_spread_fitted'].mean().unstack().sort_index()
    out = {}
    for d in dates:
        p = piv[piv.index <= d].tail(window).ffill().dropna(axis=1)
        coph = squareform(cophenet(linkage(squareform(issuer_distance(p), checks=False), method='ward')))
        out[d] = pd.DataFrame(coph, index=p.columns, columns=p.columns)
    return out


def cluster_ic(obs, pairs, window, horizons, step, threshold=1.0):
    """IC par groupe (même émetteur / même cluster / clusters différents) et par distance cophénétique."""
    coph = issuer_cophenetic(sorted(obs['date'].unique()), window)
    ia, ib = pairs['iss_a'].to_numpy()[obs['pair']], pairs['iss_b'].to_numpy()[obs['pair']]
    obs['dist'] = [0.0 if a == b else coph[d].at[a, b] if a in coph[d].index and b in coph[d].index else np.nan
                   for d, a, b in zip(obs['date'], ia, ib)]
    intra, inter = ia == ib, (ia != ib) & obs['dist'].notna().to_numpy()
    same_cl = inter & (obs['dist'] <= threshold).to_numpy()
    adf = (obs['adf'] < 0.05).to_numpy()
    fmt = lambda x: f"{x:.3f}"
    groups = {'même émetteur': intra, 'même cluster': same_cl, 'clusters différents': inter & ~same_cl,
              'univers A (app)': intra | same_cl, 'toutes': intra | inter}
    print("\n--- Étape 1 : IC par groupe (et avec ADF p < 0,05) ---")
    masks = {**groups, **{f'{k} + ADF': m & adf for k, m in groups.items()}}
    print(ic_table(obs, masks, horizons, step).to_string(index=False, float_format=fmt))
    print("\n--- Étape 2a : paires inter-émetteurs par quintile de distance cophénétique ---")
    q = pd.qcut(obs.loc[inter, 'dist'], 5)
    print(ic_table(obs, {f'Q{k + 1} {iv}': inter & (obs.index.isin(q.index[q.cat.codes == k]))
                         for k, iv in enumerate(q.cat.categories)}, horizons, step).to_string(index=False, float_format=fmt))
    print("\n--- Étape 2b : même émetteur + inter-émetteurs à distance <= d* ---")
    print(ic_table(obs, {f'd* = {d}': intra | (inter & (obs['dist'] <= d).to_numpy()) for d in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf)},
                   horizons, step).to_string(index=False, float_format=fmt))


def evaluate_ic(args, horizons=(5, 20, 60), step=21):
    """
    Rejoue le signal des spreads de l'univers A sur l'historique du résidu SvF (marché - courbe ajustée) :
    tous les `step` jours (chaque jour sur les `args.dates` dernières dates si fourni), z-score et filtres sur
    `window` jours, puis variation du spread SvF à h jours. `args.coint` restreint, à chaque date, aux paires
    cointégrées dans les deux sens.
    shortcut: titres vivants à la fin avec historique complet (biais de survie), z sur la moyenne
    empirique et non sur le μ de l'OU ; à revoir si l'IC sert à calibrer des seuils fins.
    """
    if args.dates:
        step = 1
    span = args.window + (args.dates or args.window) + max(horizons)
    mkt, cur, last = load_universe(args.metric, span, end=args.end)
    fit, _, _ = load_universe(args.metric, span, 'fitted', end=args.end)
    cols = mkt.columns.intersection(fit.columns)
    svf = mkt[cols] - fit[cols]
    clusters = json.load(open(MARKET_DATA_DIR.parent / 'clusters.json'))
    pairs = candidate_pairs(cur.loc[cols], args.max_ttm_diff)
    if not args.all_pairs:
        pairs = pairs[app_universe_flag(pairs, clusters)[0].to_numpy()].reset_index(drop=True)
    S = svf[pairs['a']].to_numpy() - svf[pairs['b']].to_numpy()
    A, B = mkt[pairs['a']].to_numpy(), mkt[pairs['b']].to_numpy()
    print(f"IC : {len(pairs)} spreads {'candidats' if args.all_pairs else "de l'univers A"}, {len(S)} jours jusqu'au {last.date()}, métrique {args.metric}"
          f"{', cointégrés seulement' if args.coint else ''}")

    ts = range(len(S) - max(horizons) - args.dates if args.dates else args.window, len(S) - max(horizons), step)
    obs = Parallel(n_jobs=-1, verbose=5)(delayed(_ic_obs)(t, S, A, B, args.window, horizons, args.coint, args.all_pairs) for t in ts)
    obs = pd.concat(obs, ignore_index=True).dropna(subset=['z'])
    obs['date'] = mkt.index[obs['t']]
    print(f"{obs['date'].nunique()} dates, {len(obs) / obs['date'].nunique():.0f} paires/date en moyenne")
    if args.all_pairs:
        cluster_ic(obs, pairs, args.window, horizons, step)
        if args.out:
            obs.to_parquet(args.out)
        return obs

    zok = obs['z'].abs() >= MIN_Z_SCORE_EDGE
    hlok = (obs['hl'] > MIN_HALF_LIFE_EDGE) & (obs['hl'] <= MAX_HALF_LIFE_EDGE)
    hok = obs['hurst'] < MAX_HURST_EDGE
    masks = {
        'tous': np.ones(len(obs), dtype=bool),
        '|z| >= 1': zok,
        'demi-vie 5-60 j': hlok,
        'Hurst < 0,40': hok,
        'ADF p < 0,05': obs['adf'] < 0.05,
        'arête app (z + demi-vie + Hurst)': zok & hlok & hok,
    }
    print(ic_table(obs, masks, horizons, step).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out:
        obs.to_parquet(args.out)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='asset_swap_spread')
    ap.add_argument('--window', type=int, default=252)
    ap.add_argument('--max-ttm-diff', type=float, default=2.0)
    ap.add_argument('--oos', type=int, default=0, help="Jours de test hors échantillon : active l'évaluation walk-forward")
    ap.add_argument('--ic', action='store_true', help="Mesure l'IC du z-score et des filtres de l'app")
    ap.add_argument('--dates', type=int, default=0, help="--ic : évalue chaque jour sur les N dernières dates")
    ap.add_argument('--coint', action='store_true', help="--ic : seulement les paires cointégrées (EG deux sens) à date")
    ap.add_argument('--all-pairs', action='store_true', help="--ic : toutes les paires candidates, IC par groupe et distance de cluster")
    ap.add_argument('--end', default=None, help="--ic : dernière date de données utilisée (ex. 2023-03-31)")
    ap.add_argument('--out', default=None, help="CSV de sortie (optionnel)")
    args = ap.parse_args()
    if args.ic:
        evaluate_ic(args)
        return
    if args.oos > 0:
        evaluate_persistence(args)
        return

    px, cur, last = load_universe(args.metric, args.window)
    clusters = json.load(open(MARKET_DATA_DIR.parent / 'clusters.json'))
    print(f"Date de référence : {last.date()} | métrique : {args.metric} | fenêtre : {len(px)} j | titres : {px.shape[1]}")
    print(f"Clusters (clusters.json) : {clusters}")

    pairs = candidate_pairs(cur, args.max_ttm_diff)
    in_a, same_iss = app_universe_flag(pairs, clusters)
    pairs['A'] = in_a.to_numpy()
    pairs['intra'] = same_iss.to_numpy()

    corr = px.corr().to_numpy()
    idx = {k: n for n, k in enumerate(px.columns)}
    pairs['corr'] = [corr[idx[a], idx[b]] for a, b in zip(pairs['a'], pairs['b'])]
    pairs['corr_ok'] = pairs['corr'] > CORR_MIN

    # Cointégration : toutes les paires de A et toutes les paires corrélées de B
    todo = pairs[pairs['A'] | pairs['corr_ok']].copy()
    print(f"Paires candidates (TTM <= {args.max_ttm_diff}a) : {len(pairs)} | à tester en cointégration : {len(todo)}")
    rows = [pair_stats(px, a, b) for a, b in zip(todo['a'], todo['b'])]
    todo[['p_ab', 'p_ba', 'hl_days', 'hurst']] = pd.DataFrame(rows, index=todo.index)
    pairs = pairs.join(todo[['p_ab', 'p_ba', 'hl_days', 'hurst']])

    pairs['coint_both'] = (pairs['p_ab'] < P_MAX) & (pairs['p_ba'] < P_MAX)
    pairs['hl_ok'] = pairs['hl_days'].between(HL_MIN, HL_MAX, inclusive='neither')
    pairs['B_corr'] = pairs['corr_ok']
    pairs['B_coint'] = pairs['corr_ok'] & pairs['coint_both']
    pairs['B'] = pairs['B_coint'] & pairs['hl_ok']
    pairs['B_hurst'] = pairs['B'] & (pairs['hurst'] < 0.5)

    A, B = pairs['A'], pairs['B']
    inter = ~pairs['intra']
    print("\n=== Entonnoir B (article) ===")
    for k in ['B_corr', 'B_coint', 'B', 'B_hurst']:
        print(f"  {k:8s}: {int(pairs[k].sum()):7d}   (dont inter-émetteurs : {int((pairs[k] & inter).sum())})")
    print(f"  attendu par hasard à p<{P_MAX} (une direction) : {P_MAX * pairs['corr_ok'].sum():.0f}")

    print("\n=== Univers A (app) ===")
    print(f"  A total : {int(A.sum())} (intra : {int((A & pairs['intra']).sum())}, inter-cluster : {int((A & inter).sum())})")
    print(f"  A qui passe coint 2 sens : {int((A & pairs['coint_both']).sum())} | + demi-vie 5-60 : {int((A & pairs['coint_both'] & pairs['hl_ok']).sum())}")

    both, a_only, b_only = int((A & B).sum()), int((A & ~B).sum()), int((~A & B).sum())
    print("\n=== Recouvrement (titre) ===")
    print(f"  A∩B : {both} | A\\B : {a_only} | B\\A : {b_only} | Jaccard : {both / max(1, both + a_only + b_only):.3f}")
    print(f"  Part de B dans A : {both / max(1, int(B.sum())):.1%} | part de A dans B : {both / max(1, int(A.sum())):.1%}")

    # Niveau paire d'émetteurs
    def issuer_pairs(mask):
        s = pairs[mask]
        return set(tuple(sorted(t)) for t in zip(s['iss_a'], s['iss_b']))
    ia, ib = issuer_pairs(A), issuer_pairs(B)
    ia_x, ib_x = {p for p in ia if p[0] != p[1]}, {p for p in ib if p[0] != p[1]}
    print("\n=== Recouvrement (paires d'émetteurs inter-émetteurs) ===")
    print(f"  A : {len(ia_x)} | B : {len(ib_x)} | A∩B : {len(ia_x & ib_x)} | Jaccard : {len(ia_x & ib_x) / max(1, len(ia_x | ib_x)):.3f}")
    print(f"  B hors clusters (exemples) : {sorted(ib_x - ia_x)[:15]}")

    print("\n=== Qualité statistique ===")
    for name, mask in [('A\\B', A & ~B), ('A∩B', A & B), ('B\\A', ~A & B)]:
        s = pairs[mask]
        print(f"  {name:4s} n={len(s):6d} | corr med {s['corr'].median():.2f} | p_EG max-dir med {np.fmax(s['p_ab'], s['p_ba']).median():.3f} "
              f"| HL med {s['hl_days'].median():.0f}j | Hurst med {s['hurst'].median():.2f}")
    sa = pairs[A]
    print(f"  A corr>0.7 : {sa['corr_ok'].mean():.1%} | A coint 2 sens : {sa['coint_both'].mean():.1%}")

    print("\n=== Corrélées mais non cointégrées (équivalent BBVA/SAN) ===")
    top = pairs[pairs['corr_ok'] & ~pairs['coint_both']].sort_values('corr', ascending=False).head(5)
    print(top[['a', 'b', 'iss_a', 'iss_b', 'corr', 'p_ab', 'p_ba', 'A']].to_string(index=False))

    print("\n=== Émetteurs sans pair inter-cluster dans A mais avec paires B ===")
    cl_members = {i for c in clusters if len(c) > 1 for i in c}
    solo = sorted({i for p in ib_x for i in p} - cl_members)
    print(f"  {solo}")

    # Stabilité de B sur la moitié la plus récente
    half = len(px) // 2
    px_half = px.tail(half)
    sub = pairs[pairs['B_corr']]
    ok = [(coint(px_half[a], px_half[b])[1] < P_MAX) and (coint(px_half[b], px_half[a])[1] < P_MAX)
          for a, b in zip(sub['a'], sub['b'])]
    sub = sub.assign(coint_half=ok)
    print("\n=== Stabilité (cointégration 2 sens sur la 2e moitié de fenêtre) ===")
    print(f"  B_coint plein-fenêtre qui reste cointégré sur {half} j : "
          f"{(sub['coint_half'] & sub['B_coint']).sum()}/{int(sub['B_coint'].sum())}")

    if args.out:
        pairs.to_csv(args.out, index=False)
        print(f"\nDétail écrit dans {args.out}")


if __name__ == '__main__':
    main()
