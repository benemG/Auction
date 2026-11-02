import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.trade_definitions import build_rv_graph
from src.clustering import issuer_distance
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def _spread(hl):
    return pd.DataFrame([dict(bond1_isin='A', bond2_isin='B', bond1_name='X 1', bond2_name='X 2',
                              zscore_12m=1.5, half_life_days=hl, hurst=0.45)])


def test_edge_without_mr_stats_keeps_out_of_range_half_life():
    assert build_rv_graph(_spread(hl=200), require_mr_stats=False).number_of_edges() == 1
    assert build_rv_graph(_spread(hl=200), require_mr_stats=True).number_of_edges() == 0


def test_edge_still_requires_min_zscore():
    df = _spread(hl=20).assign(zscore_12m=0.5)
    assert build_rv_graph(df, require_mr_stats=False).number_of_edges() == 0


def test_maxclust_cut_gives_two_groups():
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(size=252))
    piv = pd.DataFrame({'a': base, 'b': base + 1, 'c': -base + 50, 'd': -base + 52})
    Z = linkage(squareform(issuer_distance(piv), checks=False), method='ward')
    ids = fcluster(Z, t=2, criterion='maxclust')
    assert len(set(ids)) == 2 and ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
