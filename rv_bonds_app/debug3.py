import pandas as pd
from src.data_io import read_market_data, read_fitted_curves, read_opportunities
import ast

df = read_opportunities('2026-02-17')
dt = pd.to_datetime('2026-02-17')

for i in range(len(df)):
    trade = df.iloc[i]
    issuer = trade['issuer']
    isins = [trade.get(f'bond{j}_isin') for j in range(1, 5) if trade.get(f'bond{j}_isin') is not None]
    weights = ast.literal_eval(trade['leg_weights']) if isinstance(trade['leg_weights'], str) else trade['leg_weights']
    metric = trade['metric']

    df_m = read_market_data(issuer)
    df_f = read_fitted_curves(issuer)

    df_m = df_m[df_m['date'] <= dt]
    df_f = df_f[df_f['date'] <= dt]

    dates = df_m['date'].unique()
    history = []

    for d in dates:
        d_m = df_m[df_m['date'] == d].set_index('isin')
        d_f = df_f[df_f['date'] == d].set_index('isin')
        
        try:
            m_val = sum(w * d_m.loc[isin, metric] for w, isin in zip(weights, isins))
            f_val = sum(w * d_f.loc[isin, f"{metric}_fitted"] for w, isin in zip(weights, isins))
            history.append({'date': d, 'spread_market': m_val, 'spread_fitted': f_val})
        except Exception as e:
            pass
            
    df_hist = pd.DataFrame(history)
    for col in df_hist.columns:
        if any(isinstance(x, pd.Series) for x in df_hist[col]):
            print(f'Trade {trade["trade_id"]} has Series in {col}')
            
print("Done checking all trades")
