import pandas as pd
from src.data_io import read_market_data, read_fitted_curves, read_opportunities
import ast

df = read_opportunities('2026-02-17')
trade = df.iloc[0]
issuer = trade['issuer']
isins = [trade.get(f'bond{i}_isin') for i in range(1, 5) if trade.get(f'bond{i}_isin') is not None]
weights = ast.literal_eval(trade['leg_weights']) if isinstance(trade['leg_weights'], str) else trade['leg_weights']
metric = trade['metric']

df_m = read_market_data(issuer)
df_f = read_fitted_curves(issuer)

dt = pd.to_datetime('2026-02-17')
df_m = df_m[df_m['date'] <= dt]
df_f = df_f[df_f['date'] <= dt]

dates = df_m['date'].unique()
history = []

for dt in dates:
    d_m = df_m[df_m['date'] == dt].set_index('isin')
    d_f = df_f[df_f['date'] == dt].set_index('isin')
    
    try:
        m_val = sum(w * d_m.loc[isin, metric] for w, isin in zip(weights, isins))
        f_val = sum(w * d_f.loc[isin, f"{metric}_fitted"] for w, isin in zip(weights, isins))
        history.append({'date': dt, 'spread_market': m_val, 'spread_fitted': f_val})
    except Exception as e:
        pass
        
df_hist = pd.DataFrame(history)
for col in df_hist.columns:
    print(f'{col}: {type(df_hist[col].iloc[0])}')
