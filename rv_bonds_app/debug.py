import pandas as pd
from src.data_io import read_market_data, read_opportunities, read_fitted_curves
import ast

df = read_opportunities('2026-02-17')
if not df.empty:
    trade = df.iloc[0]
    print(trade['trade_id'], trade['issuer'])
    df_m = read_market_data(trade['issuer'])
    
    isins = [trade.get(f'bond{i}_isin') for i in range(1, 5) if trade.get(f'bond{i}_isin') is not None]
    weights = ast.literal_eval(trade['leg_weights']) if isinstance(trade['leg_weights'], str) else trade['leg_weights']
    
    print('isins:', isins)
    print('weights:', weights)
    
    # simulate build_trade_history
    df_f = read_fitted_curves(trade['issuer'])
    df_m = df_m[df_m['date'] <= pd.to_datetime('2026-02-17')]
    df_f = df_f[df_f['date'] <= pd.to_datetime('2026-02-17')]
    
    dates = df_m['date'].unique()
    history = []
    
    for dt in dates:
        d_m = df_m[df_m['date'] == dt].set_index('isin')
        d_f = df_f[df_f['date'] == dt].set_index('isin')
        
        try:
            m_val = sum(w * d_m.loc[isin, trade['metric']] for w, isin in zip(weights, isins))
            f_val = sum(w * d_f.loc[isin, f"{trade['metric']}_fitted"] for w, isin in zip(weights, isins))
            history.append({'date': dt, 'spread_market': m_val, 'spread_fitted': f_val})
        except Exception as e:
            print("Error at dt", dt, e)
            pass
            
    df_hist = pd.DataFrame(history)
    print(df_hist.head())
    print('df_hist size:', len(df_hist))
