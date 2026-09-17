import pandas as pd

# Market data QA
m = pd.read_parquet('data/market_data_long.parquet')
print('=== Market Data ===')
print(f'Rows: {len(m):,}')
print(f'Stocks: {m.code.nunique()}')
print(f'Items: {m.item_code.nunique()}')
print(f'Items list: {sorted(m.item_code.unique())}')
print(f'Date range: {m.date.min()} ~ {m.date.max()}')
print(f'Months: {m.date.dt.to_period("M").nunique()}')

# Check Samsung
ss = m[m.code == 'A005930']
print(f'Samsung items: {ss.item_code.nunique()}')
print(f'Samsung date range: {ss.date.min()} ~ {ss.date.max()}')

# Check coverage per item
for item in sorted(m.item_code.unique()):
    sub = m[m.item_code == item]
    print(f'  {item}: {sub.code.nunique()} stocks, {sub.date.nunique()} months')

print()

# Financial data QA
f = pd.read_parquet('data/financial_data_long.parquet')
print('=== Financial Data ===')
print(f'Rows: {len(f):,}')
print(f'Stocks: {f.code.nunique()}')
print(f'Items: {f.item_code.nunique()}')
print(f'Items list: {sorted(f.item_code.unique())}')
print(f'Date range: {f.date.min()} ~ {f.date.max()}')
print(f'Months: {f.date.dt.to_period("M").nunique()}')

# Check Samsung
ss = f[f.code == 'A005930']
print(f'Samsung items: {ss.item_code.nunique()}')
print(f'Samsung date range: {ss.date.min()} ~ {ss.date.max()}')

# Check coverage per item
for item in sorted(f.item_code.unique()):
    sub = f[f.item_code == item]
    print(f'  {item}: {sub.code.nunique()} stocks, {sub.date.nunique()} months')
