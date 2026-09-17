import pandas as pd

# Read
df = pd.read_csv('korean_ff3_v2/data/rf_bok.csv')
print(f'Rows: {len(df)}')
print(f'Date range: {df["date"].iloc[0]} to {df["date"].iloc[-1]}')
print(f'rf_annual range: {df["rf_annual"].min()} to {df["rf_annual"].max()}')
print(f'NaN count: {df["rf_annual"].isna().sum()}')

# Convert annual to monthly: rf_monthly = (1 + rf_annual/100)^(1/12) - 1
df['rf_monthly'] = (1 + df['rf_annual']/100) ** (1/12) - 1

# Save
df.to_csv('korean_ff3_v2/data/rf_final.csv', index=False)
print('Saved rf_final.csv')

# Verify
df2 = pd.read_csv('korean_ff3_v2/data/rf_final.csv')
print(f'Output rows: {len(df2)}')
print(f'Columns: {list(df2.columns)}')
print(f'NaN in rf_monthly: {df2["rf_monthly"].isna().sum()}')
print(df2.head(3))