import pandas as pd
import numpy as np
from pathlib import Path


def main():
    # Load data
    panel = pd.read_parquet('korean_ff3_v2/data/panel_data.parquet')
    mkt = pd.read_parquet('korean_ff3_v2/data/market_excess_return.parquet')
    be = pd.read_parquet('korean_ff3_v2/data/book_equity_monthly.parquet')

    panel['date'] = pd.to_datetime(panel['date'])
    mkt['date'] = pd.to_datetime(mkt['date'])
    be['date'] = pd.to_datetime(be['date'])

    # Compute formation_year for each row
    panel['year'] = panel['date'].dt.year
    panel['month'] = panel['date'].dt.month
    panel['formation_year'] = np.where(panel['month'] >= 7, panel['year'], panel['year'] - 1)

    # Compute lag ME (previous month's ME) for VW weights
    panel = panel.sort_values(['code', 'date'])
    panel['lag_me'] = panel.groupby('code')['me'].shift(1)

    # For portfolio formation, we need June ME and BM
    # Get June ME for each stock-year
    june_data = panel[panel['date'].dt.month == 6].copy()
    june_me = june_data.groupby(['code', 'year'])['me'].first().reset_index()
    june_me = june_me.rename(columns={'me': 'june_me', 'year': 'formation_year'})

    # Get BE for each be_year from BE data (first available month)
    be_first = be.sort_values('date').groupby(['code', 'be_year']).first().reset_index()
    be_first = be_first[['code', 'be_year', 'be']].copy()

    # Standard FF: formation_year t uses be_year t-1 BE
    be_first['formation_year'] = be_first['be_year'] + 1

    # Merge June ME and BE into a formation dataset
    formation = june_me.merge(be_first, on=['code', 'formation_year'], how='left')

    # For formation_year=2000, be_year=1998 doesn't exist.
    # Use be_year=1999 as the best available proxy (earliest BE data)
    be_1999 = be_first[be_first['be_year'] == 1999][['code', 'be']].copy()
    be_1999 = be_1999.rename(columns={'be': 'be_1999'})
    formation = formation.merge(be_1999, on='code', how='left')
    formation['be'] = np.where(
        formation['be'].isna() & (formation['formation_year'] == 2000),
        formation['be_1999'],
        formation['be']
    )
    formation = formation.drop(columns=['be_1999'])

    # Compute BM
    # Unit conversion: be is in 억원 (1e8 원), june_me is in 천원 (1e3 원)
    # bm = be * 1e8 / (june_me * 1e3) = be / june_me * 1e5
    formation['bm'] = formation['be'] / formation['june_me'] * 1e5

    # Also get panel's bm for comparison/validation
    panel_june = panel[panel['date'].dt.month == 6][['code', 'year', 'bm']].copy()
    panel_june = panel_june.rename(columns={'bm': 'panel_bm', 'year': 'formation_year'})
    formation = formation.merge(panel_june, on=['code', 'formation_year'], how='left')

    # Use panel's bm where available, otherwise use computed bm
    formation['bm_final'] = np.where(
        formation['panel_bm'].notna(),
        formation['panel_bm'],
        formation['bm']
    )

    # Portfolio formation each June
    portfolio_assignments = []

    for fy in sorted(formation['formation_year'].unique()):
        if fy < 2000:
            continue

        fy_data = formation[formation['formation_year'] == fy].copy()

        # Exclude negative BM and NaN
        valid = fy_data[(fy_data['bm_final'].notna()) & (fy_data['bm_final'] > 0)].copy()

        if len(valid) < 10:
            continue

        # Size sort: median of june_me
        size_median = valid['june_me'].median()
        valid['size_group'] = np.where(valid['june_me'] <= size_median, 'S', 'B')

        # BM sort: 30/70 percentiles
        bm_30 = valid['bm_final'].quantile(0.3)
        bm_70 = valid['bm_final'].quantile(0.7)

        valid['bm_group'] = pd.cut(
            valid['bm_final'],
            bins=[-np.inf, bm_30, bm_70, np.inf],
            labels=['L', 'M', 'H']
        )
        valid['portfolio'] = valid['size_group'] + '/' + valid['bm_group'].astype(str)

        portfolio_assignments.append(valid[['code', 'formation_year', 'portfolio']])

    port_assign = pd.concat(portfolio_assignments, ignore_index=True)

    # Merge portfolio assignments back to panel
    panel = panel.merge(port_assign, on=['code', 'formation_year'], how='left')

    # Compute VW returns for each portfolio-month
    # Use lag_me as weight (previous month's ME)
    valid_panel = panel[
        (panel['return'].notna()) &
        (panel['lag_me'].notna()) &
        (panel['lag_me'] > 0) &
        (panel['portfolio'].notna())
    ].copy()

    if len(valid_panel) == 0:
        raise ValueError("No valid data for portfolio returns")

    port_returns = valid_panel.groupby(['date', 'portfolio']).apply(
        lambda x: np.average(x['return'], weights=x['lag_me'])
    ).reset_index()
    port_returns.columns = ['date', 'portfolio', 'vw_return']

    # Pivot to wide format
    port_wide = port_returns.pivot(index='date', columns='portfolio', values='vw_return')

    # Ensure all 6 portfolios exist
    for p in ['S/L', 'S/M', 'S/H', 'B/L', 'B/M', 'B/H']:
        if p not in port_wide.columns:
            port_wide[p] = np.nan

    # Compute SMB and HML
    port_wide['SMB'] = (
        (port_wide['S/L'] + port_wide['S/M'] + port_wide['S/H']) / 3 -
        (port_wide['B/L'] + port_wide['B/M'] + port_wide['B/H']) / 3
    )
    port_wide['HML'] = (
        (port_wide['S/H'] + port_wide['B/H']) / 2 -
        (port_wide['S/L'] + port_wide['B/L']) / 2
    )

    # Merge with Mkt-RF
    port_wide = port_wide.reset_index()
    factors = port_wide.merge(mkt[['date', 'mkt_rf', 'rf']], on='date', how='left')
    factors = factors.rename(columns={'mkt_rf': 'Mkt-RF'})

    # Filter to 2000-07 ~ 2026-05
    factors = factors[
        (factors['date'] >= '2000-07-01') &
        (factors['date'] <= '2026-05-01')
    ]
    port_wide_filtered = port_wide[
        (port_wide['date'] >= '2000-07-01') &
        (port_wide['date'] <= '2026-05-01')
    ].copy()

    # Rename columns for output
    port_wide_filtered = port_wide_filtered.rename(columns={
        'S/L': 'SL', 'S/M': 'SM', 'S/H': 'SH',
        'B/L': 'BL', 'B/M': 'BM', 'B/H': 'BH'
    })

    # Save outputs
    Path('korean_ff3_v2/data').mkdir(parents=True, exist_ok=True)

    factors[['date', 'Mkt-RF', 'SMB', 'HML', 'rf']].to_csv(
        'korean_ff3_v2/data/factors_korea.csv', index=False
    )
    port_wide_filtered[['date', 'SL', 'SM', 'SH', 'BL', 'BM', 'BH']].to_csv(
        'korean_ff3_v2/data/portfolios_6_korea.csv', index=False
    )

    print(f"Saved factors_korea.csv: {len(factors)} months")
    print(f"Saved portfolios_6_korea.csv: {len(port_wide_filtered)} months")
    print(f"Factor means:")
    print(f"  Mkt-RF: {factors['Mkt-RF'].mean()*100:.4f}%")
    print(f"  SMB: {factors['SMB'].mean()*100:.4f}%")
    print(f"  HML: {factors['HML'].mean()*100:.4f}%")
    print(f"Portfolio means:")
    for col in ['SL', 'SM', 'SH', 'BL', 'BM', 'BH']:
        print(f"  {col}: {port_wide_filtered[col].mean()*100:.4f}%")


if __name__ == '__main__':
    main()
