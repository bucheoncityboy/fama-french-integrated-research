"""
02_section2_portfolios.py
Section 2.2: Stock Portfolios - Compute excess returns

Loads 25 value-weighted portfolios, subtracts RF to get excess returns.
"""

import pandas as pd
import config


def load_portfolios(data_dir):
    """Load 25 portfolios from Ken French data library."""
    path = f"{data_dir}/ff_25_portfolios.csv"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def load_factors(data_dir):
    """Load Fama-French factors including RF."""
    path = f"{data_dir}/ff_factors.csv"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def filter_date_range(df, start_date='1963-07', end_date='1991-12'):
    """Filter DataFrame to specified date range (inclusive)."""
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df[mask]


def compute_excess_returns(portfolio_df, rf_series):
    """Compute excess returns: portfolio - RF for each month."""
    # RF is a Series with Date index, broadcast across all 25 portfolios
    return portfolio_df.sub(rf_series, axis=0)


def print_average_excess_returns(excess_df):
    """Print average excess returns showing size and value patterns."""
    print("\n=== Average Monthly Excess Returns (%, 1963-07 to 1991-12) ===")
    print(f"\nOverall average: {excess_df.mean().mean():.4f}%")
    print(f"\n25 Portfolios:\n{excess_df.mean().round(4).to_string()}")

    # Show size effect: SMALL > BIG (compare row means)
    small_cols = [c for c in excess_df.columns if c.startswith('SMALL')]
    big_cols = [c for c in excess_df.columns if c.startswith('BIG')]

    avg_small = excess_df[small_cols].mean().mean()
    avg_big = excess_df[big_cols].mean().mean()
    print(f"\nSize Effect: SMALL avg = {avg_small:.4f}, BIG avg = {avg_big:.4f}")
    print(f"  -> SMALL > BIG: {avg_small > avg_big}")

    # Show value effect: HiBM > LoBM
    loBM_cols = [c for c in excess_df.columns if 'LoBM' in c]
    hiBM_cols = [c for c in excess_df.columns if 'HiBM' in c]

    avg_hiBM = excess_df[hiBM_cols].mean().mean()
    avg_loBM = excess_df[loBM_cols].mean().mean()
    print(f"\nValue Effect: HiBM avg = {avg_hiBM:.4f}, LoBM avg = {avg_loBM:.4f}")
    print(f"  -> HiBM > LoBM: {avg_hiBM > avg_loBM}")


def main():
    print("=" * 60)
    print("Section 2.2: Stock Portfolios - Computing Excess Returns")
    print("=" * 60)

    # Load portfolios
    print("\nLoading 25 value-weighted portfolios...")
    portfolios = load_portfolios(config.DATA_DIR)
    print(f"  Loaded {len(portfolios.columns)} portfolios, {len(portfolios)} months")

    # Load factors
    print("\nLoading Fama-French factors (including RF)...")
    factors = load_factors(config.DATA_DIR)
    print(f"  Loaded factors: {list(factors.columns)}")

    # Filter to 1963-07 to 1991-12
    print(f"\nFiltering to {config.START_DATE} to {config.END_DATE}...")
    portfolios = filter_date_range(portfolios, config.START_DATE, config.END_DATE)
    factors = filter_date_range(factors, config.START_DATE, config.END_DATE)
    print(f"  After filter: {len(portfolios)} months")

    # Extract RF
    rf = factors['RF']
    print(f"\nRisk-free rate: mean={rf.mean():.4f}%, min={rf.min():.4f}%, max={rf.max():.4f}%")

    # Compute excess returns
    print("\nComputing excess returns (portfolio - RF)...")
    excess_returns = compute_excess_returns(portfolios, rf)
    print(f"  Computed {len(excess_returns.columns)} portfolio excess returns")

    # Save output
    output_path = f"{config.OUTPUT_DIR}/stock_portfolios_excess.csv"
    excess_returns.to_csv(output_path)
    print(f"\nSaved to: {output_path}")
    print(f"  Shape: {excess_returns.shape}")

    # Print average excess returns
    print_average_excess_returns(excess_returns)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()