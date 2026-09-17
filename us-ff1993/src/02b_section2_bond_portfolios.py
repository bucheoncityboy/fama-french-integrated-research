"""
02b_section2_bond_portfolios.py
Section 2.2: Bond Portfolios - Fama-French (1993) Replication

CRITICAL DISCLAIMER:
These bond portfolio returns are YIELD-BASED PROXIES constructed from FRED data,
NOT actual historical total returns as used in the original Fama-French paper.
The 7 bond portfolios (Short-term Government, Long-term Government, AAA, AA, A,
BBB, and Low-Grade corporate bonds) are approximated using Treasury yields and
the DEF (BAA-AAA corporate bond yield spread, credit spread proxy) factor.
Numerical results will differ substantially from the original paper which used
actual corporate bond return data from Ibbotson.

PROXY METHODOLOGY:
- Government bond excess returns: yield spread (10Y Treasury - 3M Treasury)
- Corporate bond excess returns: yield spread + credit quality premium
- Short-term Government (1Y): approximated as (GS1 - TB3MS)
- Long-term Government (10Y): approximated as (DGS10 - TB3MS)
- AAA corporate: DGS10 + 0.2*DEF - TB3MS
- AA corporate: DGS10 + 0.4*DEF - TB3MS
- A corporate: DGS10 + 0.6*DEF - TB3MS
- BBB corporate: DGS10 + 0.8*DEF - TB3MS
- Low-Grade (BBB-): DGS10 + DEF - TB3MS

DEF represents the BAA-AAA corporate bond yield spread (credit spread proxy),
measured as the difference between Moody's Baa and Aaa yields. In decimal
monthly terms, DEF is approximately ~0.001 (1.2% annualized). The formulas
above combine the term premium (DGS10 - TB3MS) with a credit premium scaled
by rating quality.

These approximations assume:
1. Yield changes approximate bond returns (duration effect)
2. Credit spreads scale with DEF factor
3. Risk-free rate = 3-Month Treasury Bill Rate
"""

import os

import pandas as pd
from pandas_datareader.data import DataReader

import config

# Prominent disclaimer constant
DISCLAIMER = """
================================================================================
                         IMPORTANT DISCLAIMER
================================================================================
These bond portfolio returns are YIELD-BASED PROXIES, not actual returns.
They are constructed from FRED Treasury yields and the DEF factor to approximate
the 7 bond portfolios used in Fama-French (1993).

PROXY LIMITATIONS:
- Actual bond total returns depend on price changes, not just yields
- Duration mismatch: yield changes don't perfectly capture bond returns
- Credit spread proxies may not reflect actual default losses
- Corporate bond indices (BAML) lack pre-2023 data in FRED

NUMERICAL RESULTS WILL DIFFER from the original paper which used proprietary
Ibbotson SBBI data for actual bond portfolio returns.

Use these proxies only for demonstration/educational purposes.
================================================================================
"""

# FRED series identifiers for bond data
BOND_SERIES = {
    'gs1': 'GS1',      # 1-Year Treasury Constant Maturity Rate
    'gs10': 'DGS10',   # 10-Year Treasury Constant Maturity Rate
    'rf': 'TB3MS',     # 3-Month Treasury Bill Rate (Risk-Free)
}


def _validate_dates(start, end):
    """Validate and parse date strings."""
    try:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
    except Exception as exc:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {exc}")

    if start_dt > end_dt:
        raise ValueError(f"Start date ({start}) must be before end date ({end}).")

    return start_dt, end_dt


def _fetch_bond_yields(start, end):
    """
    Fetch Treasury yields from FRED.

    Parameters
    ----------
    start : str
        Start date in YYYY-MM-DD format
    end : str
        End date in YYYY-MM-DD format

    Returns
    -------
    pd.DataFrame
        DataFrame with GS1, DGS10, TB3MS columns
    """
    print("Fetching Treasury yields from FRED...")

    start_dt, end_dt = _validate_dates(start, end)

    yields = {}

    for key, symbol in BOND_SERIES.items():
        try:
            df = DataReader(symbol, 'fred', start_dt, end_dt)
            yields[key] = df[symbol]
            print("  Fetched %s (%s): %d observations" % (key, symbol, len(df)))
        except Exception as exc:
            raise ConnectionError(
                f"Failed to fetch FRED series '{symbol}' from {start} to {end}. "
                f"Check your internet connection. Original error: {exc}"
            )

    # Combine into DataFrame
    yields_df = pd.DataFrame(yields)

    # Convert to decimal (FRED reports as percentages)
    yields_df = yields_df / 100.0

    # Resample to monthly (month-end)
    yields_df = yields_df.resample('ME').last()

    # Drop rows with any missing values
    yields_df = yields_df.dropna()

    if yields_df.empty:
        raise ValueError("No overlapping Treasury yield data available.")

    return yields_df


def _load_bond_factors():
    """
    Load TERM and DEF factors from bond_factors.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with TERM and DEF columns, datetime index
    """
    csv_path = os.path.join(config.DATA_DIR, 'bond_factors.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Bond factors CSV not found at {csv_path}. "
            f"Please run fred_bond_fetcher.py first to download the data."
        )

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print("Loaded bond factors from %s: %d observations" % (csv_path, len(df)))

    return df


def create_bond_portfolios(start=None, end=None):
    """
    Create 7 bond portfolio excess returns using yield-based proxies.

    Parameters
    ----------
    start : str, optional
        Start date in YYYY-MM format. Defaults to config.START_DATE.
    end : str, optional
        End date in YYYY-MM format. Defaults to config.END_DATE.

    Returns
    -------
    pd.DataFrame
        DataFrame with 7 bond portfolio excess return columns:
        - SHORT_TERM: 1-Year Treasury excess return (proxy)
        - LONG_TERM: 10-Year Treasury excess return (proxy)
        - AAA: AAA corporate bond excess return (proxy)
        - AA: AA corporate bond excess return (proxy)
        - A: A corporate bond excess return (proxy)
        - BBB: BBB corporate bond excess return (proxy)
        - LOW_GRADE: Low-grade (BBB-) corporate bond excess return (proxy)
    """
    # Default date range from config
    if start is None:
        start = config.START_DATE
    if end is None:
        end = config.END_DATE

    start_str = f"{start}-01"
    end_str = f"{end}-01"

    print(DISCLAIMER)

    # ------------------------------------------------------------------
    # Step 1: Fetch Treasury yields from FRED
    # ------------------------------------------------------------------
    print("\nStep 1: Fetching Treasury yields from FRED...")
    yields_df = _fetch_bond_yields(start_str, end_str)
    yields_df.index.name = 'Date'

    # ------------------------------------------------------------------
    # Step 2: Load TERM and DEF factors from bond_factors.csv
    # ------------------------------------------------------------------
    print("\nStep 2: Loading TERM and DEF factors...")
    factors_df = _load_bond_factors()

    # Align factors with yields
    factors_df = factors_df[(factors_df.index >= yields_df.index.min()) &
                             (factors_df.index <= yields_df.index.max())]

    # ------------------------------------------------------------------
    # Step 3: Construct bond portfolio excess returns using proxies
    # ------------------------------------------------------------------
    print("\nStep 3: Constructing bond portfolio excess returns...")

    # Risk-free rate (3-Month Treasury Bill)
    rf = yields_df['rf']

    # Government bond proxies
    # SHORT_TERM (1Y Treasury): yield - RF (term spread approximation)
    short_term = yields_df['gs1'] - rf

    # LONG_TERM (10Y Treasury): yield - RF (term spread approximation)
    long_term = yields_df['gs10'] - rf

    # Corporate bond proxies using DEF factor
    # DEF represents the BAA-AAA corporate bond yield spread (credit spread proxy)
    # Scale DEF by credit quality: AAA gets smallest fraction, Low-grade gets full DEF

    # Reindex DEF to match yields_df index for proper alignment
    def_aligned = factors_df['DEF'].reindex(yields_df.index)

    # AAA: DGS10 + 0.2*DEF - RF
    aaa = yields_df['gs10'] + 0.2 * def_aligned - rf

    # AA: DGS10 + 0.4*DEF - RF
    aa = yields_df['gs10'] + 0.4 * def_aligned - rf

    # A: DGS10 + 0.6*DEF - RF
    a = yields_df['gs10'] + 0.6 * def_aligned - rf

    # BBB: DGS10 + 0.8*DEF - RF
    bbb = yields_df['gs10'] + 0.8 * def_aligned - rf

    # Low-Grade (BBB-): DGS10 + DEF - RF (full DEF premium)
    low_grade = yields_df['gs10'] + def_aligned - rf

    # ------------------------------------------------------------------
    # Step 4: Assemble into DataFrame
    # ------------------------------------------------------------------
    portfolios_df = pd.DataFrame({
        'SHORT_TERM': short_term,
        'LONG_TERM': long_term,
        'AAA': aaa,
        'AA': aa,
        'A': a,
        'BBB': bbb,
        'LOW_GRADE': low_grade,
    })

    # Convert annualized yield spreads to monthly decimals for consistency
    # with monthly return conventions (FRED yields are annual percentages)
    portfolios_df = portfolios_df / 12.0

    # Drop rows with NaN (from factor alignment)
    portfolios_df = portfolios_df.dropna()

    # Filter to date range
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    portfolios_df = portfolios_df[(portfolios_df.index >= start_dt) &
                                    (portfolios_df.index <= end_dt)]

    # ------------------------------------------------------------------
    # Step 5: Save to CSV
    # ------------------------------------------------------------------
    output_path = os.path.join(config.OUTPUT_DIR, 'bond_portfolios_excess.csv')
    portfolios_df.to_csv(output_path)
    print("\nSaved bond portfolios to %s" % output_path)

    # ------------------------------------------------------------------
    # Step 6: Print average excess returns
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("AVERAGE MONTHLY EXCESS RETURNS (yield-based proxies)")
    print("="*70)
    print("(Paper finding: excess returns near zero for all portfolios)")
    print("-"*70)

    avg_returns = portfolios_df.mean() * 100  # Convert to percentage

    for col in portfolios_df.columns:
        val = avg_returns[col]
        if pd.isna(val):
            print(f"  {col:15s}:      N/A  (insufficient data)")
        else:
            print(f"  {col:15s}: {val:7.4f}%  (proxy)")

    print("-"*70)
    if not portfolios_df.empty:
        print(f"Number of observations: {len(portfolios_df)}")
        print(f"Date range: {portfolios_df.index.min().strftime('%Y-%m')} to "
              f"{portfolios_df.index.max().strftime('%Y-%m')}")
    else:
        print("Number of observations: 0 (no overlapping data)")
        print("NOTE: bond_factors.csv may not cover the requested date range.")
        print("      The original data (1963-1991) is not available from FRED.")
    print("="*70)
    print("\nNOTE: These are YIELD-BASED PROXIES, not actual bond returns.")
    print("      Expected excess returns reflect yield spreads (small positive)")
    print("      rather than actual total returns. Yields are mean-reverting")
    print("      and don't capture capital gains/losses.")
    print("="*70 + "\n")

    return portfolios_df


if __name__ == '__main__':
    print("Running Section 2.2 Bond Portfolios...")
    print("-" * 70)

    df = create_bond_portfolios()
    print("\nFirst 5 rows of bond portfolios excess returns:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())