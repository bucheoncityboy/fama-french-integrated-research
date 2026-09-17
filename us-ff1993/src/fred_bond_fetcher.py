"""
fred_bond_fetcher.py

Fetch bond data from FRED and construct TERM and DEF factors.

CRITICAL DISCLAIMER:
TERM and DEF are yield-based proxies constructed from FRED data, not return-based
factors as in Fama-French (1993). Numerical results may differ from the original paper.
"""

import os

import pandas as pd
from pandas_datareader.data import DataReader

import config

# Prominent disclaimer constant
DISCLAIMER = (
    "TERM and DEF are yield-based proxies constructed from FRED data, "
    "not return-based factors as in FF(1993). DEF is defined as BAA-AAA "
    "yield spread. Numerical results may differ from the original paper."
)

# FRED series identifiers
SERIES = {
    'baa_yield': 'BAA',   # Moody's Seasoned Baa Corporate Bond Yield
    'aaa_yield': 'AAA',   # Moody's Seasoned Aaa Corporate Bond Yield
    'lt_yield': 'GS10',   # 10-Year Treasury Constant Maturity Rate
    'rf_yield': 'TB3MS',  # 3-Month Treasury Bill Rate
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


def _fetch_series(symbol, start, end):
    """Fetch a single series from FRED with informative error handling."""
    try:
        df = DataReader(symbol, 'fred', start, end)
    except Exception as exc:
        raise ConnectionError(
            f"Failed to fetch FRED series '{symbol}' from {start} to {end}. "
            f"Check your internet connection and that the series exists. "
            f"Original error: {exc}"
        )

    if df is None or df.empty:
        raise ValueError(
            f"FRED series '{symbol}' returned no data for {start} to {end}. "
            f"The series may not cover this date range."
        )

    return df


def fetch_bond_data(start='1963-07-01', end='1991-12-31'):
    """
    Fetch bond data from FRED and construct TERM and DEF factors.

    Parameters
    ----------
    start : str
        Start date in YYYY-MM-DD format
    end : str
        End date in YYYY-MM-DD format

    Returns
    -------
    pd.DataFrame
        DataFrame with TERM and DEF columns, datetime index (monthly)
    """
    print(DISCLAIMER)

    start_dt, end_dt = _validate_dates(start, end)

    # ------------------------------------------------------------------
    # 1. Fetch raw data from FRED
    # ------------------------------------------------------------------
    print("Fetching FRED series: %s" % list(SERIES.values()))
    raw = {}
    for key, symbol in SERIES.items():
        raw[key] = _fetch_series(symbol, start_dt, end_dt)

    # ------------------------------------------------------------------
    # 2. Align all series to monthly frequency and convert to decimal
    # ------------------------------------------------------------------
    # Corporate yields (monthly) -> month-end value, converted to decimal
    baa_yield = raw['baa_yield'][SERIES['baa_yield']].resample('ME').last() / 100.0
    aaa_yield = raw['aaa_yield'][SERIES['aaa_yield']].resample('ME').last() / 100.0

    # Government yield (monthly) -> month-end value, converted to decimal
    lt_yield = raw['lt_yield'][SERIES['lt_yield']].resample('ME').last() / 100.0

    # Risk-free yield (already monthly from FRED), converted to decimal
    rf_yield = raw['rf_yield'][SERIES['rf_yield']].resample('ME').last() / 100.0

    # ------------------------------------------------------------------
    # 3. Construct factor proxies
    # ------------------------------------------------------------------
    # TERM = long-term government yield proxy - risk-free rate proxy
    term = lt_yield - rf_yield

    # DEF = Baa corporate yield - Aaa corporate yield (credit spread)
    def_ = baa_yield - aaa_yield

    # ------------------------------------------------------------------
    # 4. Assemble output
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        'TERM': term,
        'DEF': def_,
    })

    # Drop rows where either factor is missing
    df = df.dropna()

    if df.empty:
        raise ValueError(
            "No overlapping monthly data available for the requested date range. "
            "Some FRED series may not start until later than the start date."
        )

    # Ensure index name
    df.index.name = 'Date'

    # ------------------------------------------------------------------
    # 5. Filter to requested range
    # ------------------------------------------------------------------
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    # ------------------------------------------------------------------
    # 6. Save to CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(config.DATA_DIR, 'bond_factors.csv')
    df.to_csv(csv_path)
    print("Saved bond factors to %s" % csv_path)

    return df


if __name__ == '__main__':
    print(DISCLAIMER)
    df = fetch_bond_data()
    print(df.head())
