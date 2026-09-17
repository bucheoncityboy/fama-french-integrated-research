"""
US Fama-French Data Loader
Loads US FF3 factors and 25 portfolios from fama-ff3- reference data.

US Data Reference:
- factors.csv: 342 months (1963-07 ~ 1991-12), skiprows=1 (first row is disclaimer)
- stock_portfolios_excess.csv: 342 months (1963-07 ~ 1991-12), no skiprows needed
"""

import pandas as pd
from pathlib import Path

# Path to US Fama-French data
US_DATA_DIR = Path(__file__).parent.parent / ".." / "fama-ff3-" / "appendix_output"


def load_us_factors() -> pd.DataFrame:
    """Load US FF3 factors: Mkt-RF, SMB, HML, RF.
    
    Returns:
        DataFrame with columns: Date, TERM, DEF, RF, Mkt-RF, SMB, HML
        Date range: 1963-07 ~ 1991-12 (342 months)
    
    Notes:
        - factors.csv has disclaimer line at top, requires skiprows=1
        - RF is in percentage (e.g., 0.27 = 0.27%)
    """
    path = US_DATA_DIR / "factors.csv"
    df = pd.read_csv(path, skiprows=1)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    return df


def load_us_portfolios() -> pd.DataFrame:
    """Load US 25 Size x BM portfolios (excess returns).
    
    Returns:
        DataFrame with columns: Date + 25 portfolio excess returns (SMALL LoBM ~ BIG HiBM)
        Date range: 1963-07 ~ 1991-12 (342 months)
    
    Notes:
        - stock_portfolios_excess.csv has header as first row (no skiprows needed)
        - Returns are in percentage (e.g., 0.8587 = 0.8587%)
        - Format: 5x5 (Size x BM), naming: SMALL/BIG x LoBM/HiBM
    """
    path = US_DATA_DIR / "stock_portfolios_excess.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_us_factor_stats() -> pd.DataFrame:
    """Calculate summary statistics for US factors.
    
    Returns:
        DataFrame with stats: mean, std, t_stat, sharpe, min, max, n_months
    """
    factors = load_us_factors()
    rf = factors["RF"] / 100  # Convert to decimal
    
    stats = []
    for col in ["Mkt-RF", "SMB", "HML"]:
        values = factors[col] / 100  # Convert to decimal
        mean_val = values.mean()
        std_val = values.std()
        t_stat = mean_val / std_val * (len(values) ** 0.5)
        sharpe = mean_val / std_val * (12 ** 0.5)  # Annualized
        
        stats.append({
            "factor": col,
            "period": "1963-07~1991-12",
            "mean": mean_val,
            "std": std_val,
            "t_stat": t_stat,
            "sharpe": sharpe,
            "min": values.min(),
            "max": values.max(),
            "n_months": len(values)
        })
    
    return pd.DataFrame(stats)


if __name__ == "__main__":
    # Quick verification
    print("=== US Factors ===")
    f = load_us_factors()
    print(f"Rows: {len(f)}, Columns: {list(f.columns)}")
    print(f"Date range: {f['Date'].min()} ~ {f['Date'].max()}")
    
    print("\n=== US 25 Portfolios ===")
    p = load_us_portfolios()
    print(f"Rows: {len(p)}, Columns: {len(p.columns)} (including Date)")
    print(f"Date range: {p['Date'].min()} ~ {p['Date'].max()}")
    
    print("\n=== US Factor Stats ===")
    print(get_us_factor_stats().to_string(index=False))