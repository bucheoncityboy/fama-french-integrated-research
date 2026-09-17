"""
01b_section2_bond_factors.py

Section 2.1: Bond Factors (TERM and DEF)

This script loads yield-based bond factor proxies from FRED and merges them
with the Fama-French stock factors and risk-free rate into a single
output/factors.csv file.

CRITICAL DISCLAIMER:
TERM and DEF are yield-based proxies constructed from FRED data, not
return-based factors as in Fama-French (1993). The corporate bond data
is synthetic pre-1991 because FRED lacks historical total-return indices.
Numerical results may differ from the original paper.
"""

import os

import pandas as pd

import config

# Prominent disclaimer constants
DISCLAIMER = (
    "TERM and DEF are yield-based proxy factors constructed from FRED data, "
    "not return-based factors as in FF(1993). DEF is the BAA-AAA corporate "
    "bond yield spread. Numerical results may differ from the original paper."
)

CSV_DISCLAIMER = (
    "# DISCLAIMER: " + DISCLAIMER + " "
    "Stock factors (Mkt-RF, SMB, HML) will be added when available."
)


def _load_bond_factors(path: str) -> pd.DataFrame:
    """Load bond factor CSV and normalize index to monthly period strings."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).to_period("M").astype(str)
    df.index.name = "Date"
    return df


def _load_ff_factors(path: str) -> pd.DataFrame:
    """Load Fama-French factor CSV and normalize index to monthly period strings."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).to_period("M").astype(str)
    df.index.name = "Date"
    return df


def _compute_summary(series: pd.Series) -> dict:
    """Compute mean, std, and t-statistic for a factor series."""
    mean = series.mean()
    std = series.std(ddof=1)
    n = series.count()
    t_stat = mean / (std / (n ** 0.5)) if std != 0 else float("nan")
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "t_stat": t_stat,
    }


def build_factors() -> pd.DataFrame:
    """
    Load bond factors, merge with stock factors (if available) and RF,
    and return the combined DataFrame.
    """
    # ------------------------------------------------------------------
    # 1. Load bond factors
    # ------------------------------------------------------------------
    bond_path = os.path.join(config.DATA_DIR, "bond_factors.csv")
    if not os.path.exists(bond_path):
        raise FileNotFoundError(f"Bond factors not found: {bond_path}")
    bond_df = _load_bond_factors(bond_path)
    print("Loaded bond factors: %d rows" % len(bond_df))

    # ------------------------------------------------------------------
    # 2. Load FF factors for RF
    # ------------------------------------------------------------------
    ff_path = os.path.join(config.DATA_DIR, "ff_factors.csv")
    if not os.path.exists(ff_path):
        raise FileNotFoundError(f"FF factors not found: {ff_path}")
    ff_df = _load_ff_factors(ff_path)
    print("Loaded FF factors: %d rows" % len(ff_df))

    # ------------------------------------------------------------------
    # 3. Merge bond factors with RF
    # ------------------------------------------------------------------
    # Start with bond factors and add RF from ff_factors
    combined = bond_df[["TERM", "DEF"]].copy()
    combined = combined.join(ff_df[["RF"]], how="outer")

    # ------------------------------------------------------------------
    # 4. Merge with stock factors if available
    # ------------------------------------------------------------------
    stock_path = os.path.join(config.OUTPUT_DIR, "factors.csv")
    if os.path.exists(stock_path):
        try:
            stock_df = pd.read_csv(
                stock_path, index_col=0, parse_dates=True, comment="#"
            )
            stock_df.index = pd.to_datetime(stock_df.index).to_period("M").astype(str)
            stock_df.index.name = "Date"

            # Keep only stock factor columns
            stock_cols = [c for c in stock_df.columns if c in ("Mkt-RF", "SMB", "HML")]
            if stock_cols:
                combined = combined.join(stock_df[stock_cols], how="outer")
                print("Merged stock factors: %s" % stock_cols)
        except Exception as exc:
            print(
                "WARNING: Could not read existing %s (%s). Treating as missing."
                % (stock_path, exc)
            )
    else:
        print(
            "output/factors.csv not found. Stock factors (Mkt-RF, SMB, HML) "
            "will be added later."
        )

    # Sort by date
    combined = combined.sort_index()

    # ------------------------------------------------------------------
    # 5. Add note column if stock factors are missing
    # ------------------------------------------------------------------
    stock_cols = {"Mkt-RF", "SMB", "HML"}
    has_all_stock = stock_cols.issubset(combined.columns)
    if not has_all_stock:
        combined["Note_StockFactorsPending"] = "Stock factors will be added later"

    return combined


def save_factors(df: pd.DataFrame, path: str) -> None:
    """Save combined factors to CSV with a disclaimer comment line."""
    # Write disclaimer as a comment line, then the CSV
    csv_body = df.to_csv()
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(CSV_DISCLAIMER + "\n")
        f.write(csv_body)
    print("Saved combined factors to %s" % path)


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for TERM and DEF."""
    print("\n" + "=" * 60)
    print("Bond Factor Summary Statistics")
    print("=" * 60)
    print(f"Disclaimer: {DISCLAIMER}\n")

    for factor in ("TERM", "DEF"):
        if factor in df.columns:
            stats_dict = _compute_summary(df[factor].dropna())
            print(f"{factor}:")
            print(f"  Mean     : {stats_dict['mean']:.6f}")
            print(f"  Std      : {stats_dict['std']:.6f}")
            print(f"  T-stat   : {stats_dict['t_stat']:.4f}")
            print(f"  N        : {stats_dict['n']}")
            print()

    print("=" * 60)


def main() -> None:
    """Main entry point."""
    print(DISCLAIMER)

    combined = build_factors()
    output_path = os.path.join(config.OUTPUT_DIR, "factors.csv")
    save_factors(combined, output_path)
    print_summary(combined)


if __name__ == "__main__":
    main()
