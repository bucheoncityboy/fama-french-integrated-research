"""
04b_section4_five_factor.py
Section 4.4: Five-Factor Joint Regressions - Fama-French (1993) Replication

Runs the full five-factor model on 32 portfolios (25 stock + 7 bond):
    R_i - R_f = alpha + beta*(R_m - R_f) + s*SMB + h*HML + m*TERM + d*DEF + eps

Also computes inline 1-factor, 2-factor bond, and 3-factor stock regressions
for incremental R^2 comparison.

Saves results to output/table5_five_factor.csv.
"""

import os

import pandas as pd

import config
from data_loader import load_factors, load_stock_portfolios
import regression_engine as re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOCK_COLS = [
    "SMALL LoBM",
    "ME1 BM2",
    "ME1 BM3",
    "ME1 BM4",
    "SMALL HiBM",
    "ME2 BM1",
    "ME2 BM2",
    "ME2 BM3",
    "ME2 BM4",
    "ME2 BM5",
    "ME3 BM1",
    "ME3 BM2",
    "ME3 BM3",
    "ME3 BM4",
    "ME3 BM5",
    "ME4 BM1",
    "ME4 BM2",
    "ME4 BM3",
    "ME4 BM4",
    "ME4 BM5",
    "BIG LoBM",
    "ME5 BM2",
    "ME5 BM3",
    "ME5 BM4",
    "BIG HiBM",
]

BOND_COLS = ["SHORT_TERM", "LONG_TERM", "AAA", "AA", "A", "BBB", "LOW_GRADE"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalize_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DatetimeIndex to first-of-month for consistent alignment."""
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp()
    return df


def load_bond_portfolios() -> pd.DataFrame:
    """Load 7 bond portfolio excess returns (decimal) and convert to %."""
    path = os.path.join(config.OUTPUT_DIR, "bond_portfolios_excess.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = _normalize_monthly_index(df)
    df = df * 100.0
    return df


def _tag_type(portfolio: str) -> str:
    """Return 'stock' or 'bond' based on portfolio name."""
    return "bond" if portfolio in BOND_COLS else "stock"


def run_all_specifications(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run 1-factor, 2-factor bond, 3-factor stock, and 5-factor regressions
    for all portfolios. Return combined DataFrame with Table 5 as primary
    results plus inline R^2 from other specs.
    """
    # Table 1: 1-factor (Mkt-RF)
    table1 = re.run_batch_regressions(returns_df, factors_df, ["Mkt-RF"])
    table1 = table1.rename(columns={"r_squared": "r_squared_table1"})

    # Table 3: 2-factor bond (TERM, DEF)
    table3 = re.run_batch_regressions(returns_df, factors_df, ["TERM", "DEF"])
    table3 = table3.rename(columns={"r_squared": "r_squared_table3"})

    # Table 4: 3-factor stock (Mkt-RF, SMB, HML)
    table4 = re.run_batch_regressions(returns_df, factors_df, ["Mkt-RF", "SMB", "HML"])
    table4 = table4.rename(columns={"r_squared": "r_squared_table4"})

    # Table 5: 5-factor (all)
    table5 = re.run_batch_regressions(
        returns_df, factors_df, ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]
    )

    # Merge R^2 columns from Table 1, 3, 4 into Table 5
    table5 = table5.merge(
        table1[["portfolio", "r_squared_table1"]], on="portfolio", how="left"
    )
    table5 = table5.merge(
        table3[["portfolio", "r_squared_table3"]], on="portfolio", how="left"
    )
    table5 = table5.merge(
        table4[["portfolio", "r_squared_table4"]], on="portfolio", how="left"
    )

    # Add type column
    table5["type"] = table5["portfolio"].apply(_tag_type)

    return table5


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_incremental_r2(table5: pd.DataFrame) -> None:
    """Print incremental R^2 comparison across model specifications."""
    stocks = table5[table5["type"] == "stock"]
    bonds = table5[table5["type"] == "bond"]

    avg = {
        "stock": {
            "Table 1 (1-factor)": stocks["r_squared_table1"].mean(),
            "Table 3 (2-factor bond)": stocks["r_squared_table3"].mean(),
            "Table 4 (3-factor stock)": stocks["r_squared_table4"].mean(),
            "Table 5 (5-factor)": stocks["r_squared"].mean(),
        },
        "bond": {
            "Table 1 (1-factor)": bonds["r_squared_table1"].mean(),
            "Table 3 (2-factor bond)": bonds["r_squared_table3"].mean(),
            "Table 4 (3-factor stock)": bonds["r_squared_table4"].mean(),
            "Table 5 (5-factor)": bonds["r_squared"].mean(),
        },
    }

    print("\n" + "=" * 70)
    print("Incremental R-squared Comparison")
    print("=" * 70)
    print(f"{'Model':<25} {'Stock Avg R2':>15} {'Bond Avg R2':>15}")
    print("-" * 70)
    for model in [
        "Table 1 (1-factor)",
        "Table 3 (2-factor bond)",
        "Table 4 (3-factor stock)",
        "Table 5 (5-factor)",
    ]:
        print(
            f"{model:<25} {avg['stock'][model]:>15.4f} {avg['bond'][model]:>15.4f}"
        )
    print("=" * 70)

    print("\nKey Findings:")
    print("-" * 70)
    stock_t4 = avg["stock"]["Table 4 (3-factor stock)"]
    stock_t5 = avg["stock"]["Table 5 (5-factor)"]
    print(
        f"  Stocks: Table 5 R^2 ({stock_t5:.4f}) vs Table 4 ({stock_t4:.4f})"
        f"  -> {'Little gain' if stock_t5 < stock_t4 + 0.02 else 'Improved'}"
    )

    bond_t3 = avg["bond"]["Table 3 (2-factor bond)"]
    bond_t5 = avg["bond"]["Table 5 (5-factor)"]
    print(
        f"  Bonds:  Table 5 R^2 ({bond_t5:.4f}) vs Table 3 ({bond_t3:.4f})"
        f"  -> {'Improved' if bond_t5 > bond_t3 else 'Little gain'}"
    )
    print("=" * 70)


def print_table5(table5: pd.DataFrame) -> None:
    """Print formatted Table 5 regression results."""
    print("\n" + "=" * 90)
    print("Table 5: Five-Factor Joint Regressions (Section 4.4)")
    print("=" * 90)

    # Reorder columns for display
    display_cols = [
        "portfolio",
        "alpha",
        "beta_Mkt-RF",
        "beta_SMB",
        "beta_HML",
        "beta_TERM",
        "beta_DEF",
        "t_Mkt-RF",
        "t_SMB",
        "t_HML",
        "t_TERM",
        "t_DEF",
        "r_squared",
        "type",
    ]

    display_df = table5[display_cols].copy()
    # Format numeric columns
    for col in display_cols:
        if col not in ("portfolio", "type"):
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"
            )

    print(display_df.to_string(index=False))
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Section 4.4: Five-Factor Joint Regressions")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    stock_df = load_stock_portfolios()
    bond_df = load_bond_portfolios()
    factors_df = load_factors()

    # Combine all 32 portfolios
    all_returns = pd.concat([stock_df, bond_df], axis=1)
    print(f"Combined portfolios: {all_returns.shape}")

    # Run all regression specifications
    print("\nRunning regressions...")
    table5 = run_all_specifications(all_returns, factors_df)

    # Save to CSV
    output_path = os.path.join(config.OUTPUT_DIR, "table5_five_factor.csv")
    table5.to_csv(output_path, index=False)
    print(f"\nSaved Table 5 results to: {output_path}")

    # Print incremental R^2 comparison
    print_incremental_r2(table5)

    # Print formatted regression table
    print_table5(table5)

    print("\nDone!")


if __name__ == "__main__":
    main()
