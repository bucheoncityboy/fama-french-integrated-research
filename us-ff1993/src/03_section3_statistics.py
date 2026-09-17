"""
03_section3_statistics.py
Section 3: The Playing Field - Fama-French (1993) Replication

Computes average excess returns for 25 stock portfolios, 7 bond portfolios,
and 5 factor premiums with t-statistics. Produces a formatted Table 2 summary
and saves results to output/table2_summary.csv.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

import config
from data_loader import load_factors, load_stock_portfolios, load_bond_portfolios

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

SIZE_BM_MAP = {
    "SMALL LoBM": (1, 1),
    "ME1 BM2": (1, 2),
    "ME1 BM3": (1, 3),
    "ME1 BM4": (1, 4),
    "SMALL HiBM": (1, 5),
    "ME2 BM1": (2, 1),
    "ME2 BM2": (2, 2),
    "ME2 BM3": (2, 3),
    "ME2 BM4": (2, 4),
    "ME2 BM5": (2, 5),
    "ME3 BM1": (3, 1),
    "ME3 BM2": (3, 2),
    "ME3 BM3": (3, 3),
    "ME3 BM4": (3, 4),
    "ME3 BM5": (3, 5),
    "ME4 BM1": (4, 1),
    "ME4 BM2": (4, 2),
    "ME4 BM3": (4, 3),
    "ME4 BM4": (4, 4),
    "ME4 BM5": (4, 5),
    "BIG LoBM": (5, 1),
    "ME5 BM2": (5, 2),
    "ME5 BM3": (5, 3),
    "ME5 BM4": (5, 4),
    "BIG HiBM": (5, 5),
}

BOND_COLS = ["SHORT_TERM", "LONG_TERM", "AAA", "AA", "A", "BBB", "LOW_GRADE"]

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]

SIZE_LABELS = {1: "Small", 2: "ME2", 3: "ME3", 4: "ME4", 5: "Big"}
BM_LABELS = {1: "LoBM", 2: "BM2", 3: "BM3", 4: "BM4", 5: "HiBM"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _ensure_factors() -> None:
    """
    Ensure combined factors CSV has all 5 factors (Mkt-RF, SMB, HML, TERM, DEF).
    If TERM or DEF are missing, regenerate by running 01b_section2_bond_factors.py.
    """
    factors_path = os.path.join(config.OUTPUT_DIR, "factors.csv")
    needs_regen = True
    if os.path.exists(factors_path):
        df = pd.read_csv(factors_path, comment="#")
        if {"TERM", "DEF", "Mkt-RF", "SMB", "HML"}.issubset(set(df.columns)):
            needs_regen = False

    if not needs_regen:
        return

    print("Combined factors missing TERM/DEF. Regenerating...")
    base_dir = config.BASE_DIR

    # Regenerate bond factors first (needed by 01b)
    fetcher_script = os.path.join(base_dir, "fred_bond_fetcher.py")
    if os.path.exists(fetcher_script):
        print("Running fred_bond_fetcher.py...")
        subprocess.run(
            [sys.executable, fetcher_script],
            capture_output=True,
            text=True,
            cwd=base_dir,
        )

    # Regenerate combined factors
    bond_factors_script = os.path.join(base_dir, "01b_section2_bond_factors.py")
    if os.path.exists(bond_factors_script):
        print("Running 01b_section2_bond_factors.py...")
        subprocess.run(
            [sys.executable, bond_factors_script],
            capture_output=True,
            text=True,
            cwd=base_dir,
        )


def _ensure_bond_data() -> None:
    """
    Ensure bond portfolio data is available.
    If bond_portfolios_excess.csv is empty or missing, regenerate it by
    running fred_bond_fetcher.py and 02b_section2_bond_portfolios.py.
    """
    bond_path = os.path.join(config.OUTPUT_DIR, "bond_portfolios_excess.csv")
    needs_regen = True
    if os.path.exists(bond_path):
        df = pd.read_csv(bond_path, index_col=0, parse_dates=True)
        if not df.empty:
            needs_regen = False

    if not needs_regen:
        return

    print("Bond portfolio data missing or empty. Regenerating...")
    base_dir = config.BASE_DIR

    # Regenerate bond factors
    fetcher_script = os.path.join(base_dir, "fred_bond_fetcher.py")
    if os.path.exists(fetcher_script):
        print("Running fred_bond_fetcher.py...")
        result = subprocess.run(
            [sys.executable, fetcher_script],
            capture_output=True,
            text=True,
            cwd=base_dir,
        )
        if result.returncode != 0:
            print("WARNING: fred_bond_fetcher.py failed: %s" % result.stderr)

    # Regenerate bond portfolios
    bond_script = os.path.join(base_dir, "02b_section2_bond_portfolios.py")
    if os.path.exists(bond_script):
        print("Running 02b_section2_bond_portfolios.py...")
        result = subprocess.run(
            [sys.executable, bond_script],
            capture_output=True,
            text=True,
            cwd=base_dir,
        )
        if result.returncode != 0:
            print("WARNING: 02b_section2_bond_portfolios.py failed: %s" % result.stderr)


def _compute_stats(series: pd.Series) -> dict:
    """Compute mean, std, n, and t-statistic for a series."""
    s = series.dropna()
    mean = s.mean()
    std = s.std(ddof=1)
    n = len(s)
    t_stat = mean / (std / np.sqrt(n)) if std and n > 0 else np.nan
    return {
        "Mean": mean,
        "Std": std,
        "N": n,
        "T_Stat": t_stat,
    }


def build_summary(
    stock_df: pd.DataFrame,
    bond_df: pd.DataFrame,
    factors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a flat summary DataFrame with one row per portfolio / factor.
    Columns: Type, Label, Mean, Std, N, T_Stat
    """
    rows = []

    # Stock portfolios
    for col in STOCK_COLS:
        if col in stock_df.columns:
            stats = _compute_stats(stock_df[col])
            rows.append({"Type": "Stock", "Label": col, **stats})

    # Bond portfolios
    for col in BOND_COLS:
        if col in bond_df.columns:
            stats = _compute_stats(bond_df[col])
            rows.append({"Type": "Bond", "Label": col, **stats})

    # Factor premiums (use Mkt-RF, SMB, HML, TERM, DEF)
    for col in FACTOR_COLS:
        if col in factors_df.columns:
            stats = _compute_stats(factors_df[col])
            rows.append({"Type": "Factor", "Label": col, **stats})

    summary = pd.DataFrame(rows)
    return summary


# ---------------------------------------------------------------------------
# Pretty printing (matching paper Table 2 structure)
# ---------------------------------------------------------------------------


def _print_stock_table(summary: pd.DataFrame) -> None:
    """Print the 5x5 stock portfolio average excess returns grid."""
    stock = summary[summary["Type"] == "Stock"].copy()
    stock["size_q"] = stock["Label"].map(lambda x: SIZE_BM_MAP[x][0])
    stock["bm_q"] = stock["Label"].map(lambda x: SIZE_BM_MAP[x][1])

    # Build grid
    grid = pd.pivot_table(
        stock, values="Mean", index="size_q", columns="bm_q", aggfunc="first"
    )

    print("\n" + "=" * 70)
    print("Table 2: Average Monthly Excess Returns (%)")
    print("          July 1963 - December 1991")
    print("=" * 70)
    print("\nStock Portfolios (25 Size-BE/ME portfolios)")
    print("-" * 70)
    header = f"{'Size':>8s} | {'LoBM':>6s} {'BM2':>6s} {'BM3':>6s} {'BM4':>6s} {'HiBM':>6s}"
    print(header)
    print("-" * 70)
    for sz in range(1, 6):
        label = SIZE_LABELS[sz]
        vals = [grid.loc[sz, bm] for bm in range(1, 6)]
        row_str = " ".join(f"{v:6.2f}" for v in vals)
        print(f"{label:>8s} | {row_str}")
    print("-" * 70)


def _print_bond_table(summary: pd.DataFrame) -> None:
    """Print bond portfolio average excess returns."""
    bond = summary[summary["Type"] == "Bond"].copy()
    print("\nBond Portfolios (7 portfolios)")
    print("-" * 70)
    print(f"{'Portfolio':>15s} | {'Mean':>8s} {'Std':>8s} {'N':>5s} {'T-Stat':>8s}")
    print("-" * 70)
    for _idx, row in bond.iterrows():
        print(
            f"{row['Label']:>15s} | {row['Mean']:8.2f} {row['Std']:8.2f} "
            f"{int(row['N']):5d} {row['T_Stat']:8.2f}"
        )
    print("-" * 70)


def _print_factor_table(summary: pd.DataFrame) -> None:
    """Print factor premium statistics."""
    factors = summary[summary["Type"] == "Factor"].copy()
    print("\nFactor Premiums")
    print("-" * 70)
    print(f"{'Factor':>10s} | {'Mean':>8s} {'Std':>8s} {'N':>5s} {'T-Stat':>8s}")
    print("-" * 70)
    for _idx, row in factors.iterrows():
        print(
            f"{row['Label']:>10s} | {row['Mean']:8.2f} {row['Std']:8.2f} "
            f"{int(row['N']):5d} {row['T_Stat']:8.2f}"
        )
    print("-" * 70)


def print_table2(summary: pd.DataFrame) -> None:
    """Print full Table 2 summary to stdout."""
    _print_stock_table(summary)
    _print_bond_table(summary)
    _print_factor_table(summary)
    print("=" * 70)
    print("\nPaper Pattern Verification:")
    print("-" * 70)

    # Size effect
    stock = summary[summary["Type"] == "Stock"].copy()
    stock["size_q"] = stock["Label"].map(lambda x: SIZE_BM_MAP[x][0])
    stock["bm_q"] = stock["Label"].map(lambda x: SIZE_BM_MAP[x][1])

    size_wins = 0
    for bm in range(1, 6):
        small = stock.loc[(stock["size_q"] == 1) & (stock["bm_q"] == bm), "Mean"].iloc[0]
        big = stock.loc[(stock["size_q"] == 5) & (stock["bm_q"] == bm), "Mean"].iloc[0]
        if small > big:
            size_wins += 1
    print(f"  Size effect (SMALL > BIG): {size_wins}/5 BE/ME groups")

    # Value effect
    value_wins = 0
    for sz in range(1, 6):
        hi = stock.loc[(stock["size_q"] == sz) & (stock["bm_q"] == 5), "Mean"].iloc[0]
        lo = stock.loc[(stock["size_q"] == sz) & (stock["bm_q"] == 1), "Mean"].iloc[0]
        if hi > lo:
            value_wins += 1
    print(f"  Value effect (HiBM > LoBM): {value_wins}/5 size groups")

    # Bond vs stock
    avg_stock = summary[summary["Type"] == "Stock"]["Mean"].mean()
    avg_bond = summary[summary["Type"] == "Bond"]["Mean"].mean()
    print(f"  Bond avg ({avg_bond:.2f}%) < Stock avg ({avg_stock:.2f}%): {avg_bond < avg_stock}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Section 3: The Playing Field - Descriptive Statistics")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    _ensure_factors()
    _ensure_bond_data()
    stock_df = load_stock_portfolios()
    bond_df = load_bond_portfolios()
    factors_df = load_factors()

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary = build_summary(stock_df, bond_df, factors_df)

    # Save to CSV
    output_path = os.path.join(config.OUTPUT_DIR, "table2_summary.csv")
    summary.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {output_path}")

    # Print formatted table
    print_table2(summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
