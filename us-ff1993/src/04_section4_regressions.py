"""
04_section4_regressions.py
Section 4: Common Variation - Fama-French (1993) Replication

Runs three regression specifications on 32 portfolios (25 stock + 7 bond):

Table 1 (Section 4.2): Market factor alone
    R_i - R_f = α + β(R_m - R_f) + ε
    Factors: ["Mkt-RF"]
    Save: output/table1_market.csv

Table 3 (Section 4.1): Bond-market factors alone
    R_i - R_f = α + m(TERM) + d(DEF) + ε
    Factors: ["TERM", "DEF"]
    Save: output/table3_bond.csv

Table 4 (Section 4.3): Three stock-market factors
    R_i - R_f = α + β(R_m - R_f) + s(SMB) + h(HML) + ε
    Factors: ["Mkt-RF", "SMB", "HML"]
    Save: output/table4_stock3f.csv
"""

import os
import subprocess
import sys

import pandas as pd

import config
from data_loader import load_factors, load_stock_portfolios, load_bond_portfolios
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

ALL_PORTFOLIOS = STOCK_COLS + BOND_COLS


# ---------------------------------------------------------------------------
# Data loading (mirrors 03_section3_statistics.py)
# ---------------------------------------------------------------------------


def _ensure_factors() -> None:
    """Ensure combined factors CSV has all 5 factors."""
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

    fetcher_script = os.path.join(base_dir, "fred_bond_fetcher.py")
    if os.path.exists(fetcher_script):
        print("Running fred_bond_fetcher.py...")
        subprocess.run(
            [sys.executable, fetcher_script],
            capture_output=True,
            text=True,
            cwd=base_dir,
        )

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
    """Ensure bond portfolio data is available."""
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

    fetcher_script = os.path.join(base_dir, "fred_bond_fetcher.py")
    if os.path.exists(fetcher_script):
        subprocess.run(
            [sys.executable, fetcher_script],
            capture_output=True,
            text=True,
            cwd=base_dir,
        )

    bond_script = os.path.join(base_dir, "02b_section2_bond_portfolios.py")
    if os.path.exists(bond_script):
        subprocess.run(
            [sys.executable, bond_script],
            capture_output=True,
            text=True,
            cwd=base_dir,
        )


def _portfolio_type(name: str) -> str:
    return "stock" if name in STOCK_COLS else "bond"


def run_specification(
    combined_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    factor_names: list,
    output_filename: str,
    table_title: str,
) -> pd.DataFrame:
    """
    Run batch regression for the given factor specification,
    add a 'type' column, save to CSV, and print formatted results.
    """
    results = re.run_batch_regressions(combined_df, factors_df, factor_names)

    # Add portfolio type
    results["type"] = results["portfolio"].apply(_portfolio_type)

    # Reorder columns: portfolio, type, alpha, betas, t-stats, r_squared
    cols = ["portfolio", "type", "alpha"]
    for f in factor_names:
        cols.append(f"beta_{f}")
        cols.append(f"t_{f}")
    cols.append("r_squared")
    results = results[cols]

    # Save
    out_path = os.path.join(config.OUTPUT_DIR, output_filename)
    results.to_csv(out_path, index=False)
    print(f"\nSaved {output_filename} ({len(results)} portfolios)")

    # Pretty-print
    print(re.format_regression_table(results, table_name=table_title))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Section 4: Common Variation - Time-Series Regressions")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    _ensure_factors()
    _ensure_bond_data()

    stock_df = load_stock_portfolios()
    bond_df = load_bond_portfolios()
    factors_df = load_factors()

    # Normalize all indices to month-start so stock (1st of month)
    # and bond (end of month) align properly for regression.
    stock_df.index = stock_df.index.to_period("M").to_timestamp()
    bond_df.index = bond_df.index.to_period("M").to_timestamp()
    factors_df.index = factors_df.index.to_period("M").to_timestamp()

    # Combine 25 stock + 7 bond portfolios into one wide DataFrame
    combined_df = pd.concat([stock_df, bond_df], axis=1)
    print(f"Combined portfolios: {combined_df.shape[0]} months x {combined_df.shape[1]} portfolios")

    # -------------------------------------------------------------------
    # Table 1: Market factor alone (Section 4.2)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Table 1: Market Factor Alone (Section 4.2)")
    print("=" * 70)
    t1 = run_specification(
        combined_df,
        factors_df,
        ["Mkt-RF"],
        "table1_market.csv",
        "Table 1: Market Factor Alone",
    )

    # -------------------------------------------------------------------
    # Table 3: Bond-market factors alone (Section 4.1)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Table 3: Bond-Market Factors Alone (Section 4.1)")
    print("=" * 70)
    t3 = run_specification(
        combined_df,
        factors_df,
        ["TERM", "DEF"],
        "table3_bond.csv",
        "Table 3: Bond-Market Factors Alone",
    )

    # -------------------------------------------------------------------
    # Table 4: Three stock-market factors (Section 4.3)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Table 4: Three Stock-Market Factors (Section 4.3)")
    print("=" * 70)
    t4 = run_specification(
        combined_df,
        factors_df,
        ["Mkt-RF", "SMB", "HML"],
        "table4_stock3f.csv",
        "Table 4: Three Stock-Market Factors",
    )

    # -------------------------------------------------------------------
    # Table 5: SMB and HML factors alone (FF1993 Table 5)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Table 5: SMB and HML Factors Alone (FF1993 Table 5)")
    print("=" * 70)
    t5 = run_specification(
        combined_df,
        factors_df,
        ["SMB", "HML"],
        "table5_smbhml.csv",
        "Table 5: SMB and HML Factors Alone",
    )

    # -------------------------------------------------------------------
    # Paper pattern verification
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Paper Pattern Verification")
    print("=" * 70)

    stock_t1 = t1[t1["type"] == "stock"]["r_squared"]
    bond_t1 = t1[t1["type"] == "bond"]["r_squared"]
    print(f"Table 1 stock R2: min={stock_t1.min():.4f}, max={stock_t1.max():.4f}, mean={stock_t1.mean():.4f}")
    print(f"Table 1 bond  R2: min={bond_t1.min():.4f}, max={bond_t1.max():.4f}, mean={bond_t1.mean():.4f}")

    term_sig = (t3["t_TERM"].abs() > 2.0).sum()
    print(f"Table 3 TERM |t|>2.0: {term_sig}/{len(t3)} portfolios")

    stock_t4 = t4[t4["type"] == "stock"]["r_squared"]
    print(f"Table 4 stock R2: min={stock_t4.min():.4f}, max={stock_t4.max():.4f}, mean={stock_t4.mean():.4f}")
    print(f"Table 4 vs Table 1 stock avg R2 improvement: +{stock_t4.mean() - stock_t1.mean():.4f}")

    stock_t5 = t5[t5["type"] == "stock"]["r_squared"]
    print(f"Table 5 stock R2: min={stock_t5.min():.4f}, max={stock_t5.max():.4f}, mean={stock_t5.mean():.4f}")
    print(f"Table 5 vs Table 1 stock avg R2 improvement: +{stock_t5.mean() - stock_t1.mean():.4f}")
    print(f"Table 5 vs Table 4 stock avg R2 improvement: +{stock_t5.mean() - stock_t4.mean():.4f}")

    smb_sig = (t5["t_SMB"].abs() > 2.0).sum()
    hml_sig = (t5["t_HML"].abs() > 2.0).sum()
    print(f"Table 5 SMB |t|>2.0: {smb_sig}/{len(t5)} portfolios")
    print(f"Table 5 HML |t|>2.0: {hml_sig}/{len(t5)} portfolios")

    print("=" * 70)
    print("\nDone!")


if __name__ == "__main__":
    main()
