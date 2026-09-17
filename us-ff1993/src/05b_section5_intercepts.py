"""
05b_section5_intercepts.py
Section 5: Individual Intercept Analysis - Fama-French (1993) Replication

Analyzes individual intercepts (alphas) from all regression models:
- 1-factor (Mkt-RF)
- 2-factor bond (TERM, DEF)
- 3-factor stock (Mkt-RF, SMB, HML)
- 5-factor (all)

For each model and each portfolio, we re-run the regression to obtain the
intercept and its t-statistic, then build a cross-section summary.

Key paper findings to verify:
- For 5-factor model on stocks: average |alpha| < 0.20%/month (economically small)
- Adding factors reduces average |alpha| compared to single-factor model
- Even if GRS rejects jointly, individual alphas should be small
"""

import os

import numpy as np
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

ALL_PORTFOLIOS = STOCK_COLS + BOND_COLS

MODEL_SPECS = [
    (["Mkt-RF"], "one_factor"),
    (["TERM", "DEF"], "two_factor_bond"),
    (["Mkt-RF", "SMB", "HML"], "three_factor_stock"),
    (["Mkt-RF", "SMB", "HML", "TERM", "DEF"], "five_factor"),
    (["SMB", "HML"], "two_factor_stock"),
]


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


def _portfolio_type(name: str) -> str:
    return "stock" if name in STOCK_COLS else "bond"


def run_intercept_analysis(
    combined_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    factor_names: list,
    model_label: str,
) -> pd.DataFrame:
    """
    Run identical OLS specification for every portfolio and extract
    intercept (alpha) and its t-statistic.

    Returns DataFrame with one row per portfolio:
        portfolio, type, model, alpha, t_alpha, abs_alpha, significant
    """
    aligned = combined_df.join(factors_df[factor_names], how="inner")

    rows = []
    for portfolio in combined_df.columns:
        y = aligned[portfolio]
        X = aligned[factor_names]
        res = re.run_ols(y, X, add_const=True)

        alpha = res["intercept"]
        t_alpha = res["intercept_t_stat"]

        rows.append(
            {
                "portfolio": portfolio,
                "type": _portfolio_type(portfolio),
                "model": model_label,
                "alpha": alpha,
                "t_alpha": t_alpha,
                "abs_alpha": abs(alpha) if pd.notna(alpha) else np.nan,
                "significant": abs(t_alpha) > 2.0 if pd.notna(t_alpha) else False,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_summary(intercept_df: pd.DataFrame) -> None:
    """Print formatted intercept analysis summary."""
    print("\n" + "=" * 70)
    print("Section 5: Individual Intercept Analysis")
    print("=" * 70)

    # Per-model, per-type summary
    summary = (
        intercept_df.groupby(["model", "type"])
        .agg(mean_abs_alpha=("abs_alpha", "mean"), significant_count=("significant", "sum"))
        .reset_index()
    )

    print("\nMean |alpha| and Significant Count by Model and Type:")
    print("-" * 70)
    print(f"{'Model':<20} {'Type':<8} {'Mean |α|':>12} {'|t|>2.0':>10}")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(
            f"{row['model']:<20} {row['type']:<8} "
            f"{row['mean_abs_alpha']:>12.4f} {int(row['significant_count']):>10d}"
        )
    print("=" * 70)

    # Stock-only comparison
    stock_summary = summary[summary["type"] == "stock"].copy()
    if not stock_summary.empty:
        print("\nStock Portfolio |alpha| Comparison:")
        print("-" * 70)
        for _, row in stock_summary.iterrows():
            print(f"  {row['model']:<20} {row['mean_abs_alpha']:>12.4f}")
        print("-" * 70)

        one_f = stock_summary[stock_summary["model"] == "one_factor"]["mean_abs_alpha"]
        three_f = stock_summary[stock_summary["model"] == "three_factor_stock"]["mean_abs_alpha"]
        five_f = stock_summary[stock_summary["model"] == "five_factor"]["mean_abs_alpha"]

        if not one_f.empty and not three_f.empty:
            print(f"  Reduction 1f -> 3f: {one_f.iloc[0] - three_f.iloc[0]:>12.4f}")
        if not three_f.empty and not five_f.empty:
            print(f"  Reduction 3f -> 5f: {three_f.iloc[0] - five_f.iloc[0]:>12.4f}")
        print("=" * 70)

    # Bond-only comparison
    bond_summary = summary[summary["type"] == "bond"].copy()
    if not bond_summary.empty:
        print("\nBond Portfolio |alpha| Comparison:")
        print("-" * 70)
        for _, row in bond_summary.iterrows():
            print(f"  {row['model']:<20} {row['mean_abs_alpha']:>12.4f}")
        print("=" * 70)

    # Key paper findings
    print("\nPaper Findings Verification:")
    print("-" * 70)
    ff5_stock = stock_summary[stock_summary["model"] == "five_factor"]["mean_abs_alpha"]
    if not ff5_stock.empty:
        verdict = "PASS" if ff5_stock.iloc[0] < 0.20 else "FAIL"
        print(
            f"  5-factor stock avg |alpha| < 0.20:  {ff5_stock.iloc[0]:.4f}  [{verdict}]"
        )

    if not one_f.empty and not ff5_stock.empty:
        verdict = "PASS" if ff5_stock.iloc[0] < one_f.iloc[0] else "FAIL"
        print(
            f"  5f avg |alpha| < 1f avg |alpha|:   {ff5_stock.iloc[0]:.4f} < {one_f.iloc[0]:.4f}  [{verdict}]"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Section 5: Individual Intercept Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    stock_df = load_stock_portfolios()
    bond_df = load_bond_portfolios()
    factors_df = load_factors()

    combined_df = pd.concat([stock_df, bond_df], axis=1)
    print(f"Combined portfolios: {combined_df.shape[0]} months x {combined_df.shape[1]} portfolios")

    # Run intercept analysis for each model
    print("\nRunning intercept analysis for 4 models...")
    all_results = []
    for factor_names, model_label in MODEL_SPECS:
        df = run_intercept_analysis(combined_df, factors_df, factor_names, model_label)
        all_results.append(df)
        print(f"  {model_label}: {len(df)} portfolios")

    intercept_df = pd.concat(all_results, ignore_index=True)

    # Save
    out_path = os.path.join(config.OUTPUT_DIR, "intercept_analysis.csv")
    intercept_df.to_csv(out_path, index=False)
    print(f"\nSaved intercept analysis to: {out_path}")

    # Print summary
    print_summary(intercept_df)
    print("\nDone!")


if __name__ == "__main__":
    main()
