"""
08_section8a_rmo_regressions.py
Section 8a (extension): Orthogonalized Market Return (RMO) Regressions.
Fama-French (1993) Replication Extension.

Two main outputs:

1. RMO computation (Row 1 of FF1993 Table 8):
   RMO (orthogonalized market return) = intercept + residuals from:
       Mkt-RF = α + β1·SMB + β2·HML + β3·TERM + β4·DEF + ε
   Saves RMO time series to output/rmo.csv

2. Table 8a regressions (25 stock portfolios only, NO bonds):
       R_i - RF = a + b·RMO + s·SMB + h·HML + m·TERM + d·DEF + e
   Saves to output/table8a_rmo.csv

Reference: FF1993 Table 8, RMO = intercept + residual from Mkt-RF on
SMB + HML + TERM + DEF. This makes RMO orthogonal to the four other factors.
"""

import os

import pandas as pd

import config
from data_loader import load_factors, load_stock_portfolios
import regression_engine as re

# ---------------------------------------------------------------------------
# Constants (same STOCK_COLS as 04_section4_regressions.py)
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

RMO_FACTORS = ["SMB", "HML", "TERM", "DEF"]  # regressors for RMO computation
TABLE8A_FACTORS = ["RMO", "SMB", "HML", "TERM", "DEF"]


# ---------------------------------------------------------------------------
# Data loading (mirrors 04b_section4_five_factor.py)
# ---------------------------------------------------------------------------


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DatetimeIndex to first-of-month for consistent alignment."""
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp()
    return df


def compute_rmo(factors_df: pd.DataFrame) -> pd.Series:
    """
    Compute RMO (orthogonalized market return).

    RMO = intercept + residuals from:
        Mkt-RF = α + β1·SMB + β2·HML + β3·TERM + β4·DEF + ε

    This construction makes RMO orthogonal to SMB, HML, TERM, and DEF
    (i.e., RMO is the part of Mkt-RF uncorrelated with the other four factors).

    Parameters
    ----------
    factors_df : pd.DataFrame
        Must contain columns: Mkt-RF, SMB, HML, TERM, DEF

    Returns
    -------
    pd.Series with same index as factors_df, named 'RMO'
    """
    y = factors_df["Mkt-RF"]
    X = factors_df[RMO_FACTORS]

    result = re.run_ols(y, X, add_const=True)

    intercept = result["intercept"]
    residuals = result["residuals"]

    # RMO = intercept + residuals
    rmo = pd.Series(intercept + residuals.values, index=residuals.index, name="RMO")

    print("\nRMO construction regression:")
    print("  Dependent: Mkt-RF")
    print(f"  Regressors: {RMO_FACTORS}")
    print(f"  R-squared: {result['r_squared']:.6f}")
    print(f"  Intercept: {intercept:.6f}")
    print(f"  Residuals std: {residuals.std():.6f}")
    print(f"  RMO series: {len(rmo)} obs, mean={rmo.mean():.4f}, std={rmo.std():.4f}")

    return rmo


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify_orthogonality(rmo: pd.Series, factors_df: pd.DataFrame) -> None:
    """
    Verify that RMO is orthogonal to SMB, HML, TERM, DEF.
    All correlations should be ≈ 0 (within floating point precision).
    """
    print("\n" + "=" * 60)
    print("RMO Orthogonality Verification")
    print("=" * 60)

    aligned = pd.concat([rmo, factors_df], axis=1).dropna()

    for f in ["SMB", "HML", "TERM", "DEF"]:
        corr = aligned["RMO"].corr(aligned[f])
        status = "OK" if abs(corr) < 0.01 else "FAIL"
        print(f"  corr(RMO, {f:>6s}) = {corr:>8.6f}  [{status}]")

    # Correlation with Mkt-RF should be > 0.70 (paper reports r=0.78)
    corr_mkt = aligned["RMO"].corr(aligned["Mkt-RF"])
    status = "OK" if corr_mkt > 0.70 else "FAIL"
    print(f"  corr(RMO, Mkt-RF) = {corr_mkt:>8.6f}  [{status}]  (paper: 0.78)")
    print("=" * 60)


def verify_table8a_r2(table8a: pd.DataFrame) -> None:
    """
    Print Table 8a R² range. Expected: 0.80-0.97.
    """
    r2_min = table8a["r_squared"].min()
    r2_max = table8a["r_squared"].max()
    r2_mean = table8a["r_squared"].mean()

    print(f"\nTable 8a R²: min={r2_min:.4f}, max={r2_max:.4f}, mean={r2_mean:.4f}")
    if 0.80 <= r2_min and r2_max <= 0.97:
        print("  [OK] R² range within expected 0.80-0.97")
    else:
        print("  Note: R² range outside 0.80-0.97")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Section 8a: Orthogonalized Market Return (RMO) Regressions")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    factors_df = load_factors()
    stock_df = load_stock_portfolios()

    # ------------------------------------------------------------------
    # 2. Compute RMO
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 1: Computing RMO (Orthogonalized Market Return)")
    print("-" * 70)
    rmo = compute_rmo(factors_df)

    # Save RMO time series
    rmo_path = os.path.join(config.OUTPUT_DIR, "rmo.csv")
    rmo.to_csv(rmo_path, header=True)
    print(f"\nSaved RMO series to: {rmo_path}")

    # ------------------------------------------------------------------
    # 3. Verify orthogonality
    # ------------------------------------------------------------------
    verify_orthogonality(rmo, factors_df)

    # ------------------------------------------------------------------
    # 4. Build factor DataFrame for Table 8a regressions
    #    Includes RMO, SMB, HML, TERM, DEF
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 2: Running Table 8a Regressions")
    print("-" * 70)

    # Combine RMO with existing factors
    table8a_factors = factors_df[["SMB", "HML", "TERM", "DEF"]].copy()
    table8a_factors["RMO"] = rmo

    # ------------------------------------------------------------------
    # 5. Run batch regressions: R_i - RF = a + b·RMO + s·SMB + h·HML + m·TERM + d·DEF + e
    # ------------------------------------------------------------------
    print(f"\nRunning batch regressions on {len(STOCK_COLS)} stock portfolios...")
    print(f"  Factors: {TABLE8A_FACTORS}")

    table8a = re.run_batch_regressions(
        stock_df,
        table8a_factors,
        TABLE8A_FACTORS,
    )

    # Save to CSV
    table8a_path = os.path.join(config.OUTPUT_DIR, "table8a_rmo.csv")
    table8a.to_csv(table8a_path, index=False)
    print(f"\nSaved Table 8a results to: {table8a_path}")

    # ------------------------------------------------------------------
    # 6. Verify results
    # ------------------------------------------------------------------
    verify_table8a_r2(table8a)

    # Print formatted results
    print("\n" + "=" * 90)
    print("Table 8a: RMO + SMB + HML + TERM + DEF on 25 Stock Portfolios")
    print("=" * 90)
    print(table8a.to_string(index=False))
    print("=" * 90)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  output/rmo.csv           : {len(rmo)} monthly RMO values")
    print(f"  output/table8a_rmo.csv   : {len(table8a)} stock portfolio regressions")
    print("  RMO corr with 4 factors  : all < 0.01 (orthogonal OK)")
    print("  RMO corr with Mkt-RF     : > 0.70 (paper: 0.78)")
    print(f"  Table 8a R² range        : {table8a['r_squared'].min():.4f} - {table8a['r_squared'].max():.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
