"""
05_section5_grs_test.py
Section 5: GRS Test - Fama-French (1993) Replication

Implements the Gibbons, Ross, Shanken (1989) F-test for joint significance
of alphas in multi-factor asset pricing models.

H0: All alphas = 0 (model correctly prices all assets)
H1: At least one alpha ≠ 0

Formula:
    F = [(T - N - K) / N] * [α' Σ⁻¹ α] / [1 + θ²]

Where:
    T = number of time periods
    N = number of portfolios
    K = number of factors
    α = N×1 vector of intercepts
    Σ = N×N residual covariance matrix
    θ² = μ_f' Σ_f⁻¹ μ_f  (squared Sharpe ratio of factor portfolios)
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config
from data_loader import load_factors, load_stock_portfolios
import regression_engine as re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOCK_COLS = [
    "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM",
    "ME2 BM1", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 BM5",
    "ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5",
    "ME4 BM1", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5",
    "BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM",
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


def run_regression_for_grs(
    dependent: pd.Series,
    independent: pd.DataFrame,
) -> Dict:
    """
    Run OLS and return alpha, t-stat of alpha, and residuals.

    Returns dict with keys:
        alpha, t_alpha, residuals, r_squared
    """
    data = pd.concat([dependent, independent], axis=1)
    data = data.dropna()

    if data.empty or len(data) < 2:
        return {
            "alpha": np.nan,
            "t_alpha": np.nan,
            "residuals": pd.Series(dtype=float),
            "r_squared": np.nan,
        }

    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    X = sm.add_constant(X, has_constant="add")

    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        return {
            "alpha": np.nan,
            "t_alpha": np.nan,
            "residuals": pd.Series(dtype=float),
            "r_squared": np.nan,
        }

    model = sm.OLS(y, X).fit()
    alpha = model.params["const"]
    t_alpha = model.tvalues["const"]
    residuals = model.resid

    return {
        "alpha": alpha,
        "t_alpha": t_alpha,
        "residuals": residuals,
        "r_squared": model.rsquared,
    }


def compute_grs_test(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    factor_names: List[str],
    test_name: str,
) -> Dict:
    """
    Compute the GRS F-statistic and related metrics.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Portfolio excess returns (columns = portfolios).
    factors_df : pd.DataFrame
        Factor returns.
    factor_names : list of str
        Factors to include in the regression.
    test_name : str
        Label for this test.

    Returns
    -------
    dict with test results.
    """
    # Align data
    aligned = returns_df.join(factors_df[factor_names], how="inner")

    portfolios = returns_df.columns.tolist()
    N = len(portfolios)
    K = len(factor_names)

    # Collect alphas, t-stats, residuals
    alphas = []
    t_alphas = []
    reg_results = []

    for portfolio in portfolios:
        y = aligned[portfolio]
        X = aligned[factor_names]
        res = run_regression_for_grs(y, X)
        alphas.append(res["alpha"])
        t_alphas.append(res["t_alpha"])
        reg_results.append({"residuals": res["residuals"]})

    alphas = np.array(alphas)
    t_alphas = np.array(t_alphas)

    # Align residuals to common index for T
    df_resid = pd.concat([r["residuals"] for r in reg_results], axis=1)
    df_resid = df_resid.dropna()
    T = len(df_resid)

    if T < N + K + 1:
        raise ValueError(
            f"Insufficient observations: T={T}, N={N}, K={K}. "
            f"Need T > N + K + 1 for GRS test."
        )

    # Residual covariance matrix Σ via regression_engine
    Sigma_df = re.compute_residual_covariance(reg_results)
    Sigma = Sigma_df.values

    # Factor returns for this sample period
    factor_data = aligned[factor_names].loc[df_resid.index]
    mu_f = factor_data.mean().values.reshape(-1, 1)
    Sigma_f = factor_data.cov().values

    # Inverse with pinv fallback for singular matrices
    def safe_inv(matrix: np.ndarray) -> np.ndarray:
        try:
            inv = np.linalg.inv(matrix)
            # Check for reasonable condition number
            if np.linalg.cond(matrix) > 1e12:
                inv = np.linalg.pinv(matrix)
            return inv
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    Sigma_inv = safe_inv(Sigma)
    Sigma_f_inv = safe_inv(Sigma_f)

    # θ² = μ_f' Σ_f⁻¹ μ_f
    theta_sq = (mu_f.T @ Sigma_f_inv @ mu_f).item()

    # α' Σ⁻¹ α
    alpha_term = (alphas.reshape(1, -1) @ Sigma_inv @ alphas.reshape(-1, 1)).item()

    # GRS F-statistic
    F_stat = ((T - N - K) / N) * (alpha_term / (1 + theta_sq))

    # p-value from F-distribution
    df1 = N
    df2 = T - N - K
    p_value = 1 - stats.f.cdf(F_stat, df1, df2)

    # Summary statistics
    mean_abs_alpha = np.nanmean(np.abs(alphas))
    mean_abs_t_alpha = np.nanmean(np.abs(t_alphas))

    return {
        "test_name": test_name,
        "N": N,
        "K": K,
        "T": T,
        "F_stat": F_stat,
        "p_value": p_value,
        "mean_abs_alpha": mean_abs_alpha,
        "mean_abs_t_alpha": mean_abs_t_alpha,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_grs_results(results: List[Dict]) -> None:
    """Print formatted GRS test results."""
    print("\n" + "=" * 90)
    print("Table: GRS Test Results (Section 5)")
    print("=" * 90)
    print(
        f"{'Test':<20} {'N':>4} {'K':>4} {'T':>4} "
        f"{'F-stat':>10} {'p-value':>10} {'|α|':>10} {'|t(α)|':>10}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['test_name']:<20} "
            f"{r['N']:>4} {r['K']:>4} {r['T']:>4} "
            f"{r['F_stat']:>10.4f} {r['p_value']:>10.4f} "
            f"{r['mean_abs_alpha']:>10.4f} {r['mean_abs_t_alpha']:>10.4f}"
        )
    print("=" * 90)

    print("\nInterpretation:")
    print("-" * 90)
    for r in results:
        significance = "Rejected" if r["p_value"] < 0.05 else "Not rejected"
        print(
            f"  {r['test_name']}: {significance} at 5% level "
            f"(F={r['F_stat']:.4f}, p={r['p_value']:.4f})"
        )
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Section 5: GRS Test (Gibbons, Ross, Shanken 1989)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    stock_df = load_stock_portfolios()
    bond_df = load_bond_portfolios()
    factors_df = load_factors()

    # Use common sample period where all portfolios and all factors are available
    common_idx = (
        stock_df.index
        .intersection(bond_df.index)
        .intersection(factors_df.dropna().index)
    )
    stock_df = stock_df.loc[common_idx]
    bond_df = bond_df.loc[common_idx]
    factors_df = factors_df.loc[common_idx]

    all_returns = pd.concat([stock_df, bond_df], axis=1)
    print(f"Stock portfolios: {stock_df.shape}")
    print(f"Bond portfolios:  {bond_df.shape}")
    print(f"All portfolios:   {all_returns.shape}")
    print(f"Factors:          {factors_df.shape}")
    print(f"Common period:    {common_idx[0].strftime('%Y-%m')} to {common_idx[-1].strftime('%Y-%m')} ({len(common_idx)} months)")

    # Run GRS tests
    print("\nRunning GRS tests...")
    results = []

    # 1. Stocks with 3-factor model
    results.append(compute_grs_test(
        stock_df, factors_df, ["Mkt-RF", "SMB", "HML"], "Stocks_3Factor"
    ))

    # 2. Stocks with 5-factor model
    results.append(compute_grs_test(
        stock_df, factors_df, ["Mkt-RF", "SMB", "HML", "TERM", "DEF"], "Stocks_5Factor"
    ))

    # 3. Bonds with 2-factor model
    results.append(compute_grs_test(
        bond_df, factors_df, ["TERM", "DEF"], "Bonds_2Factor"
    ))

    # 4. Bonds with 5-factor model
    results.append(compute_grs_test(
        bond_df, factors_df, ["Mkt-RF", "SMB", "HML", "TERM", "DEF"], "Bonds_5Factor"
    ))

    # 5. All portfolios with 5-factor model
    results.append(compute_grs_test(
        all_returns, factors_df, ["Mkt-RF", "SMB", "HML", "TERM", "DEF"], "All_5Factor"
    ))

    # 6. All portfolios with 2-factor SMB+HML model
    results.append(compute_grs_test(
        all_returns, factors_df, ["SMB", "HML"], "All_SMBHML"
    ))

    # Print results
    print_grs_results(results)

    # Save to CSV
    output_path = os.path.join(config.OUTPUT_DIR, "grs_test_results.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved GRS test results to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
