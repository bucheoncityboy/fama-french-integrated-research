"""
regression_engine.py
Shared regression infrastructure for Fama-French replication.

Provides:
- run_ols: single time-series OLS via statsmodels
- run_batch_regressions: run identical spec across many portfolios
- compute_residual_covariance: N×N residual covariance for GRS test
- format_regression_table: pretty-print results
"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# 1. Single OLS wrapper
# ---------------------------------------------------------------------------

def run_ols(
    dependent: pd.Series,
    independent: pd.DataFrame,
    add_const: bool = True,
) -> Dict[str, Union[Dict, float, None]]:
    """
    Run OLS regression via statsmodels.

    Parameters
    ----------
    dependent : pd.Series
        Dependent variable (e.g. portfolio excess return).
    independent : pd.DataFrame
        Independent variables (e.g. factor returns).
    add_const : bool, default True
        Whether to prepend a constant (intercept) term.

    Returns
    -------
    dict with keys:
        coefficients : {var_name: coef_value}
        t_stats      : {var_name: t_stat}
        p_values     : {var_name: p_value}
        r_squared    : float
        adj_r_squared: float
        intercept    : float or None
        residuals    : pd.Series
    """
    # Align and drop NaN rows
    data = pd.concat([dependent, independent], axis=1)
    data = data.dropna()

    if data.empty or len(data) < 2:
        # Insufficient data → return NaN shell
        coefs = {col: np.nan for col in independent.columns}
        return {
            "coefficients": coefs,
            "t_stats": {k: np.nan for k in coefs},
            "p_values": {k: np.nan for k in coefs},
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
            "intercept": np.nan if add_const else None,
            "intercept_t_stat": np.nan if add_const else None,
            "residuals": pd.Series(dtype=float),
        }

    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    if add_const:
        X = sm.add_constant(X, has_constant="add")

    # Check for rank deficiency after dropping NaNs
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        coefs = {col: np.nan for col in independent.columns}
        return {
            "coefficients": coefs,
            "t_stats": {k: np.nan for k in coefs},
            "p_values": {k: np.nan for k in coefs},
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
            "intercept": np.nan if add_const else None,
            "intercept_t_stat": np.nan if add_const else None,
            "residuals": pd.Series(dtype=float),
        }

    model = sm.OLS(y, X).fit()

    # Build output dicts
    coef_dict = model.params.to_dict()
    t_dict = model.tvalues.to_dict()
    p_dict = model.pvalues.to_dict()

    # Clean intercept extraction
    if add_const:
        intercept = coef_dict.pop("const", np.nan)
        intercept_t_stat = t_dict.pop("const", np.nan)
    else:
        intercept = None
        intercept_t_stat = None

    # Ensure only original factor names remain in coefficient dicts
    coeff_out = {k: coef_dict.get(k, np.nan) for k in independent.columns}
    t_out = {k: t_dict.get(k, np.nan) for k in independent.columns}
    p_out = {k: p_dict.get(k, np.nan) for k in independent.columns}

    return {
        "coefficients": coeff_out,
        "t_stats": t_out,
        "p_values": p_out,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "intercept": intercept,
        "intercept_t_stat": intercept_t_stat,
        "residuals": model.resid,
    }


# ---------------------------------------------------------------------------
# 2. Batch regressions across portfolios
# ---------------------------------------------------------------------------

def run_batch_regressions(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    factor_names: List[str],
) -> pd.DataFrame:
    """
    Run identical OLS specification for every portfolio in *returns_df*.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Columns = portfolio returns, Index = dates.
    factors_df : pd.DataFrame
        Columns = factor returns, Index = dates.
    factor_names : list of str
        Subset of columns in *factors_df* to use as regressors.

    Returns
    -------
    pd.DataFrame with one row per portfolio:
        portfolio, alpha, beta_<factor>, t_<factor>, r_squared
    """
    # Align indices (inner join)
    aligned = returns_df.join(factors_df[factor_names], how="inner")

    rows = []
    for portfolio in returns_df.columns:
        y = aligned[portfolio]
        X = aligned[factor_names]
        res = run_ols(y, X, add_const=True)

        row = {"portfolio": portfolio}
        row["alpha"] = res["intercept"]
        for f in factor_names:
            row[f"beta_{f}"] = res["coefficients"].get(f, np.nan)
            row[f"t_{f}"] = res["t_stats"].get(f, np.nan)
        row["r_squared"] = res["r_squared"]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Residual covariance matrix (for GRS test)
# ---------------------------------------------------------------------------

def compute_residual_covariance(
    regression_results: List[Dict],
) -> pd.DataFrame:
    """
    Build N×N covariance matrix from residuals of N regressions.

    Parameters
    ----------
    regression_results : list of dict
        Each dict must contain key 'residuals' (pd.Series).

    Returns
    -------
    pd.DataFrame (N×N) with portfolio names as index/columns.
    Returns the raw covariance matrix. Downstream consumers that need
    inversion should use np.linalg.pinv for near-singular matrices.
    """
    # Collect residuals
    resid_list = []
    names = []
    for i, res in enumerate(regression_results):
        resid = res.get("residuals", pd.Series(dtype=float))
        resid_list.append(resid)
        names.append(f"Portfolio_{i}")

    # Align all residuals to common index and drop NaN rows
    df_resid = pd.concat(resid_list, axis=1)
    df_resid = df_resid.dropna()

    if df_resid.empty or df_resid.shape[0] < 2:
        return pd.DataFrame(np.nan, index=names, columns=names)

    # Compute covariance
    cov = df_resid.cov()

    # Return the raw covariance matrix. Downstream consumers that need inversion
    # (e.g., the GRS test) should use np.linalg.pinv for near-singular matrices.
    # The GRS test in 05_section5_grs_test.py already implements safe_inv() with
    # pinv fallback, so no regularisation is needed here.

    cov.index = names
    cov.columns = names
    return cov


# ---------------------------------------------------------------------------
# 4. Pretty-print helper
# ---------------------------------------------------------------------------

def format_regression_table(
    results: Union[Dict, pd.DataFrame],
    table_name: str = "Regression Results",
) -> str:
    """
    Format regression output as a printable string.

    Parameters
    ----------
    results : dict or pd.DataFrame
        Single regression result dict, or batch DataFrame from
        run_batch_regressions.
    table_name : str
        Header title for the table.

    Returns
    -------
    str
    """
    lines = [f"{'=' * 60}", f"{table_name:^60}", f"{'=' * 60}"]

    if isinstance(results, dict):
        # Single regression
        lines.append(f"R-squared    : {results.get('r_squared', np.nan):.4f}")
        lines.append(f"Adj R-squared: {results.get('adj_r_squared', np.nan):.4f}")
        lines.append(f"Intercept    : {results.get('intercept', np.nan):.4f}")
        lines.append("-" * 60)
        lines.append(f"{'Variable':<15} {'Coef':>10} {'t-stat':>10} {'p-value':>10}")
        lines.append("-" * 60)
        for var, coef in results.get("coefficients", {}).items():
            t = results.get("t_stats", {}).get(var, np.nan)
            p = results.get("p_values", {}).get(var, np.nan)
            lines.append(
                f"{var:<15} {coef:>10.4f} {t:>10.4f} {p:>10.4f}"
            )

    elif isinstance(results, pd.DataFrame):
        # Batch DataFrame
        lines.append(results.to_string(index=False))

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)
