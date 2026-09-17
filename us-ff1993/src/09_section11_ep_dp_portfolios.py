"""
09_section11_ep_dp_portfolios.py
Section 11 (Additional): E/P and D/P Portfolio Statistics and Regressions
Fama-French (1993) Replication — Table 11

Computes for 12 portfolios (6 E/P + 6 D/P):
  Panel 1: Descriptive statistics (mean, std, t-stat) on raw returns
  Panel 2: CAPM regressions  (excess returns ~ Mkt-RF)
  Panel 3: FF3F regressions  (excess returns ~ Mkt-RF + SMB + HML)

Saves results to output/table11_ep_dp.csv in long format.
"""

import os

import numpy as np
import pandas as pd

import config
from data_loader import load_factors
import regression_engine as re

# ---------------------------------------------------------------------------
# Portfolio column definitions
# ---------------------------------------------------------------------------

EP_COLS = ["EP_Neg", "EP_Low", "EP_2", "EP_3", "EP_4", "EP_High"]
DP_COLS = ["DP_Zero", "DP_Low", "DP_2", "DP_3", "DP_4", "DP_High"]
ALL_COLS = EP_COLS + DP_COLS

# Display labels matching the paper's taxonomy (Table 11)
EP_LABELS = {
    "EP_Neg": "EP <= 0",
    "EP_Low": "Low",
    "EP_2": "2",
    "EP_3": "3",
    "EP_4": "4",
    "EP_High": "High",
}
DP_LABELS = {
    "DP_Zero": "DP = 0",
    "DP_Low": "Low",
    "DP_2": "2",
    "DP_3": "3",
    "DP_4": "4",
    "DP_High": "High",
}


def portfolio_type(col: str) -> str:
    """Return 'EP' or 'DP' for a given column name."""
    return "EP" if col in EP_COLS else "DP"


def portfolio_label(col: str) -> str:
    """Return the display label for a portfolio column."""
    if col in EP_LABELS:
        return EP_LABELS[col]
    return DP_LABELS.get(col, col)


def extreme_portfolios_by_type() -> dict[str, list[str]]:
    """Return extreme portfolios to verify for each portfolio family."""
    return {
        "EP": ["EP <= 0", "High"],
        "DP": ["DP = 0", "High"],
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ep_dp_returns() -> pd.DataFrame:
    """Load raw E/P and D/P portfolio returns (already in %/month)."""
    path = os.path.join(config.DATA_DIR, "ep_dp_portfolios.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    print(f"Loaded EP/DP portfolios: {df.shape}")
    return df


def compute_descriptive_stats(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each portfolio, compute mean, std (ddof=1), and t-stat on raw returns.

    Returns
    -------
    pd.DataFrame with columns:
        portfolio, type, stat_type, model, value
    """
    rows = []
    for col in returns_df.columns:
        s = returns_df[col].dropna()
        mean = s.mean()
        std = s.std(ddof=1)
        n = len(s)
        t_stat = mean / (std / np.sqrt(n)) if std > 0 and n > 0 else np.nan
        ptype = portfolio_type(col)
        label = portfolio_label(col)

        rows.append(
            {
                "portfolio": label,
                "type": ptype,
                "stat_type": "mean",
                "model": "descriptive",
                "value": round(mean, 4),
            }
        )
        rows.append(
            {
                "portfolio": label,
                "type": ptype,
                "stat_type": "std",
                "model": "descriptive",
                "value": round(std, 4),
            }
        )
        rows.append(
            {
                "portfolio": label,
                "type": ptype,
                "stat_type": "t",
                "model": "descriptive",
                "value": round(t_stat, 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regressions (Panels 2 & 3) — use excess returns
# ---------------------------------------------------------------------------


def run_regressions(
    excess_returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    factor_names: list,
    model_name: str,
) -> pd.DataFrame:
    """
    Run batch OLS regressions on excess returns and return long-format results.

    Parameters
    ----------
    excess_returns_df : pd.DataFrame
        Portfolio excess returns (R_raw - RF).
    factors_df : pd.DataFrame
        Factor returns (Mkt-RF, SMB, HML, etc.).
    factor_names : list of str
        Columns from factors_df to use as regressors.
    model_name : str
        Label for the model (e.g. 'capm' or 'ff3f').

    Returns
    -------
    pd.DataFrame with columns:
        portfolio, type, stat_type, model, value
    """
    # Align indices (inner join)
    aligned = excess_returns_df.join(factors_df[factor_names], how="inner")

    rows = []
    for col in excess_returns_df.columns:
        y = aligned[col]
        X = aligned[factor_names]
        res = re.run_ols(y, X, add_const=True)

        ptype = portfolio_type(col)
        label = portfolio_label(col)
        intercept = res.get("intercept")

        # Alpha (intercept)
        rows.append(
            {
                "portfolio": label,
                "type": ptype,
                "stat_type": "alpha",
                "model": model_name,
                "value": round(intercept, 4) if intercept is not None else np.nan,
            }
        )

        # Betas for each factor
        for f in factor_names:
            beta_key = f"beta_{f}"
            rows.append(
                {
                    "portfolio": label,
                    "type": ptype,
                    "stat_type": beta_key,
                    "model": model_name,
                    "value": round(res["coefficients"].get(f, np.nan), 4),
                }
            )

        # R-squared
        rows.append(
            {
                "portfolio": label,
                "type": ptype,
                "stat_type": "r_squared",
                "model": model_name,
                "value": round(res["r_squared"], 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Table 11: E/P and D/P Portfolio Statistics and Regressions")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    print("\nLoading data...")
    raw_returns = load_ep_dp_returns()       # raw returns (%/month)
    factors_df = load_factors()

    # Normalise all indices to month-start timestamps for alignment
    raw_returns.index = raw_returns.index.to_period("M").to_timestamp()
    factors_df.index = factors_df.index.to_period("M").to_timestamp()

    # Subset to the 12 portfolios of interest
    returns_12 = raw_returns[ALL_COLS]
    print(f"Portfolio returns: {returns_12.shape[0]} months x {returns_12.shape[1]} portfolios")

    # Compute excess returns: R_excess = R_raw - RF
    rf_series = factors_df["RF"]
    common_idx = returns_12.index.intersection(rf_series.index)
    returns_aligned = returns_12.loc[common_idx]
    rf_aligned = rf_series.loc[common_idx]
    excess_returns = returns_aligned.sub(rf_aligned, axis=0)
    print(f"Excess returns: {excess_returns.shape[0]} months")

    # -------------------------------------------------------------------
    # Panel 1: Descriptive statistics (raw returns)
    # -------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Panel 1: Descriptive Statistics (raw returns)")
    print("-" * 70)

    desc_stats = compute_descriptive_stats(returns_12)
    print(desc_stats.to_string(index=False))

    # -------------------------------------------------------------------
    # Panel 2: CAPM regression (excess returns ~ Mkt-RF)
    # -------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Panel 2: CAPM Regressions  R-RF = a + b*(Mkt-RF)")
    print("-" * 70)

    capm_results = run_regressions(
        excess_returns, factors_df, ["Mkt-RF"], "capm"
    )
    print(capm_results.to_string(index=False))

    # -------------------------------------------------------------------
    # Panel 3: FF3F regression (excess returns ~ Mkt-RF + SMB + HML)
    # -------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Panel 3: FF3F Regressions  R-RF = a + b*(Mkt-RF) + s*SMB + h*HML")
    print("-" * 70)

    ff3f_results = run_regressions(
        excess_returns, factors_df, ["Mkt-RF", "SMB", "HML"], "ff3f"
    )
    print(ff3f_results.to_string(index=False))

    # -------------------------------------------------------------------
    # Combine and save
    # -------------------------------------------------------------------
    combined = pd.concat([desc_stats, capm_results, ff3f_results], ignore_index=True)

    # Sort: type (EP first, then DP), portfolio order, stat_type order, model
    type_order = {"EP": 0, "DP": 1}
    ep_order = {label: i for i, label in enumerate(EP_LABELS.values())}
    dp_order = {label: i for i, label in enumerate(DP_LABELS.values())}
    stat_order = {
        "mean": 0,
        "std": 1,
        "t": 2,
        "alpha": 3,
        "beta_Mkt-RF": 4,
        "beta_SMB": 5,
        "beta_HML": 6,
        "r_squared": 7,
    }

    combined["_type_order"] = combined["type"].map(type_order)
    combined["_port_order"] = combined["portfolio"].map(
        lambda x: ep_order.get(x, dp_order.get(x, 99))
    )
    combined["_stat_order"] = combined["stat_type"].map(stat_order)
    combined = combined.sort_values(
        ["_type_order", "_port_order", "_stat_order", "model"]
    ).drop(columns=["_type_order", "_port_order", "_stat_order"])

    # Column order
    combined = combined[["portfolio", "type", "stat_type", "model", "value"]]

    # Save
    out_path = os.path.join(config.OUTPUT_DIR, "table11_ep_dp.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved Table 11 to: {out_path} ({len(combined)} rows)")

    # -------------------------------------------------------------------
    # Quick verification
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    # EP value premium: High > Low
    ep_means = desc_stats[
        (desc_stats["type"] == "EP") & (desc_stats["stat_type"] == "mean")
    ]
    ep_high_mean = ep_means[ep_means["portfolio"] == "High"]["value"].values[0]
    ep_low_mean = ep_means[ep_means["portfolio"] == "Low"]["value"].values[0]
    ep_neg_mean = ep_means[ep_means["portfolio"] == "EP <= 0"]["value"].values[0]
    print("\nEP portfolio raw means:")
    print(f"  EP_Neg  (<= 0):  {ep_neg_mean:>6.2f}%/mo")
    print(f"  EP_Low  (Low):   {ep_low_mean:>6.2f}%/mo")
    print(f"  EP_High (High):  {ep_high_mean:>6.2f}%/mo")
    print(f"  Value premium (EP_High > EP_Low): {ep_high_mean > ep_low_mean}")

    # DP value premium: High > Low
    dp_means = desc_stats[
        (desc_stats["type"] == "DP") & (desc_stats["stat_type"] == "mean")
    ]
    dp_high_mean = dp_means[dp_means["portfolio"] == "High"]["value"].values[0]
    dp_low_mean = dp_means[dp_means["portfolio"] == "Low"]["value"].values[0]
    dp_zero_mean = dp_means[dp_means["portfolio"] == "DP = 0"]["value"].values[0]
    print("\nDP portfolio raw means:")
    print(f"  DP_Zero (= 0):   {dp_zero_mean:>6.2f}%/mo")
    print(f"  DP_Low  (Low):   {dp_low_mean:>6.2f}%/mo")
    print(f"  DP_High (High):  {dp_high_mean:>6.2f}%/mo")
    print(f"  Value premium (DP_High > DP_Low): {dp_high_mean > dp_low_mean}")

    # R² improvement for extreme portfolios (CAPM vs FF3F)
    print("\nR² comparison for extreme portfolios:")
    for ptype, portfolios in extreme_portfolios_by_type().items():
        for port in portfolios:
            subset = combined[
                (combined["portfolio"] == port)
                & (combined["type"] == ptype)
                & (combined["stat_type"] == "r_squared")
            ]
            if subset.empty:
                print(f"  {ptype} {port}: skipped (no matching rows)")
                continue
            capm_r2 = subset[subset["model"] == "capm"]["value"].values[0]
            ff3f_r2 = subset[subset["model"] == "ff3f"]["value"].values[0]
            improved = ff3f_r2 > capm_r2
            print(
                f"  {ptype} {port}: CAPM R²={capm_r2:.4f}, "
                f"FF3F R²={ff3f_r2:.4f}, improved={improved}"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
