"""
run_regression_korea.py
Run FF3 regression on Korean 25 portfolios and compute GRS test.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Ensure fama-ff3- regression_engine is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fama-ff3-"))
import regression_engine as re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FACTORS_PATH = os.path.join(DATA_DIR, "factors_korea.csv")
PORTFOLIOS_PATH = os.path.join(DATA_DIR, "portfolios_25_korea.csv")

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
factors = pd.read_csv(FACTORS_PATH, parse_dates=["date"])
factors["date"] = pd.to_datetime(factors["date"])
factors = factors.set_index("date")

portfolios = pd.read_csv(PORTFOLIOS_PATH, parse_dates=["Date"])
portfolios["Date"] = pd.to_datetime(portfolios["Date"])
portfolios = portfolios.set_index("Date")

print(f"Factors shape: {factors.shape}")
print(f"Portfolios shape: {portfolios.shape}")

# Align dates (inner join)
common_dates = portfolios.index.intersection(factors.index)
portfolios = portfolios.loc[common_dates]
factors = factors.loc[common_dates]

print(f"Common dates: {len(common_dates)} ({common_dates[0].strftime('%Y-%m')} to {common_dates[-1].strftime('%Y-%m')})")

# ---------------------------------------------------------------------------
# Compute excess returns: R_p - RF
# ---------------------------------------------------------------------------
rf = factors["rf"]
excess_returns = portfolios.subtract(rf, axis=0)

# ---------------------------------------------------------------------------
# Run batch regressions using regression_engine
# ---------------------------------------------------------------------------
# run_batch_regressions gives: portfolio, alpha, beta_<factor>, t_<factor>, r_squared
# We also need standard errors, so we'll augment with run_ols per portfolio.

batch_results = re.run_batch_regressions(excess_returns, factors, FACTOR_NAMES)

# Build full regression table with SEs and t-stats
rows = []
reg_results_for_grs = []

for portfolio in excess_returns.columns:
    y = excess_returns[portfolio]
    X = factors[FACTOR_NAMES]
    res = re.run_ols(y, X, add_const=True)

    alpha = res["intercept"]
    alpha_t = res["intercept_t_stat"]
    alpha_se = alpha / alpha_t if alpha_t and not np.isnan(alpha_t) and alpha_t != 0 else np.nan

    beta_mkt = res["coefficients"]["Mkt-RF"]
    beta_mkt_t = res["t_stats"]["Mkt-RF"]
    beta_mkt_se = beta_mkt / beta_mkt_t if beta_mkt_t and not np.isnan(beta_mkt_t) and beta_mkt_t != 0 else np.nan

    beta_smb = res["coefficients"]["SMB"]
    beta_smb_t = res["t_stats"]["SMB"]
    beta_smb_se = beta_smb / beta_smb_t if beta_smb_t and not np.isnan(beta_smb_t) and beta_smb_t != 0 else np.nan

    beta_hml = res["coefficients"]["HML"]
    beta_hml_t = res["t_stats"]["HML"]
    beta_hml_se = beta_hml / beta_hml_t if beta_hml_t and not np.isnan(beta_hml_t) and beta_hml_t != 0 else np.nan

    row = {
        "portfolio": portfolio,
        "alpha": alpha,
        "alpha_t": alpha_t,
        "alpha_se": alpha_se,
        "beta_mkt": beta_mkt,
        "beta_mkt_t": beta_mkt_t,
        "beta_mkt_se": beta_mkt_se,
        "beta_smb": beta_smb,
        "beta_smb_t": beta_smb_t,
        "beta_smb_se": beta_smb_se,
        "beta_hml": beta_hml,
        "beta_hml_t": beta_hml_t,
        "beta_hml_se": beta_hml_se,
        "r_squared": res["r_squared"],
    }
    rows.append(row)
    reg_results_for_grs.append({"residuals": res["residuals"]})

reg_df = pd.DataFrame(rows)
reg_path = os.path.join(OUTPUT_DIR, "regression_korea.csv")
reg_df.to_csv(reg_path, index=False)
print(f"Saved regression results to {reg_path}")

# ---------------------------------------------------------------------------
# GRS Test (Gibbons, Ross, Shanken 1989)
# ---------------------------------------------------------------------------
N = len(excess_returns.columns)
K = len(FACTOR_NAMES)

# Collect alphas and residuals
alphas = []
t_alphas = []
resid_list = []

for portfolio in excess_returns.columns:
    y = excess_returns[portfolio]
    X = factors[FACTOR_NAMES]
    res = re.run_ols(y, X, add_const=True)
    alphas.append(res["intercept"])
    t_alphas.append(res["intercept_t_stat"])
    resid_list.append(res["residuals"])

alphas = np.array(alphas)
t_alphas = np.array(t_alphas)

# Align residuals to common index
df_resid = pd.concat(resid_list, axis=1)
df_resid = df_resid.dropna()
T = len(df_resid)

print(f"GRS test: T={T}, N={N}, K={K}")

# Residual covariance matrix
Sigma = df_resid.cov().values

# Factor returns for this sample period
factor_data = factors[FACTOR_NAMES].loc[df_resid.index]
mu_f = factor_data.mean().values.reshape(-1, 1)
Sigma_f = factor_data.cov().values

# Safe inverse with pinv fallback
def safe_inv(matrix):
    try:
        inv = np.linalg.inv(matrix)
        if np.linalg.cond(matrix) > 1e12:
            inv = np.linalg.pinv(matrix)
        return inv
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)

Sigma_inv = safe_inv(Sigma)
Sigma_f_inv = safe_inv(Sigma_f)

# theta^2 = mu_f' Sigma_f^-1 mu_f
theta_sq = (mu_f.T @ Sigma_f_inv @ mu_f).item()

# alpha' Sigma^-1 alpha
alpha_term = (alphas.reshape(1, -1) @ Sigma_inv @ alphas.reshape(-1, 1)).item()

# GRS F-statistic
F_stat = ((T - N - K) / N) * (alpha_term / (1 + theta_sq))

df1 = N
df2 = T - N - K
p_value = 1 - stats.f.cdf(F_stat, df1, df2)

grs_results = {
    "grs_stat": [F_stat],
    "p_value": [p_value],
    "df1": [df1],
    "df2": [df2],
    "T": [T],
    "N": [N],
    "K": [K],
    "mean_abs_alpha": [np.nanmean(np.abs(alphas))],
    "mean_abs_t_alpha": [np.nanmean(np.abs(t_alphas))],
}

grs_df = pd.DataFrame(grs_results)
grs_path = os.path.join(OUTPUT_DIR, "grs_test.csv")
grs_df.to_csv(grs_path, index=False)
print(f"Saved GRS test results to {grs_path}")

# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------
print("=" * 60)
print("Verification Checks")
print("=" * 60)

# All beta_mkt > 0
all_mkt_positive = (reg_df["beta_mkt"] > 0).all()
print(f"All beta_mkt > 0: {all_mkt_positive}")

# Small-cap portfolios (S1B1~S1B5) have beta_smb > 0 on average
small_cols = [c for c in reg_df["portfolio"] if c.startswith("S1")]
small_smb_mean = reg_df[reg_df["portfolio"].isin(small_cols)]["beta_smb"].mean()
print(f"Small-cap avg beta_smb: {small_smb_mean:.4f} (should be > 0)")

# High-BM portfolios (B5: S1B5, S2B5, S3B5, S4B5, S5B5) have beta_hml > 0 on average
high_bm_cols = [c for c in reg_df["portfolio"] if c.endswith("B5")]
high_bm_hml_mean = reg_df[reg_df["portfolio"].isin(high_bm_cols)]["beta_hml"].mean()
print(f"High-BM avg beta_hml: {high_bm_hml_mean:.4f} (should be > 0)")

print(f"GRS F-stat: {F_stat:.4f}, p-value: {p_value:.4f}")
print("=" * 60)
print("Done!")
