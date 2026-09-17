"""
Build appendix-facing outputs for the assignment guide
`Fama-French 1993 재현 및 정리.md`.

Goal:
- create assignment-facing filtered tables in the project output directory
- include only the stock-side tables that the guide asks to keep
- explicitly separate excluded bond-side subtables
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config
import regression_engine as re


APPENDIX_DIR = Path(config.OUTPUT_DIR)
APPENDIX_DIR.mkdir(exist_ok=True)

STOCK_ROWS = ["SMALL", "ME2", "ME3", "ME4", "BIG"]
COLS = ["Low", "2", "3", "4", "High"]
PORT_GRID = {
    "SMALL": ["SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM"],
    "ME2": ["ME2 BM1", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 BM5"],
    "ME3": ["ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5"],
    "ME4": ["ME4 BM1", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5"],
    "BIG": ["BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM"],
}
STOCK_COLS = [p for row in PORT_GRID.values() for p in row]


def normalize_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp()
    return df


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = APPENDIX_DIR / name
    df.to_csv(path, index=True)


def load_core() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table0 = pd.read_csv(Path(config.OUTPUT_DIR) / "table0_descriptive_stats.csv")
    factors = pd.read_csv(Path(config.OUTPUT_DIR) / "factors.csv", comment="#", index_col=0, parse_dates=True)
    factors = normalize_monthly_index(factors)
    stock = pd.read_csv(Path(config.OUTPUT_DIR) / "stock_portfolios_excess.csv", index_col=0, parse_dates=True)
    stock = normalize_monthly_index(stock)
    bond = pd.read_csv(Path(config.OUTPUT_DIR) / "bond_portfolios_excess.csv", index_col=0, parse_dates=True)
    bond = normalize_monthly_index(bond) * 100.0
    epdp = pd.read_csv(Path(config.OUTPUT_DIR) / "table11_ep_dp.csv")
    return table0, factors, stock, bond, epdp


def build_table1(table0: pd.DataFrame) -> None:
    table0 = table0.set_index("portfolio")
    panel1 = pd.DataFrame(index=STOCK_ROWS, columns=COLS)
    panel2 = pd.DataFrame(index=STOCK_ROWS, columns=COLS)
    for row_label, ports in PORT_GRID.items():
        for col_label, portfolio in zip(COLS, ports):
            rec = table0.loc[portfolio]
            panel1.loc[row_label, col_label] = f"{rec['avg_firm_count']:.2f} / {rec['avg_market_cap_mm']:.2f}"
            panel2.loc[row_label, col_label] = f"{rec['mkt_cap_share_pct']:.2f} / {rec['avg_firm_count']:.2f}"
    save_csv(panel1, "table1_panel1_firm_count_market_cap.csv")
    save_csv(panel2, "table1_panel2_cap_share_firm_count.csv")

    panel3_reference = """# Table 1 Panel 3 (Reference-only)

Raw firm-level E/P and D/P inputs for 25 size-BE/ME cells are not available in this repository.
To keep the assignment package complete, use the original guide values below as the appendix-facing reference table.

| Size | Low | 2 | 3 | 4 | High |
|---|---|---|---|---|---|
| Small | 2.42 / 1.00 | 7.24 / 1.94 | 8.26 / 2.60 | 9.06 / 3.13 | 2.66 / 2.82 |
| 2 | 5.20 / 1.59 | 8.61 / 2.45 | 10.16 / 3.45 | 10.95 / 4.25 | 9.28 / 4.53 |
| 3 | 5.91 / 1.56 | 8.72 / 3.03 | 10.43 / 4.04 | 11.62 / 4.68 | 10.78 / 4.64 |
| 4 | 5.85 / 1.80 | 8.94 / 3.09 | 11.64 / 4.22 | 10.45 / 5.01 | 11.39 / 4.94 |
| Big | 6.00 / 2.34 | 9.07 / 3.69 | 10.90 / 4.68 | 12.45 / 5.49 | 13.92 / 5.90 |
"""
    (APPENDIX_DIR / "table1_panel3_ep_dp_reference.md").write_text(panel3_reference, encoding="utf-8")


def compute_rmo(factors: pd.DataFrame) -> pd.Series:
    y = factors["Mkt-RF"]
    x = factors[["SMB", "HML", "TERM", "DEF"]].copy()
    x[["TERM", "DEF"]] = x[["TERM", "DEF"]] * 100.0
    result = re.run_ols(y, x, add_const=True)
    resid = result["residuals"]
    return pd.Series(result["intercept"] + resid.values, index=resid.index, name="RMO")


def build_table2(factors: pd.DataFrame, stock: pd.DataFrame, bond: pd.DataFrame) -> None:
    rf = factors["RF"]
    market_raw = factors["Mkt-RF"] + rf
    tb = rf
    long_term_raw = bond["LONG_TERM"] + rf.reindex(bond.index)
    corp_raw = bond["LOW_GRADE"] + rf.reindex(bond.index)
    rmo = compute_rmo(factors)

    summary_series = {
        "RM": market_raw,
        "TB": tb,
        "LTG": long_term_raw,
        "CB": corp_raw,
        "RM-RF": factors["Mkt-RF"],
        "RMO": rmo.reindex(factors.index),
        "SMB": factors["SMB"],
        "HML": factors["HML"],
        "TERM": factors["TERM"] * 100.0,
        "DEF": factors["DEF"] * 100.0,
    }

    rows = []
    for name, series in summary_series.items():
        s = series.dropna()
        t_stat = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
        rows.append({
            "variable": name,
            "mean": s.mean(),
            "std": s.std(ddof=1),
            "t_stat": t_stat,
            "autocorr_lag1": s.autocorr(lag=1),
            "autocorr_lag2": s.autocorr(lag=2),
            "autocorr_lag12": s.autocorr(lag=12),
        })
    save_csv(pd.DataFrame(rows).set_index("variable"), "table2_panel1_factor_summary.csv")

    corr = pd.concat(
        [factors["Mkt-RF"], rmo.reindex(factors.index), factors["SMB"], factors["HML"], factors["TERM"] * 100.0, factors["DEF"] * 100.0],
        axis=1,
    )
    corr.columns = ["RM-RF", "RMO", "SMB", "HML", "TERM", "DEF"]
    save_csv(corr.corr(), "table2_panel1_correlation_matrix.csv")

    mean_std = pd.DataFrame(index=STOCK_ROWS, columns=COLS)
    tstats = pd.DataFrame(index=STOCK_ROWS, columns=COLS)
    for row_label, ports in PORT_GRID.items():
        for col_label, portfolio in zip(COLS, ports):
            s = stock[portfolio].dropna()
            mean_std.loc[row_label, col_label] = f"{s.mean():.2f} / {s.std(ddof=1):.2f}"
            t_stat = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
            tstats.loc[row_label, col_label] = f"{t_stat:.2f}"
    save_csv(mean_std, "table2_panel2_stock_mean_std.csv")
    save_csv(tstats, "table2_panel3_stock_tstats.csv")


def run_spec(returns_df: pd.DataFrame, factors_df: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    aligned = returns_df.join(factors_df[factor_names], how="inner")
    rows = []
    for portfolio in returns_df.columns:
        y = aligned[portfolio]
        x = aligned[factor_names]
        x = sm.add_constant(x, has_constant="add")
        model = sm.OLS(y, x).fit()
        row = {
            "portfolio": portfolio,
            "alpha": model.params["const"],
            "t_alpha": model.tvalues["const"],
            "r_squared": model.rsquared,
            "s_e": float(np.sqrt(np.sum(model.resid ** 2) / model.df_resid)),
        }
        for factor in factor_names:
            row[f"beta_{factor}"] = model.params[factor]
            row[f"t_{factor}"] = model.tvalues[factor]
        rows.append(row)
    return pd.DataFrame(rows)


def to_grid(df: pd.DataFrame, value_fn, name: str) -> None:
    grid = pd.DataFrame(index=STOCK_ROWS, columns=COLS)
    df = df.set_index("portfolio")
    for row_label, ports in PORT_GRID.items():
        for col_label, portfolio in zip(COLS, ports):
            grid.loc[row_label, col_label] = value_fn(df.loc[portfolio])
    save_csv(grid, name)


def build_regression_tables(factors: pd.DataFrame, stock: pd.DataFrame) -> None:
    factor_pct = factors.copy()
    factor_pct[["TERM", "DEF"]] = factor_pct[["TERM", "DEF"]] * 100.0
    rmo = compute_rmo(factors)
    factor_pct["RMO"] = rmo

    specs = {
        "table4": (["Mkt-RF"], [
            ("panel1_b_t_b", lambda r: f"{r['beta_Mkt-RF']:.2f} / {r['t_Mkt-RF']:.2f}"),
            ("panel2_r2_se", lambda r: f"{r['r_squared']:.2f} / {r['s_e']:.2f}"),
        ]),
        "table5": (["SMB", "HML"], [
            ("panel1_s_t_s", lambda r: f"{r['beta_SMB']:.2f} / {r['t_SMB']:.2f}"),
            ("panel2_h_t_h", lambda r: f"{r['beta_HML']:.2f} / {r['t_HML']:.2f}"),
            ("panel3_r2_se", lambda r: f"{r['r_squared']:.2f} / {r['s_e']:.2f}"),
        ]),
        "table6": (["Mkt-RF", "SMB", "HML"], [
            ("panel1_b_t_b", lambda r: f"{r['beta_Mkt-RF']:.2f} / {r['t_Mkt-RF']:.2f}"),
            ("panel2_s_t_s", lambda r: f"{r['beta_SMB']:.2f} / {r['t_SMB']:.2f}"),
            ("panel3_h_t_h", lambda r: f"{r['beta_HML']:.2f} / {r['t_HML']:.2f}"),
            ("panel4_r2_se", lambda r: f"{r['r_squared']:.2f} / {r['s_e']:.2f}"),
        ]),
    }

    for table_name, (factors_used, panels) in specs.items():
        spec_df = run_spec(stock, factor_pct, factors_used)
        for suffix, formatter in panels:
            to_grid(spec_df, formatter, f"{table_name}_{suffix}.csv")


def build_table9(stock: pd.DataFrame, bond: pd.DataFrame, factors: pd.DataFrame) -> None:
    factor_pct = factors.copy()
    factor_pct[["TERM", "DEF"]] = factor_pct[["TERM", "DEF"]] * 100.0
    model_specs = [
        ("(ii)", ["Mkt-RF"]),
        ("(iii)", ["SMB", "HML"]),
        ("(iv)", ["Mkt-RF", "SMB", "HML"]),
    ]

    rows = []
    for portfolio in STOCK_COLS:
        row = {"portfolio": portfolio}
        for label, factor_names in model_specs:
            res = run_spec(stock[[portfolio]], factor_pct, factor_names).iloc[0]
            row[label] = f"{res['alpha']:.2f} / {res['t_alpha']:.2f}"
        rows.append(row)
    table9a = pd.DataFrame(rows).set_index("portfolio")
    save_csv(table9a, "table9a_stock_alphas.csv")

    all_returns = pd.concat([stock, bond], axis=1).dropna(how="all")
    common_idx = all_returns.index.intersection(factor_pct.dropna().index)
    all_returns = all_returns.loc[common_idx]
    factor_pct = factor_pct.loc[common_idx]
    grs_rows = []
    for label, factor_names in model_specs:
        grs_rows.append(compute_grs(all_returns, factor_pct, factor_names, label))
    save_csv(pd.DataFrame(grs_rows).set_index("model"), "table9c_joint_tests.csv")


def compute_grs(returns_df: pd.DataFrame, factors_df: pd.DataFrame, factor_names: list[str], label: str) -> dict:
    aligned = returns_df.join(factors_df[factor_names], how="inner").dropna()
    port_names = returns_df.columns.tolist()
    N = len(port_names)
    K = len(factor_names)

    # ---- Unrestricted regressions (with intercept / alpha) ----
    alpha_vals = []
    t_vals = []
    unrestricted_residuals = []
    for portfolio in port_names:
        y = aligned[portfolio]
        x = aligned[factor_names]
        res = re.run_ols(y, x, add_const=True)
        alpha_vals.append(res["intercept"])
        t_vals.append(res["intercept_t_stat"])
        unrestricted_residuals.append(res["residuals"])

    # ---- Restricted regressions (no intercept; H0: all alphas = 0) ----
    restricted_betas: dict[str, dict] = {}
    restricted_residuals = []
    for portfolio in port_names:
        y = aligned[portfolio]
        x = aligned[factor_names]
        res = re.run_ols(y, x, add_const=False)
        restricted_betas[portfolio] = res["coefficients"]
        restricted_residuals.append(res["residuals"])

    # ---- Original GRS F-statistic from unrestricted results ----
    resid_df = pd.concat(unrestricted_residuals, axis=1).dropna()
    T = len(resid_df)
    sigma = resid_df.cov().values
    mu_f = aligned.loc[resid_df.index, factor_names].mean().values.reshape(-1, 1)
    sigma_f = aligned.loc[resid_df.index, factor_names].cov().values
    sigma_inv = np.linalg.pinv(sigma)
    sigma_f_inv = np.linalg.pinv(sigma_f)
    alphas = np.array(alpha_vals)
    theta_sq = (mu_f.T @ sigma_f_inv @ mu_f).item()
    alpha_term = (alphas.reshape(1, -1) @ sigma_inv @ alphas.reshape(-1, 1)).item()
    f_obs = ((T - N - K) / N) * (alpha_term / (1 + theta_sq))
    f_p_value = 1 - stats.f.cdf(f_obs, N, T - N - K)

    # ---- Residual bootstrap for empirical p-value ----
    # Align restricted residuals to a common T×N matrix
    restricted_resid_df = pd.concat(restricted_residuals, axis=1).dropna()
    restricted_resid_df.columns = port_names
    T_boot = len(restricted_resid_df)

    # Factor data aligned to the bootstrap index
    factor_boot = aligned.loc[restricted_resid_df.index, factor_names]

    # θ² from the bootstrap sample (factors are fixed, not resampled)
    mu_f_boot = factor_boot.mean().values.reshape(-1, 1)
    sigma_f_boot = factor_boot.cov().values
    sigma_f_inv_boot = np.linalg.pinv(sigma_f_boot)
    theta_sq_boot = (mu_f_boot.T @ sigma_f_inv_boot @ mu_f_boot).item()

    # Fitted values from the restricted model: β̂ · f
    fitted = pd.DataFrame(index=restricted_resid_df.index, columns=port_names, dtype=float)
    for i, portfolio in enumerate(port_names):
        betas = restricted_betas[portfolio]
        fitted_vals = np.zeros(T_boot)
        for f in factor_names:
            fitted_vals += betas.get(f, 0.0) * factor_boot[f].values
        fitted[portfolio] = fitted_vals

    B = 999
    rng = np.random.default_rng(42)
    resid_values = restricted_resid_df.values   # T_boot × N
    fitted_values = fitted.values               # T_boot × N
    f_boot_list: list[float] = []

    for _ in range(B):
        # 1) Resample T_boot time indices (preserving cross‑sectional correlation)
        idx = rng.integers(0, T_boot, size=T_boot)
        eps_star = resid_values[idx]                     # T_boot × N

        # 2) Pseudo‑returns under H0
        pseudo_returns = fitted_values + eps_star        # T_boot × N

        # 3) Unrestricted regression on pseudo‑data → obtain bootstrap alphas
        boot_alphas = []
        boot_residuals = []
        for i, portfolio in enumerate(port_names):
            y_star = pd.Series(pseudo_returns[:, i], index=restricted_resid_df.index)
            x = factor_boot
            res = re.run_ols(y_star, x, add_const=True)
            boot_alphas.append(res["intercept"])
            boot_residuals.append(res["residuals"])

        # 4) Compute GRS F‑statistic on pseudo‑data
        boot_resid_df = pd.concat(boot_residuals, axis=1).dropna()
        T_eff = len(boot_resid_df)
        if T_eff < N + K + 1:
            continue

        boot_sigma = boot_resid_df.cov().values
        boot_alphas_arr = np.array(boot_alphas)
        boot_sigma_inv = np.linalg.pinv(boot_sigma)
        boot_alpha_term = (
            boot_alphas_arr.reshape(1, -1) @ boot_sigma_inv @ boot_alphas_arr.reshape(-1, 1)
        ).item()

        f_boot = ((T_eff - N - K) / N) * (boot_alpha_term / (1 + theta_sq_boot))
        if np.isfinite(f_boot) and f_boot > 0:
            f_boot_list.append(f_boot)

    n_boot = len(f_boot_list)
    if n_boot > 0:
        bootstrap_p = (np.sum(np.array(f_boot_list) >= f_obs) + 1.0) / (n_boot + 1.0)
    else:
        bootstrap_p = np.nan

    return {
        "model": label,
        "N": N,
        "K": K,
        "T": T,
        "F_stat": f_obs,
        "F_dist_p_value": f_p_value,
        "mean_abs_alpha": float(np.nanmean(np.abs(alpha_vals))),
        "mean_abs_t_alpha": float(np.nanmean(np.abs(t_vals))),
        "bootstrap_probability_level": bootstrap_p,
    }


def build_table11(epdp: pd.DataFrame) -> None:
    epdp.to_csv(APPENDIX_DIR / "table11_ep_dp_long.csv", index=False)


def write_index() -> None:
    text = """# Appendix Output Index

This directory contains assignment-facing tables derived from the repository outputs.

## Included tables exported here

- Table 1 panels 1-2
- Table 1 panel 3 reference snapshot
- Table 2 panels 1-3
- Table 3 stock-only panels
- Table 4 stock-only panels
- Table 5 stock-only panels
- Table 6 stock-only panels
- Table 7a stock-only panels
- Table 8a stock-only panels
- Table 9a stock-only alpha matrix
- Table 9c joint alpha F-tests (without bootstrap simulation)
- Table 11 long-format export

## Explicitly excluded from appendix-facing output

- Table 3 bond block
- Table 4 bond block
- Table 5 bond block
- Table 6 bond block
- Table 7b
- Table 8b
- Table 9b
- Table 10

This directory is the canonical assignment submission output directory.
"""
    index_path = APPENDIX_DIR / "README.md"
    if index_path.exists():
        # 포맷된 한국어 인덱스(README.md)가 있으면 덮어쓰지 않는다.
        return
    index_path.write_text(text, encoding="utf-8")


def main() -> None:
    print("Building appendix-facing exports...")
    table0, factors, stock, bond, epdp = load_core()
    build_table1(table0)
    build_table2(factors, stock, bond)
    build_regression_tables(factors, stock)
    build_table9(stock, bond, factors)
    build_table11(epdp)
    write_index()
    print(f"Appendix outputs written to: {APPENDIX_DIR.resolve()}")


if __name__ == "__main__":
    main()
