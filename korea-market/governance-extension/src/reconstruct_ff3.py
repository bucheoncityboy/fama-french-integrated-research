"""Reconstruct existing benchmarks; never interpret them as governance results."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .common import lag_market_values, month_start


PORTS = ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]


def build_formation(panel, book, mode):
    june = panel.loc[panel.date.dt.month.eq(6), ["code", "date", "me", "bm"]].copy()
    june["formation_year"] = june.date.dt.year
    june = june.rename(columns={"me": "june_me", "bm": "panel_bm"})
    b = book.sort_values("date").groupby(["code", "be_year"], as_index=False).first()
    b["formation_year"] = b.be_year + 1
    f = june.merge(b[["code", "formation_year", "be", "be_year", "date"]].rename(columns={"date": "be_panel_first_month"}),
                   on=["code", "formation_year"], how="left", validate="one_to_one")
    f["bm_calculated"] = f.be / f.june_me * 100000
    if mode == "legacy_compatible":
        f["bm_used"] = f.panel_bm.combine_first(f.bm_calculated)
    elif mode == "spec_aligned":
        f["bm_used"] = f.bm_calculated
    else:
        raise ValueError("unknown formation mode")
    f = f.loc[(f.june_me > 0) & (f.bm_used > 0)].copy()
    assignments = []
    for year, g in f.groupby("formation_year"):
        if len(g) < 10:
            continue
        g = g.copy()
        lo, hi = g.bm_used.quantile([.3, .7])
        g["size"] = np.where(g.june_me <= g.june_me.median(), "S", "B")
        g["value"] = np.select([g.bm_used <= lo, g.bm_used <= hi], ["L", "M"], default="H")
        g["portfolio"] = g["size"] + "/" + g["value"]
        assignments.append(g)
    if not assignments:
        raise ValueError("no valid annual FF3 formation")
    return pd.concat(assignments, ignore_index=True)


def reconstruct(panel, book, market, mode="spec_aligned"):
    f = build_formation(panel, book, mode)
    p = panel.sort_values(["code", "date"]).copy()
    if mode == "spec_aligned":
        p = lag_market_values(p)
    else:
        p["lag_me"] = p.groupby("code").me.shift()
    p["formation_year"] = p.date.dt.year - p.date.dt.month.le(6).astype(int)
    p = p.merge(f[["code", "formation_year", "portfolio"]], on=["code", "formation_year"], how="inner", validate="many_to_one")
    p = p.loc[p["return"].notna() & (p.lag_me > 0)].copy()
    p["weight"] = p.lag_me / p.groupby(["date", "portfolio"]).lag_me.transform("sum")
    p["contribution"] = p.weight * p["return"]
    long = p.groupby(["date", "portfolio"], as_index=False).agg(vw_return=("contribution", "sum"), n_stocks=("code", "size"), weight_sum=("weight", "sum"))
    wide = long.pivot(index="date", columns="portfolio", values="vw_return").reindex(columns=PORTS)
    wide["SMB"] = wide[["S/L", "S/M", "S/H"]].sum(axis=1, min_count=3)/3 - wide[["B/L", "B/M", "B/H"]].sum(axis=1, min_count=3)/3
    wide["HML"] = wide[["S/H", "B/H"]].sum(axis=1, min_count=2)/2 - wide[["S/L", "B/L"]].sum(axis=1, min_count=2)/2
    m = market.copy()
    m["date"] = month_start(m.date)
    if m.date.duplicated().any():
        raise ValueError("duplicate market factor months")
    factors = wide.reset_index().merge(m[["date", "mkt_rf", "rf"]], on="date", how="inner", validate="one_to_one")
    factors = factors.rename(columns={"mkt_rf": "Mkt-RF"})
    factors = factors.loc[factors.date.between("2000-07-01", "2026-05-01")]
    factors["specification"] = mode
    return factors, long, f


def summary(factors):
    rows = []
    for col in ["Mkt-RF", "SMB", "HML"]:
        d = factors[["date", col]].dropna()
        y = d[col].to_numpy()
        ols = sm.OLS(y, np.ones((len(y), 1))).fit()
        hac = ols.get_robustcov_results(cov_type="HAC", maxlags=3, use_correction=True, use_t=True)
        rows.append({"factor": col, "mean_monthly_decimal": float(y.mean()),
                     "mean_monthly_pct": float(100*y.mean()), "std_monthly_pct": float(100*y.std(ddof=1)),
                     "t_ols": float(ols.tvalues[0]), "t_hac3": float(hac.tvalues[0]),
                     "p_hac3": float(hac.pvalues[0]), "n_months": len(y),
                     "start": d.date.min().strftime("%Y-%m"), "end": d.date.max().strftime("%Y-%m"),
                     "specification": factors.specification.iloc[0]})
    return pd.DataFrame(rows)


def write_benchmarks(source, panel, output):
    data = Path(source) / "korea-market/data"
    book = pd.read_parquet(data / "book_equity_monthly.parquet")
    market = pd.read_parquet(data / "market_excess_return.parquet")
    if not np.allclose(market.mkt_return - market.rf, market.mkt_rf, atol=1e-12):
        raise ValueError("MKT-RF accounting mismatch")
    summaries, factors_by_mode = [], {}
    for mode in ["legacy_compatible", "spec_aligned"]:
        factors, returns, formation = reconstruct(panel, book, market, mode)
        factors.to_csv(Path(output) / f"ff3_factors_{mode}.csv", index=False)
        returns.to_csv(Path(output) / f"ff3_portfolios_{mode}.csv", index=False)
        # Only aggregate formation diagnostics are published, not a proprietary stock panel.
        formation.groupby("formation_year").agg(n_stocks=("code", "size"),
                    median_bm=("bm_used", "median"), median_fiscal_year=("be_year", "median")).to_csv(Path(output) / f"ff3_formation_{mode}.csv")
        summaries.append(summary(factors))
        factors_by_mode[mode] = factors
    out = pd.concat(summaries, ignore_index=True)
    out.to_csv(Path(output) / "ff3_summary.csv", index=False)
    common = factors_by_mode["legacy_compatible"].merge(factors_by_mode["spec_aligned"], on="date", suffixes=("_legacy", "_aligned"))
    sensitivity = [{"factor": c, "correlation": common[c+"_legacy"].corr(common[c+"_aligned"]),
                    "mean_difference_pct_points": (common[c+"_aligned"]-common[c+"_legacy"]).mean()*100,
                    "n_months": len(common)} for c in ["Mkt-RF", "SMB", "HML"]]
    pd.DataFrame(sensitivity).to_csv(Path(output) / "ff3_specification_sensitivity.csv", index=False)
    return factors_by_mode["spec_aligned"], out
