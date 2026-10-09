import numpy as np
import pandas as pd

CELLS = ["LowBM_LowG", "LowBM_HighG", "HighBM_LowG", "HighBM_HighG"]


def sort_portfolios(pit, min_cell=5, weight="VW", g_cut=.5, exclude_top=0):
    if weight not in ["VW", "EW"]:
        raise ValueError("weight must be VW or EW")
    if not 0 < g_cut < 1:
        raise ValueError("invalid G cutoff")
    returns, holdings, coverage = [], [], []
    for date, month in pit.groupby("date"):
        eligible = month.loc[(month.lag_bm > 0) & (month.lag_me > 0)].copy()
        covered = eligible.loc[eligible.g_value.notna()].copy()
        coverage.append({"date": date, "eligible_stocks": len(eligible), "covered_stocks": len(covered),
                         "coverage_ratio": len(covered)/len(eligible) if len(eligible) else np.nan,
                         "missing_ratio": 1-len(covered)/len(eligible) if len(eligible) else np.nan,
                         "weight": weight, "g_cut": g_cut, "exclude_top": exclude_top})
        if exclude_top:
            covered = covered.sort_values("lag_me", ascending=False).iloc[exclude_top:].copy()
        if covered.empty:
            continue
        bm_mid = covered.lag_bm.median()
        covered["bm_group"] = np.where(covered.lag_bm <= bm_mid, "LowBM", "HighBM")
        for label, group in covered.groupby("bm_group"):
            cutoff = group.g_value.quantile(g_cut)
            group = group.copy()
            group["cell"] = label + "_" + np.where(group.g_value <= cutoff, "LowG", "HighG")
            for cell in [label+"_LowG", label+"_HighG"]:
                g = group.loc[group.cell.eq(cell)].copy()
                enough = len(g) >= min_cell
                raw_weights = g.lag_me if weight == "VW" else pd.Series(1., index=g.index)
                g["weight"] = raw_weights / raw_weights.sum() if len(g) else np.nan
                missing = int(g["return"].isna().sum())
                # Do not silently renormalize over surviving return observations.
                ret = float((g["weight"] * g["return"]).sum()) if enough and not missing else np.nan
                returns.append({"date": date, "cell": cell, "return": ret, "n_formed": len(g),
                                "n_missing_return": missing, "weight_sum": g.weight.sum() if len(g) else np.nan,
                                "status": "estimated" if enough and not missing else "insufficient_cell_or_missing_return",
                                "weight": weight, "g_cut": g_cut, "exclude_top": exclude_top})
                holdings.extend(g[["code", "date", "formation_date", "available_at", "lag_me", "lag_bm", "g_value", "weight", "cell"]].to_dict("records"))
    return (pd.DataFrame(returns, columns=["date", "cell", "return", "n_formed", "n_missing_return", "weight_sum", "status", "weight", "g_cut", "exclude_top"]),
            pd.DataFrame(holdings), pd.DataFrame(coverage))


def strategies(returns):
    if returns.empty:
        return pd.DataFrame(columns=["date", "HighBM_HighG", "HighBM_LowG", "spread"])
    p = returns.pivot(index="date", columns="cell", values="return").reindex(columns=CELLS)
    p["spread"] = p.HighBM_HighG - p.HighBM_LowG
    return p.reset_index()
