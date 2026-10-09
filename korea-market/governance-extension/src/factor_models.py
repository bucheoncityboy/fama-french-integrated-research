import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTORS = ["Mkt-RF", "SMB", "HML"]


def fit_ff3(strategy, factors, *, long_short, min_months=24, hac_lags=3):
    """Explicit monthly decimal input; self-financing spreads never subtract RF."""
    for d in [strategy, factors]:
        if d.date.duplicated().any():
            raise ValueError("duplicate regression months")
    d = strategy[["date", "return"]].merge(factors[["date", *FACTORS, "rf"]], on="date", validate="one_to_one")
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
    if len(d) < min_months:
        return {"status": "not_estimated", "not_estimated_reason": "insufficient_common_months", "n_months": len(d)}
    ordinals = d.date.sort_values().dt.to_period("M").astype("int64")
    if not ordinals.diff().dropna().eq(1).all():
        return {"status": "not_estimated", "not_estimated_reason": "noncontiguous_months_for_HAC", "n_months": len(d)}
    y = d["return"] if long_short else d["return"] - d.rf
    X = sm.add_constant(d[FACTORS], has_constant="add")
    if not np.isfinite(X.to_numpy()).all() or np.linalg.matrix_rank(X) < 4:
        return {"status": "not_estimated", "not_estimated_reason": "rank_deficient_factors", "n_months": len(d)}
    ols = sm.OLS(y, X).fit()
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=hac_lags, use_correction=True, use_t=True)
    ci = hac.conf_int()[0]
    return {"status": "estimated", "n_months": len(d), "start": str(d.date.min().date()),
            "end": str(d.date.max().date()), "alpha_monthly_decimal": float(hac.params[0]),
            "alpha_monthly_pct": float(hac.params[0]*100), "t_hac": float(hac.tvalues[0]),
            "t_ols": float(ols.tvalues.iloc[0]), "p_hac": float(hac.pvalues[0]),
            "ci95_low_pct": float(ci[0]*100), "ci95_high_pct": float(ci[1]*100),
            "beta_mkt": float(hac.params[1]), "beta_smb": float(hac.params[2]), "beta_hml": float(hac.params[3]),
            "r_squared": float(ols.rsquared), "condition_number": float(ols.condition_number),
            "rf_subtracted": not long_short, "hac_lags": hac_lags}
