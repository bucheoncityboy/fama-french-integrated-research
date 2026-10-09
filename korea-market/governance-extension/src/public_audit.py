"""Audit published numeric claims without an API key or proprietary inputs."""
import hashlib,json
import numpy as np
import pandas as pd
from .acquire import ROOT,verify_registration
from .run_ff3_alpha import regression_for,mean_inference,metrics


def audit():
    verify_registration()
    returns=pd.read_csv(ROOT/"output/fixed_stage150_two_leg_available_returns.csv",parse_dates=["date"])
    factors=pd.read_csv(ROOT/"output/ff3_factors_spec_aligned.csv",parse_dates=["date"])
    alpha=pd.read_csv(ROOT/"output/supplementary_alpha.csv").set_index("spec_id").loc["fixed_stage150_two_leg_available"]
    calculated=regression_for(returns[["date","spread"]].rename(columns={"spread":"return"}),factors)
    for col in ("alpha_monthly_pct","t_hac","p_hac","ci95_low_pct","ci95_high_pct","r_squared","n_months"):
        if not np.isclose(calculated[col],alpha[col],rtol=1e-10,atol=1e-10):raise ValueError("public_alpha_mismatch_"+col)
    mean=pd.read_csv(ROOT/"output/supplementary_mean.csv").set_index("cohort").loc["fixed_stage150_two_leg_available"]
    calc_mean=mean_inference(returns.spread)
    for col in ("mean_pct","t_hac","p_hac","ci95_low_pct","ci95_high_pct"):
        if not np.isclose(calc_mean[col],mean[col],rtol=1e-10,atol=1e-10):raise ValueError("public_mean_mismatch_"+col)
    if not np.allclose(returns.spread,returns.HighBM_HighG-returns.HighBM_LowG):raise ValueError("spread_identity")
    performance=pd.read_csv(ROOT/"output/supplementary_performance.csv")
    performance=performance.loc[performance.cohort.eq("fixed_stage150_two_leg_available")]
    for row in performance.itertuples():
        m=metrics(returns[row.cell])
        for col in ("mean_monthly_pct","annualized_vol_pct","compounded_spread_index_pct","max_drawdown_index_pct"):
            if not np.isclose(m.get(col,np.nan),getattr(row,col),equal_nan=True,rtol=1e-10,atol=1e-10):raise ValueError("public_performance_mismatch")
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/"output").glob("*.csv"))}
    result={"status":"PASS","actual_public_months":len(returns),"alpha_mean_performance_identity":"PASS",
            "raw_data_or_key_required":False,"csv_sha256":hashes}
    (ROOT/"output/public_numeric_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="csv_sha256"}))

if __name__=="__main__":audit()
