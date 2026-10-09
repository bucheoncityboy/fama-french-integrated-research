"""Availability-driven supplementary diagnostics, distinct from frozen primary."""
import hashlib,json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from .acquire import ROOT
from .form_portfolios import form,wide,longest_block,CELLS
from .run_ff3_alpha import regression_for,mean_inference,metrics,bootstrap


def analyze():
    design=ROOT/"config/supplementary_availability_design.json"
    if hashlib.sha256(design.read_bytes()).hexdigest()!=design.with_suffix(".sha256").read_text().strip():
        raise ValueError("supplementary_design_changed")
    factors=pd.read_csv(ROOT/"output/ff3_factors_spec_aligned.csv",parse_dates=["date"])
    cohorts=[("all_two_leg_available",ROOT/"data/private/joint_panel.parquet"),
             ("fixed_stage150_two_leg_available",ROOT/"data/private/stage150_joint_panel.parquet")]
    results=[];means=[];performance=[];locks=[]
    for label,file in cohorts:
        p=pd.read_parquet(file)
        r,h=form(p);s=wide(r)
        good=r.loc[r.cell.isin(["HighBM_HighG","HighBM_LowG"])].groupby("date").status.apply(lambda v:len(v)==2 and v.eq("estimated").all())
        dates=longest_block(good.index[good])
        locks.append({"cohort":label,"rule":"two_value_legs_only_contiguous; same_min24; primary_full_four_cells_unchanged",
            "n_available_months":int(good.sum()),"window_N":len(dates),"start":str(dates[0].date()) if len(dates) else None,
            "end":str(dates[-1].date()) if len(dates) else None,"frozen_before_fitting":True})
        # Window is frozen before any alpha/mean/inference.
        (ROOT/"config/supplementary_windows.lock.json").write_text(json.dumps(locks,indent=2))
        chosen=s.loc[s.date.isin(dates)].copy()
        chosen.to_csv(ROOT/"output"/(label+"_returns.csv"),index=False)
        r.to_csv(ROOT/"output"/(label+"_cells_all_months.csv"),index=False)
        rr=regression_for(chosen[["date","spread"]].rename(columns={"spread":"return"}),factors)
        results.append({"spec_id":label,"role":"supplementary_exploratory","primary_replaced":False,
               "design_registration":"after_availability_audit_before_outcome_inspection","N_stocks_ever_joint":p.loc[p.joint_eligible,"stock_code"].nunique(),**rr})
        means.append({"cohort":label,**mean_inference(chosen.spread)})
        for cell in [*CELLS,"spread"]:performance.append({"cohort":label,"cell":cell,**metrics(chosen[cell]),
                 "note":"LowBM cells can have missing months; N disclosed individually. Spread index is a gross normalized diagnostic, not feasible wealth."})
        if rr["status"]=="estimated":
            bootstrap(chosen,factors).to_csv(ROOT/"output"/(label+"_block_bootstrap.csv"),index=False)
    pd.DataFrame(results).to_csv(ROOT/"output/supplementary_alpha.csv",index=False)
    pd.DataFrame(means).to_csv(ROOT/"output/supplementary_mean.csv",index=False)
    pd.DataFrame(performance).to_csv(ROOT/"output/supplementary_performance.csv",index=False)
    mt=pd.read_csv(ROOT/"output/multiple_testing.csv")
    extra=pd.DataFrame([{"spec_id":r["spec_id"],"role":"supplementary_exploratory","p_raw":r.get("p_hac"),
                  "effect_pct":r.get("alpha_monthly_pct"),"status":r["status"]} for r in results]+[
                  {"spec_id":r["cohort"]+"_mean","role":"supplementary_exploratory","p_raw":r.get("p_hac"),
                   "effect_pct":r.get("mean_pct"),"status":r["status"]} for r in means])
    mt=mt.loc[~mt.spec_id.isin(extra.spec_id)]
    mt=pd.concat([mt,extra],ignore_index=True)
    mask=~mt.role.eq("primary")&mt.p_raw.notna()
    mt["p_holm"]=np.nan
    if mask.any():mt.loc[mask,"p_holm"]=multipletests(mt.loc[mask,"p_raw"].astype(float),method="holm")[1]
    mt["total_secondary_tests_performed"]=int(mask.sum())
    mt.to_csv(ROOT/"output/multiple_testing.csv",index=False)
    status=json.loads((ROOT/"output/RESULT_STATUS.json").read_text())
    status["supplementary_estimation"]={"design":str(design.relative_to(ROOT)),"cohort_results":[{k:v for k,v in r.items() if isinstance(v,(str,int,float,bool,type(None)))} for r in results]}
    if status["ff3_alpha"]!="estimated":status["final_status"]="EMPIRICAL_LIMITED"
    (ROOT/"output/RESULT_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(json.dumps({"supplementary_windows":locks,"supplementary_results":results},ensure_ascii=False),flush=True)

if __name__=="__main__":analyze()
