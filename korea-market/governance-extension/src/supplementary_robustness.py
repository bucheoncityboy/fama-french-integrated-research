"""Apply the already-fixed full secondary parameter list to the fixed pilot diagnostic."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox,het_breuschpagan
from .acquire import ROOT
from .form_portfolios import form,wide,longest_block,CELLS
from .run_ff3_alpha import regression_for,controls


def run():
    p=pd.read_parquet(ROOT/"data/private/stage150_joint_panel.parquet")
    base=pd.read_csv(ROOT/"output/fixed_stage150_two_leg_available_returns.csv",parse_dates=["date"])
    dates=pd.DatetimeIndex(base.date)
    factors=pd.read_csv(ROOT/"output/ff3_factors_spec_aligned.csv",parse_dates=["date"])
    legacy=pd.read_csv(ROOT/"output/ff3_factors_legacy_compatible.csv",parse_dates=["date"])
    specs=[("base",{},factors,3),("equal_weight",{"weighting":"EW"},factors,3),
       ("g_cut_30",{"cut":.3},factors,3),("g_cut_70",{"cut":.7},factors,3),
       ("exclude_top5",{"exclude_top5":True},factors,3),("legacy_FF3",{},legacy,3),
       ("hac6",{},factors,6),("stale_365_days",{"max_age":365},factors,3),
       ("kospi_reported_only",{"market":"Y"},factors,3),("kosdaq_reported_only",{"market":"K"},factors,3),
       ("quarterly_rebalance",{"rebalance":"quarterly"},factors,3),("annual_rebalance",{"rebalance":"annual"},factors,3)]
    out=[];all_holdings={}
    for label,kwargs,ff,lag in specs:
        r,h=form(p,**kwargs);s=wide(r);s=s.loc[s.date.isin(dates)]
        good=r.loc[r.cell.isin(["HighBM_HighG","HighBM_LowG"])].groupby("date").status.apply(lambda v:len(v)==2 and v.eq("estimated").all())
        available=longest_block(dates.intersection(good.index[good]))
        chosen=s.loc[s.date.isin(available)]
        result=regression_for(chosen[["date","spread"]].rename(columns={"spread":"return"}),ff,lags=lag)
        out.append({"spec_id":"pilot_"+label,"cohort":"fixed_stage150","role":"supplementary_exploratory",
                    "parameters_pre_registered":True,"sample_design_after_availability_audit":True,
                    "spread_mean_pct":chosen.spread.mean()*100,"notes":"all variants shown; only N>=24 infer; no original primary replacement",**result})
        if label=="base":
            all_holdings["base"]=h
            r.to_csv(ROOT/"output/pilot_formation_diagnostics.csv",index=False)
            turnover=r.pivot(index="date",columns="cell",values="turnover_one_way_sum_abs")
            cost=turnover.HighBM_HighG+turnover.HighBM_LowG
            for bp in (10,30):
                ss=base.copy();ss["spread"]=ss.spread-ss.date.map(cost)*bp/10000
                rr=regression_for(ss[["date","spread"]].rename(columns={"spread":"return"}),factors)
                out.append({"spec_id":"pilot_cost_"+str(bp)+"bp","cohort":"fixed_stage150","role":"supplementary_exploratory",
                        "parameters_pre_registered":True,"spread_mean_pct":ss.spread.mean()*100,
                        "gross_mean_pct_same_valid_cost_months":ss.loc[ss.spread.notna(),"date"].map(base.set_index("date").spread).mean()*100,
                        "notes":"23 months only; compare with same-month gross, not 24-month base; unknown turnover not filled; no borrow/financing/liquidity validation",**rr})
    control=controls(p,dates);control.to_csv(ROOT/"output/pilot_verified_controls.csv",index=False)
    cr=control.iloc[0].to_dict()
    out.append({"spec_id":"pilot_verified_ROE_size_BM_leverage_controls","cohort":"fixed_stage150","role":"supplementary_exploratory",
            "parameters_pre_registered":True,"effect_type":"monthly_cross_section_G_coefficient_not_portfolio_alpha",
            "status":cr.get("status"),"n_months":cr.get("n_months",0),"p_hac":cr.get("p_hac"),
            "g_slope_pct":cr.get("mean_pct"),"not_estimated_reason":cr.get("reason")})
    pd.DataFrame(out).to_csv(ROOT/"output/supplementary_robustness.csv",index=False)
    mt=pd.read_csv(ROOT/"output/multiple_testing.csv");mt=mt.loc[~mt.spec_id.str.startswith("pilot_")]
    extra=pd.DataFrame([{"spec_id":r["spec_id"],"role":"supplementary_exploratory",
              "p_raw":r.get("p_hac"),"effect_pct":r.get("alpha_monthly_pct",r.get("g_slope_pct")),"status":r["status"]} for r in out if r["spec_id"]!="pilot_base"])
    mt=pd.concat([mt,extra],ignore_index=True);mask=~mt.role.eq("primary")&mt.p_raw.notna()
    mt["p_holm"]=np.nan
    if mask.any():mt.loc[mask,"p_holm"]=multipletests(mt.loc[mask,"p_raw"].astype(float),method="holm")[1]
    mt["total_secondary_tests_performed"]=int(mask.sum())
    mt.to_csv(ROOT/"output/multiple_testing.csv",index=False)
    d=base[["date","spread"]].merge(factors[["date","Mkt-RF","SMB","HML"]],on="date")
    X=sm.add_constant(d[["Mkt-RF","SMB","HML"]]);fit=sm.OLS(d.spread,X).fit()
    influence=[]
    for i,row in d.iterrows():
        q=d.drop(i);a=sm.OLS(q.spread,sm.add_constant(q[["Mkt-RF","SMB","HML"]])).fit().params["const"]*100
        influence.append({"removed_month":str(row.date.date()),"alpha_pct":a,"full_alpha_pct":fit.params["const"]*100})
    pd.DataFrame(influence).to_csv(ROOT/"output/pilot_leave_one_month_out.csv",index=False)
    decomposition=[{"component":"alpha","contribution_monthly_pct":fit.params["const"]*100}]
    for field in ("Mkt-RF","SMB","HML"):decomposition.append({"component":field,"contribution_monthly_pct":fit.params[field]*d[field].mean()*100})
    pd.DataFrame(decomposition).to_csv(ROOT/"output/pilot_mean_decomposition.csv",index=False)
    diag={"scope":"fixed_stage150_supplementary_only","n_months":len(d),
       "ljung_box_p_lag6":float(acorr_ljungbox(fit.resid,lags=[6],return_df=True).lb_pvalue.iloc[0]),
       "breusch_pagan_p":float(het_breuschpagan(fit.resid,fit.model.exog)[1]),"condition_number":float(fit.condition_number),
       "factor_correlation":d[["Mkt-RF","SMB","HML"]].corr().to_dict()}
    (ROOT/"output/pilot_model_diagnostics.json").write_text(json.dumps(diag,indent=2))
    print(json.dumps({"supplementary_specs":len(out),"estimated":sum(x["status"]=="estimated" for x in out),"secondary_family_N":int(mask.sum())}),flush=True)

if __name__=="__main__":run()
