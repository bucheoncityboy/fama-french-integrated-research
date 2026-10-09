"""Preregistered primary inference and complete secondary registry."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox,het_breuschpagan
from .acquire import ROOT,panel,verify_registration
from .form_portfolios import form,wide,longest_block,CELLS
from .factor_models import fit_ff3
from .reconstruct_ff3 import write_benchmarks


def mean_inference(series,lags=3):
    y=np.asarray(series,dtype=float)
    y=y[np.isfinite(y)]
    if len(y)<24:return {"status":"not_estimated","n_months":len(y),"reason":"fewer_than_24_months"}
    fit=sm.OLS(y,np.ones((len(y),1))).fit()
    hac=fit.get_robustcov_results(cov_type="HAC",maxlags=lags,use_correction=True,use_t=True)
    ci=hac.conf_int()[0]
    return {"status":"estimated","n_months":len(y),"mean_pct":100*float(y.mean()),"t_hac":float(hac.tvalues[0]),
          "p_hac":float(hac.pvalues[0]),"ci95_low_pct":float(ci[0]*100),"ci95_high_pct":float(ci[1]*100)}


def metrics(y):
    original=pd.Series(y)
    missing=int(original.isna().sum())
    y=original.dropna()
    if not len(y):return {"n_months":0}
    index=pd.concat([pd.Series([1.]),(1+y.reset_index(drop=True)).cumprod()],ignore_index=True)
    return {"n_months":len(y),"missing_months":missing,"mean_monthly_pct":float(y.mean()*100),"annualized_vol_pct":float(y.std(ddof=1)*np.sqrt(12)*100),
         "compounded_spread_index_pct":float((index.iloc[-1]-1)*100) if not missing else np.nan,
         "max_drawdown_index_pct":float((index/index.cummax()-1).min()*100) if not missing else np.nan,
         "path_status":"complete_normalized_index" if not missing else "missing_months_cannot_compound"}


def freeze_window(r):
    cfg=verify_registration()
    valid=r.groupby("date").status.apply(lambda s:len(s)==4 and s.eq("estimated").all())
    dates=longest_block(valid.index[valid])
    record={"pre_registration_sha256":hashlib.sha256((ROOT/"config/pre_registration.yaml").read_bytes()).hexdigest(),
       "rule":"longest_contiguous_all_four_cells_valid; ties_earliest; availability_only",
       "window_start":str(dates[0].date()) if len(dates) else None,"window_end":str(dates[-1].date()) if len(dates) else None,
       "n_valid_months":len(dates),"all_valid_months":int(valid.sum()),"all_attempted_months":len(valid),
       "is_end":cfg["time_split"]["is_end"],"oos_start":cfg["time_split"]["oos_start"],
       "frozen_before_performance_inspection":True}
    p=ROOT/"config/analysis_window.lock.json"
    if p.exists() and json.loads(p.read_text())!=record:raise ValueError("analysis_window_changed_after_freeze")
    p.write_text(json.dumps(record,indent=2)+"\n")
    return dates,record


def regression_for(s,factors,long_short=True,lags=3):
    return fit_ff3(s[["date","return"]],factors,long_short=long_short,min_months=24,hac_lags=lags)


def bootstrap(s,factors,nrep=2000):
    d=s[["date","spread"]].merge(factors[["date","Mkt-RF","SMB","HML"]],on="date").dropna()
    if len(d)<24:return pd.DataFrame([{"status":"not_estimated","reason":"fewer_than_24_months"}])
    x=d[["spread","Mkt-RF","SMB","HML"]].to_numpy();n=len(x);rng=np.random.default_rng(1993);rows=[]
    for block in (6,12):
        alphas=[];means=[]
        for _ in range(nrep):
            start=rng.integers(0,n,size=int(np.ceil(n/block)))
            indices=np.concatenate([(k+np.arange(block))%n for k in start])[:n]
            sample=x[indices]
            design=np.column_stack([np.ones(n),sample[:,1:]])
            alphas.append(np.linalg.lstsq(design,sample[:,0],rcond=None)[0][0]*100)
            means.append(sample[:,0].mean()*100)
        rows.append({"block_months":block,"repetitions":nrep,"seed":1993,"alpha_low_pct":np.quantile(alphas,.025),
              "alpha_high_pct":np.quantile(alphas,.975),"mean_low_pct":np.quantile(means,.025),"mean_high_pct":np.quantile(means,.975),
              "resampling_unit":"paired_monthly_spread_and_FF3_time_blocks; fixed_historical_portfolio_paths","status":"estimated"})
    return pd.DataFrame(rows)


def controls(p,dates):
    betas=[]
    for date,month in p.loc[p.date.isin(dates)].groupby("date"):
        d=month.loc[month.joint_eligible].copy()
        if not len(d):continue
        d=d.loc[d.bm_pit>d.bm_pit.median()].copy()
        d["log_me"]=np.log(d.lag_me)
        needed=["g_value","log_me","bm_pit","roe","leverage","return"]
        d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=needed)
        if len(d)<20:continue
        X=sm.add_constant(d[needed[:-1]],has_constant="add")
        if np.linalg.matrix_rank(X.to_numpy())<X.shape[1]:continue
        fit=sm.OLS(d["return"],X).fit()
        betas.append({"date":date,"g_slope":fit.params.g_value,"n_stocks":len(d)})
    d=pd.DataFrame(betas)
    if len(d)<24:return pd.DataFrame([{"status":"not_estimated","reason":"insufficient_contiguous_monthly_verified_ROE_control_sample","n_months":len(d)}])
    block=longest_block(d.date)
    d=d.loc[d.date.isin(block)]
    result=mean_inference(d.g_slope)
    return pd.DataFrame([{"control_spec":"HighBM_monthly_cross_section_G_logME_BM_ROE_leverage_then_HAC_of_monthly_G_slope",
             "not_causal":True,**result,"average_stocks":d.n_stocks.mean()}])


def analyze():
    cfg=verify_registration()
    if not (ROOT/"output/checkpoint_all.json").exists():raise ValueError("full_acquisition_checkpoint_required_before_performance")
    p=pd.read_parquet(ROOT/"data/private/joint_panel.parquet")
    r,h=form(p)
    dates,window=freeze_window(r)
    r.to_csv(ROOT/"output/primary_returns.csv",index=False)
    h.to_parquet(ROOT/"data/private/primary_holdings.parquet",index=False)
    s=wide(r)
    s.to_csv(ROOT/"output/primary_spread_all_months.csv",index=False)
    factors,summary=write_benchmarks(ROOT.parents[1],panel(),ROOT/"output")
    legacy=pd.read_csv(ROOT/"output/ff3_factors_legacy_compatible.csv",parse_dates=["date"])
    s=s.loc[s.date.isin(dates)].copy()
    s.to_csv(ROOT/"output/primary_spread_common_period.csv",index=False)
    primary=[]
    for name,ls in [("spread",True),("HighBM_HighG",False),("HighBM_LowG",False),("LowBM_HighG",False),("LowBM_LowG",False)]:
        result=regression_for(s[["date",name]].rename(columns={name:"return"}),factors,long_short=ls)
        primary.append({"strategy":name,"role":"primary" if name=="spread" else "secondary_exploratory",**result})
    pd.DataFrame(primary).to_csv(ROOT/"output/factor_alpha.csv",index=False)
    mean=mean_inference(s.spread)
    pd.DataFrame([{"hypothesis":"H1_nonzero_value_group_G_spread_mean","role":"primary",**mean}]).to_csv(ROOT/"output/primary_mean_test.csv",index=False)
    pd.DataFrame([{"strategy":name,**metrics(s[name])} for name in [*CELLS,"spread"]]).to_csv(ROOT/"output/performance_summary.csv",index=False)
    returns=[]
    specs=[
      ("equal_weight",{"weighting":"EW"},factors,3,None),
      ("g_cut_30",{"cut":.3},factors,3,None),("g_cut_70",{"cut":.7},factors,3,None),
      ("exclude_top5",{"exclude_top5":True},factors,3,None),("legacy_FF3",{},legacy,3,None),
      ("hac6",{},factors,6,None),("stale_365_days",{"max_age":365},factors,3,None),
      ("kospi_only",{"market":"Y"},factors,3,None),("kosdaq_only",{"market":"K"},factors,3,None),
      ("pre2024",{},factors,3,("2016-01-01","2023-12-01")),("post2024",{},factors,3,("2024-01-01","2026-05-01")),
      ("IS",{},factors,3,("2016-01-01",cfg["time_split"]["is_end"])),
      ("OOS",{},factors,3,(cfg["time_split"]["oos_start"],cfg["last_return_month"])),
      ("quarterly_rebalance",{"rebalance":"quarterly"},factors,3,None),
      ("annual_rebalance",{"rebalance":"annual"},factors,3,None)]
    base_cost=r.pivot(index="date",columns="cell",values="turnover_one_way_sum_abs").reindex(columns=CELLS)
    for sid,opts,ff,lag,period in specs:
        rr,_=form(p,**opts);ss=wide(rr);ss=ss.loc[ss.date.isin(dates)]
        if period:ss=ss.loc[ss.date.between(*period)]
        available=longest_block(ss.loc[ss[CELLS].notna().all(axis=1),"date"]);ss=ss.loc[ss.date.isin(available)]
        result=regression_for(ss[["date","spread"]].rename(columns={"spread":"return"}),ff,lags=lag)
        returns.append({"spec_id":sid,"pre_registered":True,"weighting":opts.get("weighting","VW"),"governance_cut":opts.get("cut",.5),
            "exclude_top5":opts.get("exclude_top5",False),"period":str(period),"spread_mean_pct":ss.spread.mean()*100,
            "n_stocks":float(rr.loc[rr.date.isin(available)].groupby("date").n_formed.sum().mean()) if len(available) else None,
            "oos_flag":sid=="OOS","notes":"exploratory; all specifications reported; reported corp_cls not independently historical exchange validated",**result})
    sized=p.copy()
    sized["size_group"]=sized.groupby("date").lag_me.transform(lambda v:np.where(v<=v.median(),"Small","Big"))
    size_spreads=[]
    for label in ("Small","Big"):
        rr,_=form(sized.loc[sized.size_group.eq(label)])
        ss=wide(rr);ss=ss.loc[ss.date.isin(dates)]
        size_spreads.append(ss[["date","spread"]].rename(columns={"spread":label}))
    sz=size_spreads[0].merge(size_spreads[1],on="date")
    sz["spread"]=(sz.Small+sz.Big)/2
    block=longest_block(sz.loc[sz.spread.notna(),"date"]);sz=sz.loc[sz.date.isin(block)]
    sr=regression_for(sz[["date","spread"]].rename(columns={"spread":"return"}),factors)
    returns.append({"spec_id":"size_stratified_2_groups","pre_registered":True,"spread_mean_pct":sz.spread.mean()*100,"oos_flag":False,
                    "notes":"equal combination of within-size BMxG spreads; whole available formation universe size median",**sr})
    for bps in (10,30):
        ss=s.copy()
        costs=base_cost.HighBM_HighG+base_cost.HighBM_LowG
        ss["spread"]=ss.spread-ss.date.map(costs)*bps/10000
        rr=regression_for(ss[["date","spread"]].rename(columns={"spread":"return"}),factors)
        returns.append({"spec_id":"cost_"+str(bps)+"bp","pre_registered":True,"weighting":"VW","governance_cut":.5,"exclude_top5":False,
                 "spread_mean_pct":ss.spread.mean()*100,"oos_flag":False,"notes":"assumed one-way cost per traded amount; short borrow/financing unverified",**rr})
    control_table=controls(p,dates)
    control_table.to_csv(ROOT/"output/verified_controls.csv",index=False)
    cr=control_table.iloc[0].to_dict()
    returns.append({"spec_id":"roe_control_if_verified","pre_registered":True,"status":cr.get("status"),"n_months":cr.get("n_months",0),
          "p_hac":cr.get("p_hac"),"t_hac":cr.get("t_hac"),"covariate_g_slope_pct":cr.get("mean_pct"),"oos_flag":False,
          "notes":"monthly cross-sectional G coefficient controlling logME, BM, ROE and leverage; differs from portfolio alpha",
          "not_estimated_reason":cr.get("reason")})
    for unavailable in ("quarterly_G_if_data_budget_allows","industry_control"):
        returns.append({"spec_id":unavailable,"pre_registered":unavailable!="industry_control","status":"not_estimated",
            "n_months":0,"not_estimated_reason":"separate_control_table" if unavailable=="roe_control_if_verified" else "not_yet_run_or_insufficient_verified_source","oos_flag":False})
    rob=pd.DataFrame(returns)
    for new,old in [("alpha_pct","alpha_monthly_pct"),("t_hac","t_hac"),("p_hac","p_hac"),("ci_low","ci95_low_pct"),("ci_high","ci95_high_pct")]:
        rob[new]=rob[old] if old in rob else np.nan
    rob.to_csv(ROOT/"output/robustness.csv",index=False)
    tests=[]
    for row in primary:
        tests.append({"spec_id":row["strategy"],"role":row["role"],"p_raw":row.get("p_hac"),"effect_pct":row.get("alpha_monthly_pct"),"status":row["status"]})
    tests += [{"spec_id":"H1_mean","role":"primary","p_raw":mean.get("p_hac"),"effect_pct":mean.get("mean_pct"),"status":mean["status"]}]
    tests += [{"spec_id":row["spec_id"],"role":"secondary_exploratory","p_raw":row.get("p_hac"),"effect_pct":row.get("alpha_monthly_pct"),"status":row["status"]} for row in returns]
    mt=pd.DataFrame(tests);mask=mt.role.eq("secondary_exploratory")&mt.p_raw.notna()
    mt["p_holm"]=np.nan
    if mask.any():mt.loc[mask,"p_holm"]=multipletests(mt.loc[mask,"p_raw"].astype(float),method="holm")[1]
    mt["total_secondary_tests_performed"]=int(mask.sum())
    mt.to_csv(ROOT/"output/multiple_testing.csv",index=False)
    bootstrap(s,factors,cfg["uncertainty"]["bootstrap_repetitions"]).to_csv(ROOT/"output/block_bootstrap.csv",index=False)
    influence=[]
    if len(s)>=24:
        d=s[["date","spread"]].merge(factors[["date","Mkt-RF","SMB","HML"]],on="date")
        X=sm.add_constant(d[["Mkt-RF","SMB","HML"]]);fit=sm.OLS(d.spread,X).fit()
        for i,row in d.iterrows():
            sub=d.drop(i);xx=sm.add_constant(sub[["Mkt-RF","SMB","HML"]])
            a=sm.OLS(sub.spread,xx).fit().params["const"]*100
            influence.append({"removed_month":str(row.date.date()),"alpha_pct":float(a),"full_alpha_pct":float(fit.params["const"]*100),"diagnostic_only":True})
        diag={"ljung_box_p_lag6":float(acorr_ljungbox(fit.resid,lags=[6],return_df=True).lb_pvalue.iloc[0]),
             "breusch_pagan_p":float(het_breuschpagan(fit.resid,fit.model.exog)[1]),
             "factor_condition_number":float(fit.condition_number),"factor_correlation":d[["Mkt-RF","SMB","HML"]].corr().to_dict()}
    else:diag={"status":"not_estimated","reason":"fewer_than_24_contiguous_primary_months"}
    pd.DataFrame(influence,columns=["removed_month","alpha_pct","full_alpha_pct","diagnostic_only"]).to_csv(ROOT/"output/leave_one_month_out.csv",index=False)
    (ROOT/"output/model_diagnostics.json").write_text(json.dumps(diag,indent=2))
    main=primary[0]
    status={"final_status":"COMPLETED_EMPIRICAL" if main["status"]=="estimated" else "EMPIRICAL_LIMITED",
      "claim_level":"exploratory_conditional_on_original_price_universe_and_FF3",
      "data_acquisition":"actual_historical_API_acquired","pit_verification":"raw_hash_values_and_filing_dates_verified",
      "portfolio_estimation":"estimated_months_with_missing_cell_months_explicit","ff3_alpha":main["status"],
      "bias_validation":"pending","oos_validation":"pending","report_status":"pending",
      "source_commit":cfg["source_commit"],"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "pre_registration_sha256":window["pre_registration_sha256"],"n_valid_months":len(s),
      "alpha_monthly_pct":main.get("alpha_monthly_pct"),"alpha_p_hac":main.get("p_hac"),
      "total_g_obs":len(pd.read_parquet(ROOT/"data/private/governance_observations.parquet")),
      "joint_panel_obs":int(p.joint_eligible.sum()),"gate_failures":[],"limitations":[],"next_actions":[]}
    (ROOT/"output/RESULT_STATUS.json").write_text(json.dumps(status,indent=2))
    print(json.dumps({"window":window,"primary_alpha":main,"H1":mean},ensure_ascii=False),flush=True)

if __name__=="__main__":analyze()
