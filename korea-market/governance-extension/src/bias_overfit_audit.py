"""Evidence-based bias diagnostics; unverified claims are never promoted to PASS."""
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .acquire import ROOT,verify_registration
from .disclosure_pit import validate_dates


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__();self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=="tr":self.row=[]
        if tag in ("td","th"):self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ("td","th") and self.cell is not None:
            if self.row is not None:self.row.append("".join(self.cell).strip())
            self.cell=None
        if tag=="tr" and self.row is not None:
            if self.row:self.rows.append(self.row)
            self.row=None


def current_listing():
    f=ROOT/"data/private/krx_current_list.html"
    if not f.exists():return pd.DataFrame()
    parser=TableParser();parser.feed(f.read_bytes().decode("euc-kr",errors="replace"))
    if not parser.rows:return pd.DataFrame()
    h=parser.rows[0];rows=[r for r in parser.rows[1:] if len(r)==len(h)]
    d=pd.DataFrame(rows,columns=h)
    if not {"회사명","시장구분","종목코드","업종","상장일"}.issubset(d):return pd.DataFrame()
    d=d.rename(columns={"회사명":"current_name","시장구분":"current_market","종목코드":"stock_code","업종":"current_industry","상장일":"current_listing_date"})
    d["stock_code"]=d.stock_code.map(lambda x:x.zfill(6) if x.isdigit() else x)
    d=d[["stock_code","current_name","current_market","current_industry","current_listing_date"]].drop_duplicates("stock_code")
    d.to_csv(ROOT/"data/private/krx_current_listing.csv",index=False)
    return d


def missing_characteristics(p):
    rows=[]
    for date,month in p.groupby("date"):
        for label,group in month.groupby("G_PIT"):
            rows.append({"date":date,"has_G":bool(label),"N":len(group),"mean_log_ME":float(np.log(group.loc[group.lag_me.gt(0),"lag_me"]).mean()),
              "median_BM_verified_only":float(group.bm_pit.median()),"verified_BM_N":int(group.bm_pit.notna().sum()),
              "return_available_fraction":float(group["return"].notna().mean()),
              "mean_return_available_pct":float(group["return"].mean()*100),
              "reported_K_fraction":float(group.corp_cls_reported.eq("K").mean()),
              "reported_Y_fraction":float(group.corp_cls_reported.eq("Y").mean())})
    return pd.DataFrame(rows)


def audit():
    cfg=verify_registration()
    p=pd.read_parquet(ROOT/"data/private/joint_panel.parquet")
    g=pd.read_parquet(ROOT/"data/private/governance_pit.parquet")
    r=pd.read_csv(ROOT/"output/primary_returns.csv",parse_dates=["date"])
    status=json.loads((ROOT/"output/RESULT_STATUS.json").read_text())
    rob=pd.read_csv(ROOT/"output/robustness.csv")
    alpha=pd.read_csv(ROOT/"output/factor_alpha.csv")
    window=json.loads((ROOT/"config/analysis_window.lock.json").read_text())
    u=pd.read_csv(ROOT/"data/private/universe.csv",dtype={"stock_code":str})
    mapped=pd.read_csv(ROOT/"data/ticker_map_audit.csv",dtype={"stock_code":str})
    listing=current_listing()
    if len(listing):
        survival=u.merge(listing,on="stock_code",how="left")
        survival["present_in_current_KRX_snapshot"]=survival.current_market.notna()
        survival["early_price_end"]=pd.to_datetime(survival.last_month)<pd.Timestamp("2026-05-01")
        survival.drop(columns=["predefined_ME_2014_12"],errors="ignore").to_csv(ROOT/"output/current_listing_survival_audit.csv",index=False)
        current_count=int(survival.present_in_current_KRX_snapshot.sum())
        exited_count=int((survival.early_price_end&~survival.present_in_current_KRX_snapshot).sum())
        market_distribution=survival.current_market.fillna("NOT_IN_CURRENT_SNAPSHOT").value_counts().to_dict()
        gp=p.merge(listing[["stock_code","current_industry","current_market"]],on="stock_code",how="left")
        industries=gp.groupby(["date","G_PIT","current_industry"],dropna=False).agg(N=("stock_code","size")).reset_index()
        industries["classification_timing"]="2026-10-09_current_snapshot_not_historical_PIT"
        industries.to_csv(ROOT/"output/current_industry_coverage_diagnostic.csv",index=False)
    else:
        current_count,exited_count,market_distribution=None,None,{}
    chars=missing_characteristics(p)
    chars.to_csv(ROOT/"output/coverage_characteristics.csv",index=False)
    selection=p.groupby("date").agg(population_N=("stock_code","size"),G_PIT_N=("G_PIT","sum"),financial_PIT_N=("financial_PIT","sum"),
             joint_N=("joint_eligible","sum"),missing_return_N=("return",lambda v:int(v.isna().sum())))
    mapped_set=set(mapped.loc[mapped.status.eq("UNIQUE_OFFICIAL"),"stock_code"])
    selection["code_matched_N"]=p.stock_code.isin(mapped_set).groupby(p.date).sum()
    raw_g_codes=set(g.stock_code)
    selection["summary_responded_company_N"]=p.stock_code.isin(raw_g_codes).groupby(p.date).sum()
    selection["summary_response_timing"]="ever_responded; PIT_N_is_the_time_valid_count"
    selection["four_cell_formed_N"]=r.groupby("date").n_formed.sum()
    selection["all_cells_valid"]=r.groupby("date").status.apply(lambda v:len(v)==4 and v.eq("estimated").all())
    selection["insufficient_cell_N"]=r.loc[r.n_formed.lt(5)].groupby("date").size().reindex(selection.index,fill_value=0)
    selection.to_csv(ROOT/"output/selection_audit.csv")
    concentration=r.groupby("cell").agg(months=("date","size"),mean_hhi=("hhi","mean"),max_hhi=("hhi","max"),
             median_effective_N=("effective_N","median"),max_top5_weight=("top5_weight","max"),
             mean_log_me=("mean_log_me","mean"),mean_bm=("mean_bm","mean"),mean_roe=("mean_roe","mean"),
             mean_leverage=("mean_leverage","mean"),mean_age_days=("mean_age_days","mean"),
             mean_G_tie_fraction=("g_tie_at_cut_fraction","mean"),mean_large_assets_fraction=("assets_ge_2trillion_fraction","mean"))
    concentration.to_csv(ROOT/"output/concentration_confounding.csv")
    gd=g.loc[g.status.eq("PASS")].copy()
    gd["ratio_bin"]=gd.g_value.round(6)
    gd.groupby(["bsns_year","ratio_bin"]).size().rename("N").to_csv(ROOT/"output/governance_ratio_distribution.csv")
    financial=pd.read_parquet(ROOT/"data/private/verified_financials.parquet")
    if "total_assets_krw" in financial:
        reg=gd.merge(financial.loc[financial.reprt_code.eq("11011"),["stock_code","be_year","total_assets_krw","fs_div"]],
                     left_on=["stock_code","bsns_year"],right_on=["stock_code","be_year"],how="left",validate="one_to_one")
        reg["asset_group"]=np.where(reg.total_assets_krw.ge(2e12),"ge_2trillion","lt_2trillion_or_unverified")
        reg.groupby(["bsns_year","asset_group"]).agg(N=("stock_code","size"),mean_G=("g_value","mean"),median_G=("g_value","median"),
             G_unique=("g_value","nunique")).to_csv(ROOT/"output/regulatory_size_diagnostic.csv")
    future_count=int((p.loc[p.joint_eligible,"filing_date"]>=p.loc[p.joint_eligible,"return_month_start"]).sum())
    validate_dates(p.loc[p.joint_eligible])
    missing_cells=int(r.n_missing_return.gt(0).sum())
    stale_count=int(p.loc[p.joint_eligible,"age_days"].gt(365).sum())
    amendments=int(gd.is_amendment.sum())
    main=alpha.loc[alpha.strategy.eq("spread")].iloc[0].to_dict()
    rows=[]
    def add(kind,method,n,metric,verdict,evidence,severity,mitigation,risk):
        rows.append({"bias_type":kind,"method":method,"n_observations":n,"metric":json.dumps(metric,ensure_ascii=False),
          "pass_fail_unverified":verdict,"evidence_path":evidence,"severity":severity,"mitigation":mitigation,"remaining_risk":risk})
    add("look_ahead_publication","raw_hash/value/receipt checks; KRX next-day month-end; financial BE value/date cross-match",
        int(p.joint_eligible.sum()),{"future_leaks":future_count,"PIT_G_rows":int(gd.raw_verified.sum())},"PASS" if future_count==0 else "FAIL",
        "governance_panel_quality.csv; data/source_audit.csv","high","quarantine unmatched receipts and late financial revisions","does not establish unobserved original historical versions")
    add("survivorship_delisting","retain original historical codes; preceding-month universe; current official KRX listing snapshot and terminal-return gaps",
        len(u),{"currently_listed":current_count,"early_end_and_not_current":exited_count,"missing_formed_cell_months":missing_cells},
        "UNVERIFIED","current_listing_survival_audit.csv; selection_audit.csv","high","no current-list survivor filter; missing return invalidates cell month; never impute terminal return",
        "original exchange-wide universe/delisting-return completeness and absence of historical companies remain unresolved")
    add("coverage_selection","monthly covered/uncovered lag-ME, verified BM, returns and current industry descriptive comparisons",
        len(p),{"mean_G_coverage":float(p.groupby("date").G_PIT.mean().mean()),"market_distribution_current":market_distribution},
        "UNVERIFIED","coverage_characteristics.csv; current_industry_coverage_diagnostic.csv; selection_audit.csv","high",
        "report full acquisition failures and flow of exclusions","non-random governance and financial matching; original universe is not the complete Korean stock market")
    add("size_industry_profitability_confounding","cell log-ME/BM/verified ROE/leverage; verified monthly cross-sectional controls; current industry descriptive only",
        int(p.joint_eligible.sum()),{"control_file":"verified_controls.csv"},"UNVERIFIED",
        "concentration_confounding.csv; verified_controls.csv","high","size stratification and verified ROE checks reported including NA",
        "historical industry classifications absent; regulation and profitability may explain G differences")
    add("statutory_minimum_ratio","G discrete/tie distribution versus annual reported total-assets >=2trillion; distinguish financial-statement scope",
        len(gd),{"amended_G_rows":amendments},"UNVERIFIED","regulatory_size_diagnostic.csv; governance_ratio_distribution.csv","high",
        "ratio interpreted as outside-director share only","consolidated assets do not certify statutory separate-entity threshold; historical company exceptions not independently verified")
    add("mega_cap_concentration","cell HHI/effective N/top5 weights; exclude top5 of original formation-month population",
        len(r),{"max_top5_weight":float(r.top5_weight.max())},"PASS","concentration_confounding.csv; robustness.csv","medium",
        "predefined top5 exclusion; publish weak and NA sensitivities","PASS means diagnostic performed; does not mean economically unconcentrated")
    add("stale_revised_data","period priority, latest receipt only at its actual publication; 365/550 day comparison",
        len(gd),{"joint_rows_older_365d":stale_count,"amended_rows":amendments},"UNVERIFIED",
        "data/source_audit.csv; robustness.csv","high","late corrections never backdated; current final value can only enter after final receipt",
        "document/original-file API unavailable; original-before-correction counts not reconstructed")
    add("threshold_mining","frozen YAML SHA256 and complete fixed secondary registry; no max-Sharpe or min-p choice",
        len(rob),{"locked_sha":window["pre_registration_sha256"]},"PASS","config/pre_registration.lock.json; multiple_testing.csv","medium",
        "report every preregistered variant including insufficient samples","initial smoke membership correction recorded before G outcomes; no claim of prospective original HML registration")
    is_row=rob.loc[rob.spec_id.eq("IS")];oos_row=rob.loc[rob.spec_id.eq("OOS")]
    is_n=int(is_row.n_months.iloc[0]) if len(is_row) else 0;oos_n=int(oos_row.n_months.iloc[0]) if len(oos_row) else 0
    independent=is_n>=24 and oos_n>=24 and is_row.status.iloc[0]=="estimated" and oos_row.status.iloc[0]=="estimated"
    add("time_series_overfit","2021-12/2022-01 fixed split; pre/post2024, contiguous windows and independent monthly observation unit",
        int(status["n_valid_months"]),{"IS_months":is_n,"OOS_months":oos_n,"independent_OOS":independent},
        "PASS" if independent else "UNVERIFIED","robustness.csv; config/analysis_window.lock.json","high",
        "no holdout tuning; fixed tests disclosed","short IS and data-availability exclusions can prevent independent OOS validation")
    mt=pd.read_csv(ROOT/"output/multiple_testing.csv")
    add("multiple_testing","all primary H1/H2 and secondary mean/alpha/control p values; Holm secondary family",
        len(mt),{"secondary_tests_performed":int(mt.total_secondary_tests_performed.iloc[0])},"PASS",
        "multiple_testing.csv","medium","all nonsignificant and NA results retained","Holm covers every performed secondary inference including supplementary mean and controls; primary H1/H2 separately disclosed")
    influence=pd.read_csv(ROOT/"output/leave_one_month_out.csv")
    add("outlier_influence","leave-one-month-out alpha; historical >100% price-return filtering audit",
        len(influence),{"leave_one_month_alpha_min":float(influence.alpha_pct.min()) if len(influence) else None,
                        "leave_one_month_alpha_max":float(influence.alpha_pct.max()) if len(influence) else None},
        "UNVERIFIED","leave_one_month_out.csv; return_filter_audit.csv","high",
        "no winsorization or deletion to improve results","independent adjusted-price/corporate-action source unavailable; original return filter inherited")
    add("transaction_implementation","drift-adjusted turnover; monthly/quarterly/annual membership; assumed 0/10/30bp per one-way traded amount",
        len(r),{"average_turnover":float(r.turnover_one_way_sum_abs.mean())},"UNVERIFIED",
        "primary_returns.csv; robustness.csv","high","gross spread and assumed-cost estimates separated",
        "short eligibility, borrowing, financing, liquidity and suspension execution unverified")
    add("factor_definition","legacy/spec-aligned factors, same-month RF treatment; value-date verification for portfolio BM only",
        int(status["n_valid_months"]),{"benchmark":"spec_aligned_FF3"},"UNVERIFIED","ff3_summary.csv; robustness.csv; ff3_specification_sensitivity.csv",
        "high","report both benchmarks; conditional-on-benchmark claim","benchmark universe and every factor stock's financial PIT/dividend metadata not independently resolved")
    add("model_uncertainty","HAC3/HAC6; circular blocks 6/12; residual autocorrelation, heteroskedasticity and condition number",
        int(status["n_valid_months"]),{"primary_alpha_status":main.get("status")},"PASS" if main.get("status")=="estimated" else "UNVERIFIED",
        "block_bootstrap.csv; model_diagnostics.json; robustness.csv","medium","all uncertainty methods shown; no favorable-method selection","short sample and temporal dependence limit power")
    pd.DataFrame(rows).to_csv(ROOT/"output/bias_audit.csv",index=False)
    if (ROOT/"output/supplementary_robustness.csv").exists():
        pilot=pd.read_csv(ROOT/"output/supplementary_robustness.csv")
        inf=pd.read_csv(ROOT/"output/pilot_leave_one_month_out.csv")
        for row in rows:
            if row["bias_type"]=="size_industry_profitability_confounding":
                row["evidence_path"]+="; pilot_verified_controls.csv; control_extraction_integrity.csv"
            if row["bias_type"]=="outlier_influence":
                row["metric"]=json.dumps({"primary_N":len(influence),"pilot_N":len(inf),"pilot_alpha_min":float(inf.alpha_pct.min()),"pilot_alpha_max":float(inf.alpha_pct.max())})
                row["evidence_path"]+="; pilot_leave_one_month_out.csv; independent_price_event_flags.csv; corporate_action_evidence.csv"
            if row["bias_type"]=="model_uncertainty":
                row["evidence_path"]+="; fixed_stage150_two_leg_available_block_bootstrap.csv; pilot_model_diagnostics.json"
        pd.DataFrame(rows).to_csv(ROOT/"output/bias_audit.csv",index=False)
    status["bias_validation"]="completed_diagnostics_with_unverified_material_biases"
    status["oos_validation"]="fixed_time_holdout_estimated" if independent else "in_sample_exploratory_no_independent_OOS"
    status["gate_failures"]=[x["bias_type"] for x in rows if x["pass_fail_unverified"]=="FAIL"]
    if main.get("status")!="estimated":status["gate_failures"].append("Gate_D_primary_contiguous_min24_not_met")
    status["limitations"]=[x["remaining_risk"] for x in rows if x["severity"]=="high" and x["pass_fail_unverified"]!="PASS"]
    status["claim_level"]="exploratory; no_causal_or_investable_alpha_claim"
    status["next_actions"]=["recover pre-2020 governance originals if available","independently audit complete historical listing/delisting and adjusted total returns",
                            "obtain historical sector and legal entity asset metadata","extend independent holdout without specification tuning"]
    if main.get("status")!="estimated":status["final_status"]="EMPIRICAL_LIMITED"
    (ROOT/"output/RESULT_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2))
    text=f"""# Overfit assessment

Primary: H1 two-sided mean spread and H2 FF3 alpha. Monthly observations, not stock-month counts, drive inference.
Frozen G registration hash: {window["pre_registration_sha256"]}.
Fixed IS end 2021-12 and OOS start 2022-01. Usable IS months {is_n}; OOS months {oos_n}.
Independent OOS validation: {independent}. No independent-OOS claim when the IS/OOS prerequisites fail.
Primary specifications were never selected by return, Sharpe or p value. Membership-only smoke correction is recorded in commit 0ada647 before strategy outcomes.
All secondary attempted specifications and unavailable cases appear in multiple_testing.csv, robustness.csv and supplementary_robustness.csv; Holm applies to every performed secondary mean, alpha and control inference.
The longest contiguous valid window is an availability rule fixed in advance; it may introduce selection of periods through missing data. All excluded months remain in primary_returns.csv and selection_audit.csv.
Leave-one-month-out and circular time-block bootstrap are diagnostics on the fixed monthly strategy/factor path, not new stock-level backtests.
Claim permission: exploratory conditional evidence within this original price universe. Historical market completeness, terminal returns, dividend/price adjustment, historical industry and implementability remain material limitations.
No causal G effect, COE reduction, new ESG factor or prospective investment recommendation is established.

The 150-stage cumulative acquisition cohort has a distinct supplementary two-value-leg window, registered after availability diagnostics and before G outcome inspection. It does not replace the original full-universe/all-four-cell primary. Its 24-month 2024-04 to 2026-03 estimates are exploratory, availability-conditioned and not independent OOS. The supplementary availability design recorded an intermediate 50/15-month count; later prior-universe/missing-terminal-return correction reduced final primary counts to 23 individually valid/3 contiguous. The locked primary YAML and minimum 24 threshold were not changed. Correcting exact repeated NI rows after outcomes recovered controls without changing any core strategy data hash.
"""
    (ROOT/"output/overfit_assessment.md").write_text(text)
    print(json.dumps({"bias_rows":len(rows),"PASS":sum(x["pass_fail_unverified"]=="PASS" for x in rows),
           "UNVERIFIED":sum(x["pass_fail_unverified"]=="UNVERIFIED" for x in rows),"independent_OOS":independent}),flush=True)

if __name__=="__main__":audit()
