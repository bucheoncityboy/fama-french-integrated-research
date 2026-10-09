import json
from pathlib import Path
import numpy as np
import pandas as pd
from .acquire import ROOT,SOURCE,verify_registration,panel
from .disclosure_pit import available_at,calendar_tables,select_observation,validate_dates
from .verify_evidence import verify_g_frame,verify_financial_frame


def build():
    cfg=verify_registration();p=panel().sort_values(["stock_code","date"]).copy()
    if p.duplicated(["stock_code","date"]).any():raise ValueError("duplicate_market_month")
    g=pd.read_parquet(ROOT/"data/private/governance_observations.parquet")
    f=pd.read_parquet(ROOT/"data/private/verified_financials.parquet")
    g=verify_g_frame(g,ROOT/"data/raw")
    f=verify_financial_frame(f,ROOT/"data/raw")
    # Annual reports only for the preregistered primary.
    g=g.loc[g.reprt_code.eq("11011")].copy()
    g["available_at"]=g.filing_date.map(lambda d:available_at(d) if pd.notna(d) else pd.NaT)
    f=f.loc[f.reprt_code.eq("11011")].copy()
    f["financial_available_at"]=f.financial_filing_date.map(available_at)
    june=p.loc[p.date.dt.month.eq(6),["stock_code","date","me"]].copy()
    june["formation_year"]=june.date.dt.year;june["be_year"]=june.formation_year-1
    ends,starts=calendar_tables()
    june["june_formation_day"]=june.date.map(lambda d:ends.get(d.to_period("M"),pd.NaT))
    f=f.drop(columns=["bsns_year"],errors="ignore")
    formation=june.merge(f,on=["stock_code","be_year"],how="left",validate="one_to_one")
    formation["financial_PIT"]=formation.financial_available_at.le(formation.june_formation_day)
    formation["bm_pit"]=formation.be_eok/(formation.me/100000)
    formation.loc[~formation.financial_PIT|(formation.me<=0)|(formation.be_eok<=0),"bm_pit"]=np.nan
    # Form from the preceding month's universe, then LEFT JOIN future returns.
    # A terminal missing month stays missing; do not condition holdings on future survival.
    realized=p[["stock_code","date","return"]].copy()
    p=p.drop(columns=["return"]).rename(columns={"me":"lag_me","date":"observation_month"})
    p["date"]=p.observation_month+pd.offsets.MonthBegin(1)
    p=p.merge(realized,on=["stock_code","date"],how="left",validate="one_to_one")
    p["formation_year"]=p.date.dt.year-p.date.dt.month.le(6).astype(int)
    for c in ("roe","leverage","total_assets_krw"):
        if c not in formation:formation[c]=np.nan
    p=p.merge(formation[["stock_code","formation_year","bm_pit","financial_filing_date","financial_rcept_no","financial_source_hash","financial_PIT","roe","leverage","total_assets_krw"]],
                  on=["stock_code","formation_year"],how="left",validate="many_to_one")
    p=p.loc[p.date.between("2016-01-01",cfg["last_return_month"])].copy()
    p["formation_month_end"]=p.date.map(lambda d:ends.get((d-pd.offsets.MonthBegin(1)).to_period("M"),pd.NaT))
    p["return_month_start"]=p.date
    p["first_return_session"]=p.date.map(lambda d:starts.get(d.to_period("M"),pd.NaT))
    outputs=[]
    groups={s:d for s,d in g.groupby("stock_code")}
    fields=["g_value","filing_date","available_at","period_end","rcept_no","source_hash","corp_cls_reported","is_amendment"]
    for stock,market in p.groupby("stock_code",sort=True):
        market=market.sort_values("formation_month_end")
        obs=groups.get(stock,g.iloc[:0]);obs=obs.loc[obs.status.eq("PASS")&obs.available_at.notna()]
        events=[]
        for at in sorted(obs.available_at.unique()):
            chosen=select_observation(obs,pd.Timestamp(at),cfg["primary"]["max_governance_age_days"])
            if chosen is not None:events.append({"event_at":pd.Timestamp(at),**{key:chosen[key] for key in fields}})
        if events:
            market=pd.merge_asof(market,pd.DataFrame(events).sort_values("event_at"),left_on="formation_month_end",right_on="event_at",direction="backward")
            market["age_days"]=(market.formation_month_end-market.filing_date).dt.days
            market["G_PIT"]=market.g_value.notna()&market.age_days.le(cfg["primary"]["max_governance_age_days"])
            market.loc[~market.G_PIT,"g_value"]=np.nan
        else:
            for key in fields:market[key]=pd.NaT if key in ("filing_date","available_at","period_end") else (False if key=="is_amendment" else np.nan)
            market["age_days"]=np.nan;market["G_PIT"]=False
        outputs.append(market)
    p=pd.concat(outputs,ignore_index=True)
    p["joint_eligible"]=p.G_PIT&p.bm_pit.gt(0)&p.lag_me.gt(0)
    validate_dates(p.loc[p.joint_eligible])
    p.to_parquet(ROOT/"data/private/joint_panel.parquet",index=False)
    g.to_parquet(ROOT/"data/private/governance_pit.parquet",index=False)
    quality=p.groupby("date").agg(population_N=("stock_code","size"),G_PIT_N=("G_PIT","sum"),financial_PIT_N=("financial_PIT","sum"),
                    joint_N=("joint_eligible","sum"),missing_returns=("return",lambda x:x.isna().sum()))
    quality.to_csv(ROOT/"output/governance_panel_quality.csv")
    quality.to_csv(ROOT/"data/coverage.csv")
    result={"joint_panel_obs":int(p.joint_eligible.sum()),"joint_stocks":p.loc[p.joint_eligible,"stock_code"].nunique(),
           "months_with_joint_data":int(quality.joint_N.gt(0).sum()),"future_leaks":0,"primary_annual_G_rows":len(g),"G_PIT_observations":int(g.status.eq("PASS").sum())}
    (ROOT/"output/pit_status.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=="__main__":build()
