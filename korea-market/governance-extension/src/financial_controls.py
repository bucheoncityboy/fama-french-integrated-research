"""Verified, scope-consistent control extraction from cached major accounts."""
import hashlib
import json
import numpy as np
import pandas as pd


def unique_amount(accounts, names, field="thstrm_amount"):
    # OpenDART major accounts repeats NI at ord 29/61. Collapse only exact
    # finite numeric duplicates; conflicting values remain unavailable.
    from .acquire import number
    hits=[number(a.get(field)) for a in accounts if a.get("account_nm", "").replace(" ", "") in names]
    if not hits or not all(np.isfinite(v) for v in hits):
        return np.nan
    values=set(hits)
    return next(iter(values)) if len(values)==1 else np.nan


def extract_controls(accounts, equity_name, equity):
    assets=unique_amount(accounts,["자산총계"])
    liabilities=unique_amount(accounts,["부채총계"])
    income=unique_amount(accounts,["당기순이익","당기순이익(손실)"])
    prior=unique_amount(accounts,[equity_name.replace(" ", "")],"frmtrm_amount")
    roe=income/((equity+prior)/2) if equity>0 and prior>0 else np.nan
    return {"roe":roe,"leverage":liabilities/assets if assets>0 else np.nan,"total_assets_krw":assets}


def refresh():
    from .acquire import ROOT
    f=pd.read_parquet(ROOT/"data/private/verified_financials.parquet")
    for i,row in f.iterrows():
        b=(ROOT/"data/raw"/(row.financial_source_hash+".json")).read_bytes()
        if hashlib.sha256(b).hexdigest()!=row.financial_source_hash:
            raise ValueError("financial_hash_mismatch")
        accounts=[a for a in json.loads(b).get("list",[]) if a.get("fs_div")==row.fs_div and
                  str(a.get("rcept_no"))==row.financial_rcept_no and a.get("currency")=="KRW" and
                  str(a.get("corp_code"))==row.corp_code and int(a.get("bsns_year",-1))==row.be_year]
        values=extract_controls(accounts,row.account_nm,row.dart_be_eok*1e8)
        for col,value in values.items():f.at[i,col]=value
    f.to_parquet(ROOT/"data/private/verified_financials.parquet",index=False)
    records=[]
    for filename in ("joint_panel.parquet","stage150_joint_panel.parquet"):
        file=ROOT/"data/private"/filename
        if not file.exists():continue
        p=pd.read_parquet(file)
        core=p.drop(columns=["roe","leverage","total_assets_krw"],errors="ignore")
        before=hashlib.sha256(pd.util.hash_pandas_object(core,index=False).values.tobytes()).hexdigest()
        keys=f[["stock_code","be_year","financial_source_hash","roe","leverage","total_assets_krw"]].copy()
        keys["formation_year"]=keys.be_year+1
        p=p.drop(columns=["roe","leverage","total_assets_krw"],errors="ignore").merge(
            keys.drop(columns="be_year"),on=["stock_code","formation_year","financial_source_hash"],how="left",validate="many_to_one",sort=False)
        after=hashlib.sha256(pd.util.hash_pandas_object(p[core.columns],index=False).values.tobytes()).hexdigest()
        if before!=after:raise ValueError("core_strategy_data_changed")
        p.to_parquet(file,index=False)
        records.append({"file":filename,"core_hash_before":before,"core_hash_after":after,"unchanged":before==after,
                        "roe_nonmissing":int(p.roe.notna().sum())})
    pd.DataFrame(records).to_csv(ROOT/"output/control_extraction_integrity.csv",index=False)
    print(json.dumps({"verified_financial_rows":len(f),"roe_nonmissing":int(f.roe.notna().sum()),"core_strategy_unchanged":True}))

if __name__=="__main__":refresh()
