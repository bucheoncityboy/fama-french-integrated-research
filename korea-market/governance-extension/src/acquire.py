"""Staged historical acquisition. Selection never depends on G strategy outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml
from .dart_client import Client, APIError

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent


def verify_registration():
    p=ROOT/"config/pre_registration.yaml"
    lock=json.loads(p.with_suffix(".lock.json").read_text())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=lock["sha256"]:
        raise ValueError("pre_registration_changed")
    return yaml.safe_load(p.read_text())


def panel():
    d=pd.read_parquet(SOURCE/"data/panel_data.parquet")
    d["stock_code"]=d.code.str.removeprefix("A")
    return d


def baseline():
    p=panel(); rows=[]
    for file in sorted((SOURCE/"data").glob("*.parquet")):
        d=pd.read_parquet(file)
        rows.append({"file":file.name,"rows":len(d),"columns":json.dumps({c:str(d[c].dtype) for c in d}),
              "date_start":str(d.date.min()),"date_end":str(d.date.max()),
              "companies":d.code.nunique() if "code" in d else None,"duplicate_code_date":d.duplicated(["code","date"]).sum() if "code" in d and "item_code" not in d else None})
    pd.DataFrame(rows).to_csv(ROOT/"data/input_schema.csv",index=False)
    u=p.groupby("stock_code").agg(name=("name","first"),first_month=("date","min"),last_month=("date","max"),
                        stock_months=("date","size"),valid_returns=("return","count"))
    s=p.loc[p.date.eq("2014-12-01"),["stock_code","me"]].set_index("stock_code").me
    u["predefined_ME_2014_12"]=s
    ranks=s.rank(method="first")
    strata=((ranks-1)*3/len(ranks)).astype(int)
    groups=[list(strata.loc[strata.eq(i)].index.sort_values()) for i in range(3)]
    order=[]
    for i in range(max(map(len,groups))):
        for g in groups:
            if i<len(g):order.append(g[i])
    order+=sorted(set(u.index)-set(order))
    u["acquisition_order"]=[order.index(c) for c in u.index]
    u.to_csv(ROOT/"data/private/universe.csv")
    book=pd.read_parquet(SOURCE/"data/book_equity_monthly.parquet")
    be=book.sort_values("date").groupby(["code","be_year"]).first().reset_index()
    be.to_parquet(ROOT/"data/private/fiscal_book.parquet",index=False)
    source_lock=json.loads((ROOT/"data/source_snapshot.json").read_text())
    unchanged=sum(hashlib.sha256((ROOT.parents[1]/r["path"]).read_bytes()).hexdigest()==r["sha256"] for r in source_lock)
    text=f"""# Baseline audit

Source commit: 81f3bd921ffcb4ea84c19123d798fbcb830ee9e3. Original tracked files: {len(source_lock)}, unchanged: {unchanged}.
Panel: {p.stock_code.nunique()} stocks, {len(p)} stock-months, {p.date.min().date()}–{p.date.max().date()}.
Schema: data/input_schema.csv. Universe membership audit retained locally: data/private/universe.csv.

The original panel is not assumed to be the complete historical KOSPI/KOSDAQ universe.
Current-label names, financial-name exclusions and ticker-suffix preferred-stock exclusions are inherited selection rules.
Ticker-digit market flags are not used for market classification. Raw price pct_change and >100% filtering require independent corporate-action evidence.
Price adjustment, dividend reinvestment, exchange-wide delisting completeness and terminal returns remain unverified until independently evidenced.
Original six parquet files, PDFs and source scripts are read-only. Original FF3 results are prior knowledge, not preregistered G outcomes.
FnGuide redistribution permission has not been established: original stock panels and new stock-level proprietary derivatives will not be newly published.
OpenDART raw responses are cached locally. Public outputs contain derived aggregate results, hashes and filing references rather than full API response dumps.
"""
    (ROOT/"AUDIT_BASELINE.md").write_text(text)
    print(json.dumps({"phase":"baseline","stocks":len(u),"original_files_unchanged":unchanged}),flush=True)


def global_filings(client,begin,end):
    out=[];page=1
    while True:
        data,m=client.request("list.json",{"bgn_de":begin,"end_de":end,"pblntf_detail_ty":"A001",
                    "last_reprt_at":"N","page_count":100,"page_no":page})
        for r in data.get("list",[]):out.append({**r,"filing_list_hash":m["source_hash"]})
        if page>=int(data.get("total_page",1)):break
        page+=1
    return out


def mapping(smoke_only=False):
    client=Client(ROOT/"data")
    periods=[("20160101","20160331")] if smoke_only else [
          (str(start.date()).replace("-",""),str(min(start+pd.offsets.QuarterEnd(0),pd.Timestamp("2026-05-31")).date()).replace("-",""))
          for start in pd.date_range("2015-01-01","2026-05-31",freq="QS")]
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(global_filings,client,b,e):(b,e) for b,e in periods}
        for future in as_completed(pending):
            r=future.result();records.extend(r)
            print(json.dumps({"mapping_period":pending[future],"records":len(r),"requests":client.count}),flush=True)
    out=pd.DataFrame(records).drop_duplicates(["corp_code","rcept_no"])
    file=ROOT/"data/private/annual_filings.parquet"
    if file.exists():
        out=pd.concat([pd.read_parquet(file),out]).drop_duplicates(["corp_code","rcept_no"])
    out.to_parquet(file,index=False)
    u=pd.read_csv(ROOT/"data/private/universe.csv",dtype={"stock_code":str})
    valid=out.loc[out.stock_code.str.fullmatch("[0-9A-Z]{6}",na=False)].copy()
    pairs=valid.groupby(["stock_code","corp_code"]).agg(first_filing=("rcept_dt","min"),last_filing=("rcept_dt","max"),filings=("rcept_no","size")).reset_index()
    pairs["corp_codes_per_ticker"]=pairs.groupby("stock_code").corp_code.transform("nunique")
    pairs.to_csv(ROOT/"data/private/ticker_map.csv",index=False)
    audit=u[["stock_code","name","first_month","last_month"]].merge(pairs,on="stock_code",how="left")
    audit["status"]=np.where(audit.corp_code.isna(),"NO_OFFICIAL_MAPPING",np.where(audit.corp_codes_per_ticker.eq(1),"UNIQUE_OFFICIAL","AMBIGUOUS_MAPPING"))
    audit.to_csv(ROOT/"data/ticker_map_audit.csv",index=False)
    print(json.dumps({"phase":"mapping","filings":len(out),"mapped_original_tickers":audit.loc[audit.status.eq("UNIQUE_OFFICIAL"),"stock_code"].nunique(),"requests":client.count}),flush=True)


def load_mapping():
    d=pd.read_csv(ROOT/"data/private/ticker_map.csv",dtype={"stock_code":str,"corp_code":str})
    return d.loc[d.corp_codes_per_ticker.eq(1)].drop_duplicates("stock_code").set_index("stock_code").corp_code.to_dict()


def number(v):
    try:
        n=float(str(v).replace(",",""))
        return n if np.isfinite(n) else np.nan
    except (ValueError,TypeError):return np.nan


def fetch_pair(client,stock,corp,year,report="11011"):
    try:
        g,gm=client.request("outcmpnyDrctrNdChangeSttus.json",{"corp_code":corp,"bsns_year":int(year),"reprt_code":report})
        f,fm=client.request("fnlttSinglAcnt.json",{"corp_code":corp,"bsns_year":int(year),"reprt_code":report})
        return {"stock_code":stock,"corp_code":corp,"bsns_year":int(year),"reprt_code":report,
                "g":g,"gm":gm,"financial":f,"fm":fm,"error":None}
    except APIError as e:
        return {"stock_code":stock,"corp_code":corp,"bsns_year":int(year),"reprt_code":report,"error":str(e)}


def assemble(pairs):
    filings=pd.read_parquet(ROOT/"data/private/annual_filings.parquet")
    by_receipt={str(r.rcept_no):r._asdict() for r in filings.itertuples(index=False)}
    book=pd.read_parquet(ROOT/"data/private/fiscal_book.parquet")
    book["stock_code"]=book.code.str.removeprefix("A")
    bmap=book.set_index(["stock_code","be_year"]).be.to_dict()
    grows=[];frows=[];audit=[]
    for p in pairs:
        key={k:p[k] for k in ("stock_code","corp_code","bsns_year","reprt_code")}
        if p["error"]:
            audit.append({**key,"status":"API_FAILED","reason":p["error"]});continue
        gs=list({json.dumps(r,sort_keys=True):r for r in p["g"].get("list",[])}.values())
        if len(gs)!=1:
            audit.append({**key,"status":"G_EXCLUDED","reason":"no_or_ambiguous_summary_rows","rows":len(gs)});continue
        g=gs[0];receipt=str(g.get("rcept_no",""));filing=by_receipt.get(receipt)
        outside,total=number(g.get("otcmp_drctr_co")),number(g.get("drctr_co"))
        period=pd.to_datetime(g.get("stlm_dt"),errors="coerce")
        reason=""
        if not filing:reason="PIT_UNVERIFIED_receipt_not_in_actual_annual_filing_list"
        elif str(filing["corp_code"])!=p["corp_code"] or str(filing["stock_code"])!=p["stock_code"] or str(g.get("corp_code"))!=p["corp_code"]:reason="official_mapping_conflict"
        elif not (np.isfinite(total) and total>0 and np.isfinite(outside) and outside.is_integer() and total.is_integer() and 0<=outside<=total):reason="invalid_counts"
        elif pd.isna(period) or period.year!=p["bsns_year"]:reason="invalid_period"
        filing_date=pd.to_datetime(filing["rcept_dt"],format="%Y%m%d") if filing else pd.NaT
        if not reason and period>filing_date:reason="economic_period_after_filing"
        gr={**key,"rcept_no":receipt,"period_end":period,"filing_date":filing_date,
             "report_title":filing.get("report_nm","") if filing else "",
             "corp_name":g.get("corp_name",""),"corp_cls_reported":g.get("corp_cls",""),
             "outside_directors":outside,"total_directors":total,"g_value":outside/total if not reason else np.nan,
             "source_hash":p["gm"]["source_hash"],"filing_list_hash":filing.get("filing_list_hash","") if filing else "",
             "source_url":"https://dart.fss.or.kr/dsaf001/main.do?rcpNo="+receipt,
             "is_amendment":"정정" in (filing.get("report_nm","") if filing else ""),
             "status":"PASS" if not reason else "PIT_UNVERIFIED","reason":reason}
        grows.append(gr)
        be=bmap.get((p["stock_code"],p["bsns_year"]),np.nan)
        candidates=[]
        for account in p["financial"].get("list",[]):
            name=account.get("account_nm","").replace(" ","")
            if account.get("sj_div")!="BS" or name not in ["자본총계","지배기업소유주지분","지배기업의소유주에게귀속되는자본","지배주주지분","지배기업소유주지분합계"]:continue
            value=number(account.get("thstrm_amount"))/1e8
            currency=account.get("currency")
            if currency!="KRW":continue
            error=abs(value-be)
            if np.isfinite(be) and error<=max(.011,abs(be)*.00001):
                fr=str(account.get("rcept_no",""));ff=by_receipt.get(fr)
                if ff and str(ff["corp_code"])==p["corp_code"]:
                    candidates.append({"be_eok":be,"dart_be_eok":value,"account_nm":account["account_nm"],"fs_div":account.get("fs_div",""),
                         "financial_rcept_no":fr,"financial_filing_date":pd.to_datetime(ff["rcept_dt"],format="%Y%m%d"),
                         "financial_source_hash":p["fm"]["source_hash"],"financial_filing_list_hash":ff["filing_list_hash"],"match_abs_error_eok":error})
        if candidates:
            candidates.sort(key=lambda r:(r["match_abs_error_eok"],r["fs_div"]!="CFS",r["account_nm"]))
            chosen=candidates[0]
            accounts=[a for a in p["financial"].get("list",[]) if a.get("fs_div")==chosen["fs_div"] and str(a.get("rcept_no"))==chosen["financial_rcept_no"] and a.get("currency")=="KRW"]
            from .financial_controls import extract_controls
            controls=extract_controls(accounts,chosen["account_nm"],chosen["dart_be_eok"]*1e8)
            frows.append({**key,**chosen,"be_year":p["bsns_year"],"status":"VALUE_AND_DATE_MATCHED",
                    **controls})
        audit.append({**key,"rcept_no":receipt,"status":gr["status"],"reason":reason,
                      "financial_value_verified":bool(candidates),"g_source_hash":p["gm"]["source_hash"],"financial_source_hash":p["fm"]["source_hash"]})
    gd=pd.DataFrame(grows);fd=pd.DataFrame(frows);ad=pd.DataFrame(audit)
    return gd,fd,ad


def fetch(stage):
    config=verify_registration();client=Client(ROOT/"data")
    mapping=load_mapping()
    universe=pd.read_csv(ROOT/"data/private/universe.csv",dtype={"stock_code":str})
    if stage=="smoke":
        stocks=config["smoke_codes"];years=config["smoke_years"]
        prior=pd.read_parquet(ROOT/"data/private/annual_filings.parquet")
        extra=[]
        for stock in stocks:
            if stock in mapping:
                rows,hashes=client.filings(mapping[stock])
                extra.extend({**r,"filing_list_hash":"|".join(hashes)} for r in rows)
        if extra:
            pd.concat([prior,pd.DataFrame(extra)]).drop_duplicates(["corp_code","rcept_no"]).to_parquet(ROOT/"data/private/annual_filings.parquet",index=False)
    else:
        stocks=list(universe.sort_values("acquisition_order").stock_code)
        if stage!="all":stocks=stocks[:int(stage)]
        years=config["years"]
    ur=universe.set_index("stock_code");jobs=[]
    for stock in stocks:
        if stock not in mapping:continue
        for year in years:
            if pd.Timestamp(ur.loc[stock,"last_month"])<pd.Timestamp(year=year+1,month=1,day=1):continue
            if pd.Timestamp(ur.loc[stock,"first_month"])>pd.Timestamp(year=year+1,month=12,day=1):continue
            jobs.append((stock,mapping[stock],year))
    if stage=="smoke":
        for stock in stocks[:2]:
            if stock in mapping:
                jobs.extend([(stock,mapping[stock],2024,"11012"),(stock,mapping[stock],2025,"11013")])
    pairs=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(fetch_pair,client,*job) for job in jobs]
        for i,future in enumerate(as_completed(pending),1):
            p=future.result();pairs.append(p)
            # Preserve every pair incrementally, including failures; cache supports reruns.
            name=hashlib.sha256(json.dumps({k:p[k] for k in ("stock_code","corp_code","bsns_year","reprt_code")},sort_keys=True).encode()).hexdigest()
            (ROOT/"data/private"/(name+".pair.json")).write_text(json.dumps(p,ensure_ascii=False))
            if i%50==0 or i==len(jobs):print(json.dumps({"phase":"fetch","stage":stage,"completed":i,"jobs":len(jobs),"new_requests":client.count}),flush=True)
    all_pairs=[json.loads(f.read_text()) for f in sorted((ROOT/"data/private").glob("*.pair.json"))]
    g,f,a=assemble(all_pairs)
    g.to_parquet(ROOT/"data/private/governance_observations.parquet",index=False)
    f.to_parquet(ROOT/"data/private/verified_financials.parquet",index=False)
    a.to_csv(ROOT/"data/source_audit.csv",index=False)
    quality={"stage":stage,"G_rows":len(g),"G_PIT_pass":int(g.status.eq("PASS").sum()) if len(g) else 0,
        "verified_financial_rows":len(f),"companies":g.stock_code.nunique() if len(g) else 0,"new_requests":client.count}
    (ROOT/"output/acquisition_status.json").write_text(json.dumps(quality,indent=2))
    if stage=="smoke":
        (ROOT/"output/smoke_test.json").write_text(json.dumps(quality,indent=2))
        g.to_csv(ROOT/"data/smoke_evidence.csv",index=False)
    print(json.dumps(quality),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("phase",choices=["baseline","mapping-smoke","mapping","smoke","fetch"]);ap.add_argument("--stage",default="50")
    args=ap.parse_args()
    if args.phase=="baseline":baseline()
    elif args.phase.startswith("mapping"):mapping(args.phase=="mapping-smoke")
    elif args.phase=="smoke":fetch("smoke")
    else:fetch(args.stage)

if __name__=="__main__":main()
