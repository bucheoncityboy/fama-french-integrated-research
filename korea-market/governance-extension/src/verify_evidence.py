"""Re-open immutable API evidence and reject value/date/hash drift."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .acquire import number
from .financial_controls import extract_controls


def verify_g_frame(d,raw):
    @lru_cache(maxsize=2048)
    def read(digest):
        p=Path(raw)/(digest+".json")
        b=p.read_bytes()
        if hashlib.sha256(b).hexdigest()!=digest:raise ValueError("raw_hash_mismatch")
        return json.loads(b)
    valid=[]
    for row in d.itertuples(index=False):
        ok=row.status=="PASS"
        if ok:
            try:
                if not (np.isfinite(row.total_directors) and row.total_directors>0 and 0<=row.outside_directors<=row.total_directors and
                    row.total_directors%1==0 and row.outside_directors%1==0 and 0<=row.g_value<=1 and
                    np.isclose(row.g_value,row.outside_directors/row.total_directors)):
                    raise ValueError("invalid_normalized_ratio")
                source=read(row.source_hash)
                hits=[r for r in source.get("list",[]) if str(r.get("rcept_no"))==str(row.rcept_no) and str(r.get("corp_code"))==str(row.corp_code)]
                if len(hits)!=1:raise ValueError("ambiguous_receipt")
                r=hits[0]
                if number(r["drctr_co"])!=row.total_directors or number(r["otcmp_drctr_co"])!=row.outside_directors or pd.Timestamp(r["stlm_dt"])!=row.period_end:
                    raise ValueError("G_value_mismatch")
                matches=[]
                for digest in row.filing_list_hash.split("|"):
                    matches.extend(r for r in read(digest).get("list",[]) if str(r.get("rcept_no"))==str(row.rcept_no) and str(r.get("corp_code"))==str(row.corp_code))
                if len(matches)!=1 or pd.to_datetime(matches[0]["rcept_dt"],format="%Y%m%d")!=row.filing_date:
                    raise ValueError("publication_evidence_mismatch")
            except (OSError,ValueError,KeyError):ok=False
        valid.append(ok)
    out=d.copy();out["raw_verified"]=valid
    out.loc[~out.raw_verified,"status"]="PIT_UNVERIFIED"
    return out


def verify_financial_frame(d,raw):
    valid=[]
    for row in d.itertuples(index=False):
        ok=False
        try:
            p=Path(raw)/(row.financial_source_hash+".json");body=p.read_bytes()
            if hashlib.sha256(body).hexdigest()!=row.financial_source_hash:raise ValueError("financial_hash_mismatch")
            data=json.loads(body)
            hits=[r for r in data.get("list",[]) if str(r.get("rcept_no"))==str(row.financial_rcept_no) and
                   str(r.get("corp_code"))==str(row.corp_code) and r.get("fs_div")==row.fs_div and r.get("account_nm")==row.account_nm]
            ok=len(hits)==1 and hits[0].get("currency")=="KRW" and int(hits[0].get("bsns_year",-1))==int(row.be_year) and abs(number(hits[0]["thstrm_amount"])/1e8-row.be_eok)<=max(.011,abs(row.be_eok)*.00001)
            if ok:
                accounts=[r for r in data.get("list",[]) if str(r.get("rcept_no"))==str(row.financial_rcept_no) and
                          str(r.get("corp_code"))==str(row.corp_code) and r.get("fs_div")==row.fs_div and r.get("currency")=="KRW"]
                expected=extract_controls(accounts,row.account_nm,getattr(row,"dart_be_eok",row.be_eok)*1e8)
                for name,value in expected.items():
                    actual=getattr(row,name,np.nan)
                    ok=ok and (pd.isna(actual) and pd.isna(value) or np.isclose(actual,value,equal_nan=True))
            matches=[]
            for digest in row.financial_filing_list_hash.split("|"):
                file=Path(raw)/(digest+".json");b=file.read_bytes()
                if hashlib.sha256(b).hexdigest()!=digest:raise ValueError("filing_hash_mismatch")
                matches.extend(r for r in json.loads(b).get("list",[]) if str(r.get("rcept_no"))==str(row.financial_rcept_no) and str(r.get("corp_code"))==str(row.corp_code))
            ok=ok and len(matches)==1 and pd.to_datetime(matches[0]["rcept_dt"],format="%Y%m%d")==row.financial_filing_date
        except (OSError,ValueError,KeyError):ok=False
        valid.append(ok)
    out=d.copy();out["raw_verified"]=valid
    return out.loc[out.raw_verified].copy()
