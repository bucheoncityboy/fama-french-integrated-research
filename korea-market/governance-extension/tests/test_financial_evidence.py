"""SYNTHETIC_TEST_ONLY: date, currency and hundredfold-unit-error rejection."""
import hashlib,json
import pandas as pd
from src.verify_evidence import verify_financial_frame


def test_financial_evidence_rejects_100fold_unit_error_and_unknown_currency(tmp_path):
    account={"rcept_no":"20240320000001","corp_code":"00000001","fs_div":"CFS",
         "account_nm":"자본총계","thstrm_amount":"100,000,000","currency":"KRW","bsns_year":"2023"}
    response=json.dumps({"status":"000","list":[account]}).encode()
    digest=hashlib.sha256(response).hexdigest();(tmp_path/(digest+".json")).write_bytes(response)
    filing=json.dumps({"list":[{"rcept_no":"20240320000001","corp_code":"00000001","rcept_dt":"20240320"}]}).encode()
    fh=hashlib.sha256(filing).hexdigest();(tmp_path/(fh+".json")).write_bytes(filing)
    row={"financial_source_hash":digest,"financial_rcept_no":"20240320000001","corp_code":"00000001","fs_div":"CFS",
          "account_nm":"자본총계","be_eok":1.,"be_year":2023,"financial_filing_list_hash":fh,"financial_filing_date":pd.Timestamp("2024-03-20")}
    d=pd.DataFrame([row]);assert len(verify_financial_frame(d,tmp_path))==1
    d["be_eok"]=100.;assert verify_financial_frame(d,tmp_path).empty
    d["be_eok"]=1.;d["financial_filing_date"]=pd.Timestamp("2024-03-19")
    assert verify_financial_frame(d,tmp_path).empty
