"""SYNTHETIC_TEST_ONLY: fixtures verify algorithms, never provide empirical returns."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.disclosure_pit import available_at,select_observation,validate_dates
from src.form_portfolios import classify,form,longest_block
from src.acquire import verify_registration
from src.dart_client import Client,APIError
from src.run_ff3_alpha import mean_inference,metrics


def synthetic_market():
    n=80
    d=pd.DataFrame({"stock_code":[str(i).zfill(6) for i in range(n)],"date":pd.Timestamp("2024-01-01"),
       "joint_eligible":True,"age_days":100,"lag_me":np.arange(1,n+1,dtype=float),"bm_pit":np.arange(1,n+1,dtype=float),
       "g_value":np.tile(np.arange(1,11)/11,8),"return":np.tile([.01,.02,.03,.04],20),"corp_cls_reported":"Y",
       "rcept_no":"SYNTHETIC_TEST_ONLY","filing_date":pd.Timestamp("2023-03-20")})
    return d


def test_real_exchange_monthend_after_holiday():
    # 2024-03-29 was the last March session; weekend disclosure cannot trade at it.
    assert available_at("2024-03-29")==pd.Timestamp("2024-04-30")
    assert available_at("2024-03-28")==pd.Timestamp("2024-03-29")
    assert available_at("2024-03-31")==pd.Timestamp("2024-04-30")


def test_late_old_period_revision_does_not_overwrite_new_period():
    d=pd.DataFrame({"period_end":pd.to_datetime(["2022-12-31","2023-12-31","2022-12-31"]),
       "filing_date":pd.to_datetime(["2023-03-10","2024-03-10","2024-05-10"]),
       "available_at":pd.to_datetime(["2023-03-31","2024-03-29","2024-05-31"]),
       "rcept_no":["1","2","3"],"g_value":[.2,.5,.9],"status":"PASS"})
    assert select_observation(d,pd.Timestamp("2024-06-28")).g_value==.5


def test_future_G_mutation_does_not_change_past_formations():
    d=synthetic_market();future=d.copy();future["date"]=pd.Timestamp("2025-01-01")
    first,_=form(pd.concat([d,future]))
    future["g_value"]=1-future.g_value
    second,_=form(pd.concat([d,future]))
    pd.testing.assert_frame_equal(first.loc[first.date.eq("2024-01-01")].reset_index(drop=True),
                                  second.loc[second.date.eq("2024-01-01")].reset_index(drop=True))


def test_future_BM_mutation_does_not_change_past_formations():
    d=synthetic_market();future=d.copy();future["date"]=pd.Timestamp("2025-01-01")
    first,_=form(pd.concat([d,future]));future["bm_pit"]=future.bm_pit*1000
    second,_=form(pd.concat([d,future]))
    pd.testing.assert_frame_equal(first.loc[first.date.eq("2024-01-01")].reset_index(drop=True),
                                  second.loc[second.date.eq("2024-01-01")].reset_index(drop=True))


def test_PIT_rejects_future_financial_or_G_publication():
    d=pd.DataFrame({"filing_date":pd.to_datetime(["2024-02-01"]),"financial_filing_date":pd.to_datetime(["2023-03-10"]),
       "return_month_start":pd.to_datetime(["2024-01-01"]),"available_at":pd.to_datetime(["2024-02-29"]),
       "formation_month_end":pd.to_datetime(["2023-12-28"])})
    with pytest.raises(ValueError,match="PIT"):validate_dates(d)
    d.filing_date=pd.Timestamp("2023-03-10");d.available_at=pd.Timestamp("2023-03-31")
    d.financial_filing_date=pd.Timestamp("2024-02-01")
    with pytest.raises(ValueError,match="PIT"):validate_dates(d)


def test_weights_and_missing_terminal_return_guard():
    d=synthetic_market();r,h=form(d)
    assert np.allclose(r.weight_sum,1)
    d.loc[10,"return"]=np.nan
    bad,_=form(d)
    assert bad.n_missing_return.sum()==1
    assert bad["return"].isna().sum()==1
    assert bad.loc[bad.n_missing_return.gt(0),"n_formed"].iloc[0]>=5


def test_equal_G_values_are_never_arbitrarily_split():
    d=synthetic_market();d["g_value"]=.5
    groups=classify(d)
    assert groups.cell.str.endswith("LowG").all()
    r,_=form(d)
    assert r.loc[r.cell.str.endswith("HighG"),"n_formed"].eq(0).all()


def test_top5_exclusion_is_entire_population_not_covered_subset():
    d=synthetic_market();d.loc[d.index[-4:],"joint_eligible"]=False
    assigned=classify(d,exclude_top5=True)
    assert set(assigned.stock_code).isdisjoint(set(d.nlargest(5,"lag_me").stock_code))
    assert d.iloc[-6].stock_code in set(assigned.stock_code)


def test_contiguous_months_not_stockmonth_count_control_inference():
    block=longest_block(pd.to_datetime(["2020-01-01","2020-02-01","2021-01-01","2021-02-01"]))
    assert len(block)==2 and block[0]==pd.Timestamp("2020-01-01")
    assert mean_inference(np.ones(23)*.01)["status"]=="not_estimated"


def test_result_reexecution_identical_dataframe_hashes():
    d=synthetic_market();a,_=form(d);b,_=form(d.copy())
    assert a.to_csv(index=False)==b.to_csv(index=False)


class FakeResponse:
    status=200
    def __init__(self,body,ctype="application/json"):self.body=body;self.headers={"Content-Type":ctype}
    def read(self):return self.body
    def __enter__(self):return self
    def __exit__(self,*args):return False


@pytest.mark.parametrize("body,ctype",[(b"<html>maintenance</html>","text/html"),(b'{"status":"800"}',"application/json"),(b'{"status":"000","list":[]}',"text/html")])
def test_API_bad_response_not_silent_zero(tmp_path,monkeypatch,body,ctype):
    monkeypatch.setenv("DART_API_KEY","SYNTHETIC_TEST_ONLY")
    monkeypatch.setattr("src.dart_client.time.sleep",lambda _:None)
    c=Client(tmp_path,opener=lambda *a,**k:FakeResponse(body,ctype))
    with pytest.raises(APIError):c.request("test.json",{})


def test_transport_exception_does_not_leak_key(tmp_path,monkeypatch):
    secret="SYNTHETIC_TEST_ONLY_CREDENTIAL"
    monkeypatch.setenv("DART_API_KEY",secret)
    monkeypatch.setattr("src.dart_client.time.sleep",lambda _:None)
    def fail(*a,**k):raise RuntimeError("url contains "+secret)
    c=Client(tmp_path,opener=fail)
    with pytest.raises(APIError) as e:c.request("test.json",{})
    assert secret not in str(e.value)
    assert not list(tmp_path.rglob("*.json"))


def test_credential_echo_discarded_without_writing_raw(tmp_path,monkeypatch):
    monkeypatch.setenv("DART_API_KEY","SYNTHETIC_TEST_ONLY_CREDENTIAL")
    c=Client(tmp_path,opener=lambda *a,**k:FakeResponse(b'{"message":"SYNTHETIC_TEST_ONLY_CREDENTIAL"}'))
    with pytest.raises(APIError,match="discarded"):c.request("test.json",{})
    assert not list(tmp_path.rglob("*.json"))


def test_real_preregistration_lock_intact():
    assert verify_registration()["primary"]["g_cut"]==.5


def test_missing_months_never_get_compounded_as_if_complete():
    result=metrics(pd.Series([.1,np.nan,-.05]))
    assert result["missing_months"]==1
    assert np.isnan(result["compounded_spread_index_pct"])
    assert np.isnan(result["max_drawdown_index_pct"])
