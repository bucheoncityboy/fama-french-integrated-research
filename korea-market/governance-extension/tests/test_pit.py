"""Synthetic fixtures validate timing logic only; they are never research data."""
import pandas as pd
import pytest

from src.build_pit_panel import build_pit_panel, validate_observations
from src.collect_governance import usable_month_end, normalize_observation


def observation(filing="2024-03-15", receipt="20240315000001", period="2023-12-31", ratio=.5):
    return {"stock_code": "005930", "corp_code": "00126380", "rcept_no": receipt,
            "filing_date": filing, "period_end": period, "available_at": usable_month_end(filing),
            "g_value": ratio, "outside_directors": ratio*10, "total_directors": 10,
            "qa_status": "PASS", "source_url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo="+receipt,
            "source_hash": "a"*64, "filing_list_hash": "b"*64, "unit": "ratio_0_1"}


def panel():
    return pd.DataFrame({"code": ["A005930"]*6, "date": pd.date_range("2024-02-01", periods=6, freq="MS"),
                         "me": [100.]*6, "bm": [1.]*6, "return": [.01]*6})


@pytest.mark.parametrize("day,expected", [("2024-03-15", "2024-03-31"), ("2024-03-31", "2024-04-30"),
                                            ("2024-02-29", "2024-03-31")])
def test_month_end_filing_not_used_same_month(day, expected):
    assert usable_month_end(day) == pd.Timestamp(expected)


def test_future_governance_never_used():
    p = build_pit_panel(panel(), pd.DataFrame([observation()]))
    assert p.loc[p.date <= "2024-03-01", "g_value"].isna().all()
    assert p.loc[p.date == "2024-04-01", "g_value"].iloc[0] == .5
    assert (p.dropna(subset=["g_value"]).available_at <= p.dropna(subset=["g_value"]).formation_date).all()


def test_period_end_cannot_be_used_as_publication_date():
    d = observation()
    d["available_at"] = pd.Timestamp("2023-12-31")
    assert not validate_observations(pd.DataFrame([d])).pit_valid.iloc[0]


def test_bad_source_url_excluded():
    d = observation(); d["source_url"] = "https://example.com/unverified"
    assert not validate_observations(pd.DataFrame([d])).pit_valid.iloc[0]


def test_zero_is_an_observed_ratio_not_missing():
    assert validate_observations(pd.DataFrame([observation(ratio=0)])).pit_valid.iloc[0]


def test_missing_is_never_zero_filled():
    d = observation(); d["g_value"] = None
    p = build_pit_panel(panel(), pd.DataFrame([d]))
    assert p.g_value.isna().all()


def test_old_period_amendment_does_not_overwrite_recent_board():
    observations = [observation("2024-03-15", ratio=.6),
                    observation("2024-05-15", receipt="20240515000001", period="2022-12-31", ratio=.2)]
    p = build_pit_panel(panel(), pd.DataFrame(observations))
    assert p.loc[p.date == "2024-06-01", "g_value"].iloc[0] == .6


def test_duplicate_receipts_fail():
    with pytest.raises(ValueError, match="duplicate governance"):
        validate_observations(pd.DataFrame([observation(), observation()]))


def test_stale_governance_expires():
    p = build_pit_panel(panel(), pd.DataFrame([observation("2020-03-15", receipt="20200315000001", period="2019-12-31")]))
    assert p.g_value.isna().all()


def test_actual_filing_list_match_required():
    row = {"corp_code": "00126380", "rcept_no": "20240315000001", "drctr_co": "10", "otcmp_drctr_co": "5", "stlm_dt": "2023-12-31"}
    result = normalize_observation(row, {"stock_code": "005930", "corp_code": "00126380"}, [], "a"*64, "b"*64, "now")
    assert result["qa_status"] == "EXCLUDED"
    assert pd.isna(result["g_value"])
