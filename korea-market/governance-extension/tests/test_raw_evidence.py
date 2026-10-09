import hashlib
import json
import pandas as pd

from src.build_pit_panel import verify_raw_evidence
from test_pit import observation


def fixture(tmp_path):
    d = observation()
    summary = {"status": "000", "list": [{"corp_code": d["corp_code"], "rcept_no": d["rcept_no"],
        "drctr_co": "10", "otcmp_drctr_co": "5", "stlm_dt": "2023-12-31"}]}
    filings = {"status": "000", "list": [{"corp_code": d["corp_code"], "rcept_no": d["rcept_no"], "rcept_dt": "20240315"}]}
    for field, obj in [("source_hash", summary), ("filing_list_hash", filings)]:
        body = json.dumps(obj).encode()
        d[field] = hashlib.sha256(body).hexdigest()
        (tmp_path/(d[field]+".json")).write_bytes(body)
    return d


def test_raw_source_hash_and_fields_are_checked(tmp_path):
    d = fixture(tmp_path)
    assert verify_raw_evidence(pd.DataFrame([d]), tmp_path).pit_valid.iloc[0]


def test_hash_without_actual_response_is_not_verified(tmp_path):
    assert not verify_raw_evidence(pd.DataFrame([observation()]), tmp_path).pit_valid.iloc[0]


def test_normalized_value_must_match_actual_response(tmp_path):
    d = fixture(tmp_path); d["g_value"] = .6; d["outside_directors"] = 6
    assert not verify_raw_evidence(pd.DataFrame([d]), tmp_path).pit_valid.iloc[0]
