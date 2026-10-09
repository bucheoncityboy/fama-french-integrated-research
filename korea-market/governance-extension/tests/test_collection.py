import io
import pandas as pd
import pytest

from src.collect_governance import DartClient, DartError
from src.corpcode_map import validate_map


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def test_credentials_not_in_manifest_or_transport_error(tmp_path):
    secret = "TEST_CREDENTIAL_NOT_REAL"
    def failed(url, timeout): raise RuntimeError(url)
    client = DartClient(tmp_path, key=secret, opener=failed)
    with pytest.raises(DartError) as e: client.request("list.json", {"corp_code": "00126380"})
    assert secret not in str(e.value)
    client.opener = lambda *a, **kw: Response(b'{"status":"000","list":[]}')
    client.request("list.json", {"corp_code": "00126380"})
    assert secret not in str(client.manifest)
    assert "crtfc_key" not in str(client.manifest)


def test_api_rate_limit_stops_without_unbounded_retry(tmp_path):
    client = DartClient(tmp_path, key="TEST", opener=lambda *a, **kw: Response(b'{"status":"020"}'))
    with pytest.raises(DartError, match="020"): client.request("list.json", {})
    assert client.count == 1


def test_ambiguous_ticker_mapping_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        validate_map(pd.DataFrame({"stock_code": ["005930"]*2, "corp_code": ["00126380", "00126381"]}))


def test_budget_fails_before_network(tmp_path):
    client = DartClient(tmp_path, key="TEST", max_requests=0, opener=lambda *a, **kw: pytest.fail("network must not be called"))
    with pytest.raises(DartError, match="request_budget"): client.request("list.json", {})
