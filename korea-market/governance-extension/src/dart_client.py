"""Cached official API client. Credentials remain in process memory."""
import csv
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone
import urllib.parse
import urllib.request


class APIError(RuntimeError):
    pass


class Client:
    def __init__(self, root=Path("data"), budget=19000, interval=.15, opener=None):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.key = os.environ.get("DART_API_KEY") or os.environ.get("OPENDART_API_KEY")
        if not self.key:
            raise APIError("credential_not_configured")
        self.budget, self.count, self.interval = budget, 0, interval
        self.opener = opener or urllib.request.urlopen
        self.lock, self.next_time = threading.Lock(), 0.
        self.log = self.root / "api_log_sanitized.csv"

    def request(self, endpoint, params, refresh=False):
        if any("key" in k.lower() for k in params):
            raise APIError("credentials_must_not_be_query_metadata")
        safe = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
        request_id = hashlib.sha256(safe.encode()).hexdigest()
        meta_path = self.raw / (request_id + ".meta.json")
        if meta_path.exists() and not refresh:
            meta = json.loads(meta_path.read_text())
            path = self.raw / meta["raw_file"]
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == meta["source_hash"]:
                if meta["status"] in ("000", "013", "ZIP"):
                    body = path.read_bytes()
                    return (body if meta["status"] == "ZIP" else json.loads(body)), meta
        last_reason = "transport_failure"
        for attempt in range(3):
            with self.lock:
                if self.count >= self.budget:
                    raise APIError("request_budget_exhausted")
                self.count += 1
                wait = max(0., self.next_time-time.monotonic())
                self.next_time = max(self.next_time, time.monotonic()) + self.interval
            if wait:
                time.sleep(wait)
            url = "https://opendart.fss.or.kr/api/" + endpoint + "?" + urllib.parse.urlencode({**params, "crtfc_key": self.key})
            try:
                with self.opener(url, timeout=25) as response:
                    body = response.read()
                    http, ctype = response.status, response.headers.get("Content-Type", "")
            except Exception:
                last_reason = "transport_failure"
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise APIError(last_reason) from None
            digest = hashlib.sha256(body).hexdigest()
            if self.key.encode() in body:
                raise APIError("credential_echo_response_discarded_without_persistence")
            is_zip = endpoint.endswith(".xml") and body.startswith(b"PK")
            try:
                data = body if is_zip else json.loads(body)
                status = "ZIP" if is_zip else str(data.get("status", "missing_status"))
                if not is_zip and "html" in ctype.lower():
                    status = "HTML_JSON_UNVERIFIED"
            except (ValueError, UnicodeError):
                data, status = {}, "NON_JSON"
            raw_file = digest + (".zip" if is_zip else ".json")
            (self.raw / raw_file).write_bytes(body)
            meta = {"request_id": request_id, "endpoint": endpoint, "params": json.dumps(params,sort_keys=True),
                    "http_status": http, "content_type": ctype, "status": status,
                    "rows": len(data.get("list", [])) if isinstance(data,dict) else 0,
                    "source_hash": digest, "raw_file": raw_file,
                    "collected_at": datetime.now(timezone.utc).isoformat()}
            with self.lock:
                meta_path.write_text(json.dumps(meta,sort_keys=True))
                exists=self.log.exists()
                with self.log.open("a",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=list(meta))
                    if not exists:w.writeheader()
                    w.writerow(meta)
            if status in ("000","013","ZIP"):
                return data,meta
            last_reason="api_status_"+status
            if status in ("010","011","012","020","901"):
                with self.lock:self.budget=self.count
                raise APIError(last_reason)
            if attempt<2:time.sleep(2**attempt)
        raise APIError(last_reason)

    def filings(self, corp_code, start="20150101", end="20260531"):
        records, hashes = [], []
        page=1
        while True:
            payload, meta=self.request("list.json",{"corp_code":corp_code,"bgn_de":start,"end_de":end,
                 "pblntf_ty":"A","last_reprt_at":"N","page_count":100,"page_no":page})
            records.extend(payload.get("list",[]));hashes.append(meta["source_hash"])
            if page>=int(payload.get("total_page",1)):break
            page+=1
        return records,hashes
