"""Official OpenDART collector. API keys never appear in files, URLs or errors."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .common import write_json
from .corpcode_map import parse_corp_codes, validate_map

BASE = "https://opendart.fss.or.kr/api/"
ENDPOINT = "outcmpnyDrctrNdChangeSttus.json"
OBS_COLUMNS = ["stock_code", "corp_code", "corp_name", "rcept_no", "report_title",
               "filing_date", "period_end", "available_at", "outside_directors", "total_directors",
               "g_value", "unit", "source_field", "raw_value", "source_url", "source_hash",
               "filing_list_hash", "collected_at", "qa_status", "missing_reason", "is_amendment"]


class DartError(RuntimeError):
    pass


class DartClient:
    def __init__(self, raw_dir, key=None, max_requests=500, opener=None):
        self.key = key or os.environ.get("DART_API_KEY")
        if not self.key:
            raise DartError("DART_API_KEY is not configured")
        self.raw = Path(raw_dir)
        self.raw.mkdir(parents=True, exist_ok=True)
        self.limit, self.count = max_requests, 0
        self.opener = opener or urllib.request.urlopen
        self.manifest = []

    def request(self, endpoint, params):
        if self.count >= self.limit:
            raise DartError("request_budget_exhausted")
        self.count += 1
        safe_url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
        request_url = BASE + endpoint + "?" + urllib.parse.urlencode({**params, "crtfc_key": self.key})
        try:
            with self.opener(request_url, timeout=20) as response:
                body = response.read()
        except Exception:
            # urllib exceptions can contain the credential-bearing URL: suppress that text.
            raise DartError("transport_failure_or_service_unavailable") from None
        digest = hashlib.sha256(body).hexdigest()
        name = digest + (".zip" if endpoint.endswith(".xml") else ".json")
        (self.raw / name).write_bytes(body)
        self.manifest.append({"endpoint": endpoint, "source_url": safe_url, "source_hash": digest,
                              "local_file": name, "collected_at": datetime.now(timezone.utc).isoformat()})
        if endpoint.endswith(".xml"):
            if not body.startswith(b"PK"):
                raise DartError("corp_code_download_failed_or_maintenance")
            return body, digest
        try:
            result = json.loads(body)
        except ValueError:
            raise DartError("non_json_service_response") from None
        status = str(result.get("status"))
        if status == "013":
            return result, digest
        if status != "000":
            raise DartError("OpenDART_status_" + status)
        return result, digest

    def filings(self, corp_code, year, cutoff):
        records, hashes = [], []
        page = 1
        while True:
            result, digest = self.request("list.json", {"corp_code": corp_code,
                        "bgn_de": f"{year}0101", "end_de": min(f"{year}1231", cutoff.replace("-", "")),
                        "last_reprt_at": "N", "page_no": page, "page_count": 100})
            records.extend(result.get("list", []))
            hashes.append(digest)
            if page >= int(result.get("total_page", 1)):
                break
            page += 1
        return records, "|".join(hashes)


def count_number(value):
    if value is None or str(value).strip() in ["", "-", "NA"]:
        return np.nan
    text = str(value).strip().replace(",", "")
    try:
        number = float(text)
        return number if number.is_integer() and number >= 0 else np.nan
    except ValueError:
        return np.nan


def usable_month_end(filing_date):
    # Filing-time intraday availability unknown: move at least one day, then ceiling month end.
    day = pd.Timestamp(filing_date).normalize() + pd.Timedelta(days=1)
    return day + pd.offsets.MonthEnd(0)


def normalize_observation(row, mapping, filing_records, source_hash, list_hash, collected_at):
    receipt = str(row.get("rcept_no", ""))
    matches = [r for r in filing_records if str(r.get("rcept_no")) == receipt
               and str(r.get("corp_code")) == str(mapping["corp_code"])]
    outside, total = count_number(row.get("otcmp_drctr_co")), count_number(row.get("drctr_co"))
    period = pd.to_datetime(row.get("stlm_dt"), errors="coerce")
    reason = ""
    filing = matches[0] if len(matches) == 1 else {}
    date = pd.to_datetime(filing.get("rcept_dt"), format="%Y%m%d", errors="coerce")
    if len(matches) != 1:
        reason = "filing_date_not_uniquely_verified"
    elif str(row.get("corp_code")) != str(mapping["corp_code"]):
        reason = "corp_code_mismatch"
    elif not np.isfinite(total) or total <= 0 or not np.isfinite(outside) or outside > total:
        reason = "invalid_director_counts"
    elif pd.isna(period) or pd.isna(date) or period > date:
        reason = "invalid_settlement_or_publication_date"
    result = {"stock_code": mapping["stock_code"], "corp_code": mapping["corp_code"],
              "corp_name": row.get("corp_name", ""), "rcept_no": receipt,
              "report_title": filing.get("report_nm", ""), "filing_date": date,
              "period_end": period, "available_at": usable_month_end(date) if pd.notna(date) else pd.NaT,
              "outside_directors": outside, "total_directors": total,
              "g_value": outside/total if not reason else np.nan, "unit": "ratio_0_1",
              "source_field": "otcmp_drctr_co/drctr_co",
              "raw_value": json.dumps({"otcmp_drctr_co": row.get("otcmp_drctr_co"), "drctr_co": row.get("drctr_co")}, ensure_ascii=False),
              "source_url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + receipt,
              "source_hash": source_hash, "filing_list_hash": list_hash, "collected_at": collected_at,
              "qa_status": "PASS" if not reason else "EXCLUDED", "missing_reason": reason,
              "is_amendment": "정정" in filing.get("report_nm", "")}
    return result


def collect(client, mapping, years, cutoff):
    observations, failures = [], []
    for entry in mapping.to_dict("records"):
        for year in years:
            try:
                payload, digest = client.request(ENDPOINT, {"corp_code": entry["corp_code"], "bsns_year": year, "reprt_code": "11011"})
                raw_rows = payload.get("list", [])
                if not raw_rows:
                    failures.append({"stock_code": entry["stock_code"], "year": year, "reason": "no_summary_data"})
                    continue
                # Different rows with the same receipt are ambiguous; do not add their counts.
                unique = {json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in raw_rows}
                if len(unique) != 1:
                    failures.append({"stock_code": entry["stock_code"], "year": year, "reason": "ambiguous_summary_rows"})
                    continue
                row = next(iter(unique.values()))
                receipt_year = int(str(row["rcept_no"])[:4])
                if receipt_year > int(cutoff[:4]):
                    failures.append({"stock_code": entry["stock_code"], "year": year, "reason": "latest_summary_after_cutoff"})
                    continue
                filings, fh = client.filings(entry["corp_code"], receipt_year, cutoff)
                observations.append(normalize_observation(row, entry, filings, digest, fh, datetime.now(timezone.utc).isoformat()))
            except (DartError, ValueError, KeyError) as e:
                reason = str(e) if isinstance(e, DartError) else "invalid_response_fields"
                failures.append({"stock_code": entry["stock_code"], "year": year, "reason": reason})
                if any(reason.endswith(s) for s in ["010", "011", "012", "020", "800", "901"]) or reason == "request_budget_exhausted":
                    return pd.DataFrame(observations, columns=OBS_COLUMNS), pd.DataFrame(failures)
    return pd.DataFrame(observations, columns=OBS_COLUMNS), pd.DataFrame(failures, columns=["stock_code", "year", "reason"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corp-map", type=Path)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--cutoff", default="2026-10-09")
    parser.add_argument("--max-requests", type=int, default=500)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    client = DartClient(args.data_dir / "raw", max_requests=args.max_requests)
    try:
        if args.corp_map:
            mapping = validate_map(pd.read_csv(args.corp_map, dtype=str))
        else:
            raw, _ = client.request("corpCode.xml", {})
            mapping = parse_corp_codes(raw)
            mapping.to_csv(args.data_dir / "corp_map.csv", index=False)
        obs, errors = collect(client, mapping, range(args.start_year, args.end_year+1), args.cutoff)
        (args.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        obs.to_csv(args.data_dir / "processed/governance_observations.csv", index=False)
        errors.to_csv(args.data_dir / "collection_failures.csv", index=False)
    finally:
        pd.DataFrame(client.manifest).to_csv(args.data_dir / "manifest.csv", index=False)
    print("Collection complete. Validate observations before research.")


if __name__ == "__main__":
    main()
