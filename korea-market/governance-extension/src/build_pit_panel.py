import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import lag_market_values, stock_code, sha256
from .collect_governance import usable_month_end


def validate_observations(obs):
    required = {"stock_code", "corp_code", "rcept_no", "filing_date", "period_end", "available_at",
                "g_value", "outside_directors", "total_directors", "qa_status", "source_url", "source_hash", "filing_list_hash", "unit"}
    if not required.issubset(obs.columns):
        raise ValueError("missing governance evidence fields: " + ",".join(sorted(required - set(obs.columns))))
    d = obs.copy()
    for c in ["filing_date", "period_end", "available_at"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["stock_code"] = d.stock_code.map(stock_code)
    for c in ["g_value", "outside_directors", "total_directors"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    expected_url = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + d.rcept_no.astype(str)
    valid = (d.qa_status.eq("PASS") & d.corp_code.astype(str).str.fullmatch(r"\d{8}") &
             d.rcept_no.astype(str).str.fullmatch(r"\d{14}") & d.source_url.eq(expected_url) &
             d.source_hash.astype(str).str.fullmatch(r"[0-9a-f]{64}") &
             d.filing_list_hash.astype(str).str.fullmatch(r"[0-9a-f]{64}(\|[0-9a-f]{64})*") &
             d.g_value.between(0, 1) & d.total_directors.gt(0) & d.outside_directors.ge(0) &
             d.outside_directors.le(d.total_directors) & d.unit.eq("ratio_0_1") &
             d.outside_directors.mod(1).eq(0) & d.total_directors.mod(1).eq(0) &
             np.isclose(d.g_value, d.outside_directors / d.total_directors, rtol=1e-8) &
             d.period_end.le(d.filing_date) & d.available_at.notna())
    earliest = d.filing_date.map(lambda v: usable_month_end(v) if pd.notna(v) else pd.NaT)
    valid &= d.available_at.ge(earliest)
    d["pit_valid"] = valid.fillna(False)
    if d.duplicated(["stock_code", "rcept_no"]).any():
        raise ValueError("duplicate governance receipt for company")
    return d


def verify_raw_evidence(observations, raw_dir):
    """Hashes must resolve to actual cached official responses, not just look like hashes."""
    d = validate_observations(observations)
    raw_dir = Path(raw_dir)
    checked = []
    for row in d.to_dict("records"):
        reason = ""
        try:
            source = raw_dir / (str(row["source_hash"]) + ".json")
            if not source.exists() or sha256(source) != row["source_hash"]:
                raise ValueError("missing_or_mismatched_summary_hash")
            response = json.loads(source.read_text())
            matches = [r for r in response.get("list", []) if str(r.get("rcept_no")) == str(row["rcept_no"])
                       and str(r.get("corp_code")) == str(row["corp_code"])]
            if len(matches) != 1:
                raise ValueError("summary_receipt_not_unique")
            from .collect_governance import count_number
            original = matches[0]
            if (count_number(original.get("drctr_co")) != row["total_directors"] or
                count_number(original.get("otcmp_drctr_co")) != row["outside_directors"] or
                pd.Timestamp(original.get("stlm_dt")) != row["period_end"]):
                raise ValueError("normalized_value_does_not_match_raw")
            filings = []
            for digest in str(row["filing_list_hash"]).split("|"):
                file = raw_dir/(digest+".json")
                if not file.exists() or sha256(file) != digest:
                    raise ValueError("missing_or_mismatched_filing_list_hash")
                filings.extend(json.loads(file.read_text()).get("list", []))
            matched = [r for r in filings if str(r.get("rcept_no")) == str(row["rcept_no"])
                       and str(r.get("corp_code")) == str(row["corp_code"])]
            if len(matched) != 1 or pd.to_datetime(matched[0]["rcept_dt"], format="%Y%m%d") != row["filing_date"]:
                raise ValueError("filing_date_does_not_match_raw_list")
        except (ValueError, KeyError, OSError):
            reason = "raw_evidence_failed_validation"
        checked.append(not reason and bool(row["pit_valid"]))
    d["raw_verified"] = checked
    d["pit_valid"] &= d.raw_verified
    return d


def build_pit_panel(panel, observations, max_age_days=550):
    obs = validate_observations(observations)
    obs = obs.loc[obs.pit_valid].copy()
    p = lag_market_values(panel)
    p["stock_code"] = p.code.map(stock_code)
    outputs = []
    for code, g in p.groupby("stock_code", sort=False):
        candidates = obs.loc[obs.stock_code.eq(code)].sort_values(["available_at", "period_end", "filing_date", "rcept_no"])
        # An amendment to an older period must not replace newer-period board composition.
        events = []
        for at in candidates.available_at.drop_duplicates():
            chosen = candidates.loc[candidates.available_at.le(at)].sort_values(["period_end", "filing_date", "rcept_no"]).iloc[-1].copy()
            chosen["event_at"] = at
            events.append(chosen)
        if events:
            right = pd.DataFrame(events).sort_values("event_at")
            g = pd.merge_asof(g.sort_values("formation_date"), right[["event_at", "g_value", "available_at", "filing_date", "period_end", "rcept_no", "source_url"]],
                              left_on="formation_date", right_on="event_at", direction="backward")
            stale = (g.formation_date - g.filing_date).dt.days > max_age_days
            g.loc[stale, "g_value"] = np.nan
            if (g.available_at > g.formation_date).any():
                raise ValueError("look-ahead governance join")
        else:
            for field in ["g_value", "available_at", "filing_date", "period_end", "rcept_no", "source_url"]:
                g[field] = np.nan if field == "g_value" else pd.NA
        outputs.append(g)
    return pd.concat(outputs, ignore_index=True)
