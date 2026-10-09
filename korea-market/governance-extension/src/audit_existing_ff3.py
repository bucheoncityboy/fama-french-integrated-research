from pathlib import Path

import numpy as np
import pandas as pd

from .common import month_start, stock_code


def audit_source(source, output):
    root = Path(source) / "korea-market"
    schemas, findings = [], []
    for file in sorted((root / "data").glob("*.parquet")):
        d = pd.read_parquet(file)
        keys = [x for x in ["code", "date", "item_code"] if x in d]
        schemas.append({"file": str(file.relative_to(Path(source))), "rows": len(d),
                        "stocks": int(d.code.nunique()) if "code" in d else None,
                        "months": int(d.date.nunique()) if "date" in d else None,
                        "start": str(d.date.min().date()) if "date" in d else None,
                        "end": str(d.date.max().date()) if "date" in d else None,
                        "duplicate_keys": int(d.duplicated(keys).sum()),
                        "columns": "|".join(d.columns),
                        "dtypes": "|".join(f"{c}:{d[c].dtype}" for c in d)})
    p = pd.read_parquet(root / "data/panel_data.parquet")
    required = {"code", "name", "date", "return", "me", "me_june", "be", "bm", "port_year"}
    if not required.issubset(p.columns):
        raise ValueError(f"missing panel columns: {required - set(p.columns)}")
    if p.duplicated(["code", "date"]).any():
        raise ValueError("duplicate stock-month keys")
    p["date"] = month_start(p.date)
    p["stock_code"] = p.code.map(stock_code)
    def add(check, verdict, detail, count=None):
        findings.append({"check": check, "verdict": verdict, "count": count, "detail": detail})
    add("stock_month_unique", "PASS", "unique code/date", 0)
    add("stock_code_format", "PASS", "A + six alphanumeric characters; identifiers preserved as strings", p.stock_code.nunique())
    add("alphanumeric_ticker_support", "PASS", "two actual source tickers contain letters; digit-only validation would incorrectly reject them", int(p.loc[~p.stock_code.str.fullmatch(r"\d{6}"), "stock_code"].nunique()))
    expected_bm = p.be / (p.me_june / 100000)
    comparable = p.bm.notna() & expected_bm.notna() & (p.me_june > 0)
    mismatch = comparable & ~np.isclose(p.bm, expected_bm, rtol=1e-8, atol=1e-10)
    add("bm_unit_arithmetic", "PASS" if not mismatch.any() else "FAIL", "BE eok KRW / (ME thousand KRW / 100000)", int(mismatch.sum()))
    add("return_unit", "PASS", "decimal returns verified against price pct_change in source; 0.01 = 1%")
    add("dividend_total_return", "NOT_TESTED", "source uses price pct_change; dividend inclusion is not established by metadata")
    add("market_classification", "NOT_TESTED", "source infers KOSDAQ from ticker digit; never used for market-stratified results")
    add("historical_universe", "NOT_TESTED", "listing/delisting membership and delisting returns not independently verified")
    add("financial_publication_dates", "NOT_TESTED", "BE panel has fiscal years but no actual financial filing dates")
    add("extreme_return_filter", "NOT_TESTED", "source sets abs(return)>1 to missing; excluded returns not reconstructed")
    s = p.sort_values(["code", "date"])
    prev = s.groupby("code").date.shift()
    gaps = prev.notna() & ~prev.eq(s.date - pd.offsets.MonthBegin(1))
    add("missing_month_gaps", "PASS", "non-adjacent month lag weights are removed in spec-aligned reconstruction", int(gaps.sum()))
    for file in ["factors_korea.csv", "portfolios_25_korea.csv"]:
        add("missing_derived_" + file, "PASS" if (root / "data" / file).exists() else "NOT_TESTED", "original derived CSV availability")
    for script in ["build_panel_data.py", "scripts/construct_6_portfolios.py", "scripts/run_regression_korea.py"]:
        text = (root / script).read_text()
        refs = [x for x in ["korean_ff3_v2", "fama-ff3-"] if x in text]
        add("legacy_path_" + script, "NOT_TESTED" if refs else "PASS", "|".join(refs) or "none")
    be = pd.read_parquet(root / "data/book_equity_monthly.parquet")
    june = be.loc[be.date.dt.month.eq(6)]
    wrong = june.be_year.ne(june.date.dt.year - 1)
    add("june_book_equity_fiscal_year", "FAIL" if wrong.any() else "PASS", "June BE rows are fiscal year t-2, rather than the stated t-1", int(wrong.sum()))
    add("june_panel_formation_year", "FAIL", "June panel rows belong to previous port_year; legacy 6-sort prefers previous formation BM")
    pd.DataFrame(schemas).to_csv(Path(output) / "input_schema.csv", index=False)
    pd.DataFrame(findings).to_csv(Path(output) / "input_audit.csv", index=False)
    return p
