import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from .audit_existing_ff3 import audit_source
from .common import snapshot, integrity, stock_code, write_json
from .reconstruct_ff3 import write_benchmarks, build_formation
from .build_pit_panel import build_pit_panel, verify_raw_evidence
from .portfolio_sorts import sort_portfolios, strategies
from .factor_models import fit_ff3
from .robustness import evaluate
from .generate_report import generate


def market_timing_panel(panel, book, financial_filings):
    """Build convention-aligned BM, then require disclosed filing evidence for each formation."""
    f = build_formation(panel, book, "spec_aligned")
    e = pd.read_csv(financial_filings, dtype={"stock_code": str}, parse_dates=["financial_filing_date"])
    required = {"stock_code", "formation_year", "financial_filing_date", "source_url", "rcept_no", "be_year"}
    if not required.issubset(e.columns):
        raise ValueError("financial_filings requires stock_code, formation_year, financial_filing_date, source_url, rcept_no, be_year")
    e.stock_code = e.stock_code.map(stock_code)
    if e.duplicated(["stock_code", "formation_year"]).any():
        raise ValueError("ambiguous financial filing evidence")
    expected_url = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + e.rcept_no.astype(str)
    if not e.source_url.eq(expected_url).all() or not e.be_year.eq(e.formation_year-1).all():
        raise ValueError("invalid financial source or fiscal year evidence")
    f["stock_code"] = f.code.map(stock_code)
    f = f.merge(e.drop(columns="be_year"), on=["stock_code", "formation_year"], validate="one_to_one")
    f["formation_date"] = pd.to_datetime(f.formation_year.astype(str) + "-06-30")
    f = f.loc[f.financial_filing_date.lt(f.formation_date)].copy()
    p = panel.drop(columns=["bm", "stock_code"], errors="ignore").copy()
    # June observation is used for July return; actual current June formation must be visible.
    p["formation_year"] = p.date.dt.year - p.date.dt.month.lt(6).astype(int)
    return p.merge(f[["code", "formation_year", "bm_used"]].rename(columns={"bm_used": "bm"}),
                   on=["code", "formation_year"], how="left", validate="many_to_one")


def run(source, output, config, governance=None, financial_filings=None, raw_dir=Path("data/raw")):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output==Path(__file__).resolve().parents[1]/"output":
        raise ValueError("legacy fallback runner cannot overwrite empirical outputs; use python -m src.cli")
    if output == source or source in output.parents:
        raise ValueError("output must be outside the read-only source checkout")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["source_commit"]:
        raise ValueError("source commit differs from pinned specification")
    output.mkdir(parents=True, exist_ok=True)
    # Prevent stale empirical outputs from surviving a fallback rerun.
    for name in ["sort_returns.csv", "factor_alpha.csv", "robustness.csv", "source_audit.csv", "data_coverage.csv"]:
        (output/name).unlink(missing_ok=True)
    before = snapshot(source)
    p = audit_source(source, output)
    factors, summary = write_benchmarks(source, p, output)
    blockers = []
    key_present = bool(__import__("os").environ.get("DART_API_KEY"))
    valid_g = None
    if governance and Path(governance).exists():
        obs = pd.read_csv(governance, dtype={"stock_code": str, "corp_code": str, "rcept_no": str})
        valid_g = verify_raw_evidence(obs, raw_dir)
        valid_g.to_csv(output/"source_audit.csv", index=False)
        if not valid_g.pit_valid.any():
            blockers.append("No valid historical governance observations")
    else:
        blockers.extend(["No historical governance evidence supplied", "DART_API_KEY absent" if not key_present else "Governance data not collected"])
        pd.DataFrame([{"source": "OpenDART", "endpoint": "outcmpnyDrctrNdChangeSttus.json",
                       "status": "not_collected", "reason": "No verified governance observations",
                       "api_key_present": key_present, "unauthenticated_probe": "HTML, not validated JSON"}]).to_csv(output/"source_audit.csv", index=False)
    empirical = valid_g is not None and valid_g.pit_valid.any() and financial_filings is not None
    alpha_estimated = False
    if empirical:
        book = pd.read_parquet(source/"korea-market/data/book_equity_monthly.parquet")
        market = market_timing_panel(p, book, financial_filings)
        pit = build_pit_panel(market, valid_g, max_age_days=config["max_governance_age_days"])
        # Only original sample months, not pre-benchmark observations.
        pit = pit.loc[pit.date.between(config["benchmark_start"], config["benchmark_end"])]
        sorts, holdings, coverage = sort_portfolios(pit, min_cell=config["minimum_stocks_per_cell"])
        sorts.to_csv(output/"sort_returns.csv", index=False)
        coverage.to_csv(output/"data_coverage.csv", index=False)
        s = strategies(sorts)
        results = []
        for col, ls in [("spread", True), ("HighBM_HighG", False)]:
            results.append({"strategy": col, **fit_ff3(s[["date", col]].rename(columns={col: "return"}), factors, long_short=ls,
                            min_months=config["minimum_regression_months"], hac_lags=config["hac_lags"])})
        pd.DataFrame(results).to_csv(output/"factor_alpha.csv", index=False)
        pd.DataFrame(evaluate(pit, factors, min_cell=config["minimum_stocks_per_cell"], min_months=config["minimum_regression_months"])).to_csv(output/"robustness.csv", index=False)
        alpha_estimated = results[0]["status"] == "estimated"
        if not alpha_estimated:
            blockers.append(results[0]["not_estimated_reason"])
    else:
        if valid_g is not None and valid_g.pit_valid.any():
            blockers.append("Verified financial filing dates required before governance backtest")
        pp = p.loc[p.date.between(config["benchmark_start"], config["benchmark_end"])].copy()
        pp["year"] = pp.date.dt.year
        cover = pp.groupby("year").agg(stocks_in_original_panel=("code", "nunique"), stock_months=("code", "size"), months=("date", "nunique")).reset_index()
        counts = {} if valid_g is None else valid_g.loc[valid_g.pit_valid].groupby(valid_g.filing_date.dt.year).size().to_dict()
        cover["verified_governance_observations"] = cover.year.map(counts).fillna(0).astype(int)
        cover["governance_coverage_status"] = "not_joined_no_verified_joint_PIT_panel"
        cover.to_csv(output/"data_coverage.csv", index=False)
        pd.DataFrame(columns=["date", "cell", "return", "n_formed", "n_missing_return", "status"]).to_csv(output/"sort_returns.csv", index=False)
        pd.DataFrame([{"strategy": "HighBM_HighG_minus_HighBM_LowG", "status": "not_estimated", "alpha_monthly_pct": np.nan,
                       "n_months": 0, "not_estimated_reason": "historical_joint_PIT_data_unavailable"}]).to_csv(output/"factor_alpha.csv", index=False)
        pd.DataFrame([{"specification": x, "status": "not_estimated", "not_estimated_reason": "historical_joint_PIT_data_unavailable"}
                      for x in ["primary_vw_median", "equal_weight", "cutoff_30", "cutoff_70", "exclude_top5", "industry_control", "profitability_control", "turnover_cost"]]).to_csv(output/"robustness.csv", index=False)
    hashes = integrity(before, snapshot(source))
    hashes.to_csv(output/"source_integrity.csv", index=False)
    if not hashes.unchanged.all():
        raise ValueError("source integrity changed")
    legacy = summary.loc[summary.specification.eq("legacy_compatible")].set_index("factor")
    reproduced = all(round(legacy.loc[c, "mean_monthly_pct"], 2) == n for c, n in [("Mkt-RF", .71), ("SMB", -.67), ("HML", .83)])
    status = {"status": "limited", "research_route": "conditional_exploratory" if empirical else "fallback_c",
              "run_date": datetime.now(timezone.utc).isoformat(), "spec_version": config["spec_version"],
              "source_commit": commit, "sample_start": "2000-07", "sample_end": "2026-05", "number_of_stocks": int(p.code.nunique()),
              "number_of_months": len(factors), "governance_proxy": config["primary_governance_proxy"],
              "data_available_at_verified": bool(empirical), "ff3_reproduced": reproduced,
              "ff3_reproduction_scope": "README rounded factor premia and OLS t; GRS not rerun",
              "primary_alpha_estimated": bool(alpha_estimated), "source_files_unchanged": bool(hashes.unchanged.all()),
              "number_of_source_files_hashed": len(hashes), "blockers": blockers,
              "remaining_limitations": ["dividend inclusion not verified", "listing/delisting and original universe not verified",
                                       "industry/profitability data unavailable", "governance causality not identified"],
              "next_required_input": ["historical governance observations with receipt/date/raw source hashes",
                                      "official corp_code/stock_code mapping", "financial filing dates for fiscal t-1 BM"]}
    write_json(output/"RESULT_STATUS.json", status)
    generate(output, status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--governance", type=Path)
    parser.add_argument("--financial-filings", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    a = parser.parse_args()
    config = yaml.safe_load(a.config.read_text())
    status = run(a.source, a.output, config, a.governance, a.financial_filings, a.raw_dir)
    print("status=" + status["status"] + "; route=" + status["research_route"] + "; ff3_reproduced=" + str(status["ff3_reproduced"]))


if __name__ == "__main__":
    main()
