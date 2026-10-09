"""Actual price-panel arithmetic and exclusion diagnostics; no corporate-action guesses."""
import hashlib
import json
import numpy as np
import pandas as pd
from .acquire import ROOT,SOURCE


def audit():
    sr=pd.read_parquet(SOURCE/"data/stock_returns.parquet").sort_values(["code","date"])
    raw=pd.read_parquet(SOURCE/"data/market_data_long.parquet")
    prices=raw.loc[raw.item_code.eq("S410000700"),["code","date","value"]].rename(columns={"value":"price"}).sort_values(["code","date"])
    prices["raw_price_return"]=prices.groupby("code").price.pct_change(fill_method=None)
    prices["previous_price_date"]=prices.groupby("code").date.shift()
    prices["contiguous_price"]=prices.previous_price_date.eq(prices.date-pd.offsets.MonthBegin(1))
    d=sr.merge(prices,on=["code","date"],how="left",validate="one_to_one")
    valid=d["return"].notna()&d.raw_price_return.notna()
    diff=(d.loc[valid,"return"]-d.loc[valid,"raw_price_return"]).abs()
    filtered=d.raw_price_return.abs().gt(1)&d["return"].isna()
    d["extreme_return_filtered"]=filtered
    d["return_missing"]=d["return"].isna()
    d["gap_returns"]=d["return"].notna()&~d.contiguous_price.fillna(False).astype(bool)
    d.groupby("date").agg(N=("code","size"),missing_returns=("return_missing","sum"),
             extreme_return_filtered=("extreme_return_filtered","sum"),noncontiguous_observed_returns=("gap_returns","sum")).to_csv(ROOT/"output/return_filter_audit.csv")
    d.loc[filtered|d.gap_returns].to_parquet(ROOT/"data/private/return_events.parquet",index=False)
    result={"observations":len(d),"valid_compared":int(valid.sum()),"max_arithmetic_error":float(diff.max()),
         "greater_100pct_returns_removed":int(filtered.sum()),"noncontiguous_price_return_rows":int(d.gap_returns.sum()),
         "rows_without_matching_raw_price":int(d.price.isna().sum()),
         "unit_check":"decimal_price_pct_change_verified" if not len(diff) or diff.max()<1e-10 else "FAIL",
         "adjusted_price_metadata":"UNVERIFIED","cash_dividend_total_return":"UNVERIFIED",
         "terminal_delisting_return":"UNVERIFIED","corporate_action_attribution_of_filtered_returns":"UNVERIFIED",
         "source_hash":hashlib.sha256((SOURCE/"data/market_data_long.parquet").read_bytes()).hexdigest()}
    (ROOT/"output/return_data_quality.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=="__main__":audit()
