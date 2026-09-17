"""
Build panel_data.parquet: merge stock_returns with book_equity,
compute June ME, BM ratios, and KOSDAQ flag.

BM = BE(t-1) / ME(June t)
- BE from Dec fiscal year t-1, available via book_equity_monthly at June t
- ME from stock_returns at June t
- Negative BE → BM = NaN
- is_kosdaq = True if code[1] != '0'
"""

import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────
sr = pd.read_parquet("korean_ff3_v2/data/stock_returns.parquet")
be = pd.read_parquet("korean_ff3_v2/data/book_equity_monthly.parquet")

print(f"stock_returns: {sr.shape}, unique codes: {sr['code'].nunique()}")
print(f"book_equity:   {be.shape}, unique codes: {be['code'].nunique()}")

# ── Step 1: Extract June ME from stock_returns ────────────────────────
june_me = sr[sr["date"].dt.month == 6].copy()
june_me = june_me[["code", "name", "date", "me"]].rename(columns={"me": "me_june"})
june_me["port_year"] = june_me["date"].dt.year
print(f"\nJune ME rows: {len(june_me)}, unique codes: {june_me['code'].nunique()}")

# ── Step 2: Extract June BE from book_equity_monthly ──────────────────
# At June of year t, book_equity_monthly has be_year = t-1 (the lagged fiscal year BE)
june_be = be[be["date"].dt.month == 6].copy()
june_be = june_be[["code", "date", "be"]].rename(columns={"be": "be"})
june_be["port_year"] = june_be["date"].dt.year
print(f"June BE rows: {len(june_be)}, unique codes: {june_be['code'].nunique()}")

# ── Step 3: Merge June ME and June BE on code + port_year ─────────────
june_merged = june_me.merge(june_be[["code", "port_year", "be"]], on=["code", "port_year"], how="inner")
print(f"\nJune merged (ME + BE): {len(june_merged)} rows, {june_merged['code'].nunique()} stocks")

# ── Step 4: Calculate BM ─────────────────────────────────────────────
# ME is in 천원 (thousands of won), BE is in 억 (hundred million won)
# Convert ME to 억: 1 억 = 100,000 천원
# BM = BE(억) / ME(억) = BE / (ME_천원 / 100,000)
june_merged["me_june_eok"] = june_merged["me_june"] / 100_000
# Negative BE → BM = NaN
june_merged["bm"] = np.where(
    june_merged["be"] < 0,
    np.nan,
    june_merged["be"] / june_merged["me_june_eok"],
)
print(f"BM stats:\n{june_merged['bm'].describe()}")
print(f"BM median: {june_merged['bm'].median():.4f}")

# ── Step 5: KOSDAQ flag ──────────────────────────────────────────────
# Korean stock codes: A0XXXXX = KOSPI (second char '0'), A1XXXXX = KOSDAQ
june_merged["is_kosdaq"] = june_merged["code"].str[1] != "0"
print(f"\nKOSDAQ stocks: {june_merged['is_kosdaq'].sum()} out of {len(june_merged)}")

# ── Step 6: Merge back into full monthly panel ────────────────────────
# The full panel has all monthly stock_returns data, with June ME, BE, BM, port_year, is_kosdaq
# We need to assign each monthly row a port_year based on which June portfolio it belongs to
# Convention: July t to June t+1 → port_year = t
# So for a row with date in month m of year y:
#   if m >= 7: port_year = y
#   if m <= 6: port_year = y - 1

sr["port_year"] = np.where(sr["date"].dt.month >= 7, sr["date"].dt.year, sr["date"].dt.year - 1)

# Merge monthly returns with June portfolio data
panel = sr.merge(
    june_merged[["code", "port_year", "me_june", "be", "bm", "is_kosdaq"]],
    on=["code", "port_year"],
    how="left",
)

# Select and order columns per spec
panel = panel[["code", "name", "date", "return", "me", "me_june", "be", "bm", "port_year", "is_kosdaq"]]

print(f"\nFinal panel shape: {panel.shape}")
print(f"Columns: {panel.columns.tolist()}")
print(f"Unique codes: {panel['code'].nunique()}")
print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")

# ── Verification ──────────────────────────────────────────────────────
print("\n=== VERIFICATION ===")

# BM median should be 0.5~1.5
bm_median = panel.groupby("date")["bm"].median()
print(f"BM median range: {bm_median.min():.4f} to {bm_median.max():.4f}")
print(f"Overall BM median: {panel['bm'].median():.4f}")

# June 2005 valid stocks > 100
june_2005 = panel[(panel["date"].dt.year == 2005) & (panel["date"].dt.month == 6)]
valid_june_2005 = june_2005["bm"].notna().sum()
print(f"June 2005 valid BM stocks: {valid_june_2005}")

# KOSDAQ count
kosdaq_count = panel[panel["is_kosdaq"] == True]["code"].nunique()
print(f"KOSDAQ unique stocks: {kosdaq_count}")

# Negative BE check
neg_be = panel[panel["be"] < 0]
print(f"Negative BE rows: {len(neg_be)}, BM NaN for those: {neg_be['bm'].isna().sum()}")

# Sample output
print(f"\nSample rows:\n{panel.head(10).to_string()}")

# ── Save ──────────────────────────────────────────────────────────────
panel.to_parquet("korean_ff3_v2/data/panel_data.parquet", index=False)
print(f"\nSaved panel_data.parquet: {panel.shape}")