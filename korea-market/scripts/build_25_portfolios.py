"""
Build 25 Size×BM portfolios (5×5) using Fama-French methodology.

Input:  korean_ff3_v2/data/panel_data.parquet
Output: korean_ff3_v2/data/portfolios_25_korea.csv

Methodology:
- Each June, sort stocks into 5 size quintiles (S1=smallest) using me_june
- Each June, sort stocks into 5 BM quintiles (B1=lowest BM) using bm
- Independent sorts: size and BM breakpoints from quintiles
- VW returns using lag ME as weights (July t ~ June t+1)
- Exclude negative BM stocks (already NaN in panel)
- Period: 2001-07 ~ 2026-05 (first valid formation year is 2001)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PANEL_PATH = DATA_DIR / "panel_data.parquet"
OUTPUT_PATH = DATA_DIR / "portfolios_25_korea.csv"

# ── Load panel data ─────────────────────────────────────────────────────────
print("Loading panel data...")
panel = pd.read_parquet(PANEL_PATH)
print(f"  Panel: {len(panel)} rows, {panel['code'].nunique()} stocks")
print(f"  Date range: {panel['date'].min()} ~ {panel['date'].max()}")

# ── Compute lag ME before any filtering ─────────────────────────────────────
# lag ME = previous month's ME, used as VW weight
panel = panel.sort_values(["code", "date"]).copy()
panel["me_lag"] = panel.groupby("code")["me"].shift(1)

# ── Formation: assign stocks to 5×5 portfolios per port_year ────────────────
# me_june and bm are constant within each port_year (set during panel construction)
# We take one row per (code, port_year) to get the formation values

# Get unique formation data: one row per (code, port_year) with valid me_june and bm
formation = (
    panel[["code", "port_year", "me_june", "bm"]]
    .dropna(subset=["me_june", "bm"])
    .drop_duplicates(subset=["code", "port_year"])
    .copy()
)

# Exclude negative BM (should already be NaN, but double-check)
formation = formation[formation["bm"] > 0].copy()

print(f"\nFormation data: {len(formation)} rows")
print(f"  port_years: {formation['port_year'].min()} ~ {formation['port_year'].max()}")
stocks_per_year = formation.groupby("port_year")["code"].nunique()
print(f"  Stocks per port_year: min={stocks_per_year.min()}, max={stocks_per_year.max()}, mean={stocks_per_year.mean():.0f}")

# ── Assign size and BM quintiles per port_year ──────────────────────────────
# FF methodology: independent sorts
# Size: 5 quantiles (quintiles) - S1=smallest, S5=biggest
# BM: 5 quantiles (quintiles) - B1=lowest, B5=highest

formation_records = []

for py, grp in formation.groupby("port_year"):
    grp = grp.copy()
    n_stocks = len(grp)

    if n_stocks < 25:
        print(f"  Warning: port_year {py} has only {n_stocks} stocks, skipping")
        continue

    # Size quintiles (independent sort)
    grp["size_q"] = pd.qcut(grp["me_june"], 5, labels=False, duplicates="drop") + 1

    # BM quintiles (independent sort)
    grp["bm_q"] = pd.qcut(grp["bm"], 5, labels=False, duplicates="drop") + 1

    n_size_bins = grp["size_q"].nunique()
    n_bm_bins = grp["bm_q"].nunique()

    if n_size_bins < 5 or n_bm_bins < 5:
        print(f"  Warning: port_year {py} - size bins: {n_size_bins}, BM bins: {n_bm_bins}")

    # Portfolio assignment: SiBj
    grp["portfolio"] = "S" + grp["size_q"].astype(int).astype(str) + "B" + grp["bm_q"].astype(int).astype(str)

    formation_records.append(grp[["code", "port_year", "me_june", "bm", "size_q", "bm_q", "portfolio"]])

formation_df = pd.concat(formation_records, ignore_index=True)
print(f"\nFormation assignments: {len(formation_df)} rows")
print(f"  Unique portfolios: {sorted(formation_df['portfolio'].unique())}")

# ── Map portfolio assignments to return months ──────────────────────────────
# For each port_year, the portfolio assignment applies to all months in that port_year
# port_year t covers July t to June t+1

portfolio_map = formation_df[["code", "port_year", "portfolio"]].drop_duplicates()

panel_with_port = panel.merge(
    portfolio_map, on=["code", "port_year"], how="inner"
)

print(f"\nPanel with portfolio: {len(panel_with_port)} rows")
print(f"  Unique portfolios: {sorted(panel_with_port['portfolio'].unique())}")

# ── Filter to target period ──────────────────────────────────────────────────
# First valid port_year is 2001 (returns from 2001-07)
# Last valid port_year is 2025 (returns through 2026-05)
start_date = pd.Timestamp("2001-07-01")
end_date = pd.Timestamp("2026-05-01")
panel_filtered = panel_with_port[
    (panel_with_port["date"] >= start_date) & (panel_with_port["date"] <= end_date)
].copy()

print(f"\nFiltered period: {start_date.date()} ~ {end_date.date()}")
print(f"  Rows: {len(panel_filtered)}")

# ── Compute VW returns ──────────────────────────────────────────────────────
# VW return = sum(ret_i * ME_lag_i) / sum(ME_lag_i)
# Only use rows where both return and me_lag are valid

valid = panel_filtered.dropna(subset=["return", "me_lag"]).copy()
print(f"  Valid rows (return + me_lag): {len(valid)}")

# Group by date and portfolio
vw_returns = (
    valid.groupby(["date", "portfolio"])
    .apply(lambda g: np.average(g["return"], weights=g["me_lag"]), include_groups=False)
    .reset_index()
)
vw_returns.columns = ["date", "portfolio", "vw_return"]

# ── Pivot to wide format ────────────────────────────────────────────────────
all_ports = [f"S{i}B{j}" for i in range(1, 6) for j in range(1, 6)]

port_wide = vw_returns.pivot(index="date", columns="portfolio", values="vw_return")

# Ensure all 25 columns exist
for p in all_ports:
    if p not in port_wide.columns:
        port_wide[p] = np.nan

# Reorder columns
port_wide = port_wide[all_ports]

# Reset index and rename
port_wide = port_wide.reset_index()
port_wide = port_wide.rename(columns={"date": "Date"})

# Sort by date
port_wide = port_wide.sort_values("Date").reset_index(drop=True)

print(f"\nOutput shape: {port_wide.shape}")
print(f"  Date range: {port_wide['Date'].min()} ~ {port_wide['Date'].max()}")
print(f"  Months: {len(port_wide)}")

# ── Coverage statistics ──────────────────────────────────────────────────────
coverage = valid.groupby(["date", "portfolio"]).size().reset_index(name="n_stocks")
avg_coverage = coverage.groupby("portfolio")["n_stocks"].mean()
print("\nAverage stocks per portfolio:")
for p in all_ports:
    if p in avg_coverage.index:
        print(f"  {p}: {avg_coverage[p]:.1f}")
    else:
        print(f"  {p}: N/A")

# NaN ratio
total_cells = len(port_wide) * 25
nan_cells = port_wide[all_ports].isna().sum().sum()
print(f"\nNaN ratio: {nan_cells}/{total_cells} = {nan_cells/total_cells:.1%}")

# ── Mean returns ────────────────────────────────────────────────────────────
print("\nMean monthly returns (%):")
means = port_wide[all_ports].mean() * 100
for p in all_ports:
    val = means[p]
    if pd.notna(val):
        print(f"  {p}: {val:.3f}%")
    else:
        print(f"  {p}: NaN")

# ── 5×5 heatmap of mean returns ────────────────────────────────────────────
print("\n5×5 Mean Returns (% per month):")
print("         B1(Low)   B2        B3        B4        B5(High)")
for i in range(1, 6):
    row = []
    for j in range(1, 6):
        p = f"S{i}B{j}"
        val = means[p]
        row.append(f"{val*100:.3f}" if pd.notna(val) else "  NaN ")
    label = f"S{i}(Small)" if i == 1 else f"S{i}(Big)  " if i == 5 else f"S{i}        "
    print(f"  {label}  {'  '.join(row)}")

# ── Save ────────────────────────────────────────────────────────────────────
port_wide.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
print(f"  Shape: {port_wide.shape}")
print(f"  Columns: {list(port_wide.columns)}")