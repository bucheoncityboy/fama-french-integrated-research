# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

results = {}

# ============================================================
# TASK 2: RF Quality Check
# ============================================================
print("=" * 60)
print("TASK 2: RF Quality Check")
print("=" * 60)
rf = pd.read_csv('korean_ff3_v2/data/rf_final.csv')
print(f'Rows: {len(rf)}')
print(f'Range: {rf.date.min()} ~ {rf.date.max()}')
print(f'RF annual: {rf.rf_annual.min():.2f}% ~ {rf.rf_annual.max():.2f}%')
print(f'RF monthly: {rf.rf_monthly.min():.4f}% ~ {rf.rf_monthly.max():.4f}%')
print(f'NaN count: {rf.rf_monthly.isna().sum()}')
t2_pass = (len(rf) == 317 and rf.rf_annual.min() >= 0.50 and rf.rf_annual.max() <= 5.25 and rf.rf_monthly.isna().sum() == 0)
print(f'VERDICT: {"PASS" if t2_pass else "FAIL"}')
results['Task2_RF'] = t2_pass

# ============================================================
# TASK 3: Long-format Verification
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: Long-format Verification")
print("=" * 60)
m = pd.read_parquet('korean_ff3_v2/data/market_data_long.parquet')
f = pd.read_parquet('korean_ff3_v2/data/financial_data_long.parquet')
print(f'Market: {len(m)} rows, {m.code.nunique()} stocks, {m.item_code.nunique()} items')
print(f'Market date range: {m.date.min()} ~ {m.date.max()}')
print(f'Market item_codes: {sorted(m.item_code.unique())}')
print(f'Financial: {len(f)} rows, {f.code.nunique()} stocks, {f.item_code.nunique()} items')
print(f'Financial date range: {f.date.min()} ~ {f.date.max()}')
print(f'Financial item_codes: {sorted(f.item_code.unique())}')
ss = m[m.code=='A005930']
print(f'Samsung items: {ss.item_code.nunique()}, date range: {ss.date.min()} ~ {ss.date.max()}')
t3_pass = (m.item_code.nunique() == 3 and f.item_code.nunique() == 6 and ss.item_code.nunique() == 3)
print(f'VERDICT: {"PASS" if t3_pass else "FAIL"}')
results['Task3_LongFormat'] = t3_pass

# ============================================================
# TASK 5: Filtering Verification
# ============================================================
print("\n" + "=" * 60)
print("TASK 5: Filtering Verification")
print("=" * 60)
r = pd.read_parquet('korean_ff3_v2/data/stock_returns.parquet')
codes = r.code.unique()
print(f'Total stocks: {len(codes)}')
pref = [c for c in codes if str(c)[-1] in ['5','6','7']]
print(f'Preferred (should be 0): {len(pref)}')
kosdaq = [c for c in codes if str(c)[1] != '0']
print(f'KOSDAQ stocks: {len(kosdaq)}')
fin_names = ['은행','증권','보험','카드','캐피탈','저축','금융','신탁','손해보험','생명보험']
fin_count = 0
for fn in fin_names:
    matches = r[r.name.str.contains(fn, na=False)]
    if len(matches) > 0:
        fin_count += matches.code.nunique()
        print(f'  Found: {fn}: {matches.code.nunique()} stocks')
print(f'Financial stocks found (should be 0): {fin_count}')
ret = r['return'].dropna()
print(f'Returns: {len(ret)} valid, mean={ret.mean()*100:.2f}%, std={ret.std()*100:.2f}%')
t5_pass = (len(pref) == 0 and len(kosdaq) > 0 and fin_count == 0)
print(f'VERDICT: {"PASS" if t5_pass else "FAIL"}')
results['Task5_Filtering'] = t5_pass

# ============================================================
# TASK 7: Panel Quality
# ============================================================
print("\n" + "=" * 60)
print("TASK 7: Panel Quality")
print("=" * 60)
p = pd.read_parquet('korean_ff3_v2/data/panel_data.parquet')
print(f'Panel: {len(p)} rows, {p.code.nunique()} stocks')
print(f'KOSDAQ: {p[p.is_kosdaq == True].code.nunique()}')
print(f'BM median: {p.bm.median():.2f}')
j2005 = p[(p.date.dt.year==2005) & (p.date.dt.month==6) & (p.bm.notna())]
print(f'June 2005 valid: {j2005.code.nunique()} stocks')
print(f'BM range: {p.bm.min():.4f} ~ {p.bm.max():.4f}')
print(f'BM NaN ratio: {p.bm.isna().sum() / len(p):.2%}')
# Check BM median in reasonable range (0.5-1.5 per plan)
bm_median = p.bm.median()
t7_pass = (0.5 <= bm_median <= 1.5 and j2005.code.nunique() >= 100)
print(f'VERDICT: {"PASS" if t7_pass else "FAIL"} (BM median in [0.5,1.5]: {0.5 <= bm_median <= 1.5}, June stocks >= 100: {j2005.code.nunique() >= 100})')
results['Task7_Panel'] = t7_pass

# ============================================================
# TASK 9: Factor Statistics
# ============================================================
print("\n" + "=" * 60)
print("TASK 9: Factor Statistics")
print("=" * 60)
factors = pd.read_csv('korean_ff3_v2/data/factors_korea.csv')
factors['date'] = pd.to_datetime(factors['date'])
print(f'Period: {factors.date.min()} ~ {factors.date.max()}, {len(factors)} months')
for col in ['Mkt-RF','SMB','HML']:
    m = factors[col].mean()
    s = factors[col].std()
    t = m/s*(len(factors)**0.5) if s > 0 else 0
    print(f'{col}: mean={m:.4f} ({m*100:.2f}%), std={s:.4f} ({s*100:.2f}%), t={t:.2f}')
t9_pass = (len(factors) >= 300)  # ~311 months expected
print(f'VERDICT: {"PASS" if t9_pass else "FAIL"} ({len(factors)} months)')
results['Task9_Factors'] = t9_pass

# ============================================================
# TASK 11: Regression Patterns
# ============================================================
print("\n" + "=" * 60)
print("TASK 11: Regression Patterns")
print("=" * 60)
reg = pd.read_csv('korean_ff3_v2/output/regression_korea.csv')
print(f'Portfolios: {len(reg)}')
print(f'All beta_mkt > 0: {(reg.beta_mkt>0).all()}')
print(f'SMB beta range: {reg.beta_smb.min():.2f} ~ {reg.beta_smb.max():.2f}')
print(f'HML beta range: {reg.beta_hml.min():.2f} ~ {reg.beta_hml.max():.2f}')
print(f'R2 range: {reg.r_squared.min():.3f} ~ {reg.r_squared.max():.3f}')
# Check small-cap beta_smb > 0
small_ports = [c for c in reg.portfolio if str(c).startswith('S1')]
if len(small_ports) > 0:
    small_betas = reg[reg.portfolio.isin(small_ports)].beta_smb
    print(f'Small-cap beta_smb > 0: {(small_betas > 0).all()} (mean={small_betas.mean():.3f})')
# Check high-BM beta_hml > 0
high_bm_ports = [c for c in reg.portfolio if str(c).endswith('B5')]
if len(high_bm_ports) > 0:
    high_bm_betas = reg[reg.portfolio.isin(high_bm_ports)].beta_hml
    print(f'High-BM beta_hml > 0: {(high_bm_betas > 0).all()} (mean={high_bm_betas.mean():.3f})')
t11_pass = ((reg.beta_mkt > 0).all() and reg.r_squared.min() > 0 and reg.r_squared.max() < 1)
print(f'VERDICT: {"PASS" if t11_pass else "FAIL"}')
results['Task11_Regression'] = t11_pass

# ============================================================
# TASK 15: v1-v2 Correlation
# ============================================================
print("\n" + "=" * 60)
print("TASK 15: v1-v2 Correlation")
print("=" * 60)
try:
    v1 = pd.read_csv('korean_ff3/data/archive/v1/kr_ff_factors.csv')
    v2 = pd.read_csv('korean_ff3_v2/data/factors_korea.csv')
    v1['Date'] = pd.to_datetime(v1['Date'])
    v2['date'] = pd.to_datetime(v2['date'])
    # v1 uses percentage, v2 uses decimal - need to align
    # Check scale
    print(f'v1 Mkt-RF sample: {v1["Mkt-RF"].head(3).tolist()}')
    print(f'v2 Mkt-RF sample: {v2["Mkt-RF"].head(3).tolist()}')
    # v1 is in % (percentage), v2 is in decimal
    # Convert v2 to % for comparison
    v2_pct = v2.copy()
    v2_pct['Mkt-RF'] = v2_pct['Mkt-RF'] * 100
    v2_pct['SMB'] = v2_pct['SMB'] * 100
    v2_pct['HML'] = v2_pct['HML'] * 100
    # Merge on date
    m = v1.merge(v2_pct, left_on='Date', right_on='date', suffixes=('_v1','_v2'))
    print(f'Overlap period: {m.Date.min()} ~ {m.Date.max()}, {len(m)} months')
    for fac in ['Mkt-RF','SMB','HML']:
        c = m[f'{fac}_v1'].corr(m[f'{fac}_v2'])
        print(f'{fac} correlation: {c:.4f}')
    mkt_corr = m['Mkt-RF_v1'].corr(m['Mkt-RF_v2'])
    smb_corr = m['SMB_v1'].corr(m['SMB_v2'])
    hml_corr = m['HML_v1'].corr(m['HML_v2'])
    t15_pass = (mkt_corr > 0.90 and smb_corr > 0.90 and hml_corr > 0.90)
    print(f'VERDICT: {"PASS" if t15_pass else "FAIL"} (Mkt-RF: {mkt_corr:.4f}, SMB: {smb_corr:.4f}, HML: {hml_corr:.4f})')
except Exception as e:
    print(f'v1-v2 comparison error: {e}')
    t15_pass = False
    print('VERDICT: FAIL')
results['Task15_Correlation'] = t15_pass

# ============================================================
# CROSS-TASK INTEGRATION
# ============================================================
print("\n" + "=" * 60)
print("CROSS-TASK INTEGRATION")
print("=" * 60)

# 1. Factors align with portfolio dates
factors = pd.read_csv('korean_ff3_v2/data/factors_korea.csv')
factors['date'] = pd.to_datetime(factors['date'])
port6 = pd.read_csv('korean_ff3_v2/data/portfolios_6_korea.csv')
port6['date'] = pd.to_datetime(port6['date'])
port25 = pd.read_csv('korean_ff3_v2/data/portfolios_25_korea.csv')
port25['Date'] = pd.to_datetime(port25['Date'])

print(f'Factors dates: {factors.date.min()} ~ {factors.date.max()}, {len(factors)} months')
print(f'Port6 dates: {port6.date.min()} ~ {port6.date.max()}, {len(port6)} months')
print(f'Port25 dates: {port25.Date.min()} ~ {port25.Date.max()}, {len(port25)} months')

f_dates = set(factors.date.dt.to_period('M'))
p6_dates = set(port6.date.dt.to_period('M'))
p25_dates = set(port25.Date.dt.to_period('M'))

int1 = len(f_dates & p6_dates) == len(f_dates)
int2 = len(f_dates & p25_dates) >= len(f_dates) - 5  # port25 may start later
print(f'Factors-Port6 date match: {len(f_dates & p6_dates)}/{len(f_dates)} - {"PASS" if int1 else "WARN"}')
print(f'Factors-Port25 date match: {len(f_dates & p25_dates)}/{len(f_dates)} - {"PASS" if int2 else "WARN"}')

# 2. Regression uses same factors as portfolio returns
mkt_rf = pd.read_parquet('korean_ff3_v2/data/market_excess_return.parquet')
mkt_rf['date'] = pd.to_datetime(mkt_rf['date'])
print(f'Mkt-RF dates: {mkt_rf.date.min()} ~ {mkt_rf.date.max()}, {len(mkt_rf)} months')

f_mkt = factors[['date','Mkt-RF']].copy()
m_mkt = mkt_rf[['date','mkt_rf']].copy()
merged = f_mkt.merge(m_mkt, on='date', how='inner')
if len(merged) > 0:
    corr = merged['Mkt-RF'].corr(merged['mkt_rf'])
    print(f'Mkt-RF factor vs market_excess_return correlation: {corr:.6f}')
    int3 = corr > 0.99
else:
    print('Could not merge Mkt-RF data for correlation check')
    int3 = False

# 3. Panel data feeds into factors
panel = pd.read_parquet('korean_ff3_v2/data/panel_data.parquet')
panel['date'] = pd.to_datetime(panel['date'])
print(f'Panel dates: {panel.date.min()} ~ {panel.date.max()}, {panel.code.nunique()} stocks')
int4 = panel.code.nunique() > 500  # Should have many stocks

integration_pass = int1 and int2 and int3 and int4
print(f'\nIntegration: {"PASS" if integration_pass else "FAIL"}')
print(f'  Factors-Port6 alignment: {"PASS" if int1 else "FAIL"}')
print(f'  Factors-Port25 alignment: {"PASS" if int2 else "FAIL"}')
print(f'  Mkt-RF consistency: {"PASS" if int3 else "FAIL"}')
print(f'  Panel stock coverage: {"PASS" if int4 else "FAIL"}')

# ============================================================
# GRS Test
# ============================================================
print("\n" + "=" * 60)
print("GRS Test")
print("=" * 60)
grs = pd.read_csv('korean_ff3_v2/output/grs_test.csv')
print(grs.to_string())

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
total = len(results)
passed = sum(1 for v in results.values() if v)
for k, v in results.items():
    print(f'  {k}: {"PASS" if v else "FAIL"}')
print(f'\nScenarios: [{passed}/{total} pass]')
print(f'Integration: [{"PASS" if integration_pass else "FAIL"}/4]')
print(f'VERDICT: {"PASS" if all(results.values()) and integration_pass else "FAIL"}')