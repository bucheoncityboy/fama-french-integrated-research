"""
06_section6_conclusions.py
Generate FF(1993) replication report from computed output CSVs.
"""

import pandas as pd
from pathlib import Path

import config

OUTPUT_DIR = Path(config.OUTPUT_DIR)
REPORT_PATH = OUTPUT_DIR / "FF1993_replication_report.md"


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name)


def fmt(x, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def fmt4(x) -> str:
    return fmt(x, 4)


def md_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report() -> str:
    # Load data
    t2 = load_csv("table2_summary.csv")
    t1 = load_csv("table1_market.csv")
    t3 = load_csv("table3_bond.csv")
    t4 = load_csv("table4_stock3f.csv")
    t5 = load_csv("table5_five_factor.csv")
    grs = load_csv("grs_test_results.csv")
    intercept = load_csv("intercept_analysis.csv")

    stocks_t2 = t2[t2["Type"] == "Stock"].copy()
    bonds_t2 = t2[t2["Type"] == "Bond"].copy()
    factors_t2 = t2[t2["Type"] == "Factor"].copy()

    stocks_t1 = t1[t1["type"] == "stock"]
    bonds_t1 = t1[t1["type"] == "bond"]

    stocks_t3 = t3[t3["type"] == "stock"]
    bonds_t3 = t3[t3["type"] == "bond"]

    stocks_t4 = t4[t4["type"] == "stock"]
    bonds_t4 = t4[t4["type"] == "bond"]

    stocks_t5 = t5[t5["type"] == "stock"]
    bonds_t5 = t5[t5["type"] == "bond"]

    # Build 5x5 stock grid from table2_summary labels
    # Labels mix prefixes: SMALL/ME1 for row 1, BIG/ME5 for row 5
    grid_labels = [
        ["SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM"],
        ["ME2 BM1", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 BM5"],
        ["ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5"],
        ["ME4 BM1", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5"],
        ["BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM"],
    ]
    size_labels = ["SMALL", "ME2", "ME3", "ME4", "BIG"]
    bm_labels = ["LoBM", "BM2", "BM3", "BM4", "HiBM"]

    # Map labels to rows
    label_to_mean = dict(zip(stocks_t2["Label"], stocks_t2["Mean"]))

    # 5x5 average excess returns table
    stock_grid_headers = ["Size / BE-ME"] + bm_labels
    stock_grid_rows = []
    for i, size in enumerate(size_labels):
        row_vals = [size]
        for label in grid_labels[i]:
            val = label_to_mean.get(label, float("nan"))
            row_vals.append(fmt(val, 2))
        stock_grid_rows.append(row_vals)

    # Bond average excess returns
    bond_rows = []
    for _, row in bonds_t2.iterrows():
        bond_rows.append([row["Label"], fmt(row["Mean"], 2), fmt(row["Std"], 2), str(int(row["N"])), fmt(row["T_Stat"], 2)])

    # Factor premium summary
    factor_rows = []
    for _, row in factors_t2.iterrows():
        factor_rows.append([row["Label"], fmt(row["Mean"], 2), fmt(row["Std"], 2), str(int(row["N"])), fmt(row["T_Stat"], 2)])

    # Section 4 summaries
    avg_r2_stock_t1 = stocks_t1["r_squared"].mean()
    avg_r2_bond_t1 = bonds_t1["r_squared"].mean()

    avg_r2_stock_t3 = stocks_t3["r_squared"].mean()
    avg_r2_bond_t3 = bonds_t3["r_squared"].mean()

    avg_r2_stock_t4 = stocks_t4["r_squared"].mean()
    avg_r2_bond_t4 = bonds_t4["r_squared"].mean()

    avg_r2_stock_t5 = stocks_t5["r_squared"].mean()
    avg_r2_bond_t5 = bonds_t5["r_squared"].mean()

    # Incremental R2
    inc_r2_stock_t4_vs_t1 = avg_r2_stock_t4 - avg_r2_stock_t1
    inc_r2_stock_t5_vs_t4 = avg_r2_stock_t5 - avg_r2_stock_t4
    inc_r2_bond_t5_vs_t4 = avg_r2_bond_t5 - avg_r2_bond_t4
    inc_r2_stock_t5_vs_t1 = avg_r2_stock_t5 - avg_r2_stock_t1
    inc_r2_bond_t5_vs_t1 = avg_r2_bond_t5 - avg_r2_bond_t1

    # GRS test table
    grs_rows = []
    for _, row in grs.iterrows():
        grs_rows.append([
            row["test_name"],
            str(int(row["N"])),
            str(int(row["K"])),
            str(int(row["T"])),
            fmt(row["F_stat"], 4),
            fmt(row["p_value"], 4),
            fmt(row["mean_abs_alpha"], 4),
            fmt(row["mean_abs_t_alpha"], 4)
        ])

    # Intercept analysis summary per model
    models = ["one_factor", "two_factor_bond", "three_factor_stock", "five_factor", "two_factor_stock"]
    model_names = {
        "one_factor": "1-Factor (Market)",
        "two_factor_bond": "2-Factor (TERM+DEF)",
        "three_factor_stock": "3-Factor (Mkt+SMB+HML)",
        "five_factor": "5-Factor (All)",
        "two_factor_stock": "2-Factor (SMB+HML)"
    }
    intercept_summary = []
    for model in models:
        sub = intercept[intercept["model"] == model]
        if len(sub) == 0:
            continue
        mean_abs = sub["abs_alpha"].mean()
        pct_sig = (sub["significant"].sum() / len(sub)) * 100
        # Split by type
        stock_sub = sub[sub["type"] == "stock"]
        bond_sub = sub[sub["type"] == "bond"]
        mean_abs_stock = stock_sub["abs_alpha"].mean() if len(stock_sub) > 0 else 0
        mean_abs_bond = bond_sub["abs_alpha"].mean() if len(bond_sub) > 0 else 0
        intercept_summary.append([
            model_names[model],
            fmt(mean_abs, 4),
            fmt(mean_abs_stock, 4),
            fmt(mean_abs_bond, 4),
            fmt(pct_sig, 1)
        ])

    # Figure references
    fig_refs = """
- **Figure 1**: Average Excess Returns Heatmap (`output/fig1_average_returns_heatmap.png`)
- **Figure 2**: Cumulative Factor Returns (`output/fig2_factor_cumulative_returns.png`)
- **Figure 3**: R-Squared Comparison Across Models (`output/fig3_r2_comparison.png`)
- **Figure 4**: Alpha Distribution (`output/fig4_alpha_distribution.png`)
- **Figure 5**: Factor Loadings Heatmap (`output/fig5_factor_loadings_heatmap.png`)
- **Figure 6**: SMB vs HML Scatter (`output/fig6_smb_hml_scatter.png`)
"""

    report = f"""# Fama-French (1993) Common Risk Factors in the Returns on Stocks and Bonds

## Replication Report

**Replication Period**: 1963-07 to 1991-12  
**Methodology**: Time-series regression approach with 25 size x BE/ME stock portfolios and 7 bond portfolios  
**Data Sources**: Compustat BE, CRSP raw data, Ken French Data Library (hybrid first 12 months), FRED

---

## Section 1: Introduction

The Capital Asset Pricing Model (CAPM) posits that a single market factor explains the cross-section of expected returns. By the early 1990s, however, extensive empirical evidence had documented clear patterns that CAPM could not capture. Small stocks earn higher average returns than large stocks. High book-to-market (value) stocks outperform low book-to-market (growth) stocks. In bond markets, term spreads and default spreads predict differences in expected returns.

Fama and French (1993) addressed these failures by proposing a multi-factor model. Their contribution was twofold. First, they showed that three stock-market factors, Market (Mkt-RF), Size (SMB), and Value (HML), explain the cross-section of stock returns. Second, they extended the framework to bond markets using two bond factors, Term (TERM) and Default (DEF), and demonstrated that a combined five-factor model captures common variation in both stock and bond returns.

This replication implements the Fama-French (1993) methodology using publicly available data. We construct 25 stock portfolios sorted on size (market equity) and book-to-market equity (BE/ME), and 7 bond portfolios spanning government and corporate maturities and credit ratings. The stock-side series are hybrid: 1963-07 to 1964-06 retain Ken French observations, and 1964-07 onward uses self-constructed Compustat BE + CRSP raw-data portfolios linked through an open gvkey-PERMCO bridge. We then estimate one-factor, two-factor, three-factor, and five-factor models, and evaluate their performance using R-squared statistics, GRS tests, and intercept analysis.

> **IMPORTANT DISCLAIMER**: The bond factors (TERM and DEF) used in this replication are yield-based proxies constructed from FRED series. They are not the return-based bond factors from the original paper, which were derived from portfolios of government and corporate bonds. This is a known and significant limitation. Results for bond regressions should be interpreted with caution.

---

## Section 2: Inputs

### Factor Construction Methodology

| Factor | Description | Construction |
| --- | --- | --- |
| Mkt-RF | Market excess return | Hybrid stock-side factor series used in the replication output |
| SMB | Small minus Big | Recomputed from 6 Size x BE/ME portfolios; hybrid first 12 months + self-constructed thereafter |
| HML | High minus Low | Recomputed from 6 Size x BE/ME portfolios; hybrid first 12 months + self-constructed thereafter |
| TERM | Term spread proxy | Difference between long-term and short-term government yields from FRED |
| DEF | Default spread (BAA-AAA) | BAA-AAA corporate bond yield spread from FRED (BAA minus AAA) |

### Portfolio Data

| Asset Class | Count | Sorting Variables |
| --- | --- | --- |
| Stock Portfolios | 25 | Size (Market Equity, 5 quintiles) x BE/ME (5 quintiles) |
| Bond Portfolios | 7 | 2 government (short-term, long-term) + 5 corporate (AAA, AA, A, BBB, Low Grade) |

### Data Sources

| Source | URL | Series / Files Used |
| --- | --- | --- |
| Ken French Data Library | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html | Hybrid seed data for 1963-07 to 1964-06 and original reference series |
| Compustat BE | Local `compustat_be.csv` | Book equity inputs used for self-constructed stock portfolios |
| CRSP raw data | Local `crsp/` extracts | PRC, RET, SHROUT, PERMNO, PERMCO used for ME and returns |
| Wenzhi-Ding/Std_Security_Code | https://github.com/Wenzhi-Ding/Std_Security_Code | Open gvkey-PERMCO-PERMNO bridge table |
| FRED | https://fred.stlouisfed.org/ | GS10 (10-Year Treasury), TB3MS (3-Month T-Bill), BAA (Moody's Baa), AAA (Moody's Aaa) |

---

## Section 3: The Playing Field

### Table: Average Monthly Excess Returns for 25 Stock Portfolios (%)

{md_table(stock_grid_headers, stock_grid_rows)}

*Note: Returns are in percent per month. Rows are size quintiles (SMALL to BIG). Columns are BE/ME quintiles (LoBM to HiBM).*

### Table: Average Monthly Excess Returns for 7 Bond Portfolios (%)

{md_table(["Portfolio", "Mean", "Std Dev", "N", "t-Stat"], bond_rows)}

### Table: Factor Premium Summary (% per month)

{md_table(["Factor", "Mean", "Std Dev", "N", "t-Stat"], factor_rows)}

### Key Findings

- **Size effect**: Within each BE/ME quintile, smaller stocks tend to have higher average excess returns than larger stocks. For example, the smallest growth portfolio (SMALL LoBM) averages {fmt(stocks_t2[stocks_t2['Label']=='SMALL LoBM']['Mean'].values[0], 2)}% per month, while the largest growth portfolio (BIG LoBM) averages {fmt(stocks_t2[stocks_t2['Label']=='BIG LoBM']['Mean'].values[0], 2)}%.
- **Value effect**: Within each size quintile, high BE/ME (value) stocks consistently outperform low BE/ME (growth) stocks. The small-value portfolio (SMALL HiBM) averages {fmt(stocks_t2[stocks_t2['Label']=='SMALL HiBM']['Mean'].values[0], 2)}% per month versus {fmt(stocks_t2[stocks_t2['Label']=='SMALL LoBM']['Mean'].values[0], 2)}% for small-growth.
- **Bond returns remain much smaller than stock returns in this sample**: The bond proxy portfolios all have positive average excess returns, but they are an order of magnitude lower than stock portfolios and inherit mechanical structure from the yield-proxy construction.

---

## Section 4: Common Variation

### Table 1: Market Factor Regressions (One-Factor Model)

| Asset Class | Avg R-Squared | Avg |t(Mkt-RF)| | Interpretation |
| --- | --- | --- | --- |
| Stocks (25) | {fmt(avg_r2_stock_t1, 4)} | {fmt(stocks_t1['t_Mkt-RF'].abs().mean(), 2)} | Market factor explains substantial common variation in stock returns |
| Bonds (7) | {fmt(avg_r2_bond_t1, 4)} | {fmt(bonds_t1['t_Mkt-RF'].abs().mean(), 2)} | Market factor has almost no explanatory power for bond returns |

The market factor alone captures on average {fmt(avg_r2_stock_t1 * 100, 1)}% of the variance in stock portfolio returns, but only {fmt(avg_r2_bond_t1 * 100, 2)}% for bond portfolios. This confirms Fama-French's finding that stocks and bonds are driven by different sources of common risk.

### Table 3: Bond Factor Regressions (Two-Factor Model: TERM + DEF)

For bond portfolios, the TERM factor loadings are the dominant driver. The average absolute t-statistic on TERM for bonds is very large, reflecting the construction methodology where TERM is derived from the same yield series that define bond returns. For stocks, TERM and DEF loadings are small and only marginally significant on average.

| Asset Class | Avg R-Squared | Notes |
| --- | --- | --- |
| Stocks (25) | {fmt(avg_r2_stock_t3, 4)} | Low explanatory power; bond factors do not capture stock variation well |
| Bonds (7) | {fmt(avg_r2_bond_t3, 4)} | Near-perfect fit for some portfolios due to collinear construction |

### Table 4: Three-Factor Stock Regressions (Mkt-RF + SMB + HML)

Adding SMB and HML to the market factor dramatically improves explanatory power for stock returns.

| Asset Class | Avg R-Squared | Avg |alpha| | Improvement over 1-Factor |
| --- | --- | --- | --- |
| Stocks (25) | {fmt(avg_r2_stock_t4, 4)} | {fmt(stocks_t4['alpha'].abs().mean(), 4)} | +{fmt(inc_r2_stock_t4_vs_t1 * 100, 1)} percentage points |
| Bonds (7) | {fmt(avg_r2_bond_t4, 4)} | {fmt(bonds_t4['alpha'].abs().mean(), 4)} | Negligible |

The three-factor model raises average R-squared for stocks from {fmt(avg_r2_stock_t1 * 100, 1)}% to {fmt(avg_r2_stock_t4 * 100, 1)}%. SMB and HML loadings are strongly significant across portfolios, confirming the size and value effects documented in the raw returns.

### Table 5: Five-Factor Regressions (All Factors)

The full five-factor model combines stock and bond factors to explain returns in both markets.

| Asset Class | Avg R-Squared | Incremental over 3-Factor | Incremental over 1-Factor |
| --- | --- | --- | --- |
| Stocks (25) | {fmt(avg_r2_stock_t5, 4)} | +{fmt(inc_r2_stock_t5_vs_t4 * 100, 2)} pp | +{fmt(inc_r2_stock_t5_vs_t1 * 100, 1)} pp |
| Bonds (7) | {fmt(avg_r2_bond_t5, 4)} | +{fmt(inc_r2_bond_t5_vs_t4 * 100, 2)} pp | +{fmt(inc_r2_bond_t5_vs_t1 * 100, 2)} pp |

For stocks, the incremental contribution of TERM and DEF beyond the three stock factors is small ({fmt(inc_r2_stock_t5_vs_t4 * 100, 2)} percentage points), consistent with the original paper. For bonds, adding stock factors to the bond factors provides little additional explanatory power beyond what the bond factors already capture.

---

## Section 5: Average Returns in Cross-Section

### GRS Test Results

The Gibbons-Ross-Shanken (GRS) test evaluates whether the intercepts from time-series regressions are jointly zero. A model that explains the cross-section of average returns should produce intercepts that are statistically indistinguishable from zero.

{md_table(
    ["Test", "N (Portfolios)", "K (Factors)", "T (Months)", "F-Stat", "p-Value", "Mean |alpha|", "Mean |t(alpha)|"],
    grs_rows
)}

### Key GRS Findings

- **Stocks 3-Factor**: F = {fmt(grs[grs['test_name']=='Stocks_3Factor']['F_stat'].values[0], 4)}, p = {fmt(grs[grs['test_name']=='Stocks_3Factor']['p_value'].values[0], 4)}. The three-factor model produces intercepts that are not jointly significant at the 1% level, suggesting it does a reasonable job explaining the cross-section of stock returns.
- **Stocks 5-Factor**: F = {fmt(grs[grs['test_name']=='Stocks_5Factor']['F_stat'].values[0], 4)}, p = {fmt(grs[grs['test_name']=='Stocks_5Factor']['p_value'].values[0], 4)}. Adding bond factors to stocks actually worsens the GRS test, likely because the bond proxies add noise rather than explanatory power.
- **Bonds 2-Factor**: F = {fmt(grs[grs['test_name']=='Bonds_2Factor']['F_stat'].values[0], 4)}, p = {fmt(grs[grs['test_name']=='Bonds_2Factor']['p_value'].values[0], 4)}. The bond factor model passes comfortably, though this is partly mechanical due to the proxy construction.
- **All 5-Factor**: F = {fmt(grs[grs['test_name']=='All_5Factor']['F_stat'].values[0], 4)}, p = {fmt(grs[grs['test_name']=='All_5Factor']['p_value'].values[0], 4)}. The joint model is rejected at conventional levels, driven by the mismatch between stock and bond return dynamics.
- **All SMB+HML only**: F = {fmt(grs[grs['test_name']=='All_SMBHML']['F_stat'].values[0], 4)}, p = {fmt(grs[grs['test_name']=='All_SMBHML']['p_value'].values[0], 4)}. A stock-style two-factor model is decisively rejected for the combined 32 portfolios, underscoring that bond risk cannot be summarized by SMB and HML alone.

### Intercept Analysis Summary

{md_table(
    ["Model", "Mean |alpha| (All)", "Mean |alpha| (Stocks)", "Mean |alpha| (Bonds)", "% Significant"],
    intercept_summary
)}

The three-factor stock model achieves the lowest mean absolute intercept for stocks ({fmt(intercept[(intercept['model']=='three_factor_stock') & (intercept['type']=='stock')]['abs_alpha'].mean(), 4)}%), confirming that Mkt-RF, SMB, and HML capture the key dimensions of stock return variation. The one-factor model leaves much larger pricing errors, especially for small and value portfolios.

For bonds, the two-factor model yields near-zero intercepts by construction, but this reflects the proxy nature of the data rather than true explanatory power.

---

## Section 6: Conclusions

This replication of Fama and French (1993) produces findings that are broadly consistent with the original paper, subject to the important caveat that bond data are yield-based proxies rather than actual bond returns.

### Summary of Findings

1. **Stock returns are driven by three factors**: The market factor explains much of the time-series variation in stock returns, but adding SMB and HML raises average R-squared from {fmt(avg_r2_stock_t1 * 100, 1)}% to {fmt(avg_r2_stock_t4 * 100, 1)}%. The size and value effects are robust in this sample.

2. **Bond returns are driven by different factors**: The market factor has almost no explanatory power for bond returns. Term and default spreads capture bond variation, though our proxies are mechanically linked to the yield data from which they are constructed.

3. **A five-factor joint model is feasible but not perfect**: Combining stock and bond factors into a single model produces modest improvements for individual asset classes but does not fully resolve the cross-sectional pricing errors when all 32 portfolios are tested jointly.

4. **Intercept analysis supports the three-factor model for stocks**: The mean absolute intercept for stocks under the three-factor model is {fmt(intercept[(intercept['model']=='three_factor_stock') & (intercept['type']=='stock')]['abs_alpha'].mean(), 4)}% per month, substantially lower than the {fmt(intercept[(intercept['model']=='one_factor') & (intercept['type']=='stock')]['abs_alpha'].mean(), 4)}% under the one-factor model.

### Limitations

- **Bond proxy data**: The TERM and DEF factors are constructed from yield spreads (GS10, TB3MS, BAA, AAA) rather than actual bond portfolio returns. This means bond regression results should not be interpreted as evidence that the model "explains" real bond returns. It is a methodological limitation acknowledged throughout this replication.
- **Sample period**: The replication covers 1963-1991, matching the original paper. Results may differ in other periods.
- **Portfolio construction**: The stock-side replacement relies on an open gvkey-PERMCO bridge and hybrid carryover for the first 12 months because 1962 BE inputs are unavailable in `compustat_be.csv`. This is close to, but not identical with, the original CCM-based construction.

### Consistency with Fama-French (1993)

Despite the bond data limitations, the patterns in this replication match the original paper:
- Stock returns show clear size and value effects.
- The three-factor model explains stocks well.
- Bond and stock returns load on different factors.
- A joint five-factor model captures most common variation but leaves some pricing errors.

### Generated Figures

{fig_refs}

---

*Report generated by 06_section6_conclusions.py*  
*Data: Compustat BE, CRSP raw data, Ken French hybrid seed data, FRED*  
*Replication of Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1), 3-56.*
"""

    return report


def main():
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
