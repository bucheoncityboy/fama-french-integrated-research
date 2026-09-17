# Data Sources Documentation

## Ken French Data Library

**URL**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

**Files Downloaded**:
- Fama-French 3 Factors (Monthly) - `F-F_Research_Data_Factors.csv`
- 25 Portfolios formed on Size and BE/ME (5 x 5) - `25_Portfolios_5x5.csv`

**Variables Used**:
- `Mkt-RF`: Market excess return (value-weighted market return minus risk-free rate)
- `SMB`: Small Minus Big return factor
- `HML`: High Minus Low return factor
- `RF`: Risk-free rate
- 25 stock portfolio returns sorted by size (market equity) and book-to-market equity (BE/ME)

## FRED (Federal Reserve Economic Data)

**URL**: https://fred.stlouisfed.org/

**Series IDs Used**:

| Series ID | Description | Use in Replication |
| --- | --- | --- |
| GS10 | 10-Year Treasury Constant Maturity Rate | Long-term government yield for TERM factor |
| TB3MS | 3-Month Treasury Bill Secondary Market Rate | Short-term government yield for TERM factor; risk-free proxy |
| BAA | Moody's Seasoned Baa Corporate Bond Yield | Corporate yield for DEF factor |
| AAA | Moody's Seasoned Aaa Corporate Bond Yield | Government yield proxy for DEF factor |

## Important Disclaimer About Bond Data

The bond factors (TERM and DEF) in this replication are **yield-based proxies**, not actual bond portfolio returns.

- **TERM** is constructed as the difference between long-term and short-term government yields (GS10 minus TB3MS).
- **DEF** is constructed as the difference between corporate and government yields (BAA minus AAA).

This differs from the original Fama-French (1993) paper, which constructed bond factors from **actual returns** on portfolios of government and corporate bonds. The original TERM factor was the return difference between long-term and short-term government bond portfolios. The original DEF factor was the return difference between corporate and government bond portfolios of similar maturity.

### Why This Matters

1. **Yield spreads are not returns**: Yield changes do not equal total returns, which also include coupon income and price appreciation/depreciation.
2. **Mechanical correlation**: Because bond portfolio returns in this replication are also derived from yield changes, regressions of bond returns on TERM and DEF can show artificially high R-squared values.
3. **Results should be interpreted with caution**: The bond regression results demonstrate methodology but do not provide independent evidence that the five-factor model "explains" real bond returns.

### Recommendation for Future Work

To replicate the bond analysis more faithfully, obtain actual bond return data from:
- CRSP Bond Files
- Barclays (Bloomberg) Bond Indices
- Ibbotson Associates SBBI data

These sources provide total returns on government and corporate bond portfolios, allowing construction of true return-based TERM and DEF factors.

## Data Period

- **Stock data**: 1963-07 to 1991-12 (342 months)
- **Bond proxy data**: 1963-07 to 1991-12 (340-341 months, depending on series availability)

## Citation

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
