"""
Task 5: Calculate individual stock monthly returns from market_data_long.parquet
- Extract S410000700 (price), S410001250 (ME), S420003800 (shares outstanding)
- Calculate monthly returns via pct_change
- Filter preferred stocks (codes ending in 5/6/7)
- Filter financial stocks (name contains financial patterns)
- Set extreme returns (|return| > 100%) to NaN
- Save as stock_returns.parquet
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent dir for config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, PREFERRED_SUFFIXES, FINANCIAL_PATTERNS

def main():
    # ============================================================
    # 1. Load market data
    # ============================================================
    print("Loading market_data_long.parquet...")
    df = pd.read_parquet(os.path.join(DATA_DIR, 'market_data_long.parquet'))
    print(f"  Raw shape: {df.shape}")
    print(f"  Stocks: {df.code.nunique()}, Items: {df.item_code.unique()}")
    print(f"  Date range: {df.date.min()} ~ {df.date.max()}")

    # ============================================================
    # 2. Pivot: wide format with prc, me, shrout columns
    # ============================================================
    item_map = {
        'S410000700': 'prc',
        'S410001250': 'me',
        'S420003800': 'shrout',
    }

    # Filter to relevant items only
    df_items = df[df['item_code'].isin(item_map.keys())].copy()
    df_items['item_name'] = df_items['item_code'].map(item_map)

    # Pivot to wide format
    print("\nPivoting to wide format...")
    wide = df_items.pivot_table(
        index=['code', 'name', 'date'],
        columns='item_name',
        values='value',
        aggfunc='first'
    ).reset_index()
    wide.columns.name = None

    print(f"  Wide shape: {wide.shape}")
    print(f"  Columns: {list(wide.columns)}")

    # ============================================================
    # 3. Calculate monthly returns
    # ============================================================
    print("\nCalculating monthly returns...")
    wide = wide.sort_values(['code', 'date']).reset_index(drop=True)
    wide['return'] = wide.groupby('code')['prc'].pct_change(fill_method=None)
    # First observation per stock is NaN (pct_change naturally does this)

    # ============================================================
    # 4. Filter extreme returns (|return| > 100%)
    # ============================================================
    extreme_mask = wide['return'].abs() > 1.0  # 100% = 1.0 in decimal
    n_extreme = extreme_mask.sum()
    print(f"  Extreme returns (|ret| > 100%): {n_extreme}")
    wide.loc[extreme_mask, 'return'] = np.nan

    # ============================================================
    # 5. Filter preferred stocks (codes ending in 5/6/7)
    # ============================================================
    total_stocks_before = wide['code'].nunique()
    preferred_mask = wide['code'].str[-1].isin(PREFERRED_SUFFIXES)
    preferred_codes = wide.loc[preferred_mask, 'code'].unique()
    n_preferred = len(preferred_codes)
    print(f"\n  Total stocks before filtering: {total_stocks_before}")
    print(f"  Preferred stock codes (ending 5/6/7): {n_preferred}")
    if n_preferred > 0:
        print(f"    Sample preferred codes: {list(preferred_codes[:10])}")

    # Remove preferred stocks
    wide = wide[~preferred_mask].copy()

    # ============================================================
    # 6. Filter financial stocks (name contains financial patterns)
    # ============================================================
    stocks_after_pref = wide['code'].nunique()
    financial_mask = pd.Series(False, index=wide.index)
    for pattern in FINANCIAL_PATTERNS:
        financial_mask |= wide['name'].str.contains(pattern, na=False)

    financial_codes = wide.loc[financial_mask, 'code'].unique()
    n_financial = len(financial_codes)
    print(f"  Financial stock codes: {n_financial}")
    if n_financial > 0:
        print(f"    Sample financial codes: {list(financial_codes[:10])}")

    # Remove financial stocks
    wide = wide[~financial_mask].copy()

    # ============================================================
    # 7. Final stats
    # ============================================================
    remaining_stocks = wide['code'].nunique()
    print(f"\n  Stocks after preferred filter: {stocks_after_pref}")
    print(f"  Stocks after financial filter: {remaining_stocks}")
    print(f"  Total excluded: {total_stocks_before - remaining_stocks}")
    print(f"    - Preferred: {n_preferred}")
    print(f"    - Financial: {n_financial}")

    # ============================================================
    # 8. Select columns and save
    # ============================================================
    result = wide[['code', 'name', 'date', 'return', 'me', 'shrout']].copy()
    result = result.sort_values(['code', 'date']).reset_index(drop=True)

    print(f"\n  Final shape: {result.shape}")
    print(f"  Final stocks: {result.code.nunique()}")
    print(f"  Date range: {result.date.min()} ~ {result.date.max()}")

    # Return stats
    ret = result['return'].dropna()
    print(f"\n  Return stats:")
    print(f"    Valid returns: {len(ret)}")
    print(f"    Mean: {ret.mean()*100:.4f}%")
    print(f"    Std: {ret.std()*100:.4f}%")
    print(f"    Min: {ret.min()*100:.4f}%")
    print(f"    Max: {ret.max()*100:.4f}%")
    print(f"    NaN ratio: {result['return'].isna().sum() / len(result):.4f}")

    # KOSDAQ check
    kosdaq = [c for c in result['code'].unique() if c[1] != '0']
    print(f"    KOSDAQ stocks: {len(kosdaq)}")

    # Preferred check (should be 0)
    pref_check = [c for c in result['code'].unique() if c[-1] in PREFERRED_SUFFIXES]
    print(f"    Preferred in output (should be 0): {len(pref_check)}")

    # Financial check (should be 0)
    fin_check = 0
    for pattern in FINANCIAL_PATTERNS:
        fin_check += result['name'].str.contains(pattern, na=False).sum()
    print(f"    Financial rows in output (should be 0): {fin_check}")

    # Save
    output_path = os.path.join(DATA_DIR, 'stock_returns.parquet')
    result.to_parquet(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print("  DONE!")


if __name__ == '__main__':
    main()