"""
test_section2_stock_portfolios.py
RED test: Verify Section 2.2 stock portfolios excess returns
"""

import pytest
import pandas as pd
import os


def test_output_file_exists(output_dir):
    """Output file stock_portfolios_excess.csv exists."""
    path = os.path.join(output_dir, 'stock_portfolios_excess.csv')
    assert os.path.exists(path), f"Expected output file at {path}"


def test_output_shape(output_dir):
    """Output has 25 portfolio columns and 342 rows."""
    path = os.path.join(output_dir, 'stock_portfolios_excess.csv')
    df = pd.read_csv(path, index_col='Date')
    assert df.shape == (342, 25), f"Expected (342, 25), got {df.shape}"


def test_date_range(output_dir):
    """Output covers 1963-07 to 1991-12."""
    path = os.path.join(output_dir, 'stock_portfolios_excess.csv')
    df = pd.read_csv(path, index_col='Date')
    first_date = str(df.index[0])[:7]  # '1963-07-01' -> '1963-07'
    last_date = str(df.index[-1])[:7]
    assert first_date == '1963-07', f"First date should be 1963-07, got {first_date}"
    assert last_date == '1991-12', f"Last date should be 1991-12, got {last_date}"


def test_column_count(output_dir):
    """Output has exactly 25 portfolio columns."""
    path = os.path.join(output_dir, 'stock_portfolios_excess.csv')
    df = pd.read_csv(path, index_col='Date')
    assert len(df.columns) == 25, f"Expected 25 portfolios, got {len(df.columns)}"


def test_has_small_loBM_and_big_hiBM(output_dir):
    """Output has SMALL LoBM and BIG HiBM columns."""
    path = os.path.join(output_dir, 'stock_portfolios_excess.csv')
    df = pd.read_csv(path, index_col='Date')
    assert 'SMALL LoBM' in df.columns, "Missing SMALL LoBM column"
    assert 'BIG HiBM' in df.columns, "Missing BIG HiBM column"


def test_size_pattern_small_gt_big(output_dir):
    """Average excess returns show size effect: SMALL > BIG.
    Compare ALL small stocks (avg across BM quintiles) vs ALL big stocks."""
    path = os.path.join(output_dir, 'stock_portfolios_excess.csv')
    df = pd.read_csv(path, index_col='Date')

    # Size effect: small caps outperform big caps on average across BM groups
    small_cols = [c for c in df.columns if c.startswith('SMALL')]
    big_cols = [c for c in df.columns if c.startswith('BIG')]

    avg_small = df[small_cols].mean().mean()
    avg_big = df[big_cols].mean().mean()

    assert avg_small > avg_big, \
        f"Size effect violated: SMALL ({avg_small:.4f}) should be > BIG ({avg_big:.4f})"


def test_value_pattern_hiBM_gt_loBM(output_dir):
    """Average excess returns show value effect: HiBM > LoBM."""
    path = os.path.join(output_dir, 'stock_portfolios_excess.csv')
    df = pd.read_csv(path, index_col='Date')

    # Compare BIG HiBM vs BIG LoBM
    avg_hiBM = df['BIG HiBM'].mean()
    avg_loBM = df['BIG LoBM'].mean()

    assert avg_hiBM > avg_loBM, \
        f"Value effect violated: HiBM ({avg_hiBM:.4f}) should be > LoBM ({avg_loBM:.4f})"


def test_all_values_numeric(output_dir):
    """All excess return values are numeric."""
    path = os.path.join(output_dir, 'stock_portfolios_excess.csv')
    df = pd.read_csv(path, index_col='Date')
    assert df.apply(pd.api.types.is_numeric_dtype).all(), \
        "All columns should be numeric"


def test_no_missing_values(output_dir):
    """No missing values in the output."""
    path = os.path.join(output_dir, 'stock_portfolios_excess.csv')
    df = pd.read_csv(path, index_col='Date')
    assert not df.isnull().any().any(), "Output should have no missing values"