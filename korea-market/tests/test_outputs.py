"""
test_outputs.py
Pattern-based tests for Korean FF3 v2 analysis outputs.
No hardcoded expected values - only structural sanity checks.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def factors_df():
    return pd.read_csv(os.path.join(DATA_DIR, 'factors_korea.csv'), parse_dates=['date'])

@pytest.fixture
def regression_df():
    return pd.read_csv(os.path.join(OUTPUT_DIR, 'regression_korea.csv'))

@pytest.fixture
def grs_df():
    return pd.read_csv(os.path.join(OUTPUT_DIR, 'grs_test.csv'))

@pytest.fixture
def summary_df():
    return pd.read_csv(os.path.join(OUTPUT_DIR, 'us_vs_korea_factor_summary.csv'), comment='#')

# ── File existence tests ────────────────────────────────────────────

def test_factors_csv_exists():
    assert os.path.exists(os.path.join(DATA_DIR, 'factors_korea.csv'))

def test_portfolios_6_exists():
    assert os.path.exists(os.path.join(DATA_DIR, 'portfolios_6_korea.csv'))

def test_portfolios_25_exists():
    assert os.path.exists(os.path.join(DATA_DIR, 'portfolios_25_korea.csv'))

def test_regression_exists():
    assert os.path.exists(os.path.join(OUTPUT_DIR, 'regression_korea.csv'))

def test_grs_exists():
    assert os.path.exists(os.path.join(OUTPUT_DIR, 'grs_test.csv'))

def test_summary_exists():
    assert os.path.exists(os.path.join(OUTPUT_DIR, 'us_vs_korea_factor_summary.csv'))

def test_charts_exist():
    charts = ['cumulative_returns.png', 'factor_premium_comparison.png', 'heatmap_25.png']
    for c in charts:
        assert os.path.exists(os.path.join(OUTPUT_DIR, 'charts', c)), f'Missing: {c}'

# ── Structural tests ───────────────────────────────────────────────

def test_factors_has_required_columns(factors_df):
    required = ['date', 'Mkt-RF', 'SMB', 'HML', 'rf']
    missing = [c for c in required if c not in factors_df.columns]
    assert not missing, f'Missing columns: {missing}'

def test_factors_311_months(factors_df):
    assert len(factors_df) == 311, f'Expected 311 months, got {len(factors_df)}'

def test_regression_25_rows(regression_df):
    assert len(regression_df) == 25, f'Expected 25 rows, got {len(regression_df)}'

def test_regression_has_beta_columns(regression_df):
    required = ['portfolio', 'alpha', 'beta_mkt', 'beta_smb', 'beta_hml', 'r_squared']
    missing = [c for c in required if c not in regression_df.columns]
    assert not missing, f'Missing columns: {missing}'

# ── Pattern-based sanity checks ────────────────────────────────────

def test_beta_mkt_all_positive(regression_df):
    """All market betas should be positive."""
    assert (regression_df['beta_mkt'] > 0).all(), 'Not all beta_mkt > 0'

def test_r_squared_in_valid_range(regression_df):
    """R-squared should be between 0 and 1."""
    assert ((regression_df['r_squared'] >= 0) & (regression_df['r_squared'] <= 1)).all()

def test_small_cap_positive_beta_smb(regression_df):
    """Small-cap portfolios (S1*) should have positive SMB beta on average."""
    small = regression_df[regression_df['portfolio'].str.startswith('S1')]
    assert small['beta_smb'].mean() > 0, 'Small-cap beta_smb not positive'

def test_high_bm_positive_beta_hml(regression_df):
    """High-BM portfolios (*B5) should have positive HML beta on average."""
    high = regression_df[regression_df['portfolio'].str.endswith('B5')]
    assert high['beta_hml'].mean() > 0, 'High-BM beta_hml not positive'

def test_grs_pval_valid(grs_df):
    """GRS p-value should be between 0 and 1."""
    pval = grs_df['p_value'].iloc[0]
    assert 0 <= pval <= 1, f'GRS p-value out of range: {pval}'

def test_grs_stat_positive(grs_df):
    """GRS statistic should be positive."""
    assert grs_df['grs_stat'].iloc[0] > 0

def test_summary_6_rows(summary_df):
    """Summary should have 6 rows (3 factors x 2 markets)."""
    assert len(summary_df) == 6, f'Expected 6 rows, got {len(summary_df)}'

def test_v1_v2_correlation_threshold():
    """v1 vs v2 Mkt-RF correlation should be > 0.90."""
    corr = pd.read_csv(os.path.join(OUTPUT_DIR, 'v1_vs_v2_comparison.csv'))
    mkt_corr = corr[corr['factor'] == 'Mkt-RF']['correlation'].iloc[0]
    assert mkt_corr > 0.90, f'Mkt-RF correlation {mkt_corr} below 0.90'

# ── KOSDAQ inclusion test ─────────────────────────────────────────

def test_kosdaq_included():
    """Panel data should include KOSDAQ stocks."""
    panel = pd.read_parquet(os.path.join(DATA_DIR, 'panel_data.parquet'))
    kosdaq = panel[panel['is_kosdaq'] == True]['code'].nunique()
    assert kosdaq > 100, f'Only {kosdaq} KOSDAQ stocks found'

# ── Preferred exclusion test ────────────────────────────────────────

def test_no_preferred_stocks():
    """No preferred stocks (ending in 5/6/7) should be in panel."""
    panel = pd.read_parquet(os.path.join(DATA_DIR, 'panel_data.parquet'))
    codes = panel['code'].unique()
    pref = [c for c in codes if c[-1] in ['5', '6', '7']]
    assert len(pref) == 0, f'Found {len(pref)} preferred stocks'
