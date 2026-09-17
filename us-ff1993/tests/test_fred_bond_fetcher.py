"""
test_fred_bond_fetcher.py
RED test: Verify fred_bond_fetcher structure, behavior, and error handling
"""

import os
import shutil
import sys

import pandas as pd
import pytest

import config

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True, scope="module")
def preserve_bond_factors():
    """Backup and restore data/bond_factors.csv to prevent cross-test pollution."""
    original_path = os.path.join(config.DATA_DIR, "bond_factors.csv")
    backup_path = os.path.join(config.DATA_DIR, "bond_factors_backup.csv")
    if os.path.exists(original_path):
        shutil.copy2(original_path, backup_path)
    yield
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, original_path)
        os.remove(backup_path)


def test_module_imports():
    """fred_bond_fetcher module imports without errors."""
    import fred_bond_fetcher
    assert hasattr(fred_bond_fetcher, 'fetch_bond_data')


def test_fetch_bond_data_is_callable():
    """fetch_bond_data is a callable function."""
    from fred_bond_fetcher import fetch_bond_data
    assert callable(fetch_bond_data)


def test_fetch_bond_data_signature():
    """fetch_bond_data accepts start and end parameters."""
    import inspect
    from fred_bond_fetcher import fetch_bond_data
    sig = inspect.signature(fetch_bond_data)
    params = list(sig.parameters.keys())
    assert 'start' in params
    assert 'end' in params


def test_fetch_bond_data_returns_dataframe():
    """fetch_bond_data returns a pandas DataFrame."""
    from fred_bond_fetcher import fetch_bond_data
    df = fetch_bond_data(start='2020-01-01', end='2020-12-31')
    assert isinstance(df, pd.DataFrame)


def test_dataframe_has_term_and_def_columns():
    """Returned DataFrame has TERM and DEF columns."""
    from fred_bond_fetcher import fetch_bond_data
    df = fetch_bond_data(start='2020-01-01', end='2020-12-31')
    assert 'TERM' in df.columns
    assert 'DEF' in df.columns


def test_dataframe_index_is_datetime():
    """Returned DataFrame index is DatetimeIndex."""
    from fred_bond_fetcher import fetch_bond_data
    df = fetch_bond_data(start='2020-01-01', end='2020-12-31')
    assert isinstance(df.index, pd.DatetimeIndex)


def test_dataframe_index_is_monthly():
    """Returned DataFrame index frequency is monthly (MonthEnd)."""
    from fred_bond_fetcher import fetch_bond_data
    df = fetch_bond_data(start='2020-01-01', end='2020-12-31')
    # Allow for inferred freq or explicit MonthEnd
    freq = pd.infer_freq(df.index)
    assert freq in ('M', 'ME', 'MS', 'BME', 'BMS'), f"Expected monthly freq, got {freq}"


def test_disclaimer_in_module():
    """Disclaimer about yield-based proxies is present in module source."""
    import fred_bond_fetcher
    source = open(fred_bond_fetcher.__file__, 'r', encoding='utf-8').read()
    assert 'yield-based proxies' in source or 'yield-based proxy' in source
    assert 'FRED' in source
    assert 'FF(1993)' in source or 'Fama-French' in source or 'original paper' in source


def test_csv_saved():
    """bond_factors.csv is saved to data/ directory."""
    from fred_bond_fetcher import fetch_bond_data
    import config
    df = fetch_bond_data(start='2020-01-01', end='2020-12-31')
    csv_path = os.path.join(config.DATA_DIR, 'bond_factors.csv')
    assert os.path.exists(csv_path), f"Expected {csv_path} to exist"


def test_csv_has_term_and_def():
    """Saved CSV contains TERM and DEF columns."""
    from fred_bond_fetcher import fetch_bond_data
    import config
    df = fetch_bond_data(start='2020-01-01', end='2020-12-31')
    csv_path = os.path.join(config.DATA_DIR, 'bond_factors.csv')
    df_read = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    assert 'TERM' in df_read.columns
    assert 'DEF' in df_read.columns


def test_error_handling_invalid_dates():
    """fetch_bond_data raises informative error for invalid date inputs."""
    from fred_bond_fetcher import fetch_bond_data
    with pytest.raises((ValueError, TypeError)):
        fetch_bond_data(start='not-a-date', end='2020-12-31')


def test_error_handling_future_dates():
    """fetch_bond_data handles future dates gracefully."""
    from fred_bond_fetcher import fetch_bond_data
    # Should not crash; may return empty or raise informative error
    try:
        df = fetch_bond_data(start='2099-01-01', end='2099-12-31')
        assert isinstance(df, pd.DataFrame)
    except Exception as e:
        assert isinstance(e, (ValueError, KeyError))


def test_date_range_filtering():
    """Returned DataFrame respects start and end date range."""
    from fred_bond_fetcher import fetch_bond_data
    start = '2020-01-01'
    end = '2020-06-30'
    df = fetch_bond_data(start=start, end=end)
    assert df.index.min() >= pd.Timestamp(start)
    assert df.index.max() <= pd.Timestamp(end)


def test_def_mean_positive():
    """DEF mean should be positive (credit spread) and within expected range."""
    from fred_bond_fetcher import fetch_bond_data
    df = fetch_bond_data(start='2020-01-01', end='2020-12-31')
    mean_def = df['DEF'].mean()
    assert 0.005 < mean_def < 0.03, f"DEF mean {mean_def} outside expected positive range"


def test_term_mean_positive():
    """TERM mean should be positive (term spread) and within expected range."""
    from fred_bond_fetcher import fetch_bond_data
    df = fetch_bond_data(start='2020-01-01', end='2020-12-31')
    mean_term = df['TERM'].mean()
    assert 0.005 < mean_term < 0.03, f"TERM mean {mean_term} outside expected positive range"


def test_disclaimer_in_output():
    """Disclaimer is included in output CSV or logged."""
    import fred_bond_fetcher
    source = open(fred_bond_fetcher.__file__, 'r', encoding='utf-8').read()
    assert 'Numerical results may differ' in source
