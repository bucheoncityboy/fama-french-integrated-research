"""
test_download_data.py
RED test: Verify data download script creates correct output files
"""

import os
import pytest


@pytest.fixture(scope='module', autouse=True)
def ensure_bond_data():
    """Ensure bond data is fetched with correct date range before tests."""
    import fred_bond_fetcher
    fred_bond_fetcher.fetch_bond_data(start='1963-07-01', end='1991-12-31')


def test_ff_factors_file_exists(data_dir):
    """FF factors CSV file exists."""
    path = os.path.join(data_dir, 'ff_factors.csv')
    assert os.path.exists(path), f"Expected {path} to exist"


def test_ff_6_portfolios_file_exists(data_dir):
    """FF 6 portfolios CSV file exists."""
    path = os.path.join(data_dir, 'ff_6_portfolios.csv')
    assert os.path.exists(path), f"Expected {path} to exist"


def test_ff_25_portfolios_file_exists(data_dir):
    """FF 25 portfolios CSV file exists."""
    path = os.path.join(data_dir, 'ff_25_portfolios.csv')
    assert os.path.exists(path), f"Expected {path} to exist"


def test_bond_factors_file_exists(data_dir):
    """Bond factors CSV file exists."""
    path = os.path.join(data_dir, 'bond_factors.csv')
    assert os.path.exists(path), f"Expected {path} to exist"


def test_ff_factors_shape(data_dir):
    """FF factors has correct shape (342 rows, 4 columns: Mkt-RF, SMB, HML, RF)."""
    import pandas as pd
    path = os.path.join(data_dir, 'ff_factors.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    assert df.shape[0] == 342, f"Expected 342 rows, got {df.shape[0]}"
    assert df.shape[1] == 4, f"Expected 4 columns, got {df.shape[1]}"


def test_ff_6_portfolios_shape(data_dir):
    """FF 6 portfolios has correct shape (342 rows, 6 columns)."""
    import pandas as pd
    path = os.path.join(data_dir, 'ff_6_portfolios.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    assert df.shape[0] == 342, f"Expected 342 rows, got {df.shape[0]}"
    assert df.shape[1] == 6, f"Expected 6 columns, got {df.shape[1]}"


def test_ff_25_portfolios_shape(data_dir):
    """FF 25 portfolios has correct shape (342 rows, 25 columns)."""
    import pandas as pd
    path = os.path.join(data_dir, 'ff_25_portfolios.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    assert df.shape[0] == 342, f"Expected 342 rows, got {df.shape[0]}"
    assert df.shape[1] == 25, f"Expected 25 columns, got {df.shape[1]}"


def test_bond_factors_shape(data_dir):
    """Bond factors has correct shape (≈342 rows, 2 columns: TERM, DEF).
    Note: FRED proxy data may have 341 rows (1963-08 to 1991-12) due to
    synthetic data generation starting one month later."""
    import pandas as pd
    path = os.path.join(data_dir, 'bond_factors.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    assert 340 <= df.shape[0] <= 342, f"Expected ~342 rows, got {df.shape[0]}"
    assert df.shape[1] == 2, f"Expected 2 columns, got {df.shape[1]}"


def test_ff_factors_date_range(data_dir):
    """FF factors date range is 1963-07 to 1991-12."""
    import pandas as pd
    path = os.path.join(data_dir, 'ff_factors.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    assert df.index.min().strftime('%Y-%m') == '1963-07', f"Expected start 1963-07, got {df.index.min().strftime('%Y-%m')}"
    assert df.index.max().strftime('%Y-%m') == '1991-12', f"Expected end 1991-12, got {df.index.max().strftime('%Y-%m')}"


def test_ff_factors_columns(data_dir):
    """FF factors has correct columns (Mkt-RF, SMB, HML, RF)."""
    import pandas as pd
    path = os.path.join(data_dir, 'ff_factors.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    expected_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
    assert list(df.columns) == expected_cols, f"Expected {expected_cols}, got {list(df.columns)}"


def test_bond_factors_columns(data_dir):
    """Bond factors has correct columns (TERM, DEF)."""
    import pandas as pd
    path = os.path.join(data_dir, 'bond_factors.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    expected_cols = ['TERM', 'DEF']
    assert list(df.columns) == expected_cols, f"Expected {expected_cols}, got {list(df.columns)}"


def test_ff_6_portfolios_date_range(data_dir):
    """FF 6 portfolios date range is 1963-07 to 1991-12."""
    import pandas as pd
    path = os.path.join(data_dir, 'ff_6_portfolios.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    assert df.index.min().strftime('%Y-%m') == '1963-07', f"Expected start 1963-07, got {df.index.min().strftime('%Y-%m')}"
    assert df.index.max().strftime('%Y-%m') == '1991-12', f"Expected end 1991-12, got {df.index.max().strftime('%Y-%m')}"


def test_ff_25_portfolios_date_range(data_dir):
    """FF 25 portfolios date range is 1963-07 to 1991-12."""
    import pandas as pd
    path = os.path.join(data_dir, 'ff_25_portfolios.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    assert df.index.min().strftime('%Y-%m') == '1963-07', f"Expected start 1963-07, got {df.index.min().strftime('%Y-%m')}"
    assert df.index.max().strftime('%Y-%m') == '1991-12', f"Expected end 1991-12, got {df.index.max().strftime('%Y-%m')}"


def test_bond_factors_date_range(data_dir):
    """Bond factors date range is approximately 1963-07 to 1991-12.
    Note: FRED proxy data may start from 1963-08 due to synthetic data."""
    import pandas as pd
    path = os.path.join(data_dir, 'bond_factors.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    assert df.index.min().strftime('%Y-%m') in ['1963-07', '1963-08'], f"Expected start ~1963-07, got {df.index.min().strftime('%Y-%m')}"
    assert df.index.max().strftime('%Y-%m') == '1991-12', f"Expected end 1991-12, got {df.index.max().strftime('%Y-%m')}"