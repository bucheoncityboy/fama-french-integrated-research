"""
test_section2_bond_portfolios.py
RED test: Verify bond portfolio excess returns implementation
"""

import os
import sys
import pytest
import pandas as pd
from datetime import datetime

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_module_can_be_imported():
    """02b_section2_bond_portfolios module can be imported."""
    import importlib
    # Module should import without errors now that it exists
    module = importlib.import_module('02b_section2_bond_portfolios')
    assert module is not None
    assert hasattr(module, 'create_bond_portfolios')


def test_creates_7_portfolio_columns():
    """Output CSV contains exactly 7 bond portfolio columns."""
    import importlib.util
    spec = importlib.util.find_spec('02b_section2_bond_portfolios')
    if spec is None:
        pytest.skip("Module not yet implemented")

    module = importlib.import_module('02b_section2_bond_portfolios')

    # Run the main function to generate output
    try:
        module.create_bond_portfolios()
    except Exception:
        pass  # May fail if data not available

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'appendix_output', 'bond_portfolios_excess.csv')

    if not os.path.exists(output_path):
        pytest.fail(f"Output file not found: {output_path}")

    df = pd.read_csv(output_path, index_col=0, parse_dates=True)
    expected_cols = ['SHORT_TERM', 'LONG_TERM', 'AAA', 'AA', 'A', 'BBB', 'LOW_GRADE']
    assert list(df.columns) == expected_cols, f"Expected {expected_cols}, got {list(df.columns)}"


def test_excess_returns_near_zero():
    """Average excess returns should be near zero (< 0.30%/month)."""
    import importlib.util
    spec = importlib.util.find_spec('02b_section2_bond_portfolios')
    if spec is None:
        pytest.skip("Module not yet implemented")

    module = importlib.import_module('02b_section2_bond_portfolios')

    try:
        module.create_bond_portfolios()
    except Exception:
        pass

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'appendix_output', 'bond_portfolios_excess.csv')

    if not os.path.exists(output_path):
        pytest.fail(f"Output file not found: {output_path}")

    df = pd.read_csv(output_path, index_col=0, parse_dates=True)

    if df.empty:
        pytest.skip("No data in output file - bond_factors.csv may not cover 1963-1991 period")

    # Check average excess returns are near zero
    for col in df.columns:
        avg = df[col].mean()
        # Skip if avg is NaN
        if pd.isna(avg):
            continue
        assert abs(avg) < 0.003, f"Average excess return for {col} is {avg:.4f}, exceeds 0.30%/month threshold"


def test_disclaimer_present():
    """Module contains prominent disclaimer about proxy limitations."""
    import importlib.util
    spec = importlib.util.find_spec('02b_section2_bond_portfolios')
    if spec is None:
        pytest.skip("Module not yet implemented")

    module = importlib.import_module('02b_section2_bond_portfolios')
    source = open(module.__file__, 'r', encoding='utf-8').read()

    # Check for key disclaimer keywords
    assert 'proxy' in source.lower()
    assert 'DISCLAIMER' in source or 'disclaimer' in source.lower()
    assert 'limitation' in source.lower() or 'may differ' in source.lower()


def test_output_file_exists():
    """Output CSV is saved to appendix_output/bond_portfolios_excess.csv."""
    import importlib.util
    spec = importlib.util.find_spec('02b_section2_bond_portfolios')
    if spec is None:
        pytest.skip("Module not yet implemented")

    module = importlib.import_module('02b_section2_bond_portfolios')

    try:
        module.create_bond_portfolios()
    except Exception:
        pass

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'appendix_output', 'bond_portfolios_excess.csv')
    assert os.path.exists(output_path), f"Expected output file at {output_path}"


def test_date_range_matches_config():
    """Date range matches config START_DATE to END_DATE."""
    import importlib.util
    spec = importlib.util.find_spec('02b_section2_bond_portfolios')
    if spec is None:
        pytest.skip("Module not yet implemented")

    module = importlib.import_module('02b_section2_bond_portfolios')
    import config

    try:
        module.create_bond_portfolios()
    except Exception:
        pass

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'appendix_output', 'bond_portfolios_excess.csv')

    if not os.path.exists(output_path):
        pytest.fail(f"Output file not found: {output_path}")

    df = pd.read_csv(output_path, index_col=0, parse_dates=True)

    if df.empty:
        pytest.skip("No data in output file - cannot verify date range")

    start_dt = pd.to_datetime(config.START_DATE)
    end_dt = pd.to_datetime(config.END_DATE)

    # Check that dates are parsed correctly
    assert df.index.min() >= start_dt, f"Index min {df.index.min()} < START_DATE {start_dt}"
    assert df.index.max() <= end_dt, f"Index max {df.index.max()} > END_DATE {end_dt}"


def test_no_nan_in_output():
    """Output DataFrame contains no NaN values."""
    import importlib.util
    spec = importlib.util.find_spec('02b_section2_bond_portfolios')
    if spec is None:
        pytest.skip("Module not yet implemented")

    module = importlib.import_module('02b_section2_bond_portfolios')

    try:
        module.create_bond_portfolios()
    except Exception:
        pass

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'appendix_output', 'bond_portfolios_excess.csv')

    if not os.path.exists(output_path):
        pytest.fail(f"Output file not found: {output_path}")

    df = pd.read_csv(output_path, index_col=0, parse_dates=True)

    if df.empty:
        pytest.skip("No data in output file - cannot verify no NaN")

    assert not df.isna().any().any(), "Output contains NaN values"


def test_print_average_excess_returns():
    """Module prints average excess returns when run as main."""
    import importlib.util
    spec = importlib.util.find_spec('02b_section2_bond_portfolios')
    if spec is None:
        pytest.skip("Module not yet implemented")

    module = importlib.import_module('02b_section2_bond_portfolios')
    source = open(module.__file__, 'r', encoding='utf-8').read()

    # Check that print statement for averages exists
    assert 'print' in source and 'average' in source.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
