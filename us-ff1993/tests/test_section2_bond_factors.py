"""
test_section2_bond_factors.py
RED test: Verify Section 2.1 bond factor construction, merging, and output
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_script():
    """
    Run 01b_section2_bond_factors.py once and return the output path.
    This fixture executes the script as a module import side-effect.
    """
    script_path = os.path.join(config.BASE_DIR, "src", "01b_section2_bond_factors.py")
    if not os.path.exists(script_path):
        pytest.fail(f"Script not found: {script_path}")

    # Execute in a clean namespace and run main()
    import importlib.util
    spec = importlib.util.spec_from_file_location("section2_bond_factors", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

    output_path = os.path.join(config.OUTPUT_DIR, "factors.csv")
    return output_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_script_exists():
    """01b_section2_bond_factors.py exists in project root."""
    script_path = os.path.join(config.BASE_DIR, "src", "01b_section2_bond_factors.py")
    assert os.path.exists(script_path), f"Expected {script_path} to exist"


def test_output_csv_created(run_script):
    """Running the script creates output/factors.csv."""
    assert os.path.exists(run_script), f"Expected {run_script} to be created"


def _read_output_csv(path: str) -> pd.DataFrame:
    """Helper to read output CSV, skipping comment lines."""
    return pd.read_csv(path, index_col=0, parse_dates=True, comment="#")


def test_output_csv_readable(run_script):
    """output/factors.csv is a valid CSV with a Date column/index."""
    df = _read_output_csv(run_script)
    assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
    assert len(df) > 0, "DataFrame should not be empty"


def test_bond_factors_present(run_script):
    """Output CSV contains TERM and DEF columns."""
    df = _read_output_csv(run_script)
    assert "TERM" in df.columns, "TERM column missing"
    assert "DEF" in df.columns, "DEF column missing"


def test_rf_present(run_script):
    """Output CSV contains RF (risk-free rate) column."""
    df = _read_output_csv(run_script)
    assert "RF" in df.columns, "RF column missing"


def test_stock_factors_or_note(run_script):
    """
    If stock factors (Mkt-RF, SMB, HML) are present, all three must exist.
    If not, a note column indicating stock factors will be added later is acceptable.
    """
    df = _read_output_csv(run_script)
    stock_cols = {"Mkt-RF", "SMB", "HML"}
    has_all_stock = stock_cols.issubset(df.columns)
    has_any_stock = bool(stock_cols & set(df.columns))

    if has_any_stock and not has_all_stock:
        pytest.fail("Partial stock factors found (some but not all of Mkt-RF, SMB, HML)")

    if not has_all_stock:
        # Allow a placeholder note column when stock factors are missing
        note_cols = [c for c in df.columns if "stock" in c.lower() or "note" in c.lower()]
        assert note_cols, (
            "Stock factors missing but no note/placeholder column found. "
            "Expected a note indicating stock factors will be added later."
        )


def test_disclaimer_in_source():
    """Disclaimer about yield-based proxy limitations is present in module source."""
    script_path = os.path.join(config.BASE_DIR, "src", "01b_section2_bond_factors.py")
    source = open(script_path, "r", encoding="utf-8").read()
    assert "yield-based" in source.lower(), "Missing yield-based disclaimer"
    assert "proxy" in source.lower(), "Missing proxy disclaimer"


def test_disclaimer_in_output(run_script):
    """
    Disclaimer is present in output CSV (as a comment/header line) or in logged output.
    We check by inspecting the raw file for a comment line or a DISCLAIMER column.
    """
    # Check first few lines for a comment
    with open(run_script, "r", encoding="utf-8") as f:
        first_lines = [f.readline() for _ in range(5)]
    combined = "".join(first_lines).lower()
    has_comment_disclaimer = "disclaimer" in combined or "yield-based" in combined or "#" in combined

    df = _read_output_csv(run_script)
    has_disclaimer_col = any("disclaimer" in c.lower() for c in df.columns)

    assert has_comment_disclaimer or has_disclaimer_col, (
        "Disclaimer not found in output CSV (expected comment line or DISCLAIMER column)"
    )


def test_summary_statistics_printed(capsys):
    """Script prints bond factor summary statistics (mean, std, t-stat)."""
    script_path = os.path.join(config.BASE_DIR, "src", "01b_section2_bond_factors.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("section2_bond_factors_test", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

    captured = capsys.readouterr()
    out = captured.out + captured.err
    out_lower = out.lower()

    assert "mean" in out_lower, "Summary output missing 'mean'"
    assert "std" in out_lower or "standard deviation" in out_lower, "Summary output missing 'std'"
    assert "t-stat" in out_lower or "tstat" in out_lower or "t statistic" in out_lower, (
        "Summary output missing t-statistic"
    )
    assert "term" in out_lower, "Summary output missing TERM"
    assert "def" in out_lower, "Summary output missing DEF"


def test_term_mean_approximate(run_script):
    """
    TERM mean should be positive (~1.25% annualized term spread) for the
    full-sample 1963-1991 period with yield-based proxies.
    """
    df = _read_output_csv(run_script)
    if "TERM" not in df.columns:
        pytest.skip("TERM not in output")
    mean_term = df["TERM"].mean()
    assert 0.005 < mean_term < 0.03, f"TERM mean {mean_term} outside expected positive range"


def test_def_mean_approximate(run_script):
    """
    DEF mean should be positive (~1.1% annualized credit spread) for the
    full-sample 1963-1991 period with yield-based proxies.
    """
    df = _read_output_csv(run_script)
    if "DEF" not in df.columns:
        pytest.skip("DEF not in output")
    mean_def = df["DEF"].mean()
    assert 0.005 < mean_def < 0.03, f"DEF mean {mean_def} outside expected positive range"
