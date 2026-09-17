"""
tests/test_section2_factors.py
RED / GREEN tests for Section 2.1 factor construction
"""

import os
import subprocess
import sys

import pandas as pd
import pytest


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(BASE_DIR, "src", "01_section2_factors.py")
FACTORS_CSV = os.path.join(BASE_DIR, 'appendix_output', 'factors.csv')


@pytest.fixture(scope='module')
def run_script():
    """Run the factor construction script once per module."""
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH],
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )
    assert result.returncode == 0, (
        f"Script failed with return code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return result


@pytest.fixture(scope='module')
def factors_df(run_script):
    """Load the produced factors.csv."""
    assert os.path.exists(FACTORS_CSV), f"{FACTORS_CSV} was not created"
    df = pd.read_csv(FACTORS_CSV)
    return df


@pytest.fixture(scope='module')
def precomputed_factors():
    """Load Ken French pre-computed factors for validation."""
    path = os.path.join(BASE_DIR, 'data', 'ff_factors.csv')
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


class TestFactorConstruction:
    """Tests for 01_section2_factors.py output."""

    def test_factors_csv_exists(self, run_script):
        assert os.path.exists(FACTORS_CSV)

    def test_factors_columns(self, factors_df):
        expected = {'Date', 'Mkt-RF', 'SMB', 'HML', 'RF'}
        assert set(factors_df.columns) == expected, (
            f"Expected columns {expected}, got {set(factors_df.columns)}"
        )

    def test_row_count(self, factors_df):
        # 1963-07 to 1991-12 inclusive = 342 months
        assert len(factors_df) == 342, f"Expected 342 rows, got {len(factors_df)}"

    def test_date_range(self, factors_df):
        factors_df['Date'] = pd.to_datetime(factors_df['Date'])
        assert factors_df['Date'].min().strftime('%Y-%m') == '1963-07'
        assert factors_df['Date'].max().strftime('%Y-%m') == '1991-12'

    def test_smb_deviation(self, factors_df, precomputed_factors):
        factors_df['Date'] = pd.to_datetime(factors_df['Date'])
        merged = pd.merge(factors_df, precomputed_factors, on='Date', suffixes=('', '_kf'))
        mad = (merged['SMB'] - merged['SMB_kf']).abs().mean()
        assert mad < 0.05, f"SMB mean absolute deviation = {mad:.4f}% (threshold 0.05%)"

    def test_hml_deviation(self, factors_df, precomputed_factors):
        factors_df['Date'] = pd.to_datetime(factors_df['Date'])
        merged = pd.merge(factors_df, precomputed_factors, on='Date', suffixes=('', '_kf'))
        mad = (merged['HML'] - merged['HML_kf']).abs().mean()
        assert mad < 0.05, f"HML mean absolute deviation = {mad:.4f}% (threshold 0.05%)"

    def test_factor_means(self, factors_df):
        means = factors_df[['Mkt-RF', 'SMB', 'HML']].mean()
        assert means['Mkt-RF'] == pytest.approx(0.43, abs=0.05)
        assert means['SMB'] == pytest.approx(0.27, abs=0.05)
        assert means['HML'] == pytest.approx(0.40, abs=0.05)

    def test_script_prints_summary(self, run_script):
        assert 'mean' in run_script.stdout.lower() or 'std' in run_script.stdout.lower()
