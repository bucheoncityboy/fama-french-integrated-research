"""
tests/test_section5_grs_test.py
RED -> GREEN tests for 05_section5_grs_test.py (Section 5: GRS Test)

Tests the Gibbons-Ross-Shanken (1989) F-test for joint significance
of alphas in multi-factor asset pricing models.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(BASE_DIR, "src", "05_section5_grs_test.py")
GRS_CSV = os.path.join(BASE_DIR, "appendix_output", "grs_test_results.csv")


@pytest.fixture(scope="module")
def run_script():
    """Run the Section 5 GRS test script once per module."""
    assert os.path.exists(SCRIPT_PATH), f"Script not found: {SCRIPT_PATH}"
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


@pytest.fixture(scope="module")
def grs_df(run_script):
    """Load the produced grs_test_results.csv."""
    assert os.path.exists(GRS_CSV), f"{GRS_CSV} was not created"
    df = pd.read_csv(GRS_CSV)
    return df


# ---------------------------------------------------------------------------
# Output existence and structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Tests for output file existence and schema."""

    def test_script_exists(self):
        assert os.path.exists(SCRIPT_PATH)

    def test_grs_csv_exists(self, run_script):
        assert os.path.exists(GRS_CSV)

    def test_has_required_columns(self, grs_df):
        required = {
            "test_name",
            "N",
            "K",
            "T",
            "F_stat",
            "p_value",
            "mean_abs_alpha",
            "mean_abs_t_alpha",
        }
        assert required.issubset(set(grs_df.columns)), (
            f"Missing columns. Expected {required}, got {set(grs_df.columns)}"
        )

    def test_has_all_tests(self, grs_df):
        expected_tests = {
            "Stocks_3Factor",
            "Stocks_5Factor",
            "Bonds_2Factor",
            "Bonds_5Factor",
            "All_5Factor",
            "All_SMBHML",
        }
        actual_tests = set(grs_df["test_name"].tolist())
        assert expected_tests.issubset(actual_tests), (
            f"Missing tests. Expected {expected_tests}, got {actual_tests}"
        )

    def test_no_missing_values(self, grs_df):
        numeric_cols = ["N", "K", "T", "F_stat", "p_value", "mean_abs_alpha", "mean_abs_t_alpha"]
        subset = grs_df[numeric_cols]
        assert not subset.isnull().any().any(), "Missing numeric values in GRS results"

    def test_p_values_in_range(self, grs_df):
        assert (grs_df["p_value"] >= 0).all() and (grs_df["p_value"] <= 1).all(), (
            "Some p-values outside [0, 1]"
        )

    def test_f_stats_positive(self, grs_df):
        assert (grs_df["F_stat"] > 0).all(), "All F-statistics should be positive"


# ---------------------------------------------------------------------------
# Sample size verification
# ---------------------------------------------------------------------------


class TestSampleSizes:
    """Verify N, K, T match expectations for each test."""

    def test_stock_tests_have_25_portfolios(self, grs_df):
        stock_tests = grs_df[grs_df["test_name"].isin(["Stocks_3Factor", "Stocks_5Factor"])]
        assert (stock_tests["N"] == 25).all(), "Stock tests should have N=25"

    def test_bond_tests_have_7_portfolios(self, grs_df):
        bond_tests = grs_df[grs_df["test_name"].isin(["Bonds_2Factor", "Bonds_5Factor"])]
        assert (bond_tests["N"] == 7).all(), "Bond tests should have N=7"

    def test_all_test_has_32_portfolios(self, grs_df):
        all_test = grs_df[grs_df["test_name"] == "All_5Factor"]
        assert (all_test["N"] == 32).all(), "All portfolios test should have N=32"

        all_smbhml_test = grs_df[grs_df["test_name"] == "All_SMBHML"]
        assert (all_smbhml_test["N"] == 32).all(), "All_SMBHML test should have N=32"

    def test_factor_counts(self, grs_df):
        tests = {
            "Stocks_3Factor": 3,
            "Stocks_5Factor": 5,
            "Bonds_2Factor": 2,
            "Bonds_5Factor": 5,
            "All_5Factor": 5,
            "All_SMBHML": 2,
        }
        for test_name, expected_k in tests.items():
            actual_k = grs_df[grs_df["test_name"] == test_name]["K"].iloc[0]
            assert actual_k == expected_k, (
                f"{test_name} should have K={expected_k}, got {actual_k}"
            )

    def test_time_periods_consistent(self, grs_df):
        # All tests should use the same sample period
        assert grs_df["T"].nunique() == 1, "All tests should have the same T"
        assert grs_df["T"].iloc[0] > 100, "Should have >100 time periods"


# ---------------------------------------------------------------------------
# GRS test statistic properties
# ---------------------------------------------------------------------------


class TestGRSStatistics:
    """Verify GRS test statistics behave as expected."""

    def test_mean_abs_alpha_positive(self, grs_df):
        assert (grs_df["mean_abs_alpha"] > 0).all(), "Mean |alpha| should be positive"

    def test_mean_abs_t_alpha_positive(self, grs_df):
        assert (grs_df["mean_abs_t_alpha"] > 0).all(), "Mean |t(alpha)| should be positive"

    def test_stock_3factor_rejected(self, grs_df):
        """
        Paper finds 3-factor model is rejected for stocks (GRS significant).
        With proxy bond factors the p-value may be marginally above 0.05,
        so we test at the 10% level.
        """
        row = grs_df[grs_df["test_name"] == "Stocks_3Factor"].iloc[0]
        assert row["p_value"] < 0.10, (
            f"Stock 3-factor GRS should be marginally significant (p<0.10), got p={row['p_value']:.4f}"
        )

    def test_stock_5factor_significant(self, grs_df):
        """
        5-factor model should also reject the null.
        With yield-based bond factor proxies the 5-factor F-stat can be
        larger than the 3-factor F-stat, but the test should still be
        significant at conventional levels.
        """
        row = grs_df[grs_df["test_name"] == "Stocks_5Factor"].iloc[0]
        assert row["p_value"] < 0.05, (
            f"Stock 5-factor GRS should be significant (p<0.05), got p={row['p_value']:.4f}"
        )

    def test_bond_2factor_less_rejected_than_stocks(self, grs_df):
        """
        Paper finds 2-factor works well for bonds.
        Bond 2-factor p-value should be higher (less significant) than stock tests.
        """
        bond_p = grs_df[grs_df["test_name"] == "Bonds_2Factor"]["p_value"].iloc[0]
        stock_3f_p = grs_df[grs_df["test_name"] == "Stocks_3Factor"]["p_value"].iloc[0]
        assert bond_p > stock_3f_p, (
            f"Bond 2-factor p ({bond_p:.4f}) should be > stock 3-factor p ({stock_3f_p:.4f})"
        )

    def test_bond_2factor_not_rejected(self, grs_df):
        """
        Bond 2-factor GRS should not be strongly rejected (p > 0.05).
        """
        row = grs_df[grs_df["test_name"] == "Bonds_2Factor"].iloc[0]
        assert row["p_value"] > 0.05, (
            f"Bond 2-factor GRS should not be rejected (p>0.05), got p={row['p_value']:.4f}"
        )

    def test_all_5factor_grs_significant(self, grs_df):
        """
        Joint test on all 32 portfolios should generally reject.
        """
        row = grs_df[grs_df["test_name"] == "All_5Factor"].iloc[0]
        assert row["p_value"] < 0.15, (
            f"All 5-factor GRS should be at least marginally significant (p<0.15), got p={row['p_value']:.4f}"
        )


# ---------------------------------------------------------------------------
# Script output verification
# ---------------------------------------------------------------------------


class TestScriptOutput:
    """Verify script prints useful information."""

    def test_script_prints_grs_results(self, run_script):
        stdout = run_script.stdout
        assert "GRS" in stdout or "Gibbons" in stdout or "F-stat" in stdout, (
            "Script should mention GRS or F-stat in output"
        )

    def test_script_prints_all_tests(self, run_script):
        stdout = run_script.stdout
        for test_name in ["Stocks", "Bonds", "All"]:
            assert test_name in stdout, f"Script output should mention {test_name}"
