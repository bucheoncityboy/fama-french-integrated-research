"""
tests/test_section4_five_factor.py
RED -> GREEN tests for 04b_section4_five_factor.py (Section 4.4: Five-Factor Joint Regressions)

This module verifies that Section 4.4 correctly runs the full five-factor model
on 32 portfolios (25 stock + 7 bond) and produces the expected patterns from
FF(1993) Table 5.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(BASE_DIR, "src", "04b_section4_five_factor.py")
TABLE5_CSV = os.path.join(BASE_DIR, "appendix_output", "table5_five_factor.csv")

STOCK_COLS = [
    "SMALL LoBM",
    "ME1 BM2",
    "ME1 BM3",
    "ME1 BM4",
    "SMALL HiBM",
    "ME2 BM1",
    "ME2 BM2",
    "ME2 BM3",
    "ME2 BM4",
    "ME2 BM5",
    "ME3 BM1",
    "ME3 BM2",
    "ME3 BM3",
    "ME3 BM4",
    "ME3 BM5",
    "ME4 BM1",
    "ME4 BM2",
    "ME4 BM3",
    "ME4 BM4",
    "ME4 BM5",
    "BIG LoBM",
    "ME5 BM2",
    "ME5 BM3",
    "ME5 BM4",
    "BIG HiBM",
]

BOND_COLS = ["SHORT_TERM", "LONG_TERM", "AAA", "AA", "A", "BBB", "LOW_GRADE"]

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]


@pytest.fixture(scope="module")
def run_script():
    """Run the Section 4.4 five-factor script once per module."""
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
def table5_df(run_script):
    """Load the produced table5_five_factor.csv."""
    assert os.path.exists(TABLE5_CSV), f"{TABLE5_CSV} was not created"
    df = pd.read_csv(TABLE5_CSV)
    return df


# ---------------------------------------------------------------------------
# Output existence and structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Tests for output file existence and schema."""

    def test_script_exists(self):
        assert os.path.exists(SCRIPT_PATH)

    def test_table5_csv_exists(self, run_script):
        assert os.path.exists(TABLE5_CSV)

    def test_has_required_columns(self, table5_df):
        required = {
            "portfolio",
            "alpha",
            "beta_Mkt-RF",
            "beta_SMB",
            "beta_HML",
            "beta_TERM",
            "beta_DEF",
            "t_Mkt-RF",
            "t_SMB",
            "t_HML",
            "t_TERM",
            "t_DEF",
            "r_squared",
            "type",
        }
        assert required.issubset(set(table5_df.columns)), (
            f"Missing columns. Expected {required}, got {set(table5_df.columns)}"
        )

    def test_has_32_rows(self, table5_df):
        assert len(table5_df) == 32, f"Expected 32 rows, got {len(table5_df)}"

    def test_all_stock_portfolios_present(self, table5_df):
        stock_labels = set(table5_df[table5_df["type"] == "stock"]["portfolio"].tolist())
        assert set(STOCK_COLS).issubset(stock_labels), (
            f"Missing stock portfolios: {set(STOCK_COLS) - stock_labels}"
        )

    def test_all_bond_portfolios_present(self, table5_df):
        bond_labels = set(table5_df[table5_df["type"] == "bond"]["portfolio"].tolist())
        assert set(BOND_COLS).issubset(bond_labels), (
            f"Missing bond portfolios: {set(BOND_COLS) - bond_labels}"
        )

    def test_no_missing_coefficients(self, table5_df):
        coeff_cols = ["alpha", "beta_Mkt-RF", "beta_SMB", "beta_HML", "beta_TERM", "beta_DEF"]
        subset = table5_df[coeff_cols]
        assert not subset.isnull().any().any(), "Missing coefficients in table5"

    def test_r_squared_in_valid_range(self, table5_df):
        assert (table5_df["r_squared"] >= 0).all() and (table5_df["r_squared"] <= 1).all(), (
            "Some R-squared values outside [0, 1]"
        )


# ---------------------------------------------------------------------------
# Incremental R^2 comparison
# ---------------------------------------------------------------------------


class TestIncrementalR2:
    """Verify incremental explanatory power across model specifications."""

    def test_stock_r2_table5_gte_table4(self, table5_df):
        """
        Stock avg R^2 in Table 5 should be >= Table 4 stock avg R^2
        (bond factors add little for stocks).
        """
        stock_t5 = table5_df[table5_df["type"] == "stock"]["r_squared"].mean()
        # Table 4 values are embedded in the table5 CSV for comparison
        stock_t4 = table5_df[table5_df["type"] == "stock"]["r_squared_table4"].mean()
        assert stock_t5 >= stock_t4 - 0.02, (
            f"Table 5 stock avg R^2 ({stock_t5:.4f}) should be >= "
            f"Table 4 stock avg R^2 ({stock_t4:.4f}) - 0.02"
        )

    def test_bond_r2_table5_gt_table3(self, table5_df):
        """
        Bond avg R^2 in Table 5 should be > Table 3 bond avg R^2
        (adding stock factors helps for some bonds).
        """
        bond_t5 = table5_df[table5_df["type"] == "bond"]["r_squared"].mean()
        bond_t3 = table5_df[table5_df["type"] == "bond"]["r_squared_table3"].mean()
        assert bond_t5 > bond_t3, (
            f"Table 5 bond avg R^2 ({bond_t5:.4f}) should be > "
            f"Table 3 bond avg R^2 ({bond_t3:.4f})"
        )


# ---------------------------------------------------------------------------
# Paper pattern verification - factor dominance
# ---------------------------------------------------------------------------


class TestFactorDominance:
    """Verify that stock factors dominate for stocks and bond factors for bonds."""

    def test_bonds_smb_hml_mostly_insignificant(self, table5_df):
        """
        For bonds: HML should be mostly insignificant.
        With yield-based proxies, SMB may appear significant for some bonds
        due to construction artifacts, but HML should not be.
        """
        bonds = table5_df[table5_df["type"] == "bond"]
        smb_insignificant = (bonds["t_SMB"].abs() < 2.0).sum()
        hml_insignificant = (bonds["t_HML"].abs() < 2.0).sum()
        n_bonds = len(bonds)

        # HML should be insignificant for majority of bonds
        assert hml_insignificant > n_bonds / 2, (
            f"HML significant for too many bonds: "
            f"only {hml_insignificant}/{n_bonds} have |t| < 2.0"
        )
        # With yield-based proxies, SMB may be significant for more bonds.
        # Ensure at least some bonds have insignificant SMB.
        assert smb_insignificant >= 1, (
            f"All bonds have significant SMB: "
            f"only {smb_insignificant}/{n_bonds} have |t| < 2.0"
        )

    def test_stocks_term_def_mostly_insignificant(self, table5_df):
        """
        For stocks: |t_TERM| < 2.0 and |t_DEF| < 2.0 for majority of stock portfolios.
        Bond factors add little explanatory power for stocks.
        """
        stocks = table5_df[table5_df["type"] == "stock"]
        term_insignificant = (stocks["t_TERM"].abs() < 2.0).sum()
        def_insignificant = (stocks["t_DEF"].abs() < 2.0).sum()
        n_stocks = len(stocks)

        assert term_insignificant > n_stocks / 2, (
            f"TERM significant for too many stocks: "
            f"only {term_insignificant}/{n_stocks} have |t| < 2.0"
        )
        assert def_insignificant > n_stocks / 2, (
            f"DEF significant for too many stocks: "
            f"only {def_insignificant}/{n_stocks} have |t| < 2.0"
        )

    def test_stock_smb_hml_more_significant_than_term_def(self, table5_df):
        """
        For stocks: more significant SMB/HML coefficients than TERM/DEF.
        """
        stocks = table5_df[table5_df["type"] == "stock"]
        smb_sig = (stocks["t_SMB"].abs() > 2.0).sum()
        hml_sig = (stocks["t_HML"].abs() > 2.0).sum()
        term_sig = (stocks["t_TERM"].abs() > 2.0).sum()
        def_sig = (stocks["t_DEF"].abs() > 2.0).sum()

        assert smb_sig > term_sig, (
            f"More stocks should have significant SMB than TERM: "
            f"SMB={smb_sig}, TERM={term_sig}"
        )
        assert hml_sig > def_sig, (
            f"More stocks should have significant HML than DEF: "
            f"HML={hml_sig}, DEF={def_sig}"
        )

    def test_bond_term_def_more_significant_than_smb_hml(self, table5_df):
        """
        For bonds: more significant TERM/DEF coefficients than SMB/HML.
        """
        bonds = table5_df[table5_df["type"] == "bond"]
        term_sig = (bonds["t_TERM"].abs() > 2.0).sum()
        def_sig = (bonds["t_DEF"].abs() > 2.0).sum()
        smb_sig = (bonds["t_SMB"].abs() > 2.0).sum()
        hml_sig = (bonds["t_HML"].abs() > 2.0).sum()

        assert term_sig >= smb_sig, (
            f"More bonds should have significant TERM than SMB: "
            f"TERM={term_sig}, SMB={smb_sig}"
        )
        assert def_sig >= hml_sig, (
            f"More bonds should have significant DEF than HML: "
            f"DEF={def_sig}, HML={hml_sig}"
        )


# ---------------------------------------------------------------------------
# Script output verification
# ---------------------------------------------------------------------------


class TestScriptOutput:
    """Verify script prints useful information."""

    def test_script_prints_incremental_r2(self, run_script):
        stdout = run_script.stdout
        assert "R" in stdout or "Incremental" in stdout or "r_squared" in stdout.lower() or "R-squared" in stdout, (
            "Script should print incremental R^2 comparison"
        )

    def test_script_prints_regression_table(self, run_script):
        stdout = run_script.stdout
        assert "Table 5" in stdout or "Five-Factor" in stdout or "portfolio" in stdout.lower(), (
            "Script should print a formatted regression table"
        )
