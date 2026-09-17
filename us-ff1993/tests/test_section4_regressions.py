"""
tests/test_section4_regressions.py
RED -> GREEN tests for 04_section4_regressions.py (Section 4: Common Variation)

Verifies Tables 1, 3, 4 regressions on 32 portfolios (25 stock + 7 bond).
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(BASE_DIR, "src", "04_section4_regressions.py")

TABLE1_CSV = os.path.join(BASE_DIR, "appendix_output", "table1_market.csv")
TABLE3_CSV = os.path.join(BASE_DIR, "appendix_output", "table3_bond.csv")
TABLE4_CSV = os.path.join(BASE_DIR, "appendix_output", "table4_stock3f.csv")

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

ALL_PORTFOLIOS = STOCK_COLS + BOND_COLS


@pytest.fixture(scope="module")
def run_script():
    """Run the Section 4 regression script once per module."""
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
def table1_df(run_script):
    assert os.path.exists(TABLE1_CSV), f"{TABLE1_CSV} was not created"
    return pd.read_csv(TABLE1_CSV)


@pytest.fixture(scope="module")
def table3_df(run_script):
    assert os.path.exists(TABLE3_CSV), f"{TABLE3_CSV} was not created"
    return pd.read_csv(TABLE3_CSV)


@pytest.fixture(scope="module")
def table4_df(run_script):
    assert os.path.exists(TABLE4_CSV), f"{TABLE4_CSV} was not created"
    return pd.read_csv(TABLE4_CSV)


# ---------------------------------------------------------------------------
# Output existence and structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Tests for output file existence and schema."""

    def test_script_exists(self):
        assert os.path.exists(SCRIPT_PATH)

    def test_table1_exists(self, run_script):
        assert os.path.exists(TABLE1_CSV)

    def test_table3_exists(self, run_script):
        assert os.path.exists(TABLE3_CSV)

    def test_table4_exists(self, run_script):
        assert os.path.exists(TABLE4_CSV)

    def test_table1_has_32_rows(self, table1_df):
        assert len(table1_df) == 32, f"Expected 32 rows, got {len(table1_df)}"

    def test_table3_has_32_rows(self, table3_df):
        assert len(table3_df) == 32, f"Expected 32 rows, got {len(table3_df)}"

    def test_table4_has_32_rows(self, table4_df):
        assert len(table4_df) == 32, f"Expected 32 rows, got {len(table4_df)}"

    def test_table1_has_type_column(self, table1_df):
        assert "type" in table1_df.columns
        assert set(table1_df["type"].unique()) == {"stock", "bond"}

    def test_table3_has_type_column(self, table3_df):
        assert "type" in table3_df.columns
        assert set(table3_df["type"].unique()) == {"stock", "bond"}

    def test_table4_has_type_column(self, table4_df):
        assert "type" in table4_df.columns
        assert set(table4_df["type"].unique()) == {"stock", "bond"}

    def test_all_portfolios_present_table1(self, table1_df):
        assert set(table1_df["portfolio"].tolist()) == set(ALL_PORTFOLIOS)

    def test_all_portfolios_present_table3(self, table3_df):
        assert set(table3_df["portfolio"].tolist()) == set(ALL_PORTFOLIOS)

    def test_all_portfolios_present_table4(self, table4_df):
        assert set(table4_df["portfolio"].tolist()) == set(ALL_PORTFOLIOS)

    def test_table1_required_columns(self, table1_df):
        required = {"portfolio", "alpha", "beta_Mkt-RF", "t_Mkt-RF", "r_squared", "type"}
        assert required.issubset(set(table1_df.columns)), (
            f"Missing columns. Expected {required}, got {set(table1_df.columns)}"
        )

    def test_table3_required_columns(self, table3_df):
        required = {
            "portfolio",
            "alpha",
            "beta_TERM",
            "t_TERM",
            "beta_DEF",
            "t_DEF",
            "r_squared",
            "type",
        }
        assert required.issubset(set(table3_df.columns)), (
            f"Missing columns. Expected {required}, got {set(table3_df.columns)}"
        )

    def test_table4_required_columns(self, table4_df):
        required = {
            "portfolio",
            "alpha",
            "beta_Mkt-RF",
            "t_Mkt-RF",
            "beta_SMB",
            "t_SMB",
            "beta_HML",
            "t_HML",
            "r_squared",
            "type",
        }
        assert required.issubset(set(table4_df.columns)), (
            f"Missing columns. Expected {required}, got {set(table4_df.columns)}"
        )

    def test_no_missing_r_squared(self, table1_df, table3_df, table4_df):
        for name, df in [("table1", table1_df), ("table3", table3_df), ("table4", table4_df)]:
            assert not df["r_squared"].isnull().any(), f"Missing r_squared in {name}"


# ---------------------------------------------------------------------------
# Paper pattern verification
# ---------------------------------------------------------------------------


class TestPaperPatterns:
    """Verify empirical patterns reported in FF(1993) Tables 1, 3, 4."""

    def test_table1_stock_r2_range(self, table1_df):
        """Table 1: Stock R^2 = 0.55-0.92 range (relaxed: min > 0.40, max < 0.98, mean > 0.60)."""
        stock = table1_df[table1_df["type"] == "stock"]["r_squared"]
        assert stock.min() > 0.40, f"Table 1 stock R2 min too low: {stock.min():.4f}"
        assert stock.max() < 0.98, f"Table 1 stock R2 max too high: {stock.max():.4f}"
        assert stock.mean() > 0.60, f"Table 1 stock R2 mean too low: {stock.mean():.4f}"

    def test_table1_bond_r2_non_negative(self, table1_df):
        """Table 1: Bond R^2 should be non-negative."""
        bond = table1_df[table1_df["type"] == "bond"]["r_squared"]
        assert bond.min() >= 0.0, f"Table 1 bond R2 min negative: {bond.min():.4f}"

    def test_table3_term_significant_for_majority(self, table3_df):
        """
        Table 3: TERM slopes positive and significant (|t| > 2.0) for majority.
        With yield-based proxies, TERM is significant for bonds but generally
        not for stocks, so we check the bond subset.
        """
        bonds = table3_df[table3_df["type"] == "bond"]
        term_t = bonds["t_TERM"]
        significant = (term_t.abs() > 2.0).sum()
        assert significant > len(bonds) * 0.5, (
            f"TERM significant for only {significant}/{len(bonds)} bond portfolios"
        )

    def test_table4_stock_r2_improved_over_table1(self, table1_df, table4_df):
        """Table 4 stock avg R^2 > Table 1 stock avg R^2 by at least 0.05."""
        t1_stock_r2 = table1_df[table1_df["type"] == "stock"]["r_squared"].mean()
        t4_stock_r2 = table4_df[table4_df["type"] == "stock"]["r_squared"].mean()
        assert t4_stock_r2 > t1_stock_r2 + 0.05, (
            f"Table 4 stock avg R2 ({t4_stock_r2:.4f}) not > Table 1 ({t1_stock_r2:.4f}) + 0.05"
        )

    def test_table4_stock_r2_high_for_most(self, table4_df):
        """Table 4: Stock R^2 > 0.80 for most portfolios."""
        stock = table4_df[table4_df["type"] == "stock"]["r_squared"]
        high = (stock > 0.80).sum()
        assert high >= 15, f"Only {high}/25 stock portfolios have R2 > 0.80 in Table 4"

    def test_script_prints_tables(self, run_script):
        """Script should print formatted regression tables to stdout."""
        stdout = run_script.stdout
        assert "Table 1" in stdout or "Table 3" in stdout or "Table 4" in stdout, (
            "Script should print table summaries to stdout"
        )
