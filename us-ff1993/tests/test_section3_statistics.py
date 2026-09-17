"""
tests/test_section3_statistics.py
RED → GREEN tests for 03_section3_statistics.py (Section 3: The Playing Field)

This module verifies that Section 3 correctly computes average excess returns
and factor premiums with t-statistics, matching the structure of FF(1993) Table 2.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(BASE_DIR, "src", "03_section3_statistics.py")
SUMMARY_CSV = os.path.join(BASE_DIR, "appendix_output", "table2_summary.csv")

# Expected stock column order for 5x5 grid
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

# Map columns to (size_quintile, bm_quintile) where 1=smallest/lowest, 5=largest/highest
SIZE_BM_MAP = {
    "SMALL LoBM": (1, 1),
    "ME1 BM2": (1, 2),
    "ME1 BM3": (1, 3),
    "ME1 BM4": (1, 4),
    "SMALL HiBM": (1, 5),
    "ME2 BM1": (2, 1),
    "ME2 BM2": (2, 2),
    "ME2 BM3": (2, 3),
    "ME2 BM4": (2, 4),
    "ME2 BM5": (2, 5),
    "ME3 BM1": (3, 1),
    "ME3 BM2": (3, 2),
    "ME3 BM3": (3, 3),
    "ME3 BM4": (3, 4),
    "ME3 BM5": (3, 5),
    "ME4 BM1": (4, 1),
    "ME4 BM2": (4, 2),
    "ME4 BM3": (4, 3),
    "ME4 BM4": (4, 4),
    "ME4 BM5": (4, 5),
    "BIG LoBM": (5, 1),
    "ME5 BM2": (5, 2),
    "ME5 BM3": (5, 3),
    "ME5 BM4": (5, 4),
    "BIG HiBM": (5, 5),
}

BOND_COLS = ["SHORT_TERM", "LONG_TERM", "AAA", "AA", "A", "BBB", "LOW_GRADE"]

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]


@pytest.fixture(scope="module")
def run_script():
    """Run the Section 3 statistics script once per module."""
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
def summary_df(run_script):
    """Load the produced table2_summary.csv."""
    assert os.path.exists(SUMMARY_CSV), f"{SUMMARY_CSV} was not created"
    df = pd.read_csv(SUMMARY_CSV)
    return df


# ---------------------------------------------------------------------------
# Output existence and structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    """Tests for output file existence and schema."""

    def test_summary_csv_exists(self, run_script):
        assert os.path.exists(SUMMARY_CSV)

    def test_has_required_columns(self, summary_df):
        required = {"Type", "Label", "Mean", "Std", "N", "T_Stat"}
        assert required.issubset(set(summary_df.columns)), (
            f"Missing columns. Expected {required}, got {set(summary_df.columns)}"
        )

    def test_all_stock_portfolios_present(self, summary_df):
        stock_labels = set(
            summary_df[summary_df["Type"] == "Stock"]["Label"].tolist()
        )
        assert set(STOCK_COLS).issubset(stock_labels), (
            f"Missing stock portfolios: {set(STOCK_COLS) - stock_labels}"
        )

    def test_all_bond_portfolios_present(self, summary_df):
        bond_labels = set(
            summary_df[summary_df["Type"] == "Bond"]["Label"].tolist()
        )
        assert set(BOND_COLS).issubset(bond_labels), (
            f"Missing bond portfolios: {set(BOND_COLS) - bond_labels}"
        )

    def test_all_factors_present(self, summary_df):
        factor_labels = set(
            summary_df[summary_df["Type"] == "Factor"]["Label"].tolist()
        )
        assert set(FACTOR_COLS).issubset(factor_labels), (
            f"Missing factors: {set(FACTOR_COLS) - factor_labels}"
        )

    def test_no_missing_statistics(self, summary_df):
        # Every row should have Mean, Std, N, T_Stat
        subset = summary_df[["Mean", "Std", "N", "T_Stat"]]
        assert not subset.isnull().any().any(), "Missing statistics in summary"

    def test_t_stats_are_finite(self, summary_df):
        assert summary_df["T_Stat"].apply(lambda x: pd.notna(x) and abs(x) < 1e6).all()


# ---------------------------------------------------------------------------
# Paper pattern verification
# ---------------------------------------------------------------------------

class TestPaperPatterns:
    """Verify empirical patterns reported in FF(1993)."""

    def test_size_effect_small_gt_big(self, summary_df):
        """
        Negative relation: size <-> average return (small > big).
        Check that smallest quintile > largest quintile in >= 3/5 BE/ME groups.
        """
        stock_df = summary_df[summary_df["Type"] == "Stock"].copy()
        stock_df["size_q"] = stock_df["Label"].map(lambda x: SIZE_BM_MAP[x][0])
        stock_df["bm_q"] = stock_df["Label"].map(lambda x: SIZE_BM_MAP[x][1])

        wins = 0
        for bm in range(1, 6):
            small_mean = stock_df.loc[
                (stock_df["size_q"] == 1) & (stock_df["bm_q"] == bm), "Mean"
            ].iloc[0]
            big_mean = stock_df.loc[
                (stock_df["size_q"] == 5) & (stock_df["bm_q"] == bm), "Mean"
            ].iloc[0]
            if small_mean > big_mean:
                wins += 1

        assert wins >= 3, (
            f"Size effect too weak: SMALL > BIG in only {wins}/5 BE/ME groups"
        )

    def test_value_effect_hiBM_gt_loBM(self, summary_df):
        """
        Positive relation: BE/ME <-> average return (value > growth).
        Check that highest BE/ME > lowest BE/ME in >= 3/5 size groups.
        """
        stock_df = summary_df[summary_df["Type"] == "Stock"].copy()
        stock_df["size_q"] = stock_df["Label"].map(lambda x: SIZE_BM_MAP[x][0])
        stock_df["bm_q"] = stock_df["Label"].map(lambda x: SIZE_BM_MAP[x][1])

        wins = 0
        for sz in range(1, 6):
            hi_mean = stock_df.loc[
                (stock_df["size_q"] == sz) & (stock_df["bm_q"] == 5), "Mean"
            ].iloc[0]
            lo_mean = stock_df.loc[
                (stock_df["size_q"] == sz) & (stock_df["bm_q"] == 1), "Mean"
            ].iloc[0]
            if hi_mean > lo_mean:
                wins += 1

        assert wins >= 3, (
            f"Value effect too weak: HiBM > LoBM in only {wins}/5 size groups"
        )

    def test_bond_returns_lt_stock_returns(self, summary_df):
        """
        Bond average excess returns should be lower than stock average excess returns.
        """
        stock_means = summary_df[summary_df["Type"] == "Stock"]["Mean"]
        bond_means = summary_df[summary_df["Type"] == "Bond"]["Mean"]

        avg_stock = stock_means.mean()
        avg_bond = bond_means.mean()

        assert avg_bond < avg_stock, (
            f"Bond avg ({avg_bond:.4f}) should be < stock avg ({avg_stock:.4f})"
        )

    def test_factor_premiums_have_t_stats(self, summary_df):
        """Every factor should have a non-NaN t-statistic."""
        factor_df = summary_df[summary_df["Type"] == "Factor"]
        for _idx, row in factor_df.iterrows():
            assert pd.notna(row["T_Stat"]), (
                f"Factor {row['Label']} has missing t-statistic"
            )

    def test_stock_means_in_paper_range(self, summary_df):
        """
        Stock avg excess returns should roughly fall in 0.3-1.1 %/month
        (paper range 0.32-1.05).
        """
        stock_means = summary_df[summary_df["Type"] == "Stock"]["Mean"]
        assert stock_means.min() >= 0.0, (
            f"Some stock mean is negative: min={stock_means.min():.4f}"
        )
        assert stock_means.max() <= 1.5, (
            f"Some stock mean too high: max={stock_means.max():.4f}"
        )

    def test_script_prints_table(self, run_script):
        """Script should print a formatted table summary."""
        stdout = run_script.stdout
        assert "Table 2" in stdout or "Average" in stdout or "Excess" in stdout, (
            "Script should print a summary table to stdout"
        )
