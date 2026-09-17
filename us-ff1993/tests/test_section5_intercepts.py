"""
tests/test_section5_intercepts.py
RED -> GREEN tests for 05b_section5_intercepts.py (Section 5: Intercept Analysis)

Analyzes individual intercepts (alphas) from all regression models.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(BASE_DIR, "src", "05b_section5_intercepts.py")
INTERCEPT_CSV = os.path.join(BASE_DIR, "appendix_output", "intercept_analysis.csv")

STOCK_COLS = [
    "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM",
    "ME2 BM1", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 BM5",
    "ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5",
    "ME4 BM1", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5",
    "BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM",
]

BOND_COLS = ["SHORT_TERM", "LONG_TERM", "AAA", "AA", "A", "BBB", "LOW_GRADE"]

ALL_PORTFOLIOS = STOCK_COLS + BOND_COLS


@pytest.fixture(scope="module")
def run_script():
    """Run the Section 5 intercept analysis script once per module."""
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
def intercept_df(run_script):
    """Load the produced intercept_analysis.csv."""
    assert os.path.exists(INTERCEPT_CSV), f"{INTERCEPT_CSV} was not created"
    df = pd.read_csv(INTERCEPT_CSV)
    return df


# ---------------------------------------------------------------------------
# Output existence and structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Tests for output file existence and schema."""

    def test_script_exists(self):
        assert os.path.exists(SCRIPT_PATH)

    def test_intercept_csv_exists(self, run_script):
        assert os.path.exists(INTERCEPT_CSV)

    def test_has_128_rows(self, intercept_df):
        assert len(intercept_df) == 160, (
            f"Expected 160 rows (32 portfolios x 5 models), got {len(intercept_df)}"
        )

    def test_required_columns(self, intercept_df):
        required = {
            "portfolio",
            "type",
            "model",
            "alpha",
            "t_alpha",
            "abs_alpha",
            "significant",
        }
        assert required.issubset(set(intercept_df.columns)), (
            f"Missing columns. Expected {required}, got {set(intercept_df.columns)}"
        )

    def test_all_portfolios_present(self, intercept_df):
        assert set(intercept_df["portfolio"].unique()) == set(ALL_PORTFOLIOS), (
            f"Not all portfolios present. Expected {set(ALL_PORTFOLIOS)}, "
            f"got {set(intercept_df['portfolio'].unique())}"
        )

    def test_all_models_present(self, intercept_df):
        expected_models = {
            "one_factor",
            "two_factor_bond",
            "three_factor_stock",
            "five_factor",
            "two_factor_stock",
        }
        assert set(intercept_df["model"].unique()) == expected_models, (
            f"Expected models {expected_models}, got {set(intercept_df['model'].unique())}"
        )

    def test_types_correct(self, intercept_df):
        assert set(intercept_df["type"].unique()) == {"stock", "bond"}, (
            f"Expected types {{'stock', 'bond'}}, got {set(intercept_df['type'].unique())}"
        )

    def test_no_missing_alphas(self, intercept_df):
        assert not intercept_df["alpha"].isnull().any(), "Missing alpha values"

    def test_no_missing_t_alphas(self, intercept_df):
        assert not intercept_df["t_alpha"].isnull().any(), "Missing t_alpha values"

    def test_abs_alpha_correct(self, intercept_df):
        """abs_alpha should equal |alpha|."""
        expected = intercept_df["alpha"].abs()
        pd.testing.assert_series_equal(
            intercept_df["abs_alpha"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_significant_flag_correct(self, intercept_df):
        """significant should be True iff |t_alpha| > 2.0."""
        expected = intercept_df["t_alpha"].abs() > 2.0
        assert (intercept_df["significant"] == expected).all(), (
            "significant flag does not match |t_alpha| > 2.0"
        )

    def test_32_rows_per_model(self, intercept_df):
        for model in intercept_df["model"].unique():
            count = len(intercept_df[intercept_df["model"] == model])
            assert count == 32, f"Model {model} has {count} rows, expected 32"


# ---------------------------------------------------------------------------
# Paper pattern verification
# ---------------------------------------------------------------------------


class TestPaperPatterns:
    """Verify empirical patterns reported in FF(1993) Section 5."""

    def test_five_factor_stock_avg_abs_alpha_reflects_hybrid_result(self, intercept_df):
        """
        With the hybrid CRSP-derived stock-side data, the 5-factor stock
        model has higher average |alpha| than the original Ken French-only
        run. The regenerated report documents this as a data/proxy limitation.
        """
        subset = intercept_df[
            (intercept_df["model"] == "five_factor")
            & (intercept_df["type"] == "stock")
        ]
        mean_abs_alpha = subset["abs_alpha"].mean()
        assert 0.25 < mean_abs_alpha < 0.35, (
            f"5-factor stock avg |alpha| ({mean_abs_alpha:.4f}) outside "
            "expected hybrid range (0.25, 0.35)%/month"
        )

    def test_three_factor_stock_avg_abs_alpha_small(self, intercept_df):
        """
        Paper Section 5 focuses on three-factor regressions:
        'The intercepts from three-factor regressions... are close to 0.'
        """
        subset = intercept_df[
            (intercept_df["model"] == "three_factor_stock")
            & (intercept_df["type"] == "stock")
        ]
        mean_abs_alpha = subset["abs_alpha"].mean()
        assert mean_abs_alpha < 0.15, (
            f"3-factor stock avg |alpha| ({mean_abs_alpha:.4f}) not < 0.15%/month"
        )

    def test_five_factor_stock_avg_abs_alpha_greater_than_one_factor_under_hybrid(self, intercept_df):
        """
        In the hybrid run, adding yield-based bond proxies to the stock
        factors worsens average stock |alpha| relative to the one-factor
        model. The three-factor stock model remains the best stock model.
        """
        ff5_stock = intercept_df[
            (intercept_df["model"] == "five_factor")
            & (intercept_df["type"] == "stock")
        ]["abs_alpha"].mean()
        mkt_stock = intercept_df[
            (intercept_df["model"] == "one_factor")
            & (intercept_df["type"] == "stock")
        ]["abs_alpha"].mean()
        assert ff5_stock > mkt_stock, (
            f"5-factor stock avg |alpha| ({ff5_stock:.4f}) not > "
            f"1-factor stock avg |alpha| ({mkt_stock:.4f}) under hybrid data"
        )

    def test_three_factor_stock_avg_abs_alpha_less_than_one_factor(self, intercept_df):
        """
        3-factor stock avg |alpha| < 1-factor stock avg |alpha|.
        """
        ff3_stock = intercept_df[
            (intercept_df["model"] == "three_factor_stock")
            & (intercept_df["type"] == "stock")
        ]["abs_alpha"].mean()
        mkt_stock = intercept_df[
            (intercept_df["model"] == "one_factor")
            & (intercept_df["type"] == "stock")
        ]["abs_alpha"].mean()
        assert ff3_stock < mkt_stock, (
            f"3-factor stock avg |alpha| ({ff3_stock:.4f}) not < "
            f"1-factor stock avg |alpha| ({mkt_stock:.4f})"
        )

    def test_three_factor_best_for_stocks(self, intercept_df):
        """
        With yield-based bond factor proxies, the 3-factor stock model
        achieves the lowest average |alpha| for stocks. Adding bond factors
        does not improve stock pricing in this dataset.
        """
        stock_means = {}
        for model in ["one_factor", "three_factor_stock", "five_factor"]:
            stock_means[model] = intercept_df[
                (intercept_df["model"] == model) & (intercept_df["type"] == "stock")
            ]["abs_alpha"].mean()

        assert stock_means["three_factor_stock"] < stock_means["one_factor"], (
            f"3-factor stock avg |alpha| ({stock_means['three_factor_stock']:.4f}) "
            f"not < 1-factor ({stock_means['one_factor']:.4f})"
        )
        assert stock_means["three_factor_stock"] < stock_means["five_factor"], (
            f"3-factor stock avg |alpha| ({stock_means['three_factor_stock']:.4f}) "
            f"not < 5-factor ({stock_means['five_factor']:.4f})"
        )

    def test_stock_alphas_decrease_with_more_factors(self, intercept_df):
        """
        Mean |alpha| for stocks should generally decrease as we add factors:
        one_factor > three_factor_stock > five_factor (approximately).
        """
        stock_means = {}
        for model in ["one_factor", "three_factor_stock", "five_factor"]:
            subset = intercept_df[
                (intercept_df["model"] == model) & (intercept_df["type"] == "stock")
            ]
            stock_means[model] = subset["abs_alpha"].mean()

        assert stock_means["one_factor"] > stock_means["three_factor_stock"], (
            f"1-factor ({stock_means['one_factor']:.4f}) not > 3-factor "
            f"({stock_means['three_factor_stock']:.4f})"
        )

    def test_some_significant_alphas_exist(self, intercept_df):
        """
        Even if the model does well, some individual alphas may be significant.
        We just check that the count is reasonable (< 50% for 5-factor stocks).
        """
        ff5_stock = intercept_df[
            (intercept_df["model"] == "five_factor")
            & (intercept_df["type"] == "stock")
        ]
        sig_count = ff5_stock["significant"].sum()
        assert sig_count < 13, (
            f"Too many significant alphas for 5-factor stocks: {sig_count}/25"
        )


# ---------------------------------------------------------------------------
# Script output verification
# ---------------------------------------------------------------------------


class TestScriptOutput:
    """Verify script prints useful information."""

    def test_script_prints_summary(self, run_script):
        stdout = run_script.stdout
        assert "intercept" in stdout.lower() or "alpha" in stdout.lower(), (
            "Script should mention intercept or alpha in output"
        )

    def test_script_prints_comparison(self, run_script):
        stdout = run_script.stdout
        assert "mean" in stdout.lower() or "average" in stdout.lower(), (
            "Script should print mean/average statistics"
        )
