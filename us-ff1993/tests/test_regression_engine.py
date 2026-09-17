"""
tests/test_regression_engine.py
RED → GREEN tests for regression_engine.py
"""

import numpy as np
import pandas as pd
import pytest

import regression_engine as re


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_ols_data():
    """Synthetic data where y = 1.0 + 2.0*x + eps."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    eps = np.random.randn(n) * 0.1
    y = 1.0 + 2.0 * x + eps
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


@pytest.fixture
def multi_factor_data():
    """Synthetic 2-factor data: y = 0.5 + 1.5*x1 - 0.8*x2 + eps."""
    np.random.seed(7)
    n = 120
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    eps = np.random.randn(n) * 0.2
    y = 0.5 + 1.5 * x1 - 0.8 * x2 + eps
    return pd.Series(y, name="y"), pd.DataFrame({"x1": x1, "x2": x2})


@pytest.fixture
def portfolio_returns():
    """3 portfolios, 60 months."""
    np.random.seed(99)
    n = 60
    dates = pd.date_range("2020-01", periods=n, freq="ME")
    p1 = np.random.randn(n) * 0.02 + 0.01
    p2 = np.random.randn(n) * 0.025 + 0.005
    p3 = np.random.randn(n) * 0.03 + 0.008
    return pd.DataFrame({"P1": p1, "P2": p2, "P3": p3}, index=dates)


@pytest.fixture
def factor_returns():
    """2 factors, 60 months."""
    np.random.seed(101)
    n = 60
    dates = pd.date_range("2020-01", periods=n, freq="ME")
    mkt = np.random.randn(n) * 0.04 + 0.008
    smb = np.random.randn(n) * 0.02 + 0.002
    return pd.DataFrame({"MKT": mkt, "SMB": smb}, index=dates)


# ---------------------------------------------------------------------------
# 1. run_ols
# ---------------------------------------------------------------------------

def test_run_ols_recover_coefficients(simple_ols_data):
    y, X = simple_ols_data
    result = re.run_ols(y, X, add_const=True)

    assert isinstance(result, dict)
    assert "coefficients" in result
    assert "t_stats" in result
    assert "p_values" in result
    assert "r_squared" in result
    assert "adj_r_squared" in result
    assert "intercept" in result

    # intercept ~ 1.0, beta ~ 2.0
    assert result["intercept"] == pytest.approx(1.0, abs=0.05)
    assert result["coefficients"]["x"] == pytest.approx(2.0, abs=0.05)


def test_run_ols_no_constant(simple_ols_data):
    y, X = simple_ols_data
    result = re.run_ols(y, X, add_const=False)
    assert result["intercept"] is None or np.isnan(result["intercept"])
    assert "x" in result["coefficients"]


def test_run_ols_nan_handling():
    np.random.seed(3)
    n = 50
    x = np.random.randn(n)
    y = 2.0 + 3.0 * x + np.random.randn(n) * 0.1
    y[5:10] = np.nan
    x[15:20] = np.nan

    s_y = pd.Series(y, name="y")
    df_x = pd.DataFrame({"x": x})
    result = re.run_ols(s_y, df_x, add_const=True)
    # Should drop NaN rows and still run
    assert result["coefficients"]["x"] == pytest.approx(3.0, abs=0.05)


def test_run_ols_insufficient_data():
    y = pd.Series([1.0, np.nan, 3.0])
    X = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    result = re.run_ols(y, X, add_const=True)
    assert all(np.isnan(v) for v in result["coefficients"].values())


# ---------------------------------------------------------------------------
# 2. run_batch_regressions
# ---------------------------------------------------------------------------

def test_run_batch_regressions_shape(portfolio_returns, factor_returns):
    df = re.run_batch_regressions(
        portfolio_returns, factor_returns, factor_names=["MKT", "SMB"]
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 3 portfolios
    assert "portfolio" in df.columns
    assert "alpha" in df.columns
    assert "beta_MKT" in df.columns
    assert "beta_SMB" in df.columns
    assert "t_MKT" in df.columns
    assert "t_SMB" in df.columns
    assert "r_squared" in df.columns


def test_run_batch_regressions_index_alignment(portfolio_returns, factor_returns):
    # Misaligned index: factor_returns shorter
    short_factors = factor_returns.iloc[10:]
    df = re.run_batch_regressions(
        portfolio_returns, short_factors, factor_names=["MKT", "SMB"]
    )
    assert len(df) == 3
    assert df["r_squared"].notna().all()


# ---------------------------------------------------------------------------
# 3. compute_residual_covariance
# ---------------------------------------------------------------------------

def test_compute_residual_covariance_shape(portfolio_returns, factor_returns):
    results = []
    for col in portfolio_returns.columns:
        res = re.run_ols(portfolio_returns[col], factor_returns, add_const=True)
        results.append(res)
    cov = re.compute_residual_covariance(results)
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (3, 3)
    assert np.allclose(cov.values, cov.values.T)  # symmetric
    assert np.all(np.linalg.eigvalsh(cov.values) > -1e-10)  # PSD


def test_compute_residual_covariance_singular_fallback():
    # Residuals that are perfectly correlated -> near-singular
    np.random.seed(11)
    n = 10
    base = np.random.randn(n)
    res1 = base
    res2 = 2 * base
    res3 = -1 * base
    results = [
        {"residuals": pd.Series(res1)},
        {"residuals": pd.Series(res2)},
        {"residuals": pd.Series(res3)},
    ]
    cov = re.compute_residual_covariance(results)
    assert cov.shape == (3, 3)
    # Should not raise despite singularity
    assert cov.notna().all().all()


# ---------------------------------------------------------------------------
# 4. format_regression_table
# ---------------------------------------------------------------------------

def test_format_regression_table_returns_string(simple_ols_data):
    y, X = simple_ols_data
    result = re.run_ols(y, X, add_const=True)
    table = re.format_regression_table(result, table_name="Test Table")
    assert isinstance(table, str)
    assert "Test Table" in table
    assert "R²" in table or "R-squared" in table


def test_format_regression_table_with_batch(portfolio_returns, factor_returns):
    df = re.run_batch_regressions(
        portfolio_returns, factor_returns, factor_names=["MKT", "SMB"]
    )
    table = re.format_regression_table(df, table_name="Batch Results")
    assert isinstance(table, str)
    assert "Batch Results" in table
    assert "P1" in table


# ---------------------------------------------------------------------------
# Integration: known coefficients recovered
# ---------------------------------------------------------------------------

def test_known_coefficients_multi_factor(multi_factor_data):
    y, X = multi_factor_data
    result = re.run_ols(y, X, add_const=True)
    assert result["intercept"] == pytest.approx(0.5, abs=0.05)
    assert result["coefficients"]["x1"] == pytest.approx(1.5, abs=0.05)
    assert result["coefficients"]["x2"] == pytest.approx(-0.8, abs=0.05)
