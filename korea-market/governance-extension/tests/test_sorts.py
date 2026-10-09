import numpy as np
import pandas as pd

from src.common import lag_market_values, stock_code
from src.portfolio_sorts import sort_portfolios, strategies


def pit():
    n = 40
    return pd.DataFrame({"code": [f"A{i:06}" for i in range(n)], "date": pd.Timestamp("2024-05-01"),
        "formation_date": pd.Timestamp("2024-04-30"), "available_at": pd.Timestamp("2024-03-31"),
        "lag_me": np.arange(1,n+1,dtype=float), "lag_bm": np.repeat([.5,2],20),
        "g_value": np.tile(np.arange(20)/20,2), "return": np.tile(np.arange(20)/1000,2)})


def test_weights_and_spread_arithmetic():
    returns, holdings, _ = sort_portfolios(pit())
    assert len(returns) == 4
    assert np.allclose(returns.weight_sum, 1)
    assert holdings.groupby("cell").weight.sum().eq(1).all()
    s = strategies(returns)
    assert np.isclose(s.spread.iloc[0], s.HighBM_HighG.iloc[0]-s.HighBM_LowG.iloc[0])


def test_missing_return_is_not_survivor_renormalized():
    p = pit(); p.loc[0, "return"] = np.nan
    returns, _, _ = sort_portfolios(p)
    assert pd.isna(returns.loc[returns.cell.eq("LowBM_LowG"), "return"].iloc[0])


def test_sorting_not_conditioned_on_future_returns():
    p = pit(); _, h1, _ = sort_portfolios(p)
    p["return"] = np.nan; _, h2, _ = sort_portfolios(p)
    pd.testing.assert_frame_equal(h1, h2)


def test_tied_director_ratios_not_randomly_split():
    p = pit(); p.g_value = .5
    r, _, _ = sort_portfolios(p)
    assert r.loc[r.cell.str.endswith("HighG"), "return"].isna().all()


def test_gap_lag_is_not_a_previous_month_weight():
    p = pd.DataFrame({"code": ["A005930"]*2, "date": pd.to_datetime(["2024-01-01", "2024-03-01"]), "me": [100,200], "bm": [1,2]})
    assert lag_market_values(p).lag_me.isna().all()


def test_stock_codes_preserve_leading_zero_and_letters():
    assert stock_code("A005930") == "005930"
    assert stock_code("A0120G0") == "0120G0"
    assert stock_code("A0126Z0") == "0126Z0"
