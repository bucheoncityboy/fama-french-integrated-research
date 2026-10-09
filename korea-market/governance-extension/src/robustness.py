from .portfolio_sorts import sort_portfolios, strategies
from .factor_models import fit_ff3


def evaluate(pit, factors, min_cell=5, min_months=24):
    specs = [("primary_vw_median", "VW", .5, 0), ("equal_weight", "EW", .5, 0),
             ("cutoff_30", "VW", .3, 0), ("cutoff_70", "VW", .7, 0),
             ("exclude_top5", "VW", .5, 5)]
    rows = []
    for name, weight, cut, top in specs:
        returns, _, _ = sort_portfolios(pit, min_cell=min_cell, weight=weight, g_cut=cut, exclude_top=top)
        spread = strategies(returns)[["date", "spread"]].rename(columns={"spread": "return"})
        rows.append({"specification": name, **fit_ff3(spread, factors, long_short=True, min_months=min_months)})
    for name in ["industry_control", "profitability_control", "verified_market_split", "turnover_cost"]:
        rows.append({"specification": name, "status": "not_estimated", "not_estimated_reason": "verified_input_not_available"})
    return rows
