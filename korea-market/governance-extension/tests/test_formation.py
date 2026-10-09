import pandas as pd

from src.reconstruct_ff3 import build_formation


def test_fiscal_year_tminus1_replaces_stale_june_panel_bm():
    n = 12
    panel = pd.DataFrame({"code": [f"A{i:06}" for i in range(n)], "date": pd.Timestamp("2024-06-01"),
                         "me": [100000.]*n, "bm": [.2]*n})
    book = pd.DataFrame({"code": panel.code, "date": pd.Timestamp("2024-07-01"), "be_year": [2023]*n, "be": [1.]*n})
    a = build_formation(panel, book, "legacy_compatible")
    b = build_formation(panel, book, "spec_aligned")
    assert a.bm_used.eq(.2).all()
    assert b.bm_used.eq(1).all()
    assert b.be_year.eq(2023).all()
