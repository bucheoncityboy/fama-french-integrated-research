import numpy as np
import pandas as pd
import pytest

from src.factor_models import fit_ff3


def data():
    rng = np.random.default_rng(42)
    f = pd.DataFrame(rng.normal(0, .03, (120,3)), columns=["Mkt-RF", "SMB", "HML"])
    f["date"] = pd.date_range("2010-01-01", periods=len(f), freq="MS")
    f["rf"] = .002
    r = .001 + f["Mkt-RF"]*.7 + f.SMB*.2 - f.HML*.3
    return f, pd.DataFrame({"date": f.date, "return": r})


def test_synthetic_alpha_and_betas():
    f, s = data()
    r = fit_ff3(s, f, long_short=True)
    assert r["alpha_monthly_decimal"] == pytest.approx(.001)
    assert r["alpha_monthly_pct"] == pytest.approx(.1)
    assert r["beta_mkt"] == pytest.approx(.7)
    assert r["beta_smb"] == pytest.approx(.2)
    assert r["beta_hml"] == pytest.approx(-.3)


def test_rf_subtracted_once_only_for_long_only():
    f, s = data()
    spread = fit_ff3(s, f, long_short=True)
    long = fit_ff3(s, f, long_short=False)
    assert spread["alpha_monthly_decimal"]-long["alpha_monthly_decimal"] == pytest.approx(.002)
    assert spread["rf_subtracted"] is False


def test_insufficient_sample_not_estimated():
    f, s = data()
    assert fit_ff3(s.iloc[:12], f, long_short=True)["status"] == "not_estimated"


def test_noncontiguous_months_not_used_for_hac():
    f, s = data(); s = s.drop(index=50)
    assert fit_ff3(s, f, long_short=True)["not_estimated_reason"] == "noncontiguous_months_for_HAC"


def test_hac_uses_chronological_order():
    f, s = data()
    a = fit_ff3(s, f, long_short=True)
    b = fit_ff3(s.sample(frac=1, random_state=5), f, long_short=True)
    assert a["alpha_monthly_decimal"] == pytest.approx(b["alpha_monthly_decimal"])


def test_collinear_factors_not_estimated():
    f, s = data(); f.HML = f.SMB
    assert fit_ff3(s, f, long_short=True)["not_estimated_reason"] == "rank_deficient_factors"


def test_duplicate_months_rejected():
    f, s = data()
    with pytest.raises(ValueError, match="duplicate"):
        fit_ff3(pd.concat([s, s.iloc[:1]]), f, long_short=True)
