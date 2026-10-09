"""SYNTHETIC_TEST_ONLY: duplicate major-account guard."""
import numpy as np
from src.financial_controls import unique_amount,extract_controls


def test_identical_ni_duplicate_is_not_missing():
    aa=[{"account_nm":"당기순이익(손실)","thstrm_amount":"100","ord":o} for o in ("29","61")]
    aa += [{"account_nm":"자본총계","frmtrm_amount":"400"},{"account_nm":"자산총계","thstrm_amount":"800"},
           {"account_nm":"부채총계","thstrm_amount":"300"}]
    result=extract_controls(aa,"자본총계",600)
    assert result["roe"]==.2
    assert result["leverage"]==.375


def test_conflicting_or_missing_duplicate_remains_unavailable():
    assert np.isnan(unique_amount([{"account_nm":"x","thstrm_amount":"100"},{"account_nm":"x","thstrm_amount":"101"}],["x"]))
    assert np.isnan(unique_amount([{"account_nm":"x","thstrm_amount":"100"},{"account_nm":"x","thstrm_amount":"-"}],["x"]))
