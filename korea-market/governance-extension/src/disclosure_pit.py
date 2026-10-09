"""Conservative publication timing using KRX session month ends."""
from functools import lru_cache
import numpy as np
import pandas as pd
import exchange_calendars as xc


@lru_cache(maxsize=1)
def calendar_tables():
    c=xc.get_calendar("XKRX",start="2015-01-01",end="2026-12-31")
    sessions=c.sessions.tz_localize(None) if c.sessions.tz is not None else c.sessions
    ends=pd.Series(sessions,index=sessions.to_period("M")).groupby(level=0).max()
    starts=pd.Series(sessions,index=sessions.to_period("M")).groupby(level=0).min()
    return ends,starts


def available_at(filing):
    ends,_=calendar_tables()
    tomorrow=pd.Timestamp(filing).normalize()+pd.Timedelta(days=1)
    eligible=ends.loc[ends.ge(tomorrow)]
    return eligible.iloc[0] if len(eligible) else pd.NaT


def select_observation(observations,formation,max_age_days=550):
    if not len(observations):return None
    eligible=observations.loc[observations.available_at.le(formation)&observations.status.eq("PASS")].copy()
    if not len(eligible):return None
    chosen=eligible.sort_values(["period_end","filing_date","rcept_no"],kind="stable").iloc[-1]
    if (formation-chosen.filing_date).days>max_age_days:return None
    if not (chosen.filing_date<formation+pd.Timedelta(days=1)):
        raise ValueError("future_G_publication")
    return chosen


def validate_dates(d):
    if len(d) and not (d.filing_date.lt(d.return_month_start)&d.financial_filing_date.lt(d.return_month_start)&
             d.available_at.le(d.formation_month_end)&d.formation_month_end.lt(d.return_month_start)).all():
        raise ValueError("PIT_future_information_leak")
    return True
