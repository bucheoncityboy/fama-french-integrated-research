# -*- coding: utf-8 -*-
"""data_loader.py — 섹션 스크립트가 공유하는 공통 데이터 로더.

load_factors / load_stock_portfolios 는 여러 섹션에 복붙돼 있던 구현을 한 곳으로
모은 것이다. load_bond_portfolios 는 CSV 의 인덱스 성격(월초/월말)이 섹션마다
다르게 쓰이므로 normalize 없이 원본 그대로 반환하고, 필요하면 호출부에서
_normalize_monthly_index 를 직접 적용한다.
"""
from __future__ import annotations

import os

import pandas as pd

import config


def _normalize_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    """DatetimeIndex 를 월초(first-of-month)로 정규화해 정렬을 맞춘다."""
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp()
    return df


def load_factors() -> pd.DataFrame:
    """통합 요인(factors.csv) 로드. TERM/DEF 는 decimal 이므로 %로 환산한다."""
    path = os.path.join(config.OUTPUT_DIR, "factors.csv")
    df = pd.read_csv(path, comment="#", index_col=0, parse_dates=True)
    df = _normalize_monthly_index(df)
    # TERM, DEF 는 채권 요인 구성에서 decimal 로 만들어졌고,
    # Mkt-RF, SMB, HML, RF 는 이미 % 단위다.
    for col in ("TERM", "DEF"):
        if col in df.columns:
            df[col] = df[col] * 100.0
    return df


def load_stock_portfolios() -> pd.DataFrame:
    """25개 주식 포트폴리오 초과수익률(%, stock_portfolios_excess.csv) 로드."""
    path = os.path.join(config.OUTPUT_DIR, "stock_portfolios_excess.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = _normalize_monthly_index(df)
    return df


def load_bond_portfolios() -> pd.DataFrame:
    """7개 채권 포트폴리오 초과수익률 로드(decimal → %).

    주의: bond_portfolios_excess.csv 의 인덱스는 월말 기준이므로 여기서 월초로
    바꾸지 않는다. 회귀 정렬이 필요한 섹션은 반환값에 _normalize_monthly_index 를
    직접 적용해 사용한다.
    """
    path = os.path.join(config.OUTPUT_DIR, "bond_portfolios_excess.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df * 100.0
    return df
