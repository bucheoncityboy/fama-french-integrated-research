"""
conftest.py
Shared pytest fixtures for Fama-French replication tests
"""

import os
import sys

import pytest

# src/ 모듈(config·data_loader·regression_engine 등)을 import 할 수 있게 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


@pytest.fixture
def base_dir():
    """Base directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def data_dir(base_dir):
    """Data directory path."""
    return os.path.join(base_dir, 'data')


@pytest.fixture
def output_dir(base_dir):
    """Output directory path."""
    return os.path.join(base_dir, 'appendix_output')


@pytest.fixture
def date_range():
    """Standard date range for Fama-French analysis."""
    return {
        'start_date': '1963-07',
        'end_date': '1991-12',
        'n_months': 342
    }
