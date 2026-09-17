"""
test_scaffold.py
RED test: Verify project scaffolding is correctly configured
"""

import os
import config


def test_config_loads():
    """config.py loads without errors."""
    assert hasattr(config, 'START_DATE')
    assert hasattr(config, 'END_DATE')
    assert hasattr(config, 'N_MONTHS')


def test_start_date_value():
    """START_DATE is correctly set to 1963-07."""
    assert config.START_DATE == '1963-07'


def test_data_dir_exists(data_dir):
    """DATA_DIR directory exists."""
    assert os.path.exists(data_dir)


def test_output_dir_exists(output_dir):
    """OUTPUT_DIR directory exists."""
    assert os.path.exists(output_dir)


def test_ff_data_url_defined():
    """Ken French Data URL is defined."""
    assert hasattr(config, 'FF_DATA_URL')
    assert 'mba.tuck.dartmouth.edu' in config.FF_DATA_URL
