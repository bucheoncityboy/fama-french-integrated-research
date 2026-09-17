"""
test_ken_french_parser.py
RED test for Ken French CSV parser
"""

import pytest
import pandas as pd
import numpy as np
from io import StringIO


def test_load_ken_french_csv_basic():
    """Test basic parsing with YYYYMM dates."""
    from ken_french_parser import load_ken_french_csv

    # Simple CSV with 2 metadata rows, 4 columns (date + 3 factors)
    csv_content = """Ken French Data Library
Fama/French Factors
MktRF,SMB,HML
196307,0.50,-1.20,0.30
196308,1.00,0.30,-0.10
196309,-0.50,0.10,0.20
"""
    result = load_ken_french_csv(StringIO(csv_content), skip_rows=3)

    assert result.shape == (3, 3)
    assert result.index[0] == pd.Timestamp('1963-07')
    assert result.index[-1] == pd.Timestamp('1963-09')
    # Column names may be preserved or not depending on pandas version
    assert len(result.columns) == 3


def test_load_ken_french_csv_missing_values():
    """Test -99.99 and -999 are replaced with NaN."""
    from ken_french_parser import load_ken_french_csv

    csv_content = """Meta
Header
1,2,3
196307,-99.99,0.50
196308,1.00,-999
196309,2.50,1.00
"""
    result = load_ken_french_csv(StringIO(csv_content), skip_rows=3)

    assert pd.isna(result.iloc[0, 0])  # -99.99 -> NaN
    assert pd.isna(result.iloc[1, 1])  # -999 -> NaN
    assert result.iloc[2, 0] == 2.50  # percentage stays as-is


def test_load_ken_french_csv_multitable_vw():
    """Test selecting value-weighted sub-table."""
    from ken_french_parser import load_ken_french_csv

    # Multi-table CSV with VW first, then EW
    # Ken French multi-table files have:
    # 1. Metadata header
    # 2. Column header row (MktRF,SMB,HML)
    # 3. Data rows
    # 4. Copyright separator
    # 5. Column header row (MktRF,SMB,HML)
    # 6. Data rows
    csv_content = """Ken French Data Library
Fama/French 3 Factors
Monthly
                            Copyright 2024
MktRF,SMB,HML
196307,0.50,1.00,2.00
196308,0.60,1.10,2.10
196309,0.70,1.20,2.20
                            Copyright 2024
MktRF,SMB,HML
196307,0.40,0.90,1.80
196308,0.50,1.00,1.90
196309,0.60,1.10,1.70
"""
    result = load_ken_french_csv(StringIO(csv_content), skip_rows=4, value_type='vw')

    # Should return first table (VW), 3 rows x 3 cols
    assert result.shape == (3, 3)
    # VW values
    assert result.iloc[0, 0] == 0.50


def test_load_ken_french_csv_multitable_ew():
    """Test selecting equal-weighted sub-table."""
    from ken_french_parser import load_ken_french_csv

    csv_content = """Ken French Data Library
Fama/French 3 Factors
Monthly
                            Copyright 2024
MktRF,SMB,HML
196307,0.50,1.00,2.00
196308,0.60,1.10,2.10
                            Copyright 2024
MktRF,SMB,HML
196307,0.40,0.90,1.80
196308,0.50,1.00,1.90
"""
    result = load_ken_french_csv(StringIO(csv_content), skip_rows=4, value_type='ew')

    # Should return second table (EW)
    assert result.shape == (2, 3)
    # EW values should be lower than VW
    assert result.iloc[0, 0] == 0.40


def test_load_ken_french_csv_date_filtering():
    """Test date filtering to START_DATE to END_DATE."""
    from ken_french_parser import load_ken_french_csv
    from config import START_DATE, END_DATE

    csv_content = """Meta
Header
1,2,3
196001,0.10,0.20
196307,0.50,0.60
196312,0.70,0.80
199106,0.30,0.40
199112,0.90,1.00
199912,1.50,1.60
"""
    result = load_ken_french_csv(StringIO(csv_content), skip_rows=3)

    # Should filter: only 196307, 196312, 199106, 199112 in range
    expected_start = pd.Timestamp(START_DATE)
    expected_end = pd.Timestamp(END_DATE)

    assert result.index[0] == expected_start  # 1963-07
    assert result.index[-1] == expected_end  # 1991-12
    assert len(result) == 4


def test_load_ken_french_csv_percentage_values_preserved():
    """Test percentage values remain as-is (2.50 = 2.50%)."""
    from ken_french_parser import load_ken_french_csv

    csv_content = """Meta
Header
MktRF,SMB,HML
196307,2.50,-1.00,0.50
196308,-3.75,1.25,-0.25
"""
    result = load_ken_french_csv(StringIO(csv_content), skip_rows=3)

    assert result.iloc[0, 0] == 2.50   # positive percentage
    assert result.iloc[1, 0] == -3.75  # negative percentage


def test_load_ken_french_csv_stringIO():
    """Test parser works with file-like object (StringIO)."""
    from ken_french_parser import load_ken_french_csv

    csv_content = """Meta
Header
1,2,3
196307,1.00,2.00
196308,3.00,4.00
"""
    result = load_ken_french_csv(StringIO(csv_content), skip_rows=3)
    assert len(result) == 2
