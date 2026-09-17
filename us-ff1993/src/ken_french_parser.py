"""
ken_french_parser.py
Ken French CSV parser utility for Fama-French factor data.
"""

import pandas as pd
import numpy as np
from config import START_DATE, END_DATE


def load_ken_french_csv(filepath, skip_rows, value_type='vw'):
    """
    Load Ken French CSV file with proper parsing.

    Parameters
    ----------
    filepath : str, Path, or file-like
        Path to CSV file or file-like object (e.g., StringIO)
    skip_rows : int
        Number of metadata header rows to skip
    value_type : str
        'vw' for value-weighted, 'ew' for equal-weighted (for multi-table files)

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and parsed factor values
    """
    import io

    # Read content from filepath
    if hasattr(filepath, 'read'):
        content = filepath.read()
    elif isinstance(filepath, str) and '\n' in filepath:
        content = filepath
    else:
        with open(filepath, 'r') as f:
            content = f.read()

    lines = content.split('\n')
    
    # Skip metadata lines
    lines = lines[skip_rows:]
    
    # For multi-table files, find the correct table
    # Tables are separated by blank lines, Copyright lines, and text headers
    table_sections = []
    current_section = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line - end of current section
            if current_section:
                table_sections.append(current_section)
                current_section = []
            continue
        if 'Copyright' in stripped:
            # Copyright line - end of current section, don't include it
            if current_section:
                table_sections.append(current_section)
                current_section = []
            continue
        current_section.append(line)
    
    if current_section:
        table_sections.append(current_section)
    
    # Select the appropriate table section
    if len(table_sections) == 0:
        return pd.DataFrame()
    
    # For single-table files, use the first section
    # For multi-table files (like portfolios), sections alternate:
    # Section 0: "Average Value Weighted Returns -- Monthly" + data
    # Section 1: "Average Equal Weighted Returns -- Monthly" + data
    # etc.
    if value_type == 'vw':
        target_section = table_sections[0]
    else:
        target_section = table_sections[1] if len(table_sections) > 1 else table_sections[0]
    
    # Find header row and data rows
    # Portfolio files have section title before CSV header:
    #   "Average Value Weighted Returns -- Monthly"
    #   ",SMALL LoBM,ME1 BM2,SMALL HiBM,BIG LoBM,ME2 BM2,BIG HiBM"
    #   "192607,   1.0866,   0.8807,  -0.1275,   5.5746,   1.9060,   2.0068"
    # Strategy: collect all non-YYYYMM lines at the beginning,
    # then the LAST one before data is the CSV header
    header_candidates = []
    data_lines = []
    data_started = False
    
    for line in target_section:
        stripped = line.strip()
        if not stripped:
            continue
        
        cols = stripped.split(',')
        first_col = cols[0].strip()
        
        # Check if first column is a valid YYYYMM (6-digit number)
        try:
            val = int(first_col)
            if 100000 <= val <= 999999:
                # This is a data row
                data_started = True
                data_lines.append(line)
            else:
                # Numeric but not YYYYMM - could be header
                if not data_started:
                    header_candidates.append(line)
        except ValueError:
            # Non-numeric first column - header or metadata
            if not data_started:
                header_candidates.append(line)
    
    if not data_lines:
        return pd.DataFrame()
    
    # The actual CSV header is the LAST header candidate (section titles come first)
    header_row = header_candidates[-1] if header_candidates else None
    
    # Parse header to get column names
    if header_row:
        header_cols = header_row.split(',')
        # First column might be empty (just a comma) - skip it
        column_names = []
        for i, col in enumerate(header_cols[1:], 1):  # Skip first column (date)
            col = col.strip()
            if col:
                column_names.append(col)
            else:
                column_names.append(f'Col{i}')
    else:
        column_names = None
    
    # Parse data lines
    data_content = '\n'.join(data_lines)
    df = pd.read_csv(io.StringIO(data_content), header=None, engine='python')
    
    # First column is date (YYYYMM)
    dates = pd.to_datetime(df.iloc[:, 0].astype(str), format='%Y%m')
    
    # Remaining columns are factor values
    data = df.iloc[:, 1:]
    
    # Assign column names
    if column_names and len(column_names) == data.shape[1]:
        data.columns = column_names
    
    # Set date as index
    data.index = dates
    data.index.name = 'Date'
    
    # Replace -99.99 and -999 with NaN
    data = data.replace(-99.99, np.nan)
    data = data.replace(-999, np.nan)
    
    # Filter to date range
    start_dt = pd.Timestamp(START_DATE)
    end_dt = pd.Timestamp(END_DATE)
    data = data[(data.index >= start_dt) & (data.index <= end_dt)]
    
    return data
