"""
download_data.py
Download and parse Ken French Factor Data and bond data for Fama-French replication.
"""

import os
import urllib.request
import zipfile
import io

import config
import ken_french_parser
import fred_bond_fetcher

# Ken French zip file definitions
FF_ZIP_FILES = {
    'ff_factors': 'F-F_Research_Data_Factors_CSV.zip',
    'ff_6_portfolios': '6_Portfolios_2x3_CSV.zip',
    'ff_25_portfolios': '25_Portfolios_5x5_CSV.zip',
}

# Parser configurations
FF_PARSER_CONFIG = {
    'ff_factors': {'skip_rows': 3, 'value_type': 'vw'},
    'ff_6_portfolios': {'skip_rows': 12, 'value_type': 'vw'},
    'ff_25_portfolios': {'skip_rows': 12, 'value_type': 'vw'},
}


def download_and_parse_ff_zip(name, zip_filename):
    """
    Download a Ken French zip file and parse it to CSV.

    Parameters
    ----------
    name : str
        Internal name for the dataset (e.g., 'ff_factors')
    zip_filename : str
        Name of the zip file on the server

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame
    """
    url = os.path.join(config.FF_DATA_URL, zip_filename)
    csv_path = os.path.join(config.DATA_DIR, f'{name}.csv')

    print("Downloading %s from %s" % (name, url))
    try:
        with urllib.request.urlopen(url) as response:
            zip_data = response.read()
    except Exception as exc:
        raise ConnectionError(f"Failed to download {url}: {exc}")

    print("Extracting %s from zip" % name)
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        # Find the CSV file inside the zip (usually first .CSV file)
        csv_filename = None
        for fname in zf.namelist():
            if fname.upper().endswith('.CSV'):
                csv_filename = fname
                break

        if csv_filename is None:
            raise ValueError(f"No CSV file found in zip {zip_filename}")

        print("Parsing %s using skip_rows=%s, value_type='%s'" % (
                    name,
                    FF_PARSER_CONFIG[name]['skip_rows'],
                    FF_PARSER_CONFIG[name]['value_type']))

        with zf.open(csv_filename) as csv_file:
            # Read bytes and decode to string since ken_french_parser expects string content
            csv_content = csv_file.read().decode('utf-8')
            df = ken_french_parser.load_ken_french_csv(
                csv_content,
                skip_rows=FF_PARSER_CONFIG[name]['skip_rows'],
                value_type=FF_PARSER_CONFIG[name]['value_type']
            )

    # Save to CSV
    df.to_csv(csv_path)
    print("Saved %s to %s (%d rows, %d columns)" % (name, csv_path, df.shape[0], df.shape[1]))

    return df


def print_summary(name, df):
    """Print summary statistics for a DataFrame."""
    start_date = df.index.min().strftime('%Y-%m')
    end_date = df.index.max().strftime('%Y-%m')
    print("  %s: %d rows x %d columns | Date range: %s to %s" %
                (name, df.shape[0], df.shape[1], start_date, end_date))
    print("  Columns: %s" % list(df.columns))


def verify_file(name, expected_rows=None, expected_cols=None):
    """Verify a file exists and has expected shape."""
    import pandas as pd

    csv_path = os.path.join(config.DATA_DIR, f'{name}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist")

    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)

    if expected_rows is not None and df.shape[0] != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} rows, got {df.shape[0]}")

    if expected_cols and df.shape[1] != expected_cols:
        raise ValueError(f"{name}: expected {expected_cols} columns, got {df.shape[1]}")

    return df


def main():
    """Download and parse all Ken French factor data and bond data."""
    print("=" * 60)
    print("Starting data download for Fama-French replication")
    print("Date range: %s to %s" % (config.START_DATE, config.END_DATE))
    print("=" * 60)

    # Download and parse Ken French data
    dataframes = {}
    for name, zip_filename in FF_ZIP_FILES.items():
        df = download_and_parse_ff_zip(name, zip_filename)
        dataframes[name] = df

    # Fetch bond data (creates data/bond_factors.csv)
    print("Fetching bond data from FRED")
    fred_bond_fetcher.fetch_bond_data(
        start=config.START_DATE + '-01',
        end=config.END_DATE + '-31'
    )
    dataframes['bond_factors'] = verify_file('bond_factors', expected_cols=2)

    # Verify all files
    print("=" * 60)
    print("Verifying output files:")
    print("=" * 60)

    verify_file('ff_factors', expected_cols=4)
    verify_file('ff_6_portfolios', expected_cols=6)
    verify_file('ff_25_portfolios', expected_cols=25)

    # Print summary
    print("=" * 60)
    print("Data download complete! Summary:")
    print("=" * 60)

    for name, df in dataframes.items():
        print_summary(name, df)

    print("=" * 60)
    print("All files verified successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()