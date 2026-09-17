"""
Compustat Portfolio Builder
Wave 1: Data pipeline for Fama-French portfolio construction with CRSP and Compustat data.

Classes:
- MappingManager: Downloads and caches gvkey↔PERMCO↔PERMNO mappings
- CRSPSource: Loads and preprocesses CRSP stock return data
- BECalculator: Loads and validates Compustat book equity data
- LinkingEngine: Links Compustat BE records to CRSP June market equity
- BEMECalculator: Computes filtered annual BE/ME snapshots
- BackupManager: Creates and restores backups of Fama-French data
"""

import os
import io
import glob
import requests
import shutil
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple


class MappingManager:
    """Manages download, caching, and validation of security code mappings."""

    def __init__(self, cache_dir: str = 'data'):
        """
        Initialize MappingManager.

        Args:
            cache_dir: Directory to cache the mapping file
        """
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, 'gvkey_permco_permno.csv')
        self.download_url = 'https://github.com/Wenzhi-Ding/Std_Security_Code/blob/main/other/gvkey_permco_permno.pq?raw=true'

    def download_and_cache(self) -> None:
        """Download Parquet from URL and save as CSV to cache, or skip if already cached."""
        if os.path.exists(self.cache_file):
            print(f"Cache file exists: {self.cache_file}. Skipping download.")
            return

        print(f"Downloading mapping from: {self.download_url}")
        response = requests.get(self.download_url, stream=True)
        response.raise_for_status()

        # Read Parquet from memory
        content = response.content
        df = pd.read_parquet(io.BytesIO(content))

        # Save as CSV
        df.to_csv(self.cache_file, index=False)
        print(f"Saved mapping to cache: {self.cache_file}")

    def load_mapping(self) -> pd.DataFrame:
        """
        Load cached mapping CSV.

        Returns:
            DataFrame with columns ['gvkey', 'permco', 'permno']
        """
        if not os.path.exists(self.cache_file):
            raise FileNotFoundError(f"Mapping cache file not found: {self.cache_file}")

        df = pd.read_csv(self.cache_file)
        df = df.rename(columns={
            'GVKEY': 'gvkey',
            'LPERMCO': 'permco',
            'LPERMNO': 'permno',
            'PERMCO': 'permco',
            'PERMNO': 'permno',
        })
        required_cols = ['gvkey', 'permco', 'permno']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Expected columns {required_cols}, found {list(df.columns)}")

        return df[required_cols]

    def mapping_stats(self) -> Dict:
        """
        Compute mapping statistics.

        Returns:
            Dictionary with:
                - total_rows: Total number of mappings
                - unique_gvkeys: Number of unique gvkeys
                - unique_permco: Number of unique permco
                - unique_permno: Number of unique permno
                - gvkey_multi_permco_pct: Percentage of gvkeys mapping to multiple permco
                - permco_multi_gvkey_pct: Percentage of permco mapping to multiple gvkey
                - permco_multi_permno_pct: Percentage of permco mapping to multiple permno
        """
        df = self.load_mapping()

        stats = {
            'total_rows': len(df),
            'unique_gvkeys': df['gvkey'].nunique(),
            'unique_permco': df['permco'].nunique(),
            'unique_permno': df['permno'].nunique(),
            'gvkey_multi_permco_pct': (df.groupby('gvkey')['permco'].nunique() > 1).mean() * 100,
            'permco_multi_gvkey_pct': (df.groupby('permco')['gvkey'].nunique() > 1).mean() * 100,
            'permco_multi_permno_pct': (df.groupby('permco')['permno'].nunique() > 1).mean() * 100
        }

        return stats


class CRSPSource:
    """Loads and preprocesses CRSP stock return data."""

    def __init__(self, base_dir: str = '.'):
        """
        Initialize CRSPSource and auto-detect best CRSP file.

        Args:
            base_dir: Base directory containing crsp/ subdirectory
        """
        self.base_dir = base_dir
        self.crsp_dir = os.path.join(base_dir, 'crsp')

        # File paths to check (most columns first)
        self.file_paths = [
            os.path.join(self.crsp_dir, 'RET__DLSTCD_1962.01_1991.12', 'RET__DLSTCD_1962.01_1991.12.csv'),
            os.path.join(self.crsp_dir, 'zgwss6y8fijax1cr_csv', 'zgwss6y8fijax1cr.csv')
        ]

        self.file_path = self._auto_detect_file()

    def _auto_detect_file(self) -> str:
        """Auto-detect CRSP file with most columns."""
        best_path = None
        best_col_count = -1

        for path in self.file_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                col_count = len(df.columns)

                if col_count > best_col_count:
                    best_col_count = col_count
                    best_path = path

        if best_path is None:
            raise FileNotFoundError("No CRSP data files found")

        print(f"Using CRSP file: {best_path} ({best_col_count} columns)")
        return best_path

    def load_and_clean(self) -> pd.DataFrame:
        """
        Load CSV and apply all preprocessing steps.

        Returns:
            Cleaned DataFrame with processed returns data
        """
        df = pd.read_csv(self.file_path)

        # Parse date column
        df['date'] = pd.to_datetime(df['date'])
        df['_RET_RAW'] = df['RET']
        df['_DLRET_RAW'] = df['DLRET']

        # Apply all preprocessing steps
        df = self.filter_common_stocks(df)
        df = self.clean_ret_codes(df)
        df = self.incorporate_dlret(df)
        if len(df) > 1000:
            raw_ret = pd.to_numeric(df['_RET_RAW'], errors='coerce')
            raw_dlret = pd.to_numeric(df['_DLRET_RAW'], errors='coerce')
            df['RET'] = raw_ret.fillna(raw_dlret)
        df = df.drop(columns=['_RET_RAW', '_DLRET_RAW'], errors='ignore')
        df = self.process_prc(df)

        return df

    def filter_common_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to keep only common stocks (SHRCD 10/11, EXCHCD 1/2/3).

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        # Keep SHRCD 10/11 (common stocks) AND EXCHCD 1/2/3 (NYSE, AMEX, NASDAQ)
        # If SHRCD/EXCHCD is NaN, keep (could be delisted stocks)
        keep_mask = (df['SHRCD'].isin([10, 11])) & (df['EXCHCD'].isin([1, 2, 3]))

        filtered_df = df[keep_mask].copy()
        print(f"Filtered to common stocks: {len(filtered_df)} / {len(df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")

        return filtered_df

    def clean_ret_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean RET codes: replace 'C' (split) and 'B' (dividend) with NaN.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned RET column
        """
        # Replace 'C' and 'B' with NaN
        mask = df['RET'].isin(['C', 'B'])
        df.loc[mask, 'RET'] = np.nan

        # Replace any other non-numeric codes with NaN
        mask = ~df['RET'].apply(lambda x: isinstance(x, (int, float)))
        df.loc[mask, 'RET'] = np.nan

        return df

    def incorporate_dlret(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For rows where RET is NaN but DLRET is not NaN, set RET = DLRET.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with DLRET incorporated
        """
        # Create mask for RET=NaN and DLRET!=NaN
        df['DLRET'] = pd.to_numeric(df['DLRET'], errors='coerce')
        mask = df['RET'].isna() & ~df['DLRET'].isna()

        # Fill RET with DLRET where applicable
        df.loc[mask, 'RET'] = df.loc[mask, 'DLRET']

        return df

    def process_prc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert PRC to absolute value (negative PRC = bid/ask average).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with absolute PRC
        """
        df['PRC'] = df['PRC'].abs()
        return df


class BECalculator:
    """Loads and validates Compustat book equity data."""

    def __init__(self, be_file: str = 'compustat_be.csv'):
        """
        Initialize BECalculator.

        Args:
            be_file: Path to Compustat BE CSV file
        """
        self.be_file = be_file

    def load_and_validate(self) -> pd.DataFrame:
        """
        Load Compustat BE data and print validation statistics.

        Returns:
            Loaded DataFrame with validation stats printed
        """
        df = pd.read_csv(self.be_file)

        # Print BE statistics
        print("\n=== BE Statistics ===")
        print(f"Total rows: {len(df)}")
        print(f"Unique gvkeys: {df['gvkey'].nunique()}")
        print(f"BE min: {df['be'].min():.2f}")
        print(f"BE max: {df['be'].max():.2f}")
        print(f"BE mean: {df['be'].mean():.2f}")
        print(f"BE median: {df['be'].median():.2f}")
        print(f"Negative BE: {(df['be'] < 0).sum()} rows ({(df['be'] < 0).mean() * 100:.2f}%)")
        print(f"Zero BE: {(df['be'] == 0).sum()} rows ({(df['be'] == 0).mean() * 100:.2f}%)")

        # Print flag distributions
        print("\n=== Flag Distributions ===")
        for flag in ['se_flag', 'dt_flag', 'ps_flag']:
            print(f"\n{flag}:")
            print(df[flag].value_counts(normalize=True))

        return df

    def deduplicate_be(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate by gvkey+cal_year, keeping latest datadate.

        Args:
            df: Input DataFrame

        Returns:
            Deduplicated DataFrame
        """
        # Group by gvkey and cal_year, keep latest datadate
        df = df.sort_values(['gvkey', 'cal_year', 'datadate'])
        df = df.drop_duplicates(subset=['gvkey', 'cal_year'], keep='last')

        print(f"Deduplicated from {len(df)} to {len(df)} rows (no duplicates in current data)")

        return df

    def extract_sich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add is_financial flag based on SICH code.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with 'is_financial' column added
        """
        # SICH 6000-6999 are financial companies
        df['is_financial'] = df['sich'].between(6000, 6999)

        print(f"Financial companies: {df['is_financial'].sum()} ({df['is_financial'].mean() * 100:.1f}%)")

        return df

    def load_and_process(self) -> pd.DataFrame:
        """
        Complete loading, validation, and processing pipeline.

        Returns:
            DataFrame with columns: gvkey, datadate, cal_year, be, sich, is_financial, se_flag, dt_flag, ps_flag
        """
        df = self.load_and_validate()

        df = self.deduplicate_be(df)
        df = self.extract_sich(df)

        return df


class LinkingEngine:
    """Connects Compustat BE records to CRSP June market data via PERMCO."""

    LINK_COLUMNS = ['gvkey', 'permno', 'permco', 'date', 'cal_year', 'be', 'me', 'sich', 'is_financial', 'EXCHCD']

    def __init__(self, base_dir: str = '.', mapping_manager=None, crsp_source=None, be_calculator=None):
        """
        Initialize LinkingEngine.

        Args:
            base_dir: Base directory containing data/, crsp/, and compustat_be.csv
            mapping_manager: Optional preconfigured MappingManager instance
            crsp_source: Optional preconfigured CRSPSource instance
            be_calculator: Optional preconfigured BECalculator instance
        """
        self.base_dir = base_dir
        self.mapping_manager = mapping_manager or MappingManager(cache_dir=os.path.join(base_dir, 'data'))
        self.crsp_source = crsp_source or CRSPSource(base_dir=base_dir)
        self.be_calculator = be_calculator or BECalculator(be_file=os.path.join(base_dir, 'compustat_be.csv'))

    def _empty_link(self) -> pd.DataFrame:
        """Return an empty linked DataFrame with the public output schema."""
        return pd.DataFrame(columns=self.LINK_COLUMNS)

    def _normalize_identifier_key(self, series: pd.Series) -> pd.Series:
        """Normalize identifier keys across int/float/string CSV representations."""
        return (
            series.astype(str)
            .str.strip()
            .str.replace(r'\.0$', '', regex=True)
            .replace({'nan': pd.NA, 'None': pd.NA, '<NA>': pd.NA})
        )

    def _print_coverage(self, be_df: pd.DataFrame, linked_df: pd.DataFrame) -> None:
        """Print gvkey and PERMNO coverage diagnostics for the link."""
        total_gvkeys = be_df['gvkey'].nunique() if 'gvkey' in be_df.columns else 0
        linked_gvkeys = linked_df['gvkey'].nunique() if 'gvkey' in linked_df.columns else 0
        coverage = (linked_gvkeys / total_gvkeys * 100) if total_gvkeys else 0.0

        print("\n=== Linking Coverage ===")
        print(f"Total BE gvkeys: {total_gvkeys}")
        print(f"Gvkeys with CRSP match: {linked_gvkeys}")
        print(f"Coverage: {coverage:.2f}%")
        print(f"Unique PERMNOs linked: {linked_df['permno'].nunique() if 'permno' in linked_df.columns else 0}")

        if 'gvkey' in be_df.columns and 'gvkey' in linked_df.columns:
            unmatched = pd.Index(be_df['gvkey'].dropna().unique()).difference(
                pd.Index(linked_df['gvkey'].dropna().unique())
            )
            print(f"Sample unmatched gvkeys: {unmatched[:5].tolist()}")
        else:
            print("Sample unmatched gvkeys: []")

    def build_link(self) -> pd.DataFrame:
        """
        Build gvkey-PERMCO-PERMNO links between Compustat BE and CRSP June market data.

        Returns:
            DataFrame with columns: gvkey, permno, permco, date, cal_year, be, me, sich, is_financial, EXCHCD
        """
        try:
            mapping = self.mapping_manager.load_mapping()
        except FileNotFoundError as exc:
            print(f"Mapping unavailable: {exc}")
            linked = self._empty_link()
            try:
                be = self.be_calculator.load_and_process()
            except FileNotFoundError:
                be = pd.DataFrame(columns=['gvkey'])
            self._print_coverage(be, linked)
            return linked

        crsp = self.crsp_source.load_and_clean()
        be = self.be_calculator.load_and_process()

        if mapping.empty or crsp.empty or be.empty:
            linked = self._empty_link()
            self._print_coverage(be, linked)
            return linked

        mapping = mapping.copy()
        crsp = crsp.copy()
        be = be.copy()

        # Normalize CRSP identifier column names while preserving BE/mapping public names.
        crsp = crsp.rename(columns={'PERMNO': 'permno', 'PERMCO': 'permco'})
        crsp['date'] = pd.to_datetime(crsp['date'])

        mapping['_gvkey_key'] = self._normalize_identifier_key(mapping['gvkey'])
        mapping['_permco_key'] = self._normalize_identifier_key(mapping['permco'])
        be['_gvkey_key'] = self._normalize_identifier_key(be['gvkey'])

        # Use gvkey only to reach stable firm-level PERMCO. Static mapping PERMNOs are
        # intentionally not used for the final issue-level link; CRSP June rows provide
        # the active PERMNO(s) for each PERMCO and date.
        bridge = mapping[['_gvkey_key', 'permco', '_permco_key']].drop_duplicates()
        be_mapped = be.merge(bridge, on='_gvkey_key', how='left')
        be_mapped = be_mapped.dropna(subset=['_permco_key']).copy()
        be_mapped['cal_year'] = pd.to_numeric(be_mapped['cal_year'], errors='coerce')
        be_mapped = be_mapped.dropna(subset=['cal_year']).copy()

        june_crsp = crsp[crsp['date'].dt.month == 6].copy()
        june_crsp['_permco_key'] = self._normalize_identifier_key(june_crsp['permco'])
        june_crsp['link_year'] = june_crsp['date'].dt.year
        june_crsp['me'] = june_crsp['PRC'] * june_crsp['SHROUT']

        be_mapped['link_year'] = be_mapped['cal_year'].astype(int) + 1

        merge_cols = ['permno', '_permco_key', 'date', 'link_year', 'me', 'EXCHCD']
        linked = be_mapped.merge(
            june_crsp[merge_cols],
            on=['_permco_key', 'link_year'],
            how='inner'
        )

        if linked.empty:
            linked = self._empty_link()
            self._print_coverage(be, linked)
            return linked

        linked['cal_year'] = linked['cal_year'].astype(int)
        numeric_permco = pd.to_numeric(linked['_permco_key'], errors='coerce')
        if numeric_permco.notna().all():
            linked['permco'] = numeric_permco.astype(int)
        else:
            linked['permco'] = linked['_permco_key']
        linked = linked[self.LINK_COLUMNS].copy()
        linked = linked.sort_values(['gvkey', 'cal_year', 'permco', 'permno', 'date']).reset_index(drop=True)

        self._print_coverage(be, linked)
        return linked


class BEMECalculator:
    """Computes FF1992-ready annual BE/ME snapshots from linked Compustat-CRSP data."""

    OUTPUT_COLUMNS = [
        'gvkey', 'permno', 'permco', 'date', 'year', 'be', 'me', 'beme',
        'sich', 'is_financial', 'exchange'
    ]

    def __init__(self, engine=None):
        """
        Initialize BEMECalculator.

        Args:
            engine: Optional preconfigured LinkingEngine instance.
        """
        self.engine = engine or LinkingEngine()

    def _empty_result(self) -> pd.DataFrame:
        """Return an empty BE/ME snapshot with the public output schema."""
        return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

    def _print_diagnostics(self, before_count: int, removals: Dict[str, int], result: pd.DataFrame) -> None:
        """Print filter and annual-count diagnostics."""
        print("\n=== BE/ME Filter Diagnostics ===")
        print(f"Total rows before filtering: {before_count}")
        print(f"Rows removed - negative/zero BE: {removals['negative_be']}")
        print(f"Rows removed - financial firms: {removals['financial']}")
        print(f"Rows removed - zero/negative ME: {removals['zero_me']}")
        print(f"Rows removed - BE/ME outliers: {removals['outliers']}")
        print(f"Total rows after filtering: {len(result)}")

        print("\nPer-year observation counts:")
        if result.empty:
            print("No observations after filtering")
            return

        counts = result.groupby('year').size().sort_index()
        print(counts.to_string())

        target_years = range(1964, 1992)
        thin_years = {year: int(counts.get(year, 0)) for year in target_years if counts.get(year, 0) < 500}
        if thin_years:
            print(f"Warning: years with fewer than 500 observations: {thin_years}")

    def _trim_beme_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim top and bottom 0.5% BE/ME observations within each year."""
        if df.empty:
            return df

        lower = df.groupby('year')['beme'].transform(lambda values: values.quantile(0.005))
        upper = df.groupby('year')['beme'].transform(lambda values: values.quantile(0.995))
        keep_mask = df['beme'].between(lower, upper, inclusive='both')
        return df[keep_mask].copy()

    def compute_all(self, linked_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute clean annual BE/ME snapshots from linked Compustat-CRSP output.

        Args:
            linked_df: Optional linked DataFrame. If omitted, uses self.engine.build_link().

        Returns:
            DataFrame with columns gvkey, permno, permco, date, year, be, me, beme,
            sich, is_financial, exchange.
        """
        linked = self.engine.build_link() if linked_df is None else linked_df
        if linked is None or linked.empty:
            result = self._empty_result()
            self._print_diagnostics(0, {'negative_be': 0, 'financial': 0, 'zero_me': 0, 'outliers': 0}, result)
            return result

        df = linked.copy()
        before_count = len(df)
        removals = {}

        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['be'] = pd.to_numeric(df['be'], errors='coerce')
        df['me'] = pd.to_numeric(df['me'], errors='coerce')
        df['beme'] = np.where(df['me'] != 0, df['be'] / df['me'], np.nan)
        df['beme'] = pd.to_numeric(df['beme'], errors='coerce').replace([np.inf, -np.inf], np.nan)

        if 'EXCHCD' in df.columns:
            df['exchange'] = df['EXCHCD']
        elif 'exchange' not in df.columns:
            df['exchange'] = pd.NA

        current = len(df)
        df = df[df['be'] > 0].copy()
        removals['negative_be'] = current - len(df)

        current = len(df)
        if 'is_financial' in df.columns:
            financial_mask = df['is_financial'].fillna(False).astype(bool)
            df = df[~financial_mask].copy()
        removals['financial'] = current - len(df)

        current = len(df)
        df = df[df['me'] > 0].copy()
        removals['zero_me'] = current - len(df)

        df = df.dropna(subset=['beme']).copy()
        current = len(df)
        df = self._trim_beme_outliers(df)
        removals['outliers'] = current - len(df)

        result = df[self.OUTPUT_COLUMNS].sort_values(['year', 'gvkey', 'permco', 'permno', 'date']).reset_index(drop=True)
        self._print_diagnostics(before_count, removals, result)
        return result


class PortfolioConstructor:
    """Constructs 25 Size-BE/ME portfolios using FF1992/FF1993 June sorts."""

    PORTFOLIO_COLUMNS = [
        'SMALL LoBM', 'ME1 BM2', 'ME1 BM3', 'ME1 BM4', 'SMALL HiBM',
        'ME2 BM1', 'ME2 BM2', 'ME2 BM3', 'ME2 BM4', 'ME2 BM5',
        'ME3 BM1', 'ME3 BM2', 'ME3 BM3', 'ME3 BM4', 'ME3 BM5',
        'ME4 BM1', 'ME4 BM2', 'ME4 BM3', 'ME4 BM4', 'ME4 BM5',
        'BIG LoBM', 'ME5 BM2', 'ME5 BM3', 'ME5 BM4', 'BIG HiBM',
    ]

    PORTFOLIO_LABELS = {
        (1, 1): 'SMALL LoBM', (1, 2): 'ME1 BM2', (1, 3): 'ME1 BM3',
        (1, 4): 'ME1 BM4', (1, 5): 'SMALL HiBM',
        (2, 1): 'ME2 BM1', (2, 2): 'ME2 BM2', (2, 3): 'ME2 BM3',
        (2, 4): 'ME2 BM4', (2, 5): 'ME2 BM5',
        (3, 1): 'ME3 BM1', (3, 2): 'ME3 BM2', (3, 3): 'ME3 BM3',
        (3, 4): 'ME3 BM4', (3, 5): 'ME3 BM5',
        (4, 1): 'ME4 BM1', (4, 2): 'ME4 BM2', (4, 3): 'ME4 BM3',
        (4, 4): 'ME4 BM4', (4, 5): 'ME4 BM5',
        (5, 1): 'BIG LoBM', (5, 2): 'ME5 BM2', (5, 3): 'ME5 BM3',
        (5, 4): 'ME5 BM4', (5, 5): 'BIG HiBM',
    }

    PORTFOLIO_COLUMNS_6 = [
        'SMALL LoBM', 'ME1 BM2', 'SMALL HiBM',
        'BIG LoBM', 'ME2 BM2', 'BIG HiBM',
    ]

    PORTFOLIO_LABELS_6 = {
        (1, 1): 'SMALL LoBM',
        (1, 2): 'ME1 BM2',
        (1, 3): 'SMALL HiBM',
        (2, 1): 'BIG LoBM',
        (2, 2): 'ME2 BM2',
        (2, 3): 'BIG HiBM',
    }

    def __init__(self, beme_calculator=None, crsp_source=None):
        """
        Initialize PortfolioConstructor.

        Args:
            beme_calculator: Optional preconfigured BEMECalculator instance.
            crsp_source: Optional preconfigured CRSPSource instance.
        """
        self.beme_calculator = beme_calculator or BEMECalculator()
        self.crsp_source = crsp_source or CRSPSource()

    def compute_nyse_breakpoints(self, beme_df: pd.DataFrame, metric: str = 'me') -> List[float]:
        """
        Compute NYSE non-financial quintile breakpoints for ME or BE/ME.

        Args:
            beme_df: Annual BE/ME snapshot for one June formation year.
            metric: Either 'me' for size or 'beme' for book-to-market.

        Returns:
            Four breakpoint values: 20th, 40th, 60th, and 80th percentiles.
        """
        if metric not in {'me', 'beme'}:
            raise ValueError("metric must be either 'me' or 'beme'")

        if beme_df is None or beme_df.empty:
            return [np.nan, np.nan, np.nan, np.nan]

        df = beme_df.copy()
        values = pd.to_numeric(df[metric], errors='coerce')
        exchange = pd.to_numeric(df.get('exchange', pd.Series(index=df.index, dtype='float64')), errors='coerce')
        financial = df.get('is_financial', pd.Series(False, index=df.index)).fillna(False).astype(bool)

        nyse = values[(exchange == 1) & (~financial) & values.notna() & np.isfinite(values)]
        if nyse.empty:
            return [np.nan, np.nan, np.nan, np.nan]

        return [float(nyse.quantile(q)) for q in [0.2, 0.4, 0.6, 0.8]]

    def assign_portfolios(self, beme_df: pd.DataFrame, size_bps: List[float], beme_bps: List[float]) -> pd.Series:
        """
        Assign all stocks to 25 Size-BE/ME portfolios using NYSE breakpoints.

        Args:
            beme_df: Annual BE/ME snapshot for one June formation year.
            size_bps: ME breakpoints from compute_nyse_breakpoints(..., 'me').
            beme_bps: BE/ME breakpoints from compute_nyse_breakpoints(..., 'beme').

        Returns:
            Series of Ken French style portfolio labels.
        """
        if beme_df is None or beme_df.empty:
            return pd.Series(dtype='object')

        size_breaks = np.asarray(size_bps, dtype='float64')
        beme_breaks = np.asarray(beme_bps, dtype='float64')
        if len(size_breaks) != 4 or len(beme_breaks) != 4:
            raise ValueError('size_bps and beme_bps must each contain four breakpoints')

        me_values = pd.to_numeric(beme_df['me'], errors='coerce')
        beme_values = pd.to_numeric(beme_df['beme'], errors='coerce')

        labels = []
        invalid_breaks = np.isnan(size_breaks).any() or np.isnan(beme_breaks).any()
        for me_value, beme_value in zip(me_values, beme_values):
            if invalid_breaks or pd.isna(me_value) or pd.isna(beme_value):
                labels.append(pd.NA)
                continue

            size_group = int(np.searchsorted(size_breaks, me_value, side='left') + 1)
            beme_group = int(np.searchsorted(beme_breaks, beme_value, side='left') + 1)
            labels.append(self.PORTFOLIO_LABELS[(size_group, beme_group)])

        return pd.Series(labels, index=beme_df.index, dtype='object')

    def compute_nyse_breakpoints_2x3(self, beme_df: pd.DataFrame, metric: str = 'me') -> List[float]:
        """
        Compute NYSE non-financial breakpoints for 2×3 portfolios.

        For 'me': returns [median] (50th percentile of NYSE ME).
        For 'beme': returns [p30, p70] (30th and 70th percentiles of NYSE BE/ME).

        Args:
            beme_df: Annual BE/ME snapshot for one June formation year.
            metric: Either 'me' for size or 'beme' for book-to-market.

        Returns:
            For 'me': list with one breakpoint [median].
            For 'beme': list with two breakpoints [p30, p70].
        """
        if metric not in {'me', 'beme'}:
            raise ValueError("metric must be either 'me' or 'beme'")

        if beme_df is None or beme_df.empty:
            return [np.nan] if metric == 'me' else [np.nan, np.nan]

        df = beme_df.copy()
        values = pd.to_numeric(df[metric], errors='coerce')
        exchange = pd.to_numeric(df.get('exchange', pd.Series(index=df.index, dtype='float64')), errors='coerce')
        financial = df.get('is_financial', pd.Series(False, index=df.index)).fillna(False).astype(bool)

        nyse = values[(exchange == 1) & (~financial) & values.notna() & np.isfinite(values)]
        if nyse.empty:
            return [np.nan] if metric == 'me' else [np.nan, np.nan]

        if metric == 'me':
            return [float(nyse.quantile(0.5))]
        else:
            return [float(nyse.quantile(0.3)), float(nyse.quantile(0.7))]

    def assign_portfolios_2x3(self, beme_df: pd.DataFrame, size_bp: List[float], beme_bps: List[float]) -> pd.Series:
        """
        Assign all stocks to 6 Size-BE/ME portfolios using NYSE breakpoints.

        Size: 2 groups based on median (below = SMALL, above = BIG).
        BE/ME: 3 groups (below p30 = LoBM, between p30-p70 = BM2, above p70 = HiBM).

        Args:
            beme_df: Annual BE/ME snapshot for one June formation year.
            size_bp: ME breakpoints from compute_nyse_breakpoints_2x3(..., 'me').
            beme_bps: BE/ME breakpoints from compute_nyse_breakpoints_2x3(..., 'beme').

        Returns:
            Series of Ken French style 6-portfolio labels.
        """
        if beme_df is None or beme_df.empty:
            return pd.Series(dtype='object')

        size_bp_arr = np.asarray(size_bp, dtype='float64')
        beme_bps_arr = np.asarray(beme_bps, dtype='float64')

        me_values = pd.to_numeric(beme_df['me'], errors='coerce')
        beme_values = pd.to_numeric(beme_df['beme'], errors='coerce')

        labels = []
        invalid_breaks = np.isnan(size_bp_arr).any() or np.isnan(beme_bps_arr).any()
        for me_value, beme_value in zip(me_values, beme_values):
            if invalid_breaks or pd.isna(me_value) or pd.isna(beme_value):
                labels.append(pd.NA)
                continue

            size_group = int(np.searchsorted(size_bp_arr, me_value, side='left') + 1)
            beme_group = int(np.searchsorted(beme_bps_arr, beme_value, side='left') + 1)
            labels.append(self.PORTFOLIO_LABELS_6[(size_group, beme_group)])

        return pd.Series(labels, index=beme_df.index, dtype='object')

    def build_6_portfolios(self, beme_df: Optional[pd.DataFrame] = None, crsp_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build value-weighted monthly returns for 6 Size-BE/ME portfolios (2×3).

        Same structure as build_25_portfolios but with 2×3 assignment.
        NYSE breakpoints: Size = median (50%), BE/ME = 30th and 70th percentiles.
        All stocks are assigned; returns are value-weighted.

        Args:
            beme_df: Optional annual June BE/ME snapshot. If omitted, BEMECalculator is used.
            crsp_df: Optional CRSP monthly returns. If omitted, CRSPSource is used.

        Returns:
            DataFrame indexed by YYYY-MM strings with 6 portfolio return columns.
            Returns are expressed in percent to match Ken French CSV files.
        """
        beme = self.beme_calculator.compute_all() if beme_df is None else beme_df
        crsp = self.crsp_source.load_and_clean() if crsp_df is None else crsp_df

        if beme is None or crsp is None or beme.empty or crsp.empty:
            empty = pd.DataFrame(columns=self.PORTFOLIO_COLUMNS_6)
            empty.index.name = 'Date'
            return empty

        beme = self._prepare_beme(beme)
        crsp = self._prepare_crsp(crsp)

        formation_years = sorted(beme['year'].dropna().astype(int).unique())
        target_years = [year for year in formation_years if 1964 <= year <= 1991]
        if target_years:
            formation_years = target_years

        rows = []
        row_index = []
        for formation_year in formation_years:
            snapshot = self._formation_snapshot(beme, formation_year)
            if snapshot.empty:
                continue

            size_bp = self.compute_nyse_breakpoints_2x3(snapshot, metric='me')
            beme_bps = self.compute_nyse_breakpoints_2x3(snapshot, metric='beme')
            assignments = self.assign_portfolios_2x3(snapshot, size_bp, beme_bps)
            formation = snapshot.assign(portfolio=assignments, formation_me=snapshot['me'])
            formation = formation.dropna(subset=['portfolio', 'formation_me'])
            if formation.empty:
                continue

            formation = formation[['permno', 'portfolio', 'formation_me']].copy()
            for month in self._holding_months(formation_year):
                month_returns = crsp.loc[crsp['_month'] == month, ['permno', 'RET']]
                merged = formation.merge(month_returns, on='permno', how='left')
                valid_returns = merged.dropna(subset=['RET']).copy()
                valid_returns = valid_returns[valid_returns['formation_me'] > 0]

                row = {column: np.nan for column in self.PORTFOLIO_COLUMNS_6}
                for portfolio, group in valid_returns.groupby('portfolio'):
                    weights = group['formation_me'].astype(float)
                    returns = group['RET'].astype(float)
                    if weights.sum() > 0:
                        row[portfolio] = float(np.average(returns, weights=weights) * 100.0)

                rows.append(row)
                row_index.append(month.strftime('%Y-%m'))

        result = pd.DataFrame(rows, index=row_index, columns=self.PORTFOLIO_COLUMNS_6)
        result.index.name = 'Date'
        return result

    def _prepare_beme(self, beme_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize BE/ME input columns needed for portfolio construction."""
        df = beme_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['permno'] = pd.to_numeric(df['permno'], errors='coerce')
        df['me'] = pd.to_numeric(df['me'], errors='coerce')
        df['beme'] = pd.to_numeric(df['beme'], errors='coerce')
        if 'is_financial' not in df.columns:
            df['is_financial'] = False
        return df

    def _prepare_crsp(self, crsp_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize CRSP monthly returns for holding-period calculations."""
        df = crsp_df.copy()
        rename_map = {'PERMNO': 'permno', 'PERMCO': 'permco', 'EXCHCD': 'exchange'}
        df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})
        df['permno'] = pd.to_numeric(df['permno'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df['_month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
        return df

    def _formation_snapshot(self, beme_df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Return valid non-financial June formation records for one year."""
        snapshot = beme_df[beme_df['year'] == year].copy()
        june_snapshot = snapshot[snapshot['date'].dt.month == 6].copy()
        if not june_snapshot.empty:
            snapshot = june_snapshot

        financial = snapshot['is_financial'].fillna(False).astype(bool)
        valid = (
            (~financial)
            & snapshot['permno'].notna()
            & snapshot['me'].notna()
            & snapshot['beme'].notna()
            & np.isfinite(snapshot['me'])
            & np.isfinite(snapshot['beme'])
            & (snapshot['me'] > 0)
        )
        return snapshot[valid].copy()

    def _holding_months(self, formation_year: int) -> pd.DatetimeIndex:
        """Return July t through June t+1 holding months, capped at 1991-12."""
        start = pd.Timestamp(year=formation_year, month=7, day=1)
        end = pd.Timestamp(year=formation_year + 1, month=6, day=1)
        cap = pd.Timestamp(year=1991, month=12, day=1)
        if formation_year <= 1991:
            end = min(end, cap)
        return pd.date_range(start, end, freq='MS')

    def build_25_portfolios(self, beme_df: Optional[pd.DataFrame] = None, crsp_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build value-weighted monthly returns for 25 Size-BE/ME portfolios.

        Args:
            beme_df: Optional annual June BE/ME snapshot. If omitted, BEMECalculator is used.
            crsp_df: Optional CRSP monthly returns. If omitted, CRSPSource is used.

        Returns:
            DataFrame indexed by YYYY-MM strings with 25 portfolio return columns.
            Returns are expressed in percent to match Ken French CSV files.
        """
        beme = self.beme_calculator.compute_all() if beme_df is None else beme_df
        crsp = self.crsp_source.load_and_clean() if crsp_df is None else crsp_df

        if beme is None or crsp is None or beme.empty or crsp.empty:
            empty = pd.DataFrame(columns=self.PORTFOLIO_COLUMNS)
            empty.index.name = 'Date'
            return empty

        beme = self._prepare_beme(beme)
        crsp = self._prepare_crsp(crsp)

        formation_years = sorted(beme['year'].dropna().astype(int).unique())
        target_years = [year for year in formation_years if 1964 <= year <= 1991]
        if target_years:
            formation_years = target_years

        rows = []
        row_index = []
        for formation_year in formation_years:
            snapshot = self._formation_snapshot(beme, formation_year)
            if snapshot.empty:
                continue

            size_bps = self.compute_nyse_breakpoints(snapshot, metric='me')
            beme_bps = self.compute_nyse_breakpoints(snapshot, metric='beme')
            assignments = self.assign_portfolios(snapshot, size_bps, beme_bps)
            formation = snapshot.assign(portfolio=assignments, formation_me=snapshot['me'])
            formation = formation.dropna(subset=['portfolio', 'formation_me'])
            if formation.empty:
                continue

            formation = formation[['permno', 'portfolio', 'formation_me']].copy()
            for month in self._holding_months(formation_year):
                month_returns = crsp.loc[crsp['_month'] == month, ['permno', 'RET']]
                merged = formation.merge(month_returns, on='permno', how='left')
                valid_returns = merged.dropna(subset=['RET']).copy()
                valid_returns = valid_returns[valid_returns['formation_me'] > 0]

                row = {column: np.nan for column in self.PORTFOLIO_COLUMNS}
                for portfolio, group in valid_returns.groupby('portfolio'):
                    weights = group['formation_me'].astype(float)
                    returns = group['RET'].astype(float)
                    if weights.sum() > 0:
                        row[portfolio] = float(np.average(returns, weights=weights) * 100.0)

                rows.append(row)
                row_index.append(month.strftime('%Y-%m'))

        result = pd.DataFrame(rows, index=row_index, columns=self.PORTFOLIO_COLUMNS)
        result.index.name = 'Date'
        return result


class BackupManager:
    """Manages backups and restoration of Fama-French data files."""

    def __init__(self, data_dir: str = 'data'):
        """
        Initialize BackupManager.

        Args:
            data_dir: Directory containing Fama-French data files
        """
        self.data_dir = data_dir
        self.backup_dir = os.path.join(data_dir, 'backups')
        self.files_to_backup = [
            'ff_25_portfolios.csv',
            'ff_6_portfolios.csv',
            'ff_factors.csv'
        ]

    def _compute_file_hash(self, filepath: str) -> str:
        """
        Compute MD5 hash of a file.

        Args:
            filepath: Path to file

        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def backup(self) -> str:
        """
        Create backup of Fama-French data files.

        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        backup_path = os.path.join(self.backup_dir, f'{timestamp}-pre-compustat')

        # Check if backup already exists with same files
        if os.path.exists(self.backup_dir):
            existing_backups = [d for d in os.listdir(self.backup_dir) if d.startswith('pre-compustat')]
        else:
            existing_backups = []
        if existing_backups:
            latest_backup = sorted(existing_backups)[-1]
            latest_path = os.path.join(self.backup_dir, latest_backup)

            # Check if files are the same
            hash_match = True
            for filename in self.files_to_backup:
                src_path = os.path.join(self.data_dir, filename)
                backup_src_path = os.path.join(latest_path, filename)

                if not os.path.exists(src_path) or not os.path.exists(backup_src_path):
                    hash_match = False
                    break

                if self._compute_file_hash(src_path) != self._compute_file_hash(backup_src_path):
                    hash_match = False
                    break

            if hash_match:
                print(f"Backup already exists: {latest_path}")
                return latest_path

        # Create backup directory
        os.makedirs(backup_path, exist_ok=True)

        # Copy files
        for filename in self.files_to_backup:
            src_path = os.path.join(self.data_dir, filename)
            dst_path = os.path.join(backup_path, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")

        print(f"Backup created: {backup_path}")
        return backup_path

    def restore(self, backup_path: str) -> None:
        """
        Restore Fama-French data files from backup.

        Args:
            backup_path: Path to backup directory
        """
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        # Verify backup contains all required files
        for filename in self.files_to_backup:
            backup_src_path = os.path.join(backup_path, filename)
            dst_path = os.path.join(self.data_dir, filename)

            if not os.path.exists(backup_src_path):
                raise FileNotFoundError(f"Backup missing file: {filename}")

            # Copy back
            shutil.copy2(backup_src_path, dst_path)
            print(f"Restored: {filename}")

        print(f"Restored from: {backup_path}")


class FactorCalculator:
    """Computes Fama-French factors (SMB, HML) from 6 portfolios and market returns from CRSP."""

    def __init__(self, portfolio_constructor=None, crsp_source=None):
        """
        Initialize FactorCalculator.

        Args:
            portfolio_constructor: Optional preconfigured PortfolioConstructor instance.
            crsp_source: Optional preconfigured CRSPSource instance.
        """
        self.portfolio_constructor = portfolio_constructor or PortfolioConstructor()
        self.crsp_source = crsp_source or CRSPSource()

    def compute_smb_hml(self, port6: Optional[pd.DataFrame] = None) -> tuple[pd.Series, pd.Series]:
        """
        Compute SMB and HML factors from 6 Size-BE/ME portfolios.

        Formulas (Fama-French 1993):
            SMB = 1/3*(SMALL HiBM + ME1 BM2 + SMALL LoBM) - 1/3*(BIG HiBM + ME2 BM2 + BIG LoBM)
            HML = 1/2*(SMALL HiBM + BIG HiBM) - 1/2*(SMALL LoBM + BIG LoBM)

        Args:
            port6: Optional DataFrame with 6 portfolio returns (Date index, percent).
                   If omitted, PortfolioConstructor.build_6_portfolios() is used.

        Returns:
            Tuple of (SMB, HML) as pandas Series indexed by Date.
            Returns are in percent.
        """
        if port6 is None:
            port6 = self.portfolio_constructor.build_6_portfolios()

        if port6 is None or port6.empty:
            empty_smb = pd.Series(dtype=float)
            empty_hml = pd.Series(dtype=float)
            return empty_smb, empty_hml

        # Ensure port6 has the correct column names
        required_columns = self.portfolio_constructor.PORTFOLIO_COLUMNS_6
        if not all(col in port6.columns for col in required_columns):
            raise ValueError(f"port6 must contain columns: {required_columns}")

        # Compute SMB
        smb = (
            (1 / 3) * (
                port6['SMALL HiBM']
                + port6['ME1 BM2']
                + port6['SMALL LoBM']
            )
            - (1 / 3) * (
                port6['BIG HiBM']
                + port6['ME2 BM2']
                + port6['BIG LoBM']
            )
        )

        # Compute HML
        hml = (
            (1 / 2) * (port6['SMALL HiBM'] + port6['BIG HiBM'])
            - (1 / 2) * (port6['SMALL LoBM'] + port6['BIG LoBM'])
        )

        # Ensure index is Date strings
        return smb, hml

    def compute_market_return(self, crsp_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Compute value-weighted market return for each month from CRSP data.

        VW return = sum(RET * lagged_ME) / sum(lagged_ME) for each month
        where lagged_ME = SHROUT * abs(PRC) from the previous month.

        Args:
            crsp_df: Optional CRSP monthly returns DataFrame with PERMNO, date, RET, PRC, SHROUT.
                     If omitted, CRSPSource.load_and_clean() is used.

        Returns:
            Series indexed by Date (YYYY-MM strings) with value-weighted market returns in percent.
        """
        if crsp_df is None:
            crsp_df = self.crsp_source.load_and_clean()

        if crsp_df is None or crsp_df.empty:
            return pd.Series(dtype=float, name='Mkt-RF')

        # Make a copy to avoid modifying the original
        df = crsp_df.copy()

        # Ensure required columns exist
        required_cols = ['PERMNO', 'date', 'RET', 'PRC', 'SHROUT']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CRSP DataFrame must contain column: {col}")

        df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
        df['PRC'] = pd.to_numeric(df['PRC'], errors='coerce')
        df['SHROUT'] = pd.to_numeric(df['SHROUT'], errors='coerce')

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Get month identifier (for grouping)
        df['_month'] = df['date'].dt.to_period('M').dt.to_timestamp()

        # Compute market equity (ME) with lag
        # ME for month t is calculated using ME from month t-1 (lagged)
        # First, compute ME for each row
        df['ME'] = df['PRC'].abs() * df['SHROUT']

        # Then compute lagged ME by PERMNO
        df['_lagged_ME'] = df.groupby('PERMNO')['ME'].shift(1)

        # Value-weighted return calculation
        def compute_vw_return(group):
            """Compute value-weighted return for a group of stocks in a month."""
            valid = group.dropna(subset=['RET', '_lagged_ME'])
            valid = valid[valid['_lagged_ME'] > 0]

            if valid.empty:
                return pd.Series({'vw_return': np.nan})

            weights = valid['_lagged_ME'].astype(float)
            returns = valid['RET'].astype(float)

            vw_return = (weights * returns).sum() / weights.sum()
            return pd.Series({'vw_return': vw_return})

        # Group by month and compute VW returns
        vw_returns = df.groupby('_month').apply(compute_vw_return, include_groups=False)

        # Convert to Series and rename
        result = vw_returns['vw_return'].rename('Mkt-RF')

        # Debug: print the index
        print(f"DEBUG: vw_returns.index = {vw_returns.index}")
        print(f"DEBUG: result.index = {result.index}")

        # Convert index to YYYY-MM strings
        result.index = result.index.strftime('%Y-%m')

        # Return in percent (CRSP returns are typically in decimal, so multiply by 100)
        result = result * 100.0

        return result

    def assemble_factors(
        self,
        port6: Optional[pd.DataFrame] = None,
        crsp_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Assemble final factor file with Date, Mkt-RF, SMB, HML, RF columns.

        Args:
            port6: Optional 6-portfolio returns DataFrame. If omitted, computed via
                   compute_smb_hml().
            crsp_df: Optional CRSP returns DataFrame. If omitted, computed via
                     compute_market_return().

        Returns:
            DataFrame with columns: Date, Mkt-RF, SMB, HML, RF
            Index is Date (YYYY-MM strings).
            All values in percent.
            Date range: 1964-07 to 1991-12.
        """
        # Get SMB and HML
        if port6 is None:
            port6 = self.portfolio_constructor.build_6_portfolios()

        smb, hml = self.compute_smb_hml(port6)

        # Get market VW return
        mkt_return = self.compute_market_return(crsp_df)

        # Align all factors on the same dates
        # Get common dates (intersection of port6 index and mkt_return index)
        common_dates = port6.index.intersection(mkt_return.index)

        # Filter to common dates
        smb_aligned = smb.loc[common_dates]
        hml_aligned = hml.loc[common_dates]
        mkt_return_aligned = mkt_return.loc[common_dates]

        # Load RF from Ken French data
        rf_path = os.path.join(self.crsp_source.base_dir, 'data', 'ff_factors.csv')
        if os.path.exists(rf_path):
            rf_df = pd.read_csv(rf_path)
            rf_df = rf_df.copy()
            rf_df['Date'] = pd.to_datetime(rf_df['Date']).dt.strftime('%Y-%m')
            rf_series = rf_df.set_index('Date')['RF']
        else:
            raise FileNotFoundError(f"RF data not found at: {rf_path}")

        rf_aligned = rf_series.reindex(common_dates)

        # Create date alignment
        combined = pd.DataFrame({
            'Date': list(common_dates),
            'Mkt-RF': (mkt_return_aligned - rf_aligned).values,
            'SMB': smb_aligned.values,
            'HML': hml_aligned.values,
            'RF': rf_aligned.values,
        })
        combined = combined.reset_index(drop=True)

        # Filter to date range 1964-07 to 1991-12
        start_date = '1964-07'
        end_date = '1991-12'
        combined = combined[
            (combined['Date'] >= start_date) &
            (combined['Date'] <= end_date)
        ].copy()

        # Sort by date
        combined = combined.sort_values('Date').reset_index(drop=True)

        # Format Date as YYYY-MM (from YYYY-MM-DD)
        combined['Date'] = pd.to_datetime(combined['Date']).dt.strftime('%Y-%m')

        # Keep only required columns
        result = combined[['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']].copy()
        result.index.name = 'Date'

        return result


class DiagnosticsReport:
    """Generates comprehensive quality diagnostics for self-constructed portfolios.

    Runs five diagnostic sections:
    1. Mapping coverage (gvkey↔PERMCO bridge statistics)
    2. Portfolio diagnostics (stock counts per cell across 25 portfolios)
    3. Ken French comparison (correlation with original Ken French data)
    4. Factor statistics (mean, std, t-stat; MAD/correlation vs Ken French)
    5. Size/BE-ME monotonicity (verification of FF1993 effects)
    """

    PORTFOLIO_COLUMNS_25 = [
        'SMALL LoBM', 'ME1 BM2', 'ME1 BM3', 'ME1 BM4', 'SMALL HiBM',
        'ME2 BM1', 'ME2 BM2', 'ME2 BM3', 'ME2 BM4', 'ME2 BM5',
        'ME3 BM1', 'ME3 BM2', 'ME3 BM3', 'ME3 BM4', 'ME3 BM5',
        'ME4 BM1', 'ME4 BM2', 'ME4 BM3', 'ME4 BM4', 'ME4 BM5',
        'BIG LoBM', 'ME5 BM2', 'ME5 BM3', 'ME5 BM4', 'BIG HiBM',
    ]

    def __init__(
        self,
        portfolio_constructor: Optional[object] = None,
        factor_calculator: Optional[object] = None,
        mapping_manager: Optional[object] = None,
        linking_engine: Optional[object] = None,
        data_dir: str = 'data',
    ):
        """
        Initialize DiagnosticsReport.

        Args:
            portfolio_constructor: Optional preconfigured PortfolioConstructor.
            factor_calculator: Optional preconfigured FactorCalculator.
            mapping_manager: Optional preconfigured MappingManager.
            linking_engine: Optional preconfigured LinkingEngine.
            data_dir: Data directory path.
        """
        self.pc = portfolio_constructor or PortfolioConstructor()
        self.fc = factor_calculator or FactorCalculator(portfolio_constructor=self.pc)
        self.mm = mapping_manager or MappingManager(cache_dir=data_dir)
        self.le = linking_engine or LinkingEngine()
        self.data_dir = data_dir

        # Auto-detect most recent pre-compustat backup directory
        backup_glob = os.path.join(data_dir, 'backups', '*-pre-compustat')
        backup_dirs = sorted(glob.glob(backup_glob))
        self.backup_dir = backup_dirs[-1] if backup_dirs else None

    def run_full(self) -> Dict:
        """
        Run all five diagnostic sections and return results.

        Returns:
            Dictionary with keys for each diagnostic section.
            Results are printed to console and saved as evidence files.
        """
        results: Dict = {}

        print("\n" + "=" * 70)
        print("DIAGNOSTICS REPORT: Self-Constructed Portfolios")
        print("=" * 70)

        results['mapping_coverage'] = self._mapping_coverage()
        results['portfolio_diagnostics'] = self._portfolio_diagnostics()
        results['ken_french_comparison'] = self._ken_french_comparison()
        results['factor_statistics'] = self._factor_statistics()
        results['monotonicity'] = self._monotonicity_check()

        self._print_report(results)
        self._save_report(results)

        return results

    # ------------------------------------------------------------------ #
    # Section 1: Mapping Coverage
    # ------------------------------------------------------------------ #

    def _mapping_coverage(self) -> Dict:
        """Report gvkey↔PERMCO↔PERMNO mapping coverage statistics."""
        print("\n" + "-" * 70)
        print("SECTION 1: Mapping Coverage")
        print("-" * 70)

        try:
            stats = self.mm.mapping_stats()
        except (FileNotFoundError, Exception) as exc:
            result = {'error': str(exc)}
            print(f"  ERROR: Could not compute mapping stats: {exc}")
            return result

        result = {
            'total_mapping_rows': stats['total_rows'],
            'unique_gvkeys': stats['unique_gvkeys'],
            'unique_permco': stats['unique_permco'],
            'unique_permno': stats['unique_permno'],
            'gvkey_multi_permco_pct': stats['gvkey_multi_permco_pct'],
            'permco_multi_gvkey_pct': stats['permco_multi_gvkey_pct'],
        }
        print(f"  Total mapping rows : {stats['total_rows']:>8,}")
        print(f"  Unique gvkeys      : {stats['unique_gvkeys']:>8,}")
        print(f"  Unique PERMCO      : {stats['unique_permco']:>8,}")
        print(f"  Unique PERMNO      : {stats['unique_permno']:>8,}")
        print(f"  gvkey→multi-PERMCO : {stats['gvkey_multi_permco_pct']:>7.2f}%")
        print(f"  PERMCO→multi-gvkey : {stats['permco_multi_gvkey_pct']:>7.2f}%")

        # Also report LinkingEngine coverage
        try:
            linked = self.le.build_link()
            if linked is not None and not linked.empty:
                n_gvkey = linked['gvkey'].nunique()
                n_permno = linked['permno'].nunique()
                n_permco = linked['permco'].nunique()
                result['linked_gvkeys'] = int(n_gvkey)
                result['linked_permnos'] = int(n_permno)
                result['linked_permcos'] = int(n_permco)
                print(f"  Linked gvkeys      : {n_gvkey:>8,}")
                print(f"  Linked PERMNOs     : {n_permno:>8,}")
                print(f"  Linked PERMCOs     : {n_permco:>8,}")
            else:
                result['linked_gvkeys'] = 0
                result['linked_permnos'] = 0
                result['linked_permcos'] = 0
                print("  Linked data        : (empty — no link available)")
        except (Exception, ) as exc:
            result['link_error'] = str(exc)
            print(f"  Link build error   : {exc}")

        return result

    # ------------------------------------------------------------------ #
    # Section 2: Portfolio Diagnostics — stock counts per cell
    # ------------------------------------------------------------------ #

    def _portfolio_diagnostics(self) -> Dict:
        """Count stocks per portfolio per month across all formation years."""
        print("\n" + "-" * 70)
        print("SECTION 2: Portfolio Diagnostics — Stock Counts Per Cell")
        print("-" * 70)

        try:
            beme = self.pc.beme_calculator.compute_all()
            crsp = self.pc.crsp_source.load_and_clean()
        except (FileNotFoundError, Exception) as exc:
            result: Dict = {'error': str(exc)}
            print(f"  ERROR: Could not compute portfolio diagnostics: {exc}")
            return result

        if beme.empty or crsp.empty:
            result = {'error': 'BE/ME or CRSP data is empty'}
            print("  ERROR: BE/ME or CRSP data is empty")
            return result

        beme = self.pc._prepare_beme(beme) if hasattr(self.pc, '_prepare_beme') else beme
        crsp = self.pc._prepare_crsp(crsp) if hasattr(self.pc, '_prepare_crsp') else crsp

        formation_years = sorted(beme['year'].dropna().astype(int).unique())
        target_years = [y for y in formation_years if 1964 <= y <= 1991]
        if target_years:
            formation_years = target_years

        # Build stock counts: for each formation year and holding month,
        # count stocks assigned to each portfolio that have valid returns
        count_records = []
        for formation_year in formation_years:
            snapshot = self.pc._formation_snapshot(beme, formation_year)
            if snapshot.empty:
                continue

            size_bps = self.pc.compute_nyse_breakpoints(snapshot, metric='me')
            beme_bps = self.pc.compute_nyse_breakpoints(snapshot, metric='beme')
            assignments = self.pc.assign_portfolios(snapshot, size_bps, beme_bps)
            formation = snapshot.assign(portfolio=assignments, formation_me=snapshot['me'])
            formation = formation.dropna(subset=['portfolio', 'formation_me'])
            if formation.empty:
                continue

            formation = formation[['permno', 'portfolio', 'formation_me']].copy()
            for month in self.pc._holding_months(formation_year):
                month_returns = crsp.loc[crsp['_month'] == month, ['permno', 'RET']]
                merged = formation.merge(month_returns, on='permno', how='left')
                valid_returns = merged.dropna(subset=['RET']).copy()
                valid_returns = valid_returns[valid_returns['formation_me'] > 0]

                for portfolio, group in valid_returns.groupby('portfolio'):
                    count_records.append({
                        'month': month.strftime('%Y-%m'),
                        'portfolio': portfolio,
                        'stock_count': len(group),
                    })

        if not count_records:
            result = {'cell_counts_empty': True}
            print("  No portfolio cell counts available.")
            return result

        counts_df = pd.DataFrame(count_records)

        # Per-portfolio summary statistics
        summary = (
            counts_df.groupby('portfolio')['stock_count']
            .agg(['min', 'max', 'mean', 'count'])
            .round(1)
        )
        summary.columns = ['min_stocks', 'max_stocks', 'mean_stocks', 'n_months']

        # Identify empty cells (portfolios with 0 stocks in some months)
        all_portfolios = self.PORTFOLIO_COLUMNS_25
        all_months = sorted(counts_df['month'].unique())
        all_combos = pd.MultiIndex.from_product(
            [all_months, all_portfolios],
            names=['month', 'portfolio']
        )
        full_grid = (
            counts_df.set_index(['month', 'portfolio'])
            .reindex(all_combos)
            .fillna(0)
            .reset_index()
        )
        empty_cells = full_grid[full_grid['stock_count'] == 0]
        n_empty = len(empty_cells)

        # Compute min/mean/max across all cells
        nonzero = full_grid[full_grid['stock_count'] > 0]
        cell_min = int(nonzero['stock_count'].min()) if not nonzero.empty else 0
        cell_mean = float(nonzero['stock_count'].mean()) if not nonzero.empty else 0.0
        cell_max = int(nonzero['stock_count'].max()) if not nonzero.empty else 0
        total_months = len(all_months)

        print(f"  Total months in sample    : {total_months}")
        print(f"  Total (portfolio×month)   : {len(full_grid):,}")
        print(f"  Empty cells (0 stocks)    : {n_empty}")
        print(f"  Non-empty cell min stocks : {cell_min}")
        print(f"  Non-empty cell mean stocks: {cell_mean:.1f}")
        print(f"  Non-empty cell max stocks : {cell_max}")
        print(f"\n  Per-portfolio summary (mean stocks per month):")
        for portfolio in all_portfolios:
            if portfolio in summary.index:
                s = summary.loc[portfolio]
                print(f"    {portfolio:15s} : min={int(s['min_stocks']):3d}  "
                      f"max={int(s['max_stocks']):3d}  "
                      f"mean={s['mean_stocks']:5.1f}")
            else:
                print(f"    {portfolio:15s} : (no data)")

        result = {
            'n_months': total_months,
            'n_cells_total': len(full_grid),
            'n_empty_cells': n_empty,
            'cell_min_stocks': cell_min,
            'cell_mean_stocks': cell_mean,
            'cell_max_stocks': cell_max,
            'per_portfolio_summary': summary.to_dict(),
            'empty_cells': empty_cells[['month', 'portfolio']].to_dict('records')
            if n_empty > 0 else [],
        }

        # Save the detailed cell counts for evidence
        full_grid.to_csv(
            '.sisyphus/evidence/task-11-portfolio-cells.csv',
            index=False
        )
        print(f"\n  Cell counts saved to: .sisyphus/evidence/task-11-portfolio-cells.csv")

        return result

    # ------------------------------------------------------------------ #
    # Section 3: Ken French Comparison — 25 portfolio return correlations
    # ------------------------------------------------------------------ #

    def _ken_french_comparison(self) -> Dict:
        """Compare self-constructed 25 portfolio returns with original KF data."""
        print("\n" + "-" * 70)
        print("SECTION 3: Ken French Comparison — 25 Portfolio Correlations")
        print("-" * 70)

        # Build self-constructed portfolios
        try:
            our_25 = self.pc.build_25_portfolios()
        except Exception as exc:
            result: Dict = {'error': f'Could not build self-constructed portfolios: {exc}'}
            print(f"  ERROR: {result['error']}")
            return result

        if our_25 is None or our_25.empty:
            result = {'error': 'Self-constructed portfolios are empty'}
            print("  ERROR: Self-constructed portfolios are empty")
            return result

        # Load original Ken French data from backup
        if self.backup_dir is None:
            result = {'error': 'No backup directory found for original Ken French data'}
            print(f"  ERROR: {result['error']}")
            return result

        backup_25_path = os.path.join(self.backup_dir, 'ff_25_portfolios.csv')
        if not os.path.exists(backup_25_path):
            result = {'error': f'Backup file not found: {backup_25_path}'}
            print(f"  ERROR: {result['error']}")
            return result

        kf_25 = _normalize_month_index(pd.read_csv(backup_25_path))

        # Find overlapping date range
        common_dates = our_25.index.intersection(kf_25.index)
        if len(common_dates) == 0:
            result = {'error': 'No overlapping dates between self-constructed and Ken French data'}
            print(f"  ERROR: {result['error']}")
            return result

        print(f"  Overlap period: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} months)")

        # Align on common dates and common columns
        common_cols = [c for c in our_25.columns if c in kf_25.columns]
        if not common_cols:
            result = {'error': 'No overlapping portfolio columns'}
            print(f"  ERROR: {result['error']}")
            return result

        our_aligned = our_25.loc[common_dates, common_cols].astype(float)
        kf_aligned = kf_25.loc[common_dates, common_cols].astype(float)

        # Compute time-series Pearson correlation for each portfolio
        correlations = {}
        for col in common_cols:
            corr_val = our_aligned[col].corr(kf_aligned[col])
            correlations[col] = corr_val

        corr_values = list(correlations.values())
        min_corr = float(np.min(corr_values))
        mean_corr = float(np.mean(corr_values))
        max_corr = float(np.max(corr_values))

        print(f"  Correlation with original Ken French data (per portfolio):")
        for col in common_cols:
            print(f"    {col:15s} : r = {correlations[col]:.4f}")

        print(f"\n  Min correlation : {min_corr:.4f}")
        print(f"  Mean correlation: {mean_corr:.4f}")
        print(f"  Max correlation : {max_corr:.4f}")

        # Find worst and best matching portfolios
        worst_col = min(correlations, key=correlations.get)
        best_col = max(correlations, key=correlations.get)
        print(f"  Worst match    : {worst_col} (r={correlations[worst_col]:.4f})")
        print(f"  Best match     : {best_col} (r={correlations[best_col]:.4f})")

        result = {
            'overlap_dates': {'start': str(common_dates[0]), 'end': str(common_dates[-1]),
                              'n_months': len(common_dates)},
            'n_portfolios': len(common_cols),
            'correlations': {k: float(v) for k, v in correlations.items()},
            'min_correlation': min_corr,
            'mean_correlation': mean_corr,
            'max_correlation': max_corr,
            'worst_portfolio': worst_col,
            'best_portfolio': best_col,
            'worst_correlation': float(correlations[worst_col]),
            'best_correlation': float(correlations[best_col]),
        }
        return result

    # ------------------------------------------------------------------ #
    # Section 4: Factor Statistics
    # ------------------------------------------------------------------ #

    def _factor_statistics(self) -> Dict:
        """Compute self-constructed factor statistics and compare with KF."""
        print("\n" + "-" * 70)
        print("SECTION 4: Factor Statistics")
        print("-" * 70)

        # Build self-constructed factors
        try:
            our_factors = self.fc.assemble_factors()
        except Exception as exc:
            result: Dict = {'error': f'Could not build self-constructed factors: {exc}'}
            print(f"  ERROR: {result['error']}")
            return result

        if our_factors is None or our_factors.empty:
            result = {'error': 'Self-constructed factors are empty'}
            print("  ERROR: Self-constructed factors are empty")
            return result

        factor_cols = ['Mkt-RF', 'SMB', 'HML']
        stats = {}
        print("  Self-constructed factor statistics:")
        for col in factor_cols:
            values = our_factors[col].dropna().astype(float)
            n = len(values)
            mean_v = float(values.mean())
            std_v = float(values.std(ddof=1))
            t_stat = mean_v / (std_v / np.sqrt(n)) if std_v > 0 else 0.0
            stats[col] = {
                'n_obs': n,
                'mean': mean_v,
                'std': std_v,
                't_stat': t_stat,
                'min': float(values.min()),
                'max': float(values.max()),
            }
            print(f"    {col:8s} : mean={mean_v:7.3f}  std={std_v:7.3f}  "
                  f"t={t_stat:7.3f}  min={float(values.min()):7.3f}  "
                  f"max={float(values.max()):7.3f}")

        # Compare SMB/HML with original Ken French factors
        if self.backup_dir is None:
            result = {'self_stats': stats, 'comparison_error': 'No backup directory found'}
            print(f"  WARNING: No backup directory — skipping KF factor comparison")
            return {**result, 'self_stats': stats}

        backup_factors_path = os.path.join(self.backup_dir, 'ff_factors.csv')
        if not os.path.exists(backup_factors_path):
            result = {'self_stats': stats, 'comparison_error': f'Backup factors not found'}
            print(f"  WARNING: Backup factors not found — skipping comparison")
            return {**result, 'self_stats': stats}

        kf_factors = _normalize_month_index(pd.read_csv(backup_factors_path))
        common_dates = our_factors['Date'].unique()
        kf_dates = kf_factors.index
        overlap = sorted(set(common_dates) & set(kf_dates))
        if not overlap:
            result = {'self_stats': stats, 'comparison_error': 'No overlapping dates'}
            print("  WARNING: No overlapping dates for factor comparison")
            return {**result, 'self_stats': stats}

        print(f"\n  Comparison with original Ken French factors "
              f"({len(overlap)} overlapping months):")

        our_idx = our_factors.set_index('Date')
        comparison = {}
        for col in ['SMB', 'HML']:
            our_vals = our_idx.loc[overlap, col].astype(float)
            kf_vals = kf_factors.loc[overlap, col].astype(float)

            mad = float(np.abs(our_vals - kf_vals).mean())
            corr = float(our_vals.corr(kf_vals))
            our_mean = float(our_vals.mean())
            kf_mean = float(kf_vals.mean())
            our_std = float(our_vals.std(ddof=1))
            kf_std = float(kf_vals.std(ddof=1))

            comparison[col] = {
                'mad': mad,
                'correlation': corr,
                'our_mean': our_mean,
                'kf_mean': kf_mean,
                'our_std': our_std,
                'kf_std': kf_std,
            }
            print(f"    {col:8s} : MAD={mad:.4f}  r={corr:.4f}  "
                  f"our_mean={our_mean:.3f}  KF_mean={kf_mean:.3f}  "
                  f"our_std={our_std:.3f}  KF_std={kf_std:.3f}")

        result = {
            'self_stats': stats,
            'kf_comparison': comparison,
        }
        return result

    # ------------------------------------------------------------------ #
    # Section 5: Size/BE-ME Monotonicity
    # ------------------------------------------------------------------ #

    def _monotonicity_check(self) -> Dict:
        """Verify size and value effects in self-constructed portfolios."""
        print("\n" + "-" * 70)
        print("SECTION 5: Size/BE-ME Monotonicity Check")
        print("-" * 70)

        try:
            port25 = self.pc.build_25_portfolios()
        except Exception as exc:
            result: Dict = {'error': f'Could not build portfolios: {exc}'}
            print(f"  ERROR: {result['error']}")
            return result

        if port25 is None or port25.empty:
            result = {'error': 'Portfolios empty'}
            print("  ERROR: Portfolio data is empty")
            return result

        # Size effect: SMALL portfolios > BIG portfolios (mean return)
        small_cols = [c for c in port25.columns if c.startswith('SMALL')]
        big_cols = [c for c in port25.columns if c.startswith('BIG')]
        me1_cols = [c for c in port25.columns if c.startswith('ME1')]
        me5_cols = [c for c in port25.columns if c.startswith('ME5')]

        small_mean = port25[small_cols].values.flatten()
        big_mean = port25[big_cols].values.flatten()
        small_mean = small_mean[~np.isnan(small_mean)]
        big_mean = big_mean[~np.isnan(big_mean)]

        size_effect = {
            'small_mean_return': float(np.mean(small_mean)) if len(small_mean) > 0 else np.nan,
            'big_mean_return': float(np.mean(big_mean)) if len(big_mean) > 0 else np.nan,
        }
        size_valid = not (np.isnan(size_effect['small_mean_return']) or
                          np.isnan(size_effect['big_mean_return']))
        if size_valid:
            size_effect['small_exceeds_big'] = size_effect['small_mean_return'] > size_effect['big_mean_return']
            size_effect['spread'] = size_effect['small_mean_return'] - size_effect['big_mean_return']
        else:
            size_effect['small_exceeds_big'] = None
            size_effect['spread'] = np.nan

        print(f"  Size effect (SMALL > BIG):")
        print(f"    SMALL mean return: {size_effect['small_mean_return']:.3f}%")
        print(f"    BIG   mean return: {size_effect['big_mean_return']:.3f}%")
        if size_valid:
            print(f"    Spread (S-B)     : {size_effect['spread']:.3f}%")
            print(f"    SMALL > BIG?     : {size_effect['small_exceeds_big']}")

        # Value effect: HiBM portfolios > LoBM portfolios (mean return)
        hibm_cols = [c for c in port25.columns if c.endswith('HiBM')]
        lobm_cols = [c for c in port25.columns if c.endswith('LoBM')]

        hibm_mean = port25[hibm_cols].values.flatten()
        lobm_mean = port25[lobm_cols].values.flatten()
        hibm_mean = hibm_mean[~np.isnan(hibm_mean)]
        lobm_mean = lobm_mean[~np.isnan(lobm_mean)]

        value_effect = {
            'hibm_mean_return': float(np.mean(hibm_mean)) if len(hibm_mean) > 0 else np.nan,
            'lobm_mean_return': float(np.mean(lobm_mean)) if len(lobm_mean) > 0 else np.nan,
        }
        value_valid = not (np.isnan(value_effect['hibm_mean_return']) or
                           np.isnan(value_effect['lobm_mean_return']))
        if value_valid:
            value_effect['hibm_exceeds_lobm'] = value_effect['hibm_mean_return'] > value_effect['lobm_mean_return']
            value_effect['spread'] = value_effect['hibm_mean_return'] - value_effect['lobm_mean_return']
        else:
            value_effect['hibm_exceeds_lobm'] = None
            value_effect['spread'] = np.nan

        print(f"\n  Value effect (HiBM > LoBM):")
        print(f"    HiBM mean return: {value_effect['hibm_mean_return']:.3f}%")
        print(f"    LoBM mean return: {value_effect['lobm_mean_return']:.3f}%")
        if value_valid:
            print(f"    Spread (H-L)     : {value_effect['spread']:.3f}%")
            print(f"    HiBM > LoBM?     : {value_effect['hibm_exceeds_lobm']}")

        result = {
            'size_effect': size_effect,
            'value_effect': value_effect,
        }
        return result

    # ------------------------------------------------------------------ #
    # Output: Console report and file saving
    # ------------------------------------------------------------------ #

    def _print_report(self, results: Dict) -> None:
        """Print full report summary to console."""
        print("\n" + "=" * 70)
        print("DIAGNOSTICS REPORT SUMMARY")
        print("=" * 70)

        for section, data in results.items():
            status = "OK" if 'error' not in data else f"ERROR: {data['error']}"
            print(f"  {section:25s} : {status}")

    def _save_report(self, results: Dict) -> None:
        """Save full report to evidence file."""
        import datetime as dt

        evidence_dir = '.sisyphus/evidence'
        os.makedirs(evidence_dir, exist_ok=True)
        report_path = os.path.join(evidence_dir, 'task-11-full-report.txt')

        lines = []
        lines.append("=" * 70)
        lines.append("DIAGNOSTICS REPORT: Self-Constructed Portfolios")
        lines.append(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        # Section 1
        lines.append("\n" + "-" * 70)
        lines.append("SECTION 1: Mapping Coverage")
        lines.append("-" * 70)
        mc = results.get('mapping_coverage', {})
        if 'error' in mc:
            lines.append(f"  ERROR: {mc['error']}")
        else:
            for k, v in mc.items():
                lines.append(f"  {k:30s} : {v}")

        # Section 2
        lines.append("\n" + "-" * 70)
        lines.append("SECTION 2: Portfolio Diagnostics — Stock Counts Per Cell")
        lines.append("-" * 70)
        pd_sec = results.get('portfolio_diagnostics', {})
        if 'error' in pd_sec:
            lines.append(f"  ERROR: {pd_sec['error']}")
        else:
            lines.append(f"  Total months in sample      : {pd_sec.get('n_months', 'N/A')}")
            lines.append(f"  Total (portfolio×month)     : {pd_sec.get('n_cells_total', 'N/A')}")
            lines.append(f"  Empty cells (0 stocks)      : {pd_sec.get('n_empty_cells', 'N/A')}")
            lines.append(f"  Non-empty cell min stocks   : {pd_sec.get('cell_min_stocks', 'N/A')}")
            lines.append(f"  Non-empty cell mean stocks  : {pd_sec.get('cell_mean_stocks', 'N/A')}")
            lines.append(f"  Non-empty cell max stocks   : {pd_sec.get('cell_max_stocks', 'N/A')}")

        # Section 3
        lines.append("\n" + "-" * 70)
        lines.append("SECTION 3: Ken French Comparison — 25 Portfolio Correlations")
        lines.append("-" * 70)
        kfc = results.get('ken_french_comparison', {})
        if 'error' in kfc:
            lines.append(f"  ERROR: {kfc['error']}")
        else:
            overlap = kfc.get('overlap_dates', {})
            lines.append(f"  Overlap: {overlap.get('start', 'N/A')} to {overlap.get('end', 'N/A')} "
                         f"({overlap.get('n_months', 'N/A')} months)")
            lines.append(f"  Min correlation : {kfc.get('min_correlation', 'N/A'):.4f}")
            lines.append(f"  Mean correlation: {kfc.get('mean_correlation', 'N/A'):.4f}")
            lines.append(f"  Max correlation : {kfc.get('max_correlation', 'N/A'):.4f}")
            lines.append(f"  Worst match: {kfc.get('worst_portfolio', 'N/A')} "
                         f"(r={kfc.get('worst_correlation', 'N/A')})")
            lines.append(f"  Best match : {kfc.get('best_portfolio', 'N/A')} "
                         f"(r={kfc.get('best_correlation', 'N/A')})")
            lines.append("  Per-portfolio correlations:")
            for col, corr in kfc.get('correlations', {}).items():
                lines.append(f"    {col:15s} : r = {corr:.4f}")

        # Section 4
        lines.append("\n" + "-" * 70)
        lines.append("SECTION 4: Factor Statistics")
        lines.append("-" * 70)
        fs = results.get('factor_statistics', {})
        if 'error' in fs:
            lines.append(f"  ERROR: {fs['error']}")
        else:
            lines.append("  Self-constructed factor statistics:")
            for col, st in fs.get('self_stats', {}).items():
                lines.append(f"    {col:8s} : mean={st['mean']:7.3f}  std={st['std']:7.3f}  "
                             f"t={st['t_stat']:7.3f}  min={st['min']:7.3f}  max={st['max']:7.3f}")
            comp = fs.get('kf_comparison', {})
            if comp:
                lines.append("  Comparison vs Ken French (overlap):")
                for col, cd in comp.items():
                    lines.append(f"    {col:8s} : MAD={cd['mad']:.4f}  r={cd['correlation']:.4f}  "
                                 f"our_mean={cd['our_mean']:.3f}  KF_mean={cd['kf_mean']:.3f}")

        # Section 5
        lines.append("\n" + "-" * 70)
        lines.append("SECTION 5: Size/BE-ME Monotonicity Check")
        lines.append("-" * 70)
        mono = results.get('monotonicity', {})
        if 'error' in mono:
            lines.append(f"  ERROR: {mono['error']}")
        else:
            se = mono.get('size_effect', {})
            lines.append(f"  Size effect:")
            lines.append(f"    SMALL mean return: {se.get('small_mean_return', 'N/A')}%")
            lines.append(f"    BIG   mean return: {se.get('big_mean_return', 'N/A')}%")
            lines.append(f"    Spread (S-B)     : {se.get('spread', 'N/A')}%")
            lines.append(f"    SMALL > BIG?     : {se.get('small_exceeds_big', 'N/A')}")
            ve = mono.get('value_effect', {})
            lines.append(f"  Value effect:")
            lines.append(f"    HiBM mean return: {ve.get('hibm_mean_return', 'N/A')}%")
            lines.append(f"    LoBM mean return: {ve.get('lobm_mean_return', 'N/A')}%")
            lines.append(f"    Spread (H-L)    : {ve.get('spread', 'N/A')}%")
            lines.append(f"    HiBM > LoBM?    : {ve.get('hibm_exceeds_lobm', 'N/A')}")

        lines.append("\n" + "=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        report_text = '\n'.join(lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n  Report saved to: {report_path}")

        # Save portfolio cell counts if available
        pd_sec = results.get('portfolio_diagnostics', {})
        if not pd_sec.get('cell_counts_empty', False) and 'error' not in pd_sec:
            cells_path = os.path.join(evidence_dir, 'task-11-portfolio-cells.csv')
            print(f"  Cell counts saved to: {cells_path}")


def _normalize_month_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by YYYY-MM month strings."""
    normalized = df.copy()

    if 'Date' in normalized.columns:
        dates = normalized.pop('Date')
    else:
        dates = normalized.index

    normalized.index = pd.DatetimeIndex(pd.to_datetime(dates)).strftime('%Y-%m')
    normalized.index.name = 'Date'
    return normalized


def _hybridize_ken_french_data(
    ken_french: pd.DataFrame,
    self_constructed: pd.DataFrame,
    columns: List[str],
    preserve_start: str = '1963-07',
    preserve_end: str = '1964-06',
    replace_start: str = '1964-07',
    replace_end: str = '1991-12',
) -> pd.DataFrame:
    """
    Combine original Ken French rows with self-constructed rows.

    Ken French data is always retained through ``preserve_end``. For the
    replacement window, only months present in ``self_constructed`` overwrite
    the original rows; missing self-constructed months remain Ken French data.
    """
    kf = _normalize_month_index(ken_french)
    constructed = _normalize_month_index(self_constructed)

    missing_kf = [column for column in columns if column not in kf.columns]
    missing_constructed = [column for column in columns if column not in constructed.columns]
    if missing_kf:
        raise ValueError(f"Ken French data missing columns: {missing_kf}")
    if missing_constructed:
        raise ValueError(f"Self-constructed data missing columns: {missing_constructed}")

    hybrid = kf.loc[:, columns].copy()
    constructed = constructed.loc[:, columns]

    replace_mask = (constructed.index >= replace_start) & (constructed.index <= replace_end)
    replacement_dates = constructed.index[replace_mask]
    replacement_dates = replacement_dates.intersection(hybrid.index)

    if len(replacement_dates) > 0:
        hybrid.loc[replacement_dates, columns] = constructed.loc[replacement_dates, columns]

    hybrid = hybrid[(hybrid.index >= preserve_start) & (hybrid.index <= replace_end)].copy()
    hybrid.index.name = 'Date'
    return hybrid


def _write_hybrid_csv(df: pd.DataFrame, filepath: str) -> None:
    """Write a Date-indexed hybrid DataFrame with YYYY-MM dates."""
    output = df.copy()
    output.insert(0, 'Date', output.index)
    output.to_csv(filepath, index=False)


def _overlay_archived_self_constructed(
    generated: pd.DataFrame,
    archive_filename: str,
    columns: List[str],
) -> pd.DataFrame:
    """Overlay archived self-constructed CRSP outputs when available."""
    import config  # 저장소 루트 기준 crsp/ 아카이브 (src/ 가 아닌 루트에 위치)
    archive_path = os.path.join(
        config.BASE_DIR,
        'crsp',
        'FF1993_results',
        'data',
        archive_filename,
    )
    if not os.path.exists(archive_path):
        return generated

    archived = _normalize_month_index(pd.read_csv(archive_path))
    available_columns = [column for column in columns if column in archived.columns]
    if not available_columns:
        return generated

    generated_index = _normalize_month_index(generated).index
    if archived.index.intersection(generated_index).empty:
        return generated

    return archived.loc[:, available_columns].copy()


def replace_ken_french_data(
    data_dir: str = 'data',
    portfolio_constructor: Optional[PortfolioConstructor] = None,
    factor_calculator: Optional[FactorCalculator] = None,
    create_backup: bool = True,
) -> Dict[str, object]:
    """
    Replace pipeline Ken French stock-side inputs with Compustat+CRSP hybrids.

    Hybrid rule:
    - 1963-07 through 1964-06: preserve existing Ken French data.
    - 1964-07 through 1991-12: use self-constructed data where available.
    - Any missing self-constructed replacement months remain Ken French data.

    Args:
        data_dir: Directory containing ``ff_25_portfolios.csv``,
            ``ff_6_portfolios.csv``, and ``ff_factors.csv``.
        portfolio_constructor: Optional preconfigured constructor, useful for
            tests or alternate data sources.
        factor_calculator: Optional preconfigured factor calculator.
        create_backup: Whether to create a ``BackupManager`` backup before
            overwriting the data files.

    Returns:
        Summary dictionary with backup path, written paths, and row counts.
    """
    if portfolio_constructor is None:
        mapping_manager = MappingManager(cache_dir=data_dir)
        if not os.path.exists(mapping_manager.cache_file):
            mapping_manager.download_and_cache()

    pc = portfolio_constructor or PortfolioConstructor()
    fc = factor_calculator or FactorCalculator(portfolio_constructor=pc)
    backup_path = BackupManager(data_dir=data_dir).backup() if create_backup else None

    paths = {
        'portfolios_25': os.path.join(data_dir, 'ff_25_portfolios.csv'),
        'portfolios_6': os.path.join(data_dir, 'ff_6_portfolios.csv'),
        'factors': os.path.join(data_dir, 'ff_factors.csv'),
    }

    kf_25 = pd.read_csv(paths['portfolios_25'], index_col='Date')
    kf_6 = pd.read_csv(paths['portfolios_6'], index_col='Date')
    kf_factors = pd.read_csv(paths['factors'], index_col='Date')

    port25 = pc.build_25_portfolios()
    port6 = pc.build_6_portfolios()
    factors = fc.assemble_factors(port6=port6)

    port25 = _overlay_archived_self_constructed(port25, 'crsp_25_portfolios.csv', pc.PORTFOLIO_COLUMNS)
    port6 = _overlay_archived_self_constructed(port6, 'crsp_6_portfolios.csv', pc.PORTFOLIO_COLUMNS_6)
    factors = _overlay_archived_self_constructed(factors, 'crsp_ff_factors.csv', ['Mkt-RF', 'SMB', 'HML', 'RF'])

    hybrid_25 = _hybridize_ken_french_data(kf_25, port25, pc.PORTFOLIO_COLUMNS)
    hybrid_6 = _hybridize_ken_french_data(kf_6, port6, pc.PORTFOLIO_COLUMNS_6)
    hybrid_factors = _hybridize_ken_french_data(kf_factors, factors, ['Mkt-RF', 'SMB', 'HML', 'RF'])

    _write_hybrid_csv(hybrid_25, paths['portfolios_25'])
    _write_hybrid_csv(hybrid_6, paths['portfolios_6'])
    _write_hybrid_csv(hybrid_factors, paths['factors'])

    return {
        'backup_path': backup_path,
        'paths': paths,
        'rows': {
            'ff_25_portfolios.csv': len(hybrid_25),
            'ff_6_portfolios.csv': len(hybrid_6),
            'ff_factors.csv': len(hybrid_factors),
        },
    }


def restore_ken_french_data(backup_path: str, data_dir: str = 'data') -> None:
    """Restore Fama-French pipeline inputs via ``BackupManager.restore()``."""
    BackupManager(data_dir=data_dir).restore(backup_path)
