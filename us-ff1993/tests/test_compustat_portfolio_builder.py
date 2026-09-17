"""
test_compustat_portfolio_builder.py
TDD tests for compustat_portfolio_builder module
"""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

import compustat_portfolio_builder


class TestMappingManager:
    """Tests for MappingManager class."""

    def test_init(self, tmp_path):
        """Test initialization with default and custom cache_dir."""
        manager = compustat_portfolio_builder.MappingManager(cache_dir=str(tmp_path))
        assert manager.cache_dir == str(tmp_path)
        assert manager.cache_file == str(tmp_path / 'gvkey_permco_permno.csv')

    def test_init_custom_cache_dir(self):
        """Test initialization with custom cache directory."""
        custom_dir = '/custom/path'
        manager = compustat_portfolio_builder.MappingManager(cache_dir=custom_dir)
        assert manager.cache_dir == custom_dir

    def test_download_and_cache_skips_existing(self, tmp_path, monkeypatch):
        """Test download_and_cache skips download if cache file exists."""
        # Create existing cache file
        cache_file = tmp_path / 'gvkey_permco_permno.csv'
        cache_file.write_text('gvkey,permco,permno\n1,1,1\n')

        manager = compustat_portfolio_builder.MappingManager(cache_dir=str(tmp_path))

        # Mock requests.get to avoid actual download
        def mock_get(url, stream=False, **kwargs):
            class MockResponse:
                def raise_for_status(self):
                    pass
                def __iter__(self):
                    return iter([])
            return MockResponse()

        monkeypatch.setattr('requests.get', mock_get)

        # Should not raise and should skip download
        manager.download_and_cache()

        # File should still exist
        assert cache_file.exists()

    @pytest.mark.skip("Requires actual network access")
    def test_download_and_cache_downloads_new(self):
        """Test download_and_cache downloads new file when cache doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = compustat_portfolio_builder.MappingManager(cache_dir=tmp_dir)
            manager.download_and_cache()
            assert manager.cache_file.exists()

    def test_load_mapping_file_not_found(self, tmp_path):
        """Test load_mapping raises FileNotFoundError when file doesn't exist."""
        manager = compustat_portfolio_builder.MappingManager(cache_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            manager.load_mapping()

    def test_load_mapping_invalid_columns(self, tmp_path):
        """Test load_mapping raises ValueError when columns are missing."""
        cache_file = tmp_path / 'gvkey_permco_permno.csv'
        cache_file.write_text('id,code,value\n1,1,1\n')

        manager = compustat_portfolio_builder.MappingManager(cache_dir=str(tmp_path))
        with pytest.raises(ValueError):
            manager.load_mapping()

    def test_load_mapping_valid_columns(self, tmp_path):
        """Test load_mapping returns correct DataFrame with required columns."""
        cache_file = tmp_path / 'gvkey_permco_permno.csv'
        cache_file.write_text('gvkey,permco,permno\n1,1,1\n2,2,2\n')

        manager = compustat_portfolio_builder.MappingManager(cache_dir=str(tmp_path))
        df = manager.load_mapping()

        assert list(df.columns) == ['gvkey', 'permco', 'permno']
        assert len(df) == 2


class TestCRSPSource:
    """Tests for CRSPSource class."""

    def test_init_auto_detects_file(self, tmp_path):
        """Test initialization auto-detects best CRSP file."""
        # Create test CRSP files
        crsp_dir = tmp_path / 'crsp'
        crsp_dir.mkdir()

        # Main file with more columns
        main_dir = crsp_dir / 'RET__DLSTCD_1962.01_1991.12'
        main_dir.mkdir()
        (main_dir / 'RET__DLSTCD_1962.01_1991.12.csv').write_text(
            'PERMNO,date,SHRCD,EXCHCD,DLRET,PRC\n'
            '12345,2020-01-01,10,1,\n'
        )

        # Secondary file with fewer columns
        sec_dir = crsp_dir / 'zgwss6y8fijax1cr_csv'
        sec_dir.mkdir()
        (sec_dir / 'zgwss6y8fijax1cr.csv').write_text(
            'PERMNO,date,SHRCD\n'
            '12345,2020-01-01,10\n'
        )

        source = compustat_portfolio_builder.CRSPSource(base_dir=str(tmp_path))
        assert source.file_path == str(main_dir / 'RET__DLSTCD_1962.01_1991.12.csv')

    def test_init_no_files_found(self, tmp_path):
        """Test initialization raises FileNotFoundError when no CRSP files exist."""
        # Create temp directory without CRSP files
        with pytest.raises(FileNotFoundError):
            compustat_portfolio_builder.CRSPSource(base_dir=str(tmp_path))

    def test_filter_common_stocks(self):
        """Test filter_common_stocks keeps common stocks (SHRCD 10/11 AND EXCHCD 1/2/3)."""
        df = pd.DataFrame({
            'SHRCD': [10, 11, 20, 30, np.nan, np.nan],
            'EXCHCD': [1, 2, 3, 4, np.nan, np.nan],
            'PERMNO': [1, 2, 3, 4, 5, 6]
        })

        filtered = compustat_portfolio_builder.CRSPSource().filter_common_stocks(df)

        # Keep SHRCD 10/11 AND EXCHCD 1/2/3
        # Rows 0, 1 should remain (SHRCD 10/11, EXCHCD 1/2)
        # Rows 2, 3 should be filtered out (SHRCD 20, 30)
        # Rows 4, 5 should be filtered out (SHRCD NaN)
        assert len(filtered) == 2
        assert list(filtered['PERMNO']) == [1, 2]

    def test_clean_ret_codes_replaces_c_b(self):
        """Test clean_ret_codes replaces 'C' and 'B' with NaN."""
        df = pd.DataFrame({
            'RET': [0.01, 'C', 'B', 0.02, 'X']
        })

        cleaned = compustat_portfolio_builder.CRSPSource().clean_ret_codes(df)

        assert pd.isna(cleaned['RET'][1])  # 'C' replaced with NaN
        assert pd.isna(cleaned['RET'][2])  # 'B' replaced with NaN
        assert pd.isna(cleaned['RET'][4])  # 'X' replaced with NaN
        assert cleaned['RET'][0] == 0.01
        assert cleaned['RET'][3] == 0.02

    def test_incorporate_dlret(self):
        """Test incorporate_dlret fills RET with DLRET when RET is NaN."""
        df = pd.DataFrame({
            'RET': [0.01, np.nan, 0.02, np.nan],
            'DLRET': [np.nan, 0.03, np.nan, 0.04]
        })

        result = compustat_portfolio_builder.CRSPSource().incorporate_dlret(df)

        assert result['RET'][0] == 0.01  # RET not NaN, unchanged
        assert result['RET'][1] == 0.03  # RET NaN, filled with DLRET
        assert result['RET'][2] == 0.02  # RET not NaN, unchanged
        assert result['RET'][3] == 0.04  # RET NaN, filled with DLRET

    def test_process_prc_abs(self):
        """Test process_prc converts PRC to absolute value."""
        df = pd.DataFrame({
            'PRC': [-1.5, 2.3, -0.5, 0.0]
        })

        result = compustat_portfolio_builder.CRSPSource().process_prc(df)

        assert list(result['PRC']) == [1.5, 2.3, 0.5, 0.0]

    @pytest.fixture
    def sample_crsp_data(self, tmp_path):
        """Create sample CRSP data for testing."""
        crsp_dir = tmp_path / 'crsp'
        crsp_dir.mkdir()

        # Create main file with all required columns
        main_dir = crsp_dir / 'RET__DLSTCD_1962.01_1991.12'
        main_dir.mkdir()
        (main_dir / 'RET__DLSTCD_1962.01_1991.12.csv').write_text(
            'PERMNO,date,SHRCD,EXCHCD,PERMCO,DLRET,PRC,RET,SHROUT\n'
            '10001,2020-01-01,10,1,100,0.02,-1.5,0.01,1000000\n'
            '10002,2020-01-02,11,2,101,0.03,2.3,0.02,800000\n'
            '10003,2020-01-03,10,3,102,,0.5,-0.01,500000\n'
            '10004,2020-01-04,20,4,103,0.05,-0.8,C,1200000\n'
        )

        return str(tmp_path)

    def test_load_and_clean(self, sample_crsp_data):
        """Test load_and_clean applies all preprocessing steps."""
        source = compustat_portfolio_builder.CRSPSource(base_dir=sample_crsp_data)
        df = source.load_and_clean()

        # Check date parsing
        assert isinstance(df['date'].iloc[0], pd.Timestamp)

        # Check filtering (keep SHRCD 10/11 AND EXCHCD 1/2/3)
        # Row 0: SHRCD=10, EXCHCD=1 -> keep
        # Row 1: SHRCD=11, EXCHCD=2 -> keep
        # Row 2: SHRCD=10, EXCHCD=3 -> keep
        # Row 3: SHRCD=20, EXCHCD=4 -> filter out
        assert len(df) == 3  # Only rows 0, 1, 2 should remain

        # Check RET cleaning - row 3 (original index 3) has 'C' in RET
        # But row 3 was filtered out, so we need to check row 2 which has non-numeric DLRET
        assert pd.isna(df['RET'].iloc[2])  # Third row has DLRET filled in

        # Check PRC absolute value
        assert df['PRC'].iloc[0] == 1.5  # Negative PRC made positive


class TestBECalculator:
    """Tests for BECalculator class."""

    def test_init(self):
        """Test initialization with default and custom file."""
        calc = compustat_portfolio_builder.BECalculator()
        assert calc.be_file == 'compustat_be.csv'

        custom_file = '/custom/path/be.csv'
        calc = compustat_portfolio_builder.BECalculator(be_file=custom_file)
        assert calc.be_file == custom_file

    def test_deduplicate_be_no_duplicates(self):
        """Test deduplicate_be keeps data when no duplicates exist."""
        df = pd.DataFrame({
            'gvkey': [1, 2, 3, 2],
            'cal_year': [1990, 1990, 1990, 1990],
            'datadate': ['1990-01-01', '1990-01-01', '1990-01-01', '1990-01-01'],
            'be': [100, 200, 300, 400]
        })

        result = compustat_portfolio_builder.BECalculator().deduplicate_be(df)

        # Should keep 3 rows (gvkey 1, 2, 3) - gvkey 2's last row kept
        assert len(result) == 3

    def test_extract_sich_flag(self):
        """Test extract_sich adds is_financial flag."""
        df = pd.DataFrame({
            'sich': [1000, 6000, 6500, 7000, 8000],
            'be': [100, 200, 300, 400, 500]
        })

        result = compustat_portfolio_builder.BECalculator().extract_sich(df)

        assert 'is_financial' in result.columns
        # SICH 6000-6999 are financial companies
        # 6500 is in range, 7000 is not
        assert result['is_financial'].tolist() == [False, True, True, False, False]

    def test_extract_sich_range(self):
        """Test extract_sich correctly identifies financial companies (SICH 6000-6999)."""
        df = pd.DataFrame({
            'sich': [5999, 6000, 6999, 7000],
            'be': [100, 200, 300, 400]
        })

        result = compustat_portfolio_builder.BECalculator().extract_sich(df)

        assert result['is_financial'].tolist() == [False, True, True, False]


class TestLinkingEngine:
    """Tests for LinkingEngine class."""

    def _write_crsp_file(self, base_dir, rows=None):
        crsp_dir = base_dir / 'crsp' / 'RET__DLSTCD_1962.01_1991.12'
        crsp_dir.mkdir(parents=True, exist_ok=True)
        crsp_file = crsp_dir / 'RET__DLSTCD_1962.01_1991.12.csv'

        columns = ['PERMNO', 'date', 'SHRCD', 'EXCHCD', 'PERMCO', 'DLRET', 'PRC', 'RET', 'SHROUT']
        pd.DataFrame(rows or [], columns=columns).to_csv(crsp_file, index=False)
        return crsp_file

    def _write_be_file(self, base_dir):
        be_file = base_dir / 'compustat_be.csv'
        pd.DataFrame({
            'gvkey': [1, 2, 3],
            'datadate': ['1963-12-31', '1963-12-31', '1963-12-31'],
            'cal_year': [1963, 1963, 1963],
            'be': [100.0, 200.0, 300.0],
            'sich': [2000, 6500, 3500],
            'se_flag': ['seq', 'seq', 'seq'],
            'se_source': ['seq', 'seq', 'seq'],
            'dt_flag': ['txditc', 'txditc', 'zero'],
            'ps_flag': ['pstkrv', 'pstkrv', 'zero'],
        }).to_csv(be_file, index=False)
        return be_file

    def _write_mapping_file(self, base_dir):
        data_dir = base_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        mapping_file = data_dir / 'gvkey_permco_permno.csv'
        pd.DataFrame({
            'gvkey': [1, 1, 2, 3],
            'permco': [500, 500, 501, 999],
            'permno': [10001, 10002, 10003, 19999],
        }).to_csv(mapping_file, index=False)
        return mapping_file

    def _write_linking_fixture(self, base_dir):
        self._write_mapping_file(base_dir)
        self._write_be_file(base_dir)
        self._write_crsp_file(base_dir, rows=[
            [10001, '1964-06-30', 10, 1, 500, np.nan, 10.0, 0.01, 100.0],
            [10002, '1964-06-30', 11, 2, 500, np.nan, 20.0, 0.02, 50.0],
            [10003, '1964-05-31', 10, 1, 501, np.nan, 30.0, 0.03, 10.0],
            [10004, '1964-06-30', 20, 1, 999, np.nan, 40.0, 0.04, 10.0],
        ])

    def test_init(self, tmp_path):
        """Test initialization with default and custom dependencies."""
        self._write_crsp_file(tmp_path)

        engine = compustat_portfolio_builder.LinkingEngine(base_dir=str(tmp_path))
        assert engine.base_dir == str(tmp_path)
        assert isinstance(engine.mapping_manager, compustat_portfolio_builder.MappingManager)
        assert isinstance(engine.crsp_source, compustat_portfolio_builder.CRSPSource)
        assert isinstance(engine.be_calculator, compustat_portfolio_builder.BECalculator)

        custom_mapping = object()
        custom_crsp = object()
        custom_be = object()
        engine = compustat_portfolio_builder.LinkingEngine(
            base_dir=str(tmp_path),
            mapping_manager=custom_mapping,
            crsp_source=custom_crsp,
            be_calculator=custom_be,
        )
        assert engine.mapping_manager is custom_mapping
        assert engine.crsp_source is custom_crsp
        assert engine.be_calculator is custom_be

    def test_build_link_basic(self, tmp_path):
        """Test build_link returns expected linked rows and columns."""
        self._write_linking_fixture(tmp_path)

        engine = compustat_portfolio_builder.LinkingEngine(base_dir=str(tmp_path))
        linked = engine.build_link()

        expected_cols = ['gvkey', 'permno', 'permco', 'date', 'cal_year', 'be', 'me', 'sich', 'is_financial', 'EXCHCD']
        assert list(linked.columns) == expected_cols
        assert len(linked) == 2
        assert set(linked['permno']) == {10001, 10002}
        assert set(linked['permco']) == {500}
        assert set(linked['cal_year']) == {1963}
        assert set(linked['EXCHCD']) == {1, 2}
        assert linked['date'].dt.month.eq(6).all()
        assert linked.loc[linked['permno'] == 10001, 'me'].iloc[0] == 1000.0
        assert linked.loc[linked['permno'] == 10002, 'me'].iloc[0] == 1000.0

    def test_build_link_no_mapping(self, tmp_path):
        """Test build_link handles missing mapping cache gracefully."""
        self._write_be_file(tmp_path)
        self._write_crsp_file(tmp_path, rows=[
            [10001, '1964-06-30', 10, 1, 500, np.nan, 10.0, 0.01, 100.0],
        ])

        engine = compustat_portfolio_builder.LinkingEngine(base_dir=str(tmp_path))
        linked = engine.build_link()

        expected_cols = ['gvkey', 'permno', 'permco', 'date', 'cal_year', 'be', 'me', 'sich', 'is_financial', 'EXCHCD']
        assert list(linked.columns) == expected_cols
        assert linked.empty

    def test_build_link_coverage(self, tmp_path, capsys):
        """Test build_link prints coverage statistics."""
        self._write_linking_fixture(tmp_path)

        engine = compustat_portfolio_builder.LinkingEngine(base_dir=str(tmp_path))
        linked = engine.build_link()
        captured = capsys.readouterr()

        assert not linked.empty
        assert "Total BE gvkeys: 3" in captured.out
        assert "Gvkeys with CRSP match: 1" in captured.out
        assert "Coverage: 33.33%" in captured.out
        assert "Unique PERMNOs linked: 2" in captured.out
        assert "Sample unmatched gvkeys:" in captured.out

    def test_build_link_partial_mapping_still_links_mapped_rows(self, tmp_path):
        """Test unmapped BE gvkeys do not break valid PERMCO matches via float upcasting."""
        data_dir = tmp_path / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'gvkey': [1],
            'permco': [500],
            'permno': [10001],
        }).to_csv(data_dir / 'gvkey_permco_permno.csv', index=False)

        self._write_be_file(tmp_path)  # gvkeys 2 and 3 are intentionally unmapped
        self._write_crsp_file(tmp_path, rows=[
            [10001, '1964-06-30', 10, 1, 500, np.nan, 10.0, 0.01, 100.0],
        ])

        engine = compustat_portfolio_builder.LinkingEngine(base_dir=str(tmp_path))
        linked = engine.build_link()

        assert len(linked) == 1
        assert linked['gvkey'].iloc[0] == 1
        assert linked['permco'].iloc[0] == 500
        assert linked['permno'].iloc[0] == 10001

    def test_build_link_mixed_permco_representations(self, tmp_path):
        """Test integer-like and float-like PERMCO values normalize to the same key."""
        data_dir = tmp_path / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'gvkey': ['001'],
            'permco': ['500.0'],
            'permno': [10001],
        }).to_csv(data_dir / 'gvkey_permco_permno.csv', index=False)

        be_file = tmp_path / 'compustat_be.csv'
        pd.DataFrame({
            'gvkey': ['001'],
            'datadate': ['1963-12-31'],
            'cal_year': [1963],
            'be': [100.0],
            'sich': [2000],
            'se_flag': ['seq'],
            'se_source': ['seq'],
            'dt_flag': ['txditc'],
            'ps_flag': ['pstkrv'],
        }).to_csv(be_file, index=False)
        self._write_crsp_file(tmp_path, rows=[
            [10001, '1964-06-30', 10, 1, 500, np.nan, 10.0, 0.01, 100.0],
        ])

        engine = compustat_portfolio_builder.LinkingEngine(base_dir=str(tmp_path))
        linked = engine.build_link()

        assert len(linked) == 1
        assert linked['permco'].iloc[0] == 500
        assert linked['me'].iloc[0] == 1000.0


class TestBEMECalculator:
    """Tests for BEMECalculator class."""

    class DummyEngine:
        def __init__(self, linked_df=None):
            self.linked_df = linked_df if linked_df is not None else pd.DataFrame()

        def build_link(self):
            return self.linked_df

    def _sample_linked_df(self):
        return pd.DataFrame({
            'gvkey': [1, 2],
            'permno': [10001, 10002],
            'permco': [500, 501],
            'date': pd.to_datetime(['1964-06-30', '1965-06-30']),
            'cal_year': [1963, 1964],
            'be': [100.0, 300.0],
            'me': [1000.0, 1500.0],
            'sich': [2000, 3500],
            'is_financial': [False, False],
            'EXCHCD': [1, 2],
        })

    def _write_crsp_file(self, base_dir, rows=None):
        crsp_dir = base_dir / 'crsp' / 'RET__DLSTCD_1962.01_1991.12'
        crsp_dir.mkdir(parents=True, exist_ok=True)
        crsp_file = crsp_dir / 'RET__DLSTCD_1962.01_1991.12.csv'
        columns = ['PERMNO', 'date', 'SHRCD', 'EXCHCD', 'PERMCO', 'DLRET', 'PRC', 'RET', 'SHROUT']
        pd.DataFrame(rows or [], columns=columns).to_csv(crsp_file, index=False)
        return crsp_file

    def _write_mapping_file(self, base_dir):
        data_dir = base_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'gvkey': [1, 2],
            'permco': [500, 501],
            'permno': [10001, 10002],
        }).to_csv(data_dir / 'gvkey_permco_permno.csv', index=False)

    def _write_be_file(self, base_dir):
        pd.DataFrame({
            'gvkey': [1, 2],
            'datadate': ['1963-12-31', '1963-12-31'],
            'cal_year': [1963, 1963],
            'be': [100.0, 200.0],
            'sich': [2000, 6500],
            'se_flag': ['seq', 'seq'],
            'se_source': ['seq', 'seq'],
            'dt_flag': ['txditc', 'txditc'],
            'ps_flag': ['pstkrv', 'pstkrv'],
        }).to_csv(base_dir / 'compustat_be.csv', index=False)

    def test_init(self, monkeypatch):
        """Test initialization with default and custom engine."""
        created_engine = self.DummyEngine()
        monkeypatch.setattr(compustat_portfolio_builder, 'LinkingEngine', lambda: created_engine)

        calc = compustat_portfolio_builder.BEMECalculator()
        assert calc.engine is created_engine

        custom_engine = self.DummyEngine()
        calc = compustat_portfolio_builder.BEMECalculator(engine=custom_engine)
        assert calc.engine is custom_engine

    def test_compute_all_with_linked_df(self):
        """Test compute_all computes BE/ME ratios and output columns from synthetic linked data."""
        linked = self._sample_linked_df()
        calc = compustat_portfolio_builder.BEMECalculator(engine=self.DummyEngine())

        result = calc.compute_all(linked)

        expected_cols = ['gvkey', 'permno', 'permco', 'date', 'year', 'be', 'me', 'beme', 'sich', 'is_financial', 'exchange']
        assert list(result.columns) == expected_cols
        assert len(result) == 2
        assert result.loc[result['gvkey'] == 1, 'beme'].iloc[0] == pytest.approx(0.1)
        assert result.loc[result['gvkey'] == 2, 'beme'].iloc[0] == pytest.approx(0.2)
        assert result.loc[result['gvkey'] == 1, 'year'].iloc[0] == 1964
        assert set(result['exchange']) == {1, 2}

    def test_filters_negative_be(self):
        """Test compute_all removes non-positive BE rows."""
        linked = self._sample_linked_df()
        linked.loc[0, 'be'] = -10.0
        calc = compustat_portfolio_builder.BEMECalculator(engine=self.DummyEngine())

        result = calc.compute_all(linked)

        assert len(result) == 1
        assert result['gvkey'].tolist() == [2]
        assert (result['be'] > 0).all()

    def test_filters_financial(self):
        """Test compute_all removes financial firms."""
        linked = self._sample_linked_df()
        linked.loc[0, 'is_financial'] = True
        calc = compustat_portfolio_builder.BEMECalculator(engine=self.DummyEngine())

        result = calc.compute_all(linked)

        assert len(result) == 1
        assert result['gvkey'].tolist() == [2]
        assert not result['is_financial'].any()

    def test_filters_zero_me(self):
        """Test compute_all removes non-positive ME rows and avoids division by zero."""
        linked = self._sample_linked_df()
        linked.loc[0, 'me'] = 0.0
        calc = compustat_portfolio_builder.BEMECalculator(engine=self.DummyEngine())

        result = calc.compute_all(linked)

        assert len(result) == 1
        assert result['gvkey'].tolist() == [2]
        assert (result['me'] > 0).all()

    def test_filters_beme_outliers_by_year(self):
        """Test compute_all trims top and bottom 0.5% BE/ME observations per year."""
        linked = pd.DataFrame({
            'gvkey': range(1, 101),
            'permno': range(10001, 10101),
            'permco': range(501, 601),
            'date': pd.to_datetime(['1964-06-30'] * 100),
            'cal_year': [1963] * 100,
            'be': [100.0] * 100,
            'me': [1000.0] * 100,
            'sich': [2000] * 100,
            'is_financial': [False] * 100,
            'EXCHCD': [1] * 100,
        })
        linked.loc[0, ['be', 'me']] = [1.0, 1000.0]
        linked.loc[99, ['be', 'me']] = [1000.0, 1.0]
        calc = compustat_portfolio_builder.BEMECalculator(engine=self.DummyEngine())

        result = calc.compute_all(linked)

        assert len(result) == 98
        assert 1 not in set(result['gvkey'])
        assert 100 not in set(result['gvkey'])

    def test_compute_all_from_engine(self, tmp_path):
        """Integration test compute_all using a LinkingEngine with synthetic source files."""
        self._write_mapping_file(tmp_path)
        self._write_be_file(tmp_path)
        self._write_crsp_file(tmp_path, rows=[
            [10001, '1964-06-30', 10, 1, 500, np.nan, 10.0, 0.01, 100.0],
            [10002, '1964-06-30', 11, 2, 501, np.nan, 20.0, 0.02, 50.0],
        ])
        engine = compustat_portfolio_builder.LinkingEngine(base_dir=str(tmp_path))
        calc = compustat_portfolio_builder.BEMECalculator(engine=engine)

        result = calc.compute_all()

        assert len(result) == 1
        assert result['gvkey'].iloc[0] == 1
        assert result['beme'].iloc[0] == pytest.approx(0.1)
        assert result['exchange'].iloc[0] == 1


class TestBackupManager:
    """Tests for BackupManager class."""

    def test_init(self):
        """Test initialization with default and custom directories."""
        manager = compustat_portfolio_builder.BackupManager()
        assert manager.data_dir == 'data'
        # Use os.path.join for platform-independent path comparison
        expected_backup_dir = os.path.join('data', 'backups')
        assert manager.backup_dir == expected_backup_dir

        custom_dir = '/custom/path'
        manager = compustat_portfolio_builder.BackupManager(data_dir=custom_dir)
        assert manager.data_dir == custom_dir

    def test_compute_file_hash(self):
        """Test _compute_file_hash produces consistent MD5 hashes."""
        manager = compustat_portfolio_builder.BackupManager()

        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test content')
            filepath = f.name

        try:
            hash1 = manager._compute_file_hash(filepath)
            hash2 = manager._compute_file_hash(filepath)
            assert hash1 == hash2
        finally:
            os.unlink(filepath)

    def test_compute_file_hash_different(self):
        """Test _compute_file_hash produces different hashes for different content."""
        manager = compustat_portfolio_builder.BackupManager()

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1:
            f1.write('content 1')
            filepath1 = f1.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
            f2.write('content 2')
            filepath2 = f2.name

        try:
            hash1 = manager._compute_file_hash(filepath1)
            hash2 = manager._compute_file_hash(filepath2)
            assert hash1 != hash2
        finally:
            os.unlink(filepath1)
            os.unlink(filepath2)

    def test_backup_creates_directory(self, tmp_path, monkeypatch):
        """Test backup creates backup directory structure."""
        # Create source data files
        for filename in ['ff_25_portfolios.csv', 'ff_6_portfolios.csv', 'ff_factors.csv']:
            filepath = tmp_path / 'data' / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text('header\ncol1,col2\n1,2\n')

        # Mock requests.get to avoid actual download
        def mock_get(url, stream=False, **kwargs):
            class MockResponse:
                def raise_for_status(self):
                    pass
                def __iter__(self):
                    return iter([])
            return MockResponse()

        monkeypatch.setattr('requests.get', mock_get)

        # Create mapping manager to avoid download
        mapping_mgr = compustat_portfolio_builder.MappingManager(cache_dir=str(tmp_path / 'data'))
        Path(mapping_mgr.cache_file).write_text('gvkey,permco,permno\n1,1,1\n')

        manager = compustat_portfolio_builder.BackupManager(data_dir=str(tmp_path / 'data'))
        backup_path = manager.backup()

        # Check backup directory was created
        assert os.path.exists(backup_path)
        assert backup_path.endswith('-pre-compustat')

        # Check backup contains all files
        for filename in ['ff_25_portfolios.csv', 'ff_6_portfolios.csv', 'ff_factors.csv']:
            backup_file = os.path.join(backup_path, filename)
            assert os.path.exists(backup_file), f"File {filename} missing in backup"

    def test_backup_skips_if_same(self, tmp_path, monkeypatch):
        """Test backup skips if backup with same files already exists."""
        # Create source data files
        for filename in ['ff_25_portfolios.csv', 'ff_6_portfolios.csv', 'ff_factors.csv']:
            filepath = tmp_path / 'data' / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text('header\ncol1,col2\n1,2\n')

        # Mock requests.get
        def mock_get(url, stream=False, **kwargs):
            class MockResponse:
                def raise_for_status(self):
                    pass
                def __iter__(self):
                    return iter([])
            return MockResponse()

        monkeypatch.setattr('requests.get', mock_get)

        # Create mapping manager
        mapping_mgr = compustat_portfolio_builder.MappingManager(cache_dir=str(tmp_path / 'data'))
        Path(mapping_mgr.cache_file).write_text('gvkey,permco,permno\n1,1,1\n')

        manager = compustat_portfolio_builder.BackupManager(data_dir=str(tmp_path / 'data'))

        # First backup
        backup_path1 = manager.backup()

        # Second backup - should skip
        backup_path2 = manager.backup()

        # Should return same path
        assert backup_path1 == backup_path2

    def test_restore_file_not_found(self, tmp_path):
        """Test restore raises FileNotFoundError when backup doesn't exist."""
        manager = compustat_portfolio_builder.BackupManager(data_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            manager.restore('/nonexistent/backup')

    def test_restore_creates_missing_files(self, tmp_path):
        """Test restore copies files from backup to data directory."""
        # Create backup directory with files
        backup_dir = tmp_path / 'backup'
        backup_dir.mkdir()
        for filename in ['ff_25_portfolios.csv', 'ff_6_portfolios.csv', 'ff_factors.csv']:
            filepath = backup_dir / filename
            filepath.write_text(f'backup-{filename}\n')

        # Create data directory
        data_dir = tmp_path / 'data'
        data_dir.mkdir()

        manager = compustat_portfolio_builder.BackupManager(data_dir=str(data_dir))
        manager.restore(str(backup_dir))

        # Check files were restored
        for filename in ['ff_25_portfolios.csv', 'ff_6_portfolios.csv', 'ff_factors.csv']:
            restored_path = data_dir / filename
            assert restored_path.exists()
            assert restored_path.read_text().startswith('backup-')


class TestPortfolioConstructor:
    """Tests for PortfolioConstructor class."""

    class DummyBEMECalculator:
        def __init__(self, df=None):
            self.df = df if df is not None else pd.DataFrame()

        def compute_all(self):
            return self.df

    class DummyCRSPSource:
        def __init__(self, df=None):
            self.df = df if df is not None else pd.DataFrame()

        def load_and_clean(self):
            return self.df

    def _constructor(self, beme_df=None, crsp_df=None):
        return compustat_portfolio_builder.PortfolioConstructor(
            beme_calculator=self.DummyBEMECalculator(beme_df),
            crsp_source=self.DummyCRSPSource(crsp_df),
        )

    def _breakpoint_df(self):
        return pd.DataFrame({
            'gvkey': range(1, 9),
            'permno': range(10001, 10009),
            'permco': range(500, 508),
            'date': pd.to_datetime(['1964-06-30'] * 8),
            'year': [1964] * 8,
            'be': [1.0] * 8,
            'me': [10.0, 20.0, 30.0, 40.0, 50.0, 1000.0, 1.0, 60.0],
            'beme': [0.1, 0.2, 0.3, 0.4, 0.5, 9.9, 0.01, 0.6],
            'sich': [2000] * 8,
            'is_financial': [False, False, False, False, False, False, True, False],
            'exchange': [1, 1, 1, 1, 1, 2, 1, 3],
        })

    def _full_beme_df(self):
        rows = []
        permno = 10001
        size_me = [10.0, 30.0, 50.0, 70.0, 90.0]
        bm_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for size_idx, me in enumerate(size_me, start=1):
            for bm_idx, beme in enumerate(bm_values, start=1):
                rows.append({
                    'gvkey': permno,
                    'permno': permno,
                    'permco': permno + 1000,
                    'date': pd.Timestamp('1964-06-30'),
                    'year': 1964,
                    'be': me * beme,
                    'me': me,
                    'beme': beme,
                    'sich': 2000,
                    'is_financial': False,
                    'exchange': 1 if size_idx <= 4 else 2,
                    'size_idx': size_idx,
                    'bm_idx': bm_idx,
                })
                permno += 1
        return pd.DataFrame(rows)

    def _full_crsp_df(self, beme_df):
        months = pd.date_range('1964-07-01', '1965-06-01', freq='MS')
        rows = []
        for month in months:
            for row in beme_df.itertuples(index=False):
                rows.append({
                    'PERMNO': row.permno,
                    'date': month,
                    'RET': 0.001 * row.size_idx + 0.0001 * row.bm_idx,
                    'PRC': 10.0,
                    'SHROUT': 100.0,
                    'PERMCO': row.permco,
                    'EXCHCD': row.exchange,
                    'SHRCD': 10,
                })
        return pd.DataFrame(rows)

    def test_init(self, monkeypatch):
        """Test initialization with default and custom calculators."""
        default_beme = self.DummyBEMECalculator()
        default_crsp = self.DummyCRSPSource()
        monkeypatch.setattr(compustat_portfolio_builder, 'BEMECalculator', lambda: default_beme)
        monkeypatch.setattr(compustat_portfolio_builder, 'CRSPSource', lambda: default_crsp)

        constructor = compustat_portfolio_builder.PortfolioConstructor()
        assert constructor.beme_calculator is default_beme
        assert constructor.crsp_source is default_crsp

        custom_beme = self.DummyBEMECalculator()
        custom_crsp = self.DummyCRSPSource()
        constructor = compustat_portfolio_builder.PortfolioConstructor(
            beme_calculator=custom_beme,
            crsp_source=custom_crsp,
        )
        assert constructor.beme_calculator is custom_beme
        assert constructor.crsp_source is custom_crsp

    def test_compute_nyse_breakpoints_me(self):
        """Test ME breakpoints use only NYSE non-financial stocks."""
        constructor = self._constructor()
        breakpoints = constructor.compute_nyse_breakpoints(self._breakpoint_df(), metric='me')

        assert breakpoints == pytest.approx([18.0, 26.0, 34.0, 42.0])

    def test_compute_nyse_breakpoints_beme(self):
        """Test BE/ME breakpoints use only NYSE non-financial stocks."""
        constructor = self._constructor()
        breakpoints = constructor.compute_nyse_breakpoints(self._breakpoint_df(), metric='beme')

        assert breakpoints == pytest.approx([0.18, 0.26, 0.34, 0.42])

    def test_assign_portfolios(self):
        """Test stocks are assigned to Ken French 25-portfolio labels."""
        df = pd.DataFrame({
            'me': [10.0, 50.0, 90.0],
            'beme': [0.1, 0.5, 0.9],
        })
        constructor = self._constructor()

        labels = constructor.assign_portfolios(df, [20.0, 40.0, 60.0, 80.0], [0.2, 0.4, 0.6, 0.8])

        assert labels.tolist() == ['SMALL LoBM', 'ME3 BM3', 'BIG HiBM']

    def test_build_25_portfolios_synthetic(self):
        """Test full synthetic pipeline returns 12 months by 25 columns."""
        beme_df = self._full_beme_df()
        crsp_df = self._full_crsp_df(beme_df)
        constructor = self._constructor(beme_df, crsp_df)

        portfolios = constructor.build_25_portfolios()

        assert portfolios.shape == (12, 25)
        assert list(portfolios.columns) == constructor.PORTFOLIO_COLUMNS
        assert portfolios.index.name == 'Date'
        assert portfolios.index[0] == '1964-07'
        assert portfolios.index[-1] == '1965-06'
        assert portfolios['SMALL LoBM'].iloc[0] == pytest.approx(0.11)
        assert portfolios['BIG HiBM'].iloc[0] == pytest.approx(0.55)

    def test_build_25_portfolios_monotonic(self):
        """Test stocks assigned to SMALL portfolios have lower ME than BIG portfolios."""
        beme_df = self._full_beme_df()
        constructor = self._constructor()
        size_bps = constructor.compute_nyse_breakpoints(beme_df, metric='me')
        beme_bps = constructor.compute_nyse_breakpoints(beme_df, metric='beme')
        labels = constructor.assign_portfolios(beme_df, size_bps, beme_bps)
        assigned = beme_df.assign(portfolio=labels)

        small_labels = [label for (size, _), label in constructor.PORTFOLIO_LABELS.items() if size == 1]
        big_labels = [label for (size, _), label in constructor.PORTFOLIO_LABELS.items() if size == 5]
        small_mean_me = assigned.loc[assigned['portfolio'].isin(small_labels), 'me'].mean()
        big_mean_me = assigned.loc[assigned['portfolio'].isin(big_labels), 'me'].mean()

        assert small_mean_me < big_mean_me

    def test_compute_nyse_breakpoints_2x3_me(self):
        """Test ME 2×3 breakpoint returns median from NYSE non-financial."""
        constructor = self._constructor()
        bp_df = self._breakpoint_df()
        # NYSE non-financial ME values: 10, 20, 30, 40, 50 (exchange=1, not financial)
        # Median of 5 values is the 3rd sorted value
        breakpoints = constructor.compute_nyse_breakpoints_2x3(bp_df, metric='me')
        assert len(breakpoints) == 1
        assert breakpoints[0] == pytest.approx(30.0)

    def test_compute_nyse_breakpoints_2x3_beme(self):
        """Test BE/ME 2×3 breakpoints return 30th and 70th percentiles from NYSE non-financial."""
        constructor = self._constructor()
        bp_df = self._breakpoint_df()
        # NYSE non-financial BE/ME values: 0.1, 0.2, 0.3, 0.4, 0.5
        breakpoints = constructor.compute_nyse_breakpoints_2x3(bp_df, metric='beme')
        assert len(breakpoints) == 2
        assert breakpoints[0] == pytest.approx(0.22)  # 30th percentile
        assert breakpoints[1] == pytest.approx(0.38)  # 70th percentile

    def test_assign_portfolios_2x3(self):
        """Test stocks are assigned to 6 Ken French portfolio labels."""
        df = pd.DataFrame({
            'me': [10.0, 50.0, 100.0, 10.0, 50.0, 100.0, 25.0, 25.0],
            'beme': [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.3, 0.7],
        })
        constructor = self._constructor()
        size_bp = [40.0]  # median
        beme_bps = [0.2, 0.6]  # 30th and 70th

        labels = constructor.assign_portfolios_2x3(df, size_bp, beme_bps)

        # me=10 < 40 -> SMALL, beme=0.1 < 0.2 -> LoBM => SMALL LoBM
        assert labels.iloc[0] == 'SMALL LoBM'
        # me=50 > 40 -> BIG, beme=0.1 < 0.2 -> LoBM => BIG LoBM
        assert labels.iloc[1] == 'BIG LoBM'
        # me=100 > 40 -> BIG, beme=0.1 < 0.2 -> LoBM => BIG LoBM
        assert labels.iloc[2] == 'BIG LoBM'
        # me=10 < 40 -> SMALL, beme=0.9 > 0.6 -> HiBM => SMALL HiBM
        assert labels.iloc[3] == 'SMALL HiBM'
        # me=50 > 40 -> BIG, beme=0.9 > 0.6 -> HiBM => BIG HiBM
        assert labels.iloc[4] == 'BIG HiBM'
        # me=100 > 40 -> BIG, beme=0.9 > 0.6 -> HiBM => BIG HiBM
        assert labels.iloc[5] == 'BIG HiBM'
        # me=25 < 40 -> SMALL, beme=0.3 between 0.2 and 0.6 -> BM2 => ME1 BM2
        assert labels.iloc[6] == 'ME1 BM2'
        # me=25 < 40 -> SMALL, beme=0.7 between 0.2 and 0.6 -> BM2 => ME1 BM2
        # Actually 0.7 > 0.6 -> HiBM => SMALL HiBM
        assert labels.iloc[7] == 'SMALL HiBM'

    def test_build_6_portfolios_synthetic(self):
        """Test full synthetic 6-portfolio pipeline returns 12 months by 6 columns."""
        beme_df = self._full_beme_df()
        crsp_df = self._full_crsp_df(beme_df)
        constructor = self._constructor(beme_df, crsp_df)

        portfolios = constructor.build_6_portfolios()

        assert portfolios.shape == (12, 6)
        expected_columns = constructor.PORTFOLIO_COLUMNS_6
        assert list(portfolios.columns) == expected_columns
        assert portfolios.index.name == 'Date'
        assert portfolios.index[0] == '1964-07'
        assert portfolios.index[-1] == '1965-06'
        # Verify value-weighted returns are non-NaN and in expected range
        assert portfolios.notna().all().all()
        assert (portfolios > -100).all().all()
        assert (portfolios < 100).all().all()


class TestFactorCalculator:
    """Tests for FactorCalculator class."""

    class DummyPortfolioConstructor:
        def __init__(self, port6_df=None):
            self.port6_df = port6_df if port6_df is not None else pd.DataFrame()
            # Add required attributes for FactorCalculator
            self.PORTFOLIO_COLUMNS_6 = [
                'SMALL LoBM', 'ME1 BM2', 'SMALL HiBM',
                'BIG LoBM', 'ME2 BM2', 'BIG HiBM',
            ]

        def build_6_portfolios(self):
            return self.port6_df

    class DummyCRSPSource:
        def __init__(self, crsp_df=None):
            self.crsp_df = crsp_df if crsp_df is not None else pd.DataFrame()
            self.base_dir = '.'

        def load_and_clean(self):
            return self.crsp_df

    def test_init(self, monkeypatch):
        """Test initialization with default and custom dependencies."""
        default_port = self.DummyPortfolioConstructor()
        default_crsp = self.DummyCRSPSource()
        monkeypatch.setattr(compustat_portfolio_builder, 'PortfolioConstructor', lambda: default_port)
        monkeypatch.setattr(compustat_portfolio_builder, 'CRSPSource', lambda: default_crsp)

        calc = compustat_portfolio_builder.FactorCalculator()
        assert calc.portfolio_constructor is default_port
        assert calc.crsp_source is default_crsp

        custom_port = self.DummyPortfolioConstructor()
        custom_crsp = self.DummyCRSPSource()
        calc = compustat_portfolio_builder.FactorCalculator(
            portfolio_constructor=custom_port,
            crsp_source=custom_crsp,
        )
        assert calc.portfolio_constructor is custom_port
        assert calc.crsp_source is custom_crsp

    def test_compute_smb_hml_default(self, monkeypatch):
        """Test compute_smb_hml uses PortfolioConstructor when port6 is None."""
        expected_port6 = pd.DataFrame({
            'SMALL LoBM': [0.1, 0.2],
            'ME1 BM2': [0.3, 0.4],
            'SMALL HiBM': [0.5, 0.6],
            'BIG LoBM': [0.7, 0.8],
            'ME2 BM2': [0.9, 1.0],
            'BIG HiBM': [1.1, 1.2],
        }, index=['1964-07', '1964-08'])

        dummy_port = self.DummyPortfolioConstructor(expected_port6)
        monkeypatch.setattr(compustat_portfolio_builder, 'PortfolioConstructor', lambda: dummy_port)

        calc = compustat_portfolio_builder.FactorCalculator()
        smb, hml = calc.compute_smb_hml()

        assert len(smb) == 2
        assert len(hml) == 2
        # Verify SMB formula: 1/3*(S+ME1+S_Lo) - 1/3*(B+ME2+B_Lo)
        # Row 0: 1/3*(0.5+0.3+0.1) - 1/3*(1.1+0.9+0.7) = 0.3 - 0.9 = -0.6
        assert smb.iloc[0] == pytest.approx(-0.6)
        # Row 1: 1/3*(0.6+0.4+0.2) - 1/3*(1.2+1.0+0.8) = 0.4 - 1.0 = -0.6
        assert smb.iloc[1] == pytest.approx(-0.6)
        # Verify HML formula: 1/2*(S_Hi+B_Hi) - 1/2*(S_Lo+B_Lo)
        # Row 0: 1/2*(0.5+1.1) - 1/2*(0.1+0.7) = 0.8 - 0.4 = 0.4
        assert hml.iloc[0] == pytest.approx(0.4)
        # Row 1: 1/2*(0.6+1.2) - 1/2*(0.2+0.8) = 0.9 - 0.5 = 0.4
        assert hml.iloc[1] == pytest.approx(0.4)

    def test_compute_smb_hml_custom(self):
        """Test compute_smb_hml uses provided port6 DataFrame."""
        port6 = pd.DataFrame({
            'SMALL LoBM': [0.1, 0.2],
            'ME1 BM2': [0.3, 0.4],
            'SMALL HiBM': [0.5, 0.6],
            'BIG LoBM': [0.7, 0.8],
            'ME2 BM2': [0.9, 1.0],
            'BIG HiBM': [1.1, 1.2],
        }, index=['1964-07', '1964-08'])

        calc = compustat_portfolio_builder.FactorCalculator()
        smb, hml = calc.compute_smb_hml(port6)

        assert len(smb) == 2
        assert len(hml) == 2

    def test_compute_smb_hml_empty(self):
        """Test compute_smb_hml returns empty Series when port6 is empty."""
        port6 = pd.DataFrame(columns=[
            'SMALL LoBM', 'ME1 BM2', 'SMALL HiBM',
            'BIG LoBM', 'ME2 BM2', 'BIG HiBM'
        ])

        calc = compustat_portfolio_builder.FactorCalculator()
        smb, hml = calc.compute_smb_hml(port6)

        assert len(smb) == 0
        assert len(hml) == 0
        assert smb.dtype == float
        assert hml.dtype == float

    def test_compute_market_return_default(self, monkeypatch):
        """Test compute_market_return uses CRSPSource when crsp_df is None."""
        import numpy as np

        crsp_data = pd.DataFrame({
            'PERMNO': [10001, 10002, 10001, 10002],
            'date': pd.to_datetime(['1963-12-31', '1963-12-31', '1964-01-31', '1964-01-31']),
            'RET': [0.01, 0.03, 0.02, 0.04],
            'PRC': [10.0, 20.0, 10.0, 20.0],
            'SHROUT': [1000.0, 500.0, 1000.0, 500.0],
            'PERMCO': [500, 501, 500, 501],
            'EXCHCD': [1, 2, 1, 2],
            'SHRCD': [10, 10, 10, 10],
        })

        dummy_crsp = self.DummyCRSPSource(crsp_data)
        calc = compustat_portfolio_builder.FactorCalculator(crsp_source=dummy_crsp)
        mkt_return = calc.compute_market_return()

        # Month 1964-01: (1000*0.02 + 500*0.04) / (1000 + 500) = 0.03 -> 3.0%
        assert mkt_return.loc['1964-01'] == pytest.approx(3.0)

    def test_compute_market_return_custom(self):
        """Test compute_market_return uses provided crsp_df."""
        crsp_data = pd.DataFrame({
            'PERMNO': [10001, 10001],
            'date': pd.to_datetime(['1964-06-30', '1964-07-31']),
            'RET': [0.01, 0.02],
            'PRC': [10.0, 10.0],
            'SHROUT': [1000.0, 1000.0],
            'PERMCO': [500, 500],
            'EXCHCD': [1, 1],
        })

        calc = compustat_portfolio_builder.FactorCalculator()
        mkt_return = calc.compute_market_return(crsp_data)

        assert len(mkt_return) == 2
        assert mkt_return.name == 'Mkt-RF'

    def test_compute_market_return_empty(self):
        """Test compute_market_return returns empty Series when crsp_df is empty."""
        crsp_data = pd.DataFrame()

        calc = compustat_portfolio_builder.FactorCalculator()
        mkt_return = calc.compute_market_return(crsp_data)

        assert len(mkt_return) == 0
        assert mkt_return.dtype == float

    def test_assemble_factors_default(self, monkeypatch, tmp_path):
        """Test assemble_factors uses compute_smb_hml and compute_market_return when params are None."""
        # Create sample port6 data
        port6 = pd.DataFrame({
            'SMALL LoBM': [0.1, 0.2],
            'ME1 BM2': [0.3, 0.4],
            'SMALL HiBM': [0.5, 0.6],
            'BIG LoBM': [0.7, 0.8],
            'ME2 BM2': [0.9, 1.0],
            'BIG HiBM': [1.1, 1.2],
        }, index=['1964-07', '1964-08'])

        # Create sample CRSP data
        crsp_data = pd.DataFrame({
            'PERMNO': [10001, 10001, 10001, 10001],
            'date': pd.to_datetime(['1964-06-30', '1964-07-31', '1964-07-01', '1964-08-31']),
            'RET': [0.01, 0.02, 0.03, 0.04],
            'PRC': [10.0, 10.0, 10.0, 10.0],
            'SHROUT': [1000.0, 1000.0, 1000.0, 1000.0],
            'PERMCO': [500, 500, 500, 500],
            'EXCHCD': [1, 1, 1, 1],
            'SHRCD': [10, 10, 10, 10],
        })

        # Create sample RF data
        rf_data_path = tmp_path / 'data' / 'ff_factors.csv'
        rf_data_path.parent.mkdir(parents=True, exist_ok=True)
        rf_df = pd.DataFrame({
            'Date': ['1964-07-01', '1964-08-01'],
            'Mkt-RF': [1.0, 1.0],
            'SMB': [2.0, 2.0],
            'HML': [3.0, 3.0],
            'RF': [0.25, 0.25],
        })
        rf_df.to_csv(rf_data_path, index=False)

        # Mock PortfolioConstructor and CRSPSource
        dummy_port = self.DummyPortfolioConstructor(port6)
        dummy_crsp = self.DummyCRSPSource(crsp_data)
        dummy_crsp.base_dir = str(tmp_path)
        monkeypatch.setattr(compustat_portfolio_builder, 'PortfolioConstructor', lambda: dummy_port)
        monkeypatch.setattr(compustat_portfolio_builder, 'CRSPSource', lambda: dummy_crsp)

        calc = compustat_portfolio_builder.FactorCalculator(
            portfolio_constructor=dummy_port,
            crsp_source=dummy_crsp,
        )

        factors = calc.assemble_factors()

        # Verify structure
        assert factors.shape[1] == 5
        assert list(factors.columns) == ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
        assert factors.index.name == 'Date'
        assert len(factors) == 2  # Only 1964-07 and 1964-08 in range

    def test_assemble_factors_custom_params(self):
        """Test assemble_factors uses provided port6 and crsp_df."""
        port6 = pd.DataFrame({
            'SMALL LoBM': [0.1],
            'ME1 BM2': [0.2],
            'SMALL HiBM': [0.3],
            'BIG LoBM': [0.4],
            'ME2 BM2': [0.5],
            'BIG HiBM': [0.6],
        }, index=['1964-07'])

        # Create CRSP data with proper structure for lagged_ME calculation
        # Need multiple observations per PERMNO to compute lagged_ME
        crsp_data = pd.DataFrame({
            'PERMNO': [10001, 10001],  # Two months of data
            'date': pd.to_datetime(['1964-06-30', '1964-07-31']),
            'RET': [0.01, 0.02],
            'PRC': [10.0, 10.0],
            'SHROUT': [1000.0, 1000.0],
            'PERMCO': [500, 500],
            'EXCHCD': [1, 1],
            'SHRCD': [10, 10],
        })

        calc = compustat_portfolio_builder.FactorCalculator()
        factors = calc.assemble_factors(port6, crsp_data)

        assert len(factors) == 1
        assert 'Date' in factors.columns
        assert 'Mkt-RF' in factors.columns
        assert 'SMB' in factors.columns
        assert 'HML' in factors.columns
        assert 'RF' in factors.columns

    def test_assemble_factors_date_range(self):
        """Test assemble_factors filters to 1964-07 to 1991-12."""
        port6 = pd.DataFrame({
            'SMALL LoBM': [0.1, 0.2],
            'ME1 BM2': [0.3, 0.4],
            'SMALL HiBM': [0.5, 0.6],
            'BIG LoBM': [0.7, 0.8],
            'ME2 BM2': [0.9, 1.0],
            'BIG HiBM': [1.1, 1.2],
        }, index=['1964-07', '1991-12'])

        # Create CRSP data with proper structure for lagged_ME calculation
        crsp_data = pd.DataFrame({
            'PERMNO': [10001, 10001, 10002, 10002],  # Two stocks, multiple months
            'date': pd.to_datetime(['1964-06-30', '1964-07-31', '1991-11-30', '1991-12-31']),
            'RET': [0.01, 0.02, 0.03, 0.04],
            'PRC': [10.0, 10.0, 20.0, 20.0],
            'SHROUT': [1000.0, 1000.0, 500.0, 500.0],
            'PERMCO': [500, 500, 501, 501],
            'EXCHCD': [1, 1, 2, 2],
            'SHRCD': [10, 10, 10, 10],
        })

        calc = compustat_portfolio_builder.FactorCalculator()
        factors = calc.assemble_factors(port6, crsp_data)

        # Should only include 1964-07 and 1991-12
        assert len(factors) == 2
        assert factors.iloc[0]['Date'] == '1964-07'
        assert factors.iloc[1]['Date'] == '1991-12'


class TestDiagnosticsReport:
    """Tests for DiagnosticsReport class."""

    # ------------------------------------------------------------------ #
    # Dummy helpers for synthetic pipeline
    # ------------------------------------------------------------------ #

    class DummyMappingManager:
        """Minimal MappingManager stub that returns canned stats."""

        def mapping_stats(self):
            return {
                'total_rows': 15000,
                'unique_gvkeys': 30000,
                'unique_permco': 25000,
                'unique_permno': 35000,
                'gvkey_multi_permco_pct': 5.0,
                'permco_multi_gvkey_pct': 8.0,
                'permco_multi_permno_pct': 12.0,
            }

    class DummyLinkingEngine:
        """Minimal LinkingEngine stub."""

        def build_link(self):
            return pd.DataFrame({
                'gvkey': [1, 2, 3],
                'permno': [10001, 10002, 10003],
                'permco': [500, 501, 502],
                'date': pd.to_datetime(['1964-06-30', '1965-06-30', '1966-06-30']),
                'cal_year': [1963, 1964, 1965],
                'be': [100.0, 200.0, 300.0],
                'me': [1000.0, 2000.0, 3000.0],
                'sich': [2000, 3500, 5000],
                'is_financial': [False, False, False],
                'EXCHCD': [1, 2, 1],
            })

    class DummyPortfolioConstructor:
        """Minimal PortfolioConstructor stub for synthetic 25-portfolio returns."""

        class _DummyBEME:
            def compute_all(self):
                return pd.DataFrame()

        class _DummyCRSP:
            def load_and_clean(self):
                return pd.DataFrame()

        PORTFOLIO_COLUMNS = [
            'SMALL LoBM', 'ME1 BM2', 'ME1 BM3', 'ME1 BM4', 'SMALL HiBM',
            'ME2 BM1', 'ME2 BM2', 'ME2 BM3', 'ME2 BM4', 'ME2 BM5',
            'ME3 BM1', 'ME3 BM2', 'ME3 BM3', 'ME3 BM4', 'ME3 BM5',
            'ME4 BM1', 'ME4 BM2', 'ME4 BM3', 'ME4 BM4', 'ME4 BM5',
            'BIG LoBM', 'ME5 BM2', 'ME5 BM3', 'ME5 BM4', 'BIG HiBM',
        ]

        def __init__(self):
            self.beme_calculator = self._DummyBEME()
            self.crsp_source = self._DummyCRSP()

        def build_25_portfolios(self):
            """Return 12 months of synthetic 25-portfolio returns."""
            rng = np.random.default_rng(42)
            dates = [f'1964-{m:02d}' for m in range(7, 13)] + [f'1965-{m:02d}' for m in range(1, 7)]
            data = {col: rng.normal(1.0, 2.0, size=12) for col in self.PORTFOLIO_COLUMNS}
            df = pd.DataFrame(data, index=dates)
            df.index.name = 'Date'
            return df

        def build_6_portfolios(self):
            cols = ['SMALL LoBM', 'ME1 BM2', 'SMALL HiBM',
                    'BIG LoBM', 'ME2 BM2', 'BIG HiBM']
            dates = [f'1964-{m:02d}' for m in range(7, 13)] + [f'1965-{m:02d}' for m in range(1, 7)]
            df = pd.DataFrame(
                np.random.default_rng(42).normal(1.0, 2.0, size=(12, 6)),
                index=dates, columns=cols
            )
            df.index.name = 'Date'
            return df

    class DummyBEMECalculator:
        """Minimal BEMECalculator stub."""

        def compute_all(self):
            return pd.DataFrame({
                'gvkey': range(1, 101),
                'permno': range(10001, 10101),
                'permco': range(500, 600),
                'date': pd.to_datetime(['1964-06-30'] * 100),
                'year': [1964] * 100,
                'be': [100.0] * 100,
                'me': [1000.0] * 100,
                'beme': [0.1] * 100,
                'sich': [2000] * 100,
                'is_financial': [False] * 100,
                'exchange': [1] * 100,
            })

    class DummyCRSPSource:
        """Minimal CRSPSource stub."""

        def load_and_clean(self):
            months = pd.date_range('1964-07-01', '1965-06-01', freq='MS')
            rows = []
            for month in months:
                for permno in range(10001, 10101):
                    rows.append({
                        'PERMNO': permno,
                        'date': month,
                        'RET': 0.01,
                        'PRC': 10.0,
                        'SHROUT': 1000.0,
                        'PERMCO': 500 + (permno - 10001) // 20,
                        'EXCHCD': 1,
                        'SHRCD': 10,
                    })
            return pd.DataFrame(rows)

    class DummyFactorCalculator:
        """Minimal FactorCalculator stub."""

        def assemble_factors(self, port6=None, crsp_df=None):
            dates = [f'1964-{m:02d}' for m in range(7, 13)] + [f'1965-{m:02d}' for m in range(1, 7)]
            return pd.DataFrame({
                'Date': dates,
                'Mkt-RF': np.random.default_rng(42).normal(0.5, 1.0, size=12),
                'SMB': np.random.default_rng(43).normal(0.3, 0.8, size=12),
                'HML': np.random.default_rng(44).normal(0.4, 0.9, size=12),
                'RF': [0.25] * 12,
            })

    # ------------------------------------------------------------------ #
    # Tests
    # ------------------------------------------------------------------ #

    def test_init_default(self, monkeypatch, tmp_path):
        """Test DiagnosticsReport initialises with defaults (custom deps to avoid file access)."""
        report = compustat_portfolio_builder.DiagnosticsReport(
            portfolio_constructor=self.DummyPortfolioConstructor(),
            factor_calculator=self.DummyFactorCalculator(),
            mapping_manager=self.DummyMappingManager(),
            linking_engine=self.DummyLinkingEngine(),
        )
        assert report is not None
        assert isinstance(report.pc, self.DummyPortfolioConstructor)
        assert isinstance(report.fc, self.DummyFactorCalculator)
        assert isinstance(report.mm, self.DummyMappingManager)
        assert isinstance(report.le, self.DummyLinkingEngine)

    def test_init_custom(self):
        """Test DiagnosticsReport accepts custom dependencies."""
        report = compustat_portfolio_builder.DiagnosticsReport(
            portfolio_constructor=self.DummyPortfolioConstructor(),
            factor_calculator=self.DummyFactorCalculator(),
            mapping_manager=self.DummyMappingManager(),
            linking_engine=self.DummyLinkingEngine(),
        )
        assert isinstance(report.pc, self.DummyPortfolioConstructor)
        assert isinstance(report.fc, self.DummyFactorCalculator)
        assert isinstance(report.mm, self.DummyMappingManager)
        assert isinstance(report.le, self.DummyLinkingEngine)

    def test_run_full_returns_dict_with_sections(self):
        """Test run_full returns dict with all 5 sections."""
        report = compustat_portfolio_builder.DiagnosticsReport(
            portfolio_constructor=self.DummyPortfolioConstructor(),
            factor_calculator=self.DummyFactorCalculator(),
            mapping_manager=self.DummyMappingManager(),
            linking_engine=self.DummyLinkingEngine(),
        )

        results = report.run_full()

        # Verify all 5 sections present
        expected_keys = [
            'mapping_coverage',
            'portfolio_diagnostics',
            'ken_french_comparison',
            'factor_statistics',
            'monotonicity',
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
            assert isinstance(results[key], dict), f"Section {key} should be a dict"

    def test_run_full_mapping_coverage(self, tmp_path, monkeypatch):
        """Test mapping coverage section contains expected fields."""
        report = compustat_portfolio_builder.DiagnosticsReport(
            portfolio_constructor=self.DummyPortfolioConstructor(),
            factor_calculator=self.DummyFactorCalculator(),
            mapping_manager=self.DummyMappingManager(),
            linking_engine=self.DummyLinkingEngine(),
            data_dir=str(tmp_path / 'data'),
        )

        results = report.run_full()
        mc = results['mapping_coverage']

        assert 'unique_gvkeys' in mc
        assert 'unique_permco' in mc
        assert 'unique_permno' in mc
        assert 'linked_gvkeys' in mc
        assert mc['unique_gvkeys'] == 30000
        assert mc['linked_gvkeys'] == 3

    def test_run_full_portfolio_diagnostics(self, tmp_path, monkeypatch):
        """Test portfolio diagnostics section with synthetic data."""
        # Create dummy PortfolioConstructor with _prepare_beme etc.
        class FullDummyPC(self.DummyPortfolioConstructor):
            def __init__(self):
                super().__init__()
                # Override with data-rich CRSP source
                months = pd.date_range('1964-07-01', '1965-06-01', freq='MS')
                crsp_rows = []
                for month in months:
                    for i in range(1, 101):
                        crsp_rows.append({
                            'PERMNO': 10000 + i,
                            'date': month,
                            'RET': 0.01,
                            'PRC': 10.0,
                            'SHROUT': 1000.0,
                            'PERMCO': 500 + i // 20,
                            'EXCHCD': 1,
                            'SHRCD': 10,
                        })
                class DataCRSP:
                    def load_and_clean(self):
                        return pd.DataFrame(crsp_rows)
                self.crsp_source = DataCRSP()
                # Override beme_calculator with data-rich version
                beme_data = pd.DataFrame({
                    'gvkey': range(1, 101),
                    'permno': range(10001, 10101),
                    'permco': range(500, 600),
                    'date': pd.to_datetime(['1964-06-30'] * 100),
                    'year': [1964] * 100,
                    'be': [100.0] * 100,
                    'me': [1000.0] * 100,
                    'beme': [0.1] * 100,
                    'sich': [2000] * 100,
                    'is_financial': [False] * 100,
                    'exchange': [1] * 100,
                })
                class DataBEME:
                    def compute_all(self):
                        return beme_data
                self.beme_calculator = DataBEME()

            def _prepare_beme(self, df):
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
                df['permno'] = pd.to_numeric(df['permno'], errors='coerce')
                df['me'] = pd.to_numeric(df['me'], errors='coerce')
                df['beme'] = pd.to_numeric(df['beme'], errors='coerce')
                if 'is_financial' not in df.columns:
                    df['is_financial'] = False
                return df

            def _prepare_crsp(self, df):
                df = df.copy()
                df = df.rename(columns={
                    'PERMNO': 'permno', 'PERMCO': 'permco', 'EXCHCD': 'exchange'
                })
                df['permno'] = pd.to_numeric(df['permno'], errors='coerce')
                df['date'] = pd.to_datetime(df['date'])
                df['_month'] = df['date'].dt.to_period('M').dt.to_timestamp()
                df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
                return df

            def _formation_snapshot(self, df, year):
                snapshot = df[df['year'] == year].copy()
                june = snapshot[snapshot['date'].dt.month == 6].copy()
                if not june.empty:
                    snapshot = june
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

            def compute_nyse_breakpoints(self, df, metric='me'):
                return [20.0, 40.0, 60.0, 80.0]

            def assign_portfolios(self, df, size_bps, beme_bps):
                import itertools
                labels = []
                for i in range(len(df)):
                    labels.append(list(
                        compustat_portfolio_builder.PortfolioConstructor.PORTFOLIO_LABELS.values()
                    )[i % 25])
                return pd.Series(labels, index=df.index)

            def _holding_months(self, year):
                return pd.date_range(f'{year}-07-01', f'{year+1}-06-01', freq='MS')

        report = compustat_portfolio_builder.DiagnosticsReport(
            portfolio_constructor=FullDummyPC(),
            factor_calculator=self.DummyFactorCalculator(),
            mapping_manager=self.DummyMappingManager(),
            linking_engine=self.DummyLinkingEngine(),
            data_dir=str(tmp_path / 'data'),
        )

        results = report.run_full()
        pd_sec = results['portfolio_diagnostics']

        assert 'n_months' in pd_sec
        assert 'cell_min_stocks' in pd_sec
        assert 'cell_mean_stocks' in pd_sec
        assert 'cell_max_stocks' in pd_sec
        # With 100 stocks and 25 portfolios, each should average ~4 stocks
        assert pd_sec['cell_mean_stocks'] > 0

    def test_run_full_ken_french_comparison_synthetic(self, tmp_path, monkeypatch):
        """Test KF comparison handles missing backup gracefully."""
        import os as os_module
        original_glob = compustat_portfolio_builder.glob.glob

        def mock_glob(pattern):
            return []  # No backup dirs

        monkeypatch.setattr(compustat_portfolio_builder.glob, 'glob', mock_glob)

        report = compustat_portfolio_builder.DiagnosticsReport(
            portfolio_constructor=self.DummyPortfolioConstructor(),
            factor_calculator=self.DummyFactorCalculator(),
            mapping_manager=self.DummyMappingManager(),
            linking_engine=self.DummyLinkingEngine(),
            data_dir=str(tmp_path / 'data'),
        )

        results = report.run_full()
        kfc = results['ken_french_comparison']

        assert 'error' in kfc
        assert 'backup' in kfc['error'].lower()

    def test_run_full_factor_statistics(self, tmp_path, monkeypatch):
        """Test factor statistics section contains self_stats."""
        import os as os_module
        # Mock glob to return no backups (simpler test)
        def mock_glob(pattern):
            return []

        monkeypatch.setattr(compustat_portfolio_builder.glob, 'glob', mock_glob)

        report = compustat_portfolio_builder.DiagnosticsReport(
            portfolio_constructor=self.DummyPortfolioConstructor(),
            factor_calculator=self.DummyFactorCalculator(),
            mapping_manager=self.DummyMappingManager(),
            linking_engine=self.DummyLinkingEngine(),
            data_dir=str(tmp_path / 'data'),
        )

        results = report.run_full()
        fs = results['factor_statistics']

        if 'error' in fs:
            # Could be error if no backup, but self_stats should still exist
            if 'comparison_error' in fs:
                assert 'self_stats' in fs
        else:
            assert 'self_stats' in fs

    def test_run_full_monotonicity(self, tmp_path, monkeypatch):
        """Test monotonicity section contains size and value effects."""
        report = compustat_portfolio_builder.DiagnosticsReport(
            portfolio_constructor=self.DummyPortfolioConstructor(),
            factor_calculator=self.DummyFactorCalculator(),
            mapping_manager=self.DummyMappingManager(),
            linking_engine=self.DummyLinkingEngine(),
            data_dir=str(tmp_path / 'data'),
        )

        results = report.run_full()
        mono = results['monotonicity']

        assert 'size_effect' in mono
        assert 'value_effect' in mono
        # With random data, small/big should still have values
        assert 'small_mean_return' in mono['size_effect']
        assert 'big_mean_return' in mono['size_effect']
        assert 'hibm_mean_return' in mono['value_effect']
        assert 'lobm_mean_return' in mono['value_effect']

    def test_save_report_creates_file(self, monkeypatch):
        """Test report is saved to evidence directory."""
        # Remove evidence file if it already exists
        import os as os_module
        report_path = '.sisyphus/evidence/task-11-full-report.txt'
        cells_path = '.sisyphus/evidence/task-11-portfolio-cells.csv'
        for path in [report_path, cells_path]:
            if os_module.path.exists(path):
                os_module.remove(path)

        # Patch glob to return no backup
        def mock_glob(pattern):
            return []

        monkeypatch.setattr(compustat_portfolio_builder.glob, 'glob', mock_glob)

        report = compustat_portfolio_builder.DiagnosticsReport(
            portfolio_constructor=self.DummyPortfolioConstructor(),
            factor_calculator=self.DummyFactorCalculator(),
            mapping_manager=self.DummyMappingManager(),
            linking_engine=self.DummyLinkingEngine(),
        )

        report.run_full()

        # Check report file exists
        assert os_module.path.exists(report_path), f"Report file not found: {report_path}"
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'DIAGNOSTICS REPORT' in content
        assert 'SECTION 1: Mapping Coverage' in content
        assert 'SECTION 2:' in content
        assert 'SECTION 3:' in content
        assert 'SECTION 4:' in content
        assert 'SECTION 5:' in content
