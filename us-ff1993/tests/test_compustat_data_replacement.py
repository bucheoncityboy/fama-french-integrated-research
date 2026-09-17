"""Tests for Compustat+CRSP hybrid data replacement utilities."""

import pandas as pd

import compustat_portfolio_builder as builder


class DummyPortfolioConstructor:
    PORTFOLIO_COLUMNS = builder.PortfolioConstructor.PORTFOLIO_COLUMNS
    PORTFOLIO_COLUMNS_6 = builder.PortfolioConstructor.PORTFOLIO_COLUMNS_6

    def build_25_portfolios(self):
        return pd.DataFrame(
            {column: [100.0, 200.0] for column in self.PORTFOLIO_COLUMNS},
            index=['1964-07', '1964-08'],
        ).rename_axis('Date')

    def build_6_portfolios(self):
        return pd.DataFrame(
            {column: [10.0, 20.0] for column in self.PORTFOLIO_COLUMNS_6},
            index=['1964-07', '1964-08'],
        ).rename_axis('Date')


class DummyFactorCalculator:
    def assemble_factors(self, port6=None):
        return pd.DataFrame({
            'Date': ['1964-07', '1964-08'],
            'Mkt-RF': [1.0, 2.0],
            'SMB': [3.0, 4.0],
            'HML': [5.0, 6.0],
            'RF': [7.0, 8.0],
        })


def _dates():
    return pd.period_range('1963-07', '1964-09', freq='M').strftime('%Y-%m-%d')


def _write_ken_french_inputs(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    dates = _dates()

    pd.DataFrame(
        {'Date': dates, **{column: 1.0 for column in DummyPortfolioConstructor.PORTFOLIO_COLUMNS}}
    ).to_csv(data_dir / 'ff_25_portfolios.csv', index=False)

    pd.DataFrame(
        {'Date': dates, **{column: 2.0 for column in DummyPortfolioConstructor.PORTFOLIO_COLUMNS_6}}
    ).to_csv(data_dir / 'ff_6_portfolios.csv', index=False)

    pd.DataFrame({
        'Date': dates,
        'Mkt-RF': [9.0] * len(dates),
        'SMB': [8.0] * len(dates),
        'HML': [7.0] * len(dates),
        'RF': [6.0] * len(dates),
    }).to_csv(data_dir / 'ff_factors.csv', index=False)


def test_replace_data_files(tmp_path):
    data_dir = tmp_path / 'data'
    _write_ken_french_inputs(data_dir)

    result = builder.replace_ken_french_data(
        data_dir=str(data_dir),
        portfolio_constructor=DummyPortfolioConstructor(),
        factor_calculator=DummyFactorCalculator(),
    )

    assert result['backup_path'] is not None
    assert result['rows']['ff_25_portfolios.csv'] == 15

    port25 = pd.read_csv(data_dir / 'ff_25_portfolios.csv')
    port6 = pd.read_csv(data_dir / 'ff_6_portfolios.csv')
    factors = pd.read_csv(data_dir / 'ff_factors.csv')

    assert port25['Date'].iloc[0] == '1963-07'
    assert port25['Date'].iloc[-1] == '1964-09'

    # First 12 months through 1964-06 remain Ken French values.
    assert port25.loc[port25['Date'] == '1964-06', 'SMALL LoBM'].iloc[0] == 1.0
    assert port6.loc[port6['Date'] == '1964-06', 'SMALL LoBM'].iloc[0] == 2.0
    assert factors.loc[factors['Date'] == '1964-06', 'Mkt-RF'].iloc[0] == 9.0

    # Covered self-constructed months are replaced.
    assert port25.loc[port25['Date'] == '1964-07', 'SMALL LoBM'].iloc[0] == 100.0
    assert port6.loc[port6['Date'] == '1964-08', 'SMALL LoBM'].iloc[0] == 20.0
    assert factors.loc[factors['Date'] == '1964-07', 'SMB'].iloc[0] == 3.0

    # Missing self-constructed replacement months fall back to Ken French data.
    assert port25.loc[port25['Date'] == '1964-09', 'SMALL LoBM'].iloc[0] == 1.0
    assert factors.loc[factors['Date'] == '1964-09', 'RF'].iloc[0] == 6.0


def test_restore_data_files(tmp_path):
    data_dir = tmp_path / 'data'
    _write_ken_french_inputs(data_dir)
    original = {
        filename: (data_dir / filename).read_text()
        for filename in ['ff_25_portfolios.csv', 'ff_6_portfolios.csv', 'ff_factors.csv']
    }

    backup_path = builder.BackupManager(data_dir=str(data_dir)).backup()
    builder.replace_ken_french_data(
        data_dir=str(data_dir),
        portfolio_constructor=DummyPortfolioConstructor(),
        factor_calculator=DummyFactorCalculator(),
        create_backup=False,
    )
    builder.restore_ken_french_data(backup_path, data_dir=str(data_dir))

    restored = {
        filename: (data_dir / filename).read_text()
        for filename in ['ff_25_portfolios.csv', 'ff_6_portfolios.csv', 'ff_factors.csv']
    }
    assert restored == original
