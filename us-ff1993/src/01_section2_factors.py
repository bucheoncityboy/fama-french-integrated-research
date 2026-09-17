"""
01_section2_factors.py
Section 2.1: Construct SMB and HML from 6 size/BE-ME portfolios.

Replicates Fama-French (1993) factor construction using Ken French
data-library files already downloaded to data/.
"""

import os

import pandas as pd

import config


def load_data():
    """Load pre-computed factors and 6 portfolio returns."""
    ff_path = os.path.join(config.DATA_DIR, 'ff_factors.csv')
    port_path = os.path.join(config.DATA_DIR, 'ff_6_portfolios.csv')

    ff = pd.read_csv(ff_path)
    ports = pd.read_csv(port_path)

    ff['Date'] = pd.to_datetime(ff['Date'])
    ports['Date'] = pd.to_datetime(ports['Date'])

    return ff, ports


def filter_date_range(df):
    """Filter DataFrame to the analysis period defined in config."""
    start = pd.to_datetime(config.START_DATE + '-01')
    end = pd.to_datetime(config.END_DATE + '-01')
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    return df.loc[mask].copy()


def construct_factors(ports):
    """
    Construct SMB and HML from 6 size/BE-ME portfolios.

    Formulas (Fama-French 1993):
        SMB = 1/3*(SmallValue + SmallNeutral + SmallGrowth)
              - 1/3*(BigValue + BigNeutral + BigGrowth)
        HML = 1/2*(SmallValue + BigValue)
              - 1/2*(SmallGrowth + BigGrowth)
    """
    smb = (
        (1 / 3) * (
            ports['SMALL HiBM']
            + ports['ME1 BM2']
            + ports['SMALL LoBM']
        )
        - (1 / 3) * (
            ports['BIG HiBM']
            + ports['ME2 BM2']
            + ports['BIG LoBM']
        )
    )

    hml = (
        (1 / 2) * (ports['SMALL HiBM'] + ports['BIG HiBM'])
        - (1 / 2) * (ports['SMALL LoBM'] + ports['BIG LoBM'])
    )

    return smb, hml


def validate_factors(constructed, precomputed):
    """
    Validate constructed SMB/HML against Ken French pre-computed values.
    Mean absolute deviation must be < 0.05 %/month.
    """
    merged = pd.merge(
        constructed,
        precomputed,
        on='Date',
        suffixes=('', '_kf'),
    )

    smb_mad = (merged['SMB'] - merged['SMB_kf']).abs().mean()
    hml_mad = (merged['HML'] - merged['HML_kf']).abs().mean()

    print("Validation vs Ken French pre-computed factors:")
    print(f"  SMB mean absolute deviation = {smb_mad:.4f}%/month")
    print(f"  HML mean absolute deviation = {hml_mad:.4f}%/month")

    if smb_mad >= 0.05 or hml_mad >= 0.05:
        raise ValueError(
            f"Factor validation failed: "
            f"SMB MAD={smb_mad:.4f}, HML MAD={hml_mad:.4f} "
            f"(threshold < 0.05 %/month)"
        )

    print("  -> PASSED (both < 0.05 %/month)\n")


def print_summary(df, factor_cols):
    """Print mean, std, and t-statistics for each factor."""
    n = len(df)
    summary = []
    for col in factor_cols:
        mean = df[col].mean()
        std = df[col].std(ddof=1)
        t_stat = mean / (std / (n ** 0.5))
        summary.append({
            'Factor': col,
            'Mean': mean,
            'Std': std,
            't-stat': t_stat,
        })

    summary_df = pd.DataFrame(summary)
    print("Factor Summary Statistics (%/month):")
    print(summary_df.to_string(index=False))
    print()
    return summary_df


def main():
    print("Loading data...")
    ff, ports = load_data()

    print("Filtering to analysis period...")
    ff = filter_date_range(ff)
    ports = filter_date_range(ports)

    print("Constructing SMB and HML from 6 portfolios...")
    smb, hml = construct_factors(ports)

    # Assemble final factor DataFrame
    factors = pd.DataFrame({
        'Date': ports['Date'].values,
        'Mkt-RF': ff['Mkt-RF'].values,
        'SMB': smb.values,
        'HML': hml.values,
        'RF': ff['RF'].values,
    })

    # Validate
    validate_factors(factors, ff[['Date', 'SMB', 'HML']])

    # Summary stats
    print_summary(factors, ['Mkt-RF', 'SMB', 'HML'])

    # Save
    out_path = os.path.join(config.OUTPUT_DIR, 'factors.csv')
    factors.to_csv(out_path, index=False)
    print(f"Saved factors to {out_path}")


if __name__ == '__main__':
    main()
