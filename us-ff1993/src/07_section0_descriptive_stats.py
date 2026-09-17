"""
Section 0: Descriptive Statistics for 25 Size×BE/ME Portfolios

This script computes firm counts, average market cap, and market cap share
per portfolio cell for the 1964-1991 formation period.

Output:
    output/table0_descriptive_stats.csv: 25 rows × 4 columns

Columns:
    - portfolio: Portfolio name (SMALL LoBM, ME1 BM2, ..., BIG HiBM)
    - avg_firm_count: Average number of stocks per cell across years
    - avg_market_cap_mm: Average market cap (in millions) per cell
    - mkt_cap_share_pct: Market cap share as percentage of total

Note: E/P and D/P statistics are NOT computed (no data available).
"""

import os
import sys
import pandas as pd
from datetime import datetime

import config

# Import from compustat_portfolio_builder
from compustat_portfolio_builder import BEMECalculator, PortfolioConstructor


def compute_descriptive_stats():
    """Compute descriptive statistics for 25 portfolios across 1964-1991."""
    print("=" * 70)
    print("SECTION 0: Descriptive Statistics for 25 Portfolios")
    print("=" * 70)
    print()

    # Initialize portfolio builder
    pc = PortfolioConstructor()
    beme_calculator = BEMECalculator()

    # Load data
    print("Loading BE/ME data...")
    beme = beme_calculator.compute_all()
    print(f"  BE/ME data loaded: {len(beme)} records")

    # Get formation years (1964-1991)
    formation_years = sorted(beme['year'].dropna().astype(int).unique())
    target_years = [y for y in formation_years if 1964 <= y <= 1991]
    if target_years:
        formation_years = target_years

    print(f"  Formation years: {formation_years}")
    print()

    # Storage for per-year statistics
    year_stats = []

    # Process each formation year
    for formation_year in formation_years:
        print(f"Processing formation year {formation_year}...")

        # Get formation snapshot (June)
        snapshot = pc._formation_snapshot(beme, formation_year)
        if snapshot.empty:
            print(f"  No valid snapshot for year {formation_year}")
            continue

        # Compute NYSE breakpoints for Size (ME) and BE/ME
        size_bps = pc.compute_nyse_breakpoints(snapshot, metric='me')
        beme_bps = pc.compute_nyse_breakpoints(snapshot, metric='beme')

        # Assign stocks to 25 portfolios
        assignments = pc.assign_portfolios(snapshot, size_bps, beme_bps)

        # Add portfolio and formation_me to snapshot
        snapshot = snapshot.assign(portfolio=assignments, formation_me=snapshot['me'])

        # Drop invalid entries
        snapshot = snapshot.dropna(subset=['portfolio', 'formation_me'])
        snapshot = snapshot[snapshot['formation_me'] > 0]

        if snapshot.empty:
            print(f"  No valid assignments for year {formation_year}")
            continue

        # Group by portfolio and compute statistics
        portfolio_stats = snapshot.groupby('portfolio').agg({
            'formation_me': ['count', 'mean']
        })

        # Flatten column names
        portfolio_stats.columns = ['firm_count', 'avg_market_cap']

        # Reset index to make portfolio a column
        portfolio_stats = portfolio_stats.reset_index()

        year_stats.append(portfolio_stats)
        print(f"  Processed {len(portfolio_stats)} portfolio cells")

    print()

    if not year_stats:
        print("ERROR: No portfolio statistics computed!")
        return None

    # Combine all years
    all_stats = pd.concat(year_stats, ignore_index=True)
    print(f"Total records across all years: {len(all_stats)}")

    # Compute statistics across years
    result = all_stats.groupby('portfolio').agg({
        'firm_count': 'mean',
        'avg_market_cap': 'mean'
    }).round(2)

    # Reset index to make portfolio a column
    result = result.reset_index()

    # Rename avg_market_cap to avg_market_cap_mm
    result = result.rename(columns={'avg_market_cap': 'avg_market_cap_mm'})

    # Reorder columns
    result = result[['portfolio', 'firm_count', 'avg_market_cap_mm']]
    result = result.rename(columns={'firm_count': 'avg_firm_count'})

    # Compute and add market cap share
    total_avg_mkt_cap = result['avg_market_cap_mm'].sum()
    result['mkt_cap_share_pct'] = (result['avg_market_cap_mm'] / total_avg_mkt_cap * 100).round(2)

    return result


def print_summary_table(df):
    """Print the summary table with verification."""
    print("\n" + "-" * 70)
    print("DESCRIPTIVE STATISTICS SUMMARY")
    print("-" * 70)

    print(df.to_string(index=False))

    # Verify row count
    print(f"\nRow count: {len(df)} (expected: 25)")

    # Verify column count
    print(f"Column count: {len(df.columns)} (expected: 4)")

    # Verify sum of market cap shares
    mkt_cap_share_sum = df['mkt_cap_share_pct'].sum()
    print(f"Sum of market cap shares: {mkt_cap_share_sum:.2f}% (expected: ~100%)")

    # Verify SIZE pattern (SMALL < BIG for same BE/ME - SMALL firms are smaller)
    print("\n" + "-" * 70)
    print("SIZE PATTERN VERIFICATION (SMALL < BIG?)")
    print("-" * 70)

    size_pattern_ok = True
    for beme_group in ['LoBM', 'BM2', 'BM3', 'BM4', 'HiBM']:
        small_mkt_cap = df[df['portfolio'].str.contains('SMALL') & df['portfolio'].str.contains(beme_group)]['avg_market_cap_mm'].values
        big_mkt_cap = df[df['portfolio'].str.contains('BIG') & df['portfolio'].str.contains(beme_group)]['avg_market_cap_mm'].values

        if len(small_mkt_cap) > 0 and len(big_mkt_cap) > 0:
            small_avg = small_mkt_cap[0]
            big_avg = big_mkt_cap[0]
            pattern_ok = small_avg < big_avg

            status = "[PASS]" if pattern_ok else "[FAIL]"
            print(f"{beme_group:6s} | SMALL: {small_avg:10.0f} | BIG: {big_avg:10.0f} | {status}")

            if not pattern_ok:
                size_pattern_ok = False

    print(f"\nSize pattern verification: {'PASS' if size_pattern_ok else 'FAIL'}")

    # Verify BE/ME pattern (HiBM < LoBM for same Size - value stocks are smaller)
    print("\n" + "-" * 70)
    print("BE/ME PATTERN VERIFICATION (HiBM < LoBM?)")
    print("-" * 70)

    be_pattern_ok = True
    for size_group in ['SMALL', 'ME1', 'ME2', 'ME3', 'ME4', 'ME5']:
        low_bm = df[df['portfolio'].str.contains(size_group) & df['portfolio'].str.contains('LoBM')]['avg_market_cap_mm'].values
        high_bm = df[df['portfolio'].str.contains(size_group) & df['portfolio'].str.contains('HiBM')]['avg_market_cap_mm'].values

        if len(low_bm) > 0 and len(high_bm) > 0:
            low_avg = low_bm[0]
            high_avg = high_bm[0]
            pattern_ok = high_avg < low_avg

            status = "[PASS]" if pattern_ok else "[FAIL]"
            print(f"{size_group:6s} | LoBM: {low_avg:10.0f} | HiBM: {high_avg:10.0f} | {status}")

            if not pattern_ok:
                be_pattern_ok = False

    print(f"\nBE/ME pattern verification: {'PASS' if be_pattern_ok else 'FAIL'}")

    # Overall result
    overall_ok = size_pattern_ok and be_pattern_ok
    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 70)

    return overall_ok


def save_output(df, overall_ok):
    """Save output files."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Save CSV
    output_path = os.path.join(config.OUTPUT_DIR, 'table0_descriptive_stats.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved CSV to: {output_path}")

    # Save QA evidence
    os.makedirs('.sisyphus/evidence', exist_ok=True)

    evidence_path = '.sisyphus/evidence/task-5-descriptive-stats.txt'
    with open(evidence_path, 'w', encoding='utf-8') as f:
        f.write("Task 5: Descriptive Statistics for 25 Size×BE/ME Portfolios\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Output file:\n")
        f.write(f"  {output_path}\n\n")

        f.write("Summary statistics:\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        f.write("Verification results:\n")
        f.write(f"  Row count: {len(df)} (expected: 25)\n")
        f.write(f"  Column count: {len(df.columns)} (expected: 4)\n")
        f.write(f"  Sum of market cap shares: {df['mkt_cap_share_pct'].sum():.2f}% (expected: ~100%)\n")
        f.write(f"  Overall pattern verification: {'PASS ✓' if overall_ok else 'FAIL ✗'}\n\n")

        f.write("Note: E/P and D/P statistics are NOT computed (no data available).\n")
        f.write("      Only Size (ME) and BE/ME statistics are provided.\n")

    print(f"Saved QA evidence to: {evidence_path}")

    # Save size pattern verification separately
    pattern_path = '.sisyphus/evidence/task-5-size-pattern.txt'
    with open(pattern_path, 'w', encoding='utf-8') as f:
        f.write("Task 5: Size Pattern Verification\n")
        f.write("=" * 70 + "\n\n")

        f.write("Verification that SMALL < BIG for same BE/ME groups:\n\n")

        for beme_group in ['LoBM', 'BM2', 'BM3', 'BM4', 'HiBM']:
            small_mkt_cap = df[df['portfolio'].str.contains('SMALL') & df['portfolio'].str.contains(beme_group)]['avg_market_cap_mm'].values
            big_mkt_cap = df[df['portfolio'].str.contains('BIG') & df['portfolio'].str.contains(beme_group)]['avg_market_cap_mm'].values

            if len(small_mkt_cap) > 0 and len(big_mkt_cap) > 0:
                small_avg = small_mkt_cap[0]
                big_avg = big_mkt_cap[0]
                pattern_ok = small_avg < big_avg

                status = "✓ PASS" if pattern_ok else "✗ FAIL"
                f.write(f"{beme_group:6s} | SMALL: {small_avg:10.0f} | BIG: {big_avg:10.0f} | {status}\n")

    print(f"Saved size pattern verification to: {pattern_path}")

    return overall_ok


def main():
    """Main execution function."""
    print()

    # Compute descriptive statistics
    df = compute_descriptive_stats()

    if df is None:
        print("\nERROR: Failed to compute descriptive statistics!")
        sys.exit(1)

    # Print summary table with verification
    overall_ok = print_summary_table(df)

    # Save output files
    save_output(df, overall_ok)

    print("\n" + "=" * 70)
    print("SECTION 0 COMPLETED")
    print("=" * 70)
    print()

    if not overall_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
