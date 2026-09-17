# CRSP Hybrid Data Substitution Design

Date: 2026-05-15

## Goal

Replace the project's stock-side input data with a hybrid data set while keeping the project's target period, 1963-07 through 1991-12.

The project code should remain unchanged. The implementation should modify data files and regenerate outputs only.

## Current Project Contract

The project reads input data from `data/` through the existing filenames:

- `data/ff_factors.csv`
- `data/ff_6_portfolios.csv`
- `data/ff_25_portfolios.csv`
- `data/bond_factors.csv`

The project writes analysis outputs to `output/`. The existing scripts assume the `ff_*` filenames, so the replacement should preserve those names.

## Available CRSP-Derived Data

The CRSP-derived files currently available under `crsp/FF1993_results/data/` are:

- `crsp_ff_factors.csv`
- `crsp_6_portfolios.csv`
- `crsp_25_portfolios.csv`

These files cover 1968-07 through 1991-12. The existing project `data/ff_*` files cover 1963-07 through 1991-12.

## Chosen Approach

Use a hybrid input-data replacement:

- 1963-07 through 1968-06: keep existing project data from `data/ff_*`.
- 1968-07 through 1991-12: replace with CRSP-derived data from `crsp/FF1993_results/data/crsp_*`.

This keeps the full target period while using the CRSP-derived data where available.

## Files To Replace

Create new 342-month hybrid versions of:

- `data/ff_factors.csv`
- `data/ff_6_portfolios.csv`
- `data/ff_25_portfolios.csv`

Do not change `data/bond_factors.csv`; it is unrelated to the CRSP stock-side substitution.

Before replacement, preserve backups of the original files in a clearly named backup directory, for example:

- `data/backups/2026-05-15-pre-crsp-hybrid/ff_factors.csv`
- `data/backups/2026-05-15-pre-crsp-hybrid/ff_6_portfolios.csv`
- `data/backups/2026-05-15-pre-crsp-hybrid/ff_25_portfolios.csv`

If existing `output/` files will be overwritten, preserve them in a parallel backup directory:

- `output/backups/2026-05-15-pre-crsp-hybrid/`

## Data Rules

Normalize all dates to monthly timestamps before merging. The existing project accepts date strings that parse with pandas, but the final CSVs should use a consistent `Date` column.

The replacement cutoff is inclusive:

- Dates before 1968-07 use the original project data.
- Dates from 1968-07 onward use the CRSP-derived data.

The final hybrid files must preserve the same column names expected by the scripts:

- `ff_factors.csv`: `Date`, `Mkt-RF`, `SMB`, `HML`, `RF`
- `ff_6_portfolios.csv`: `Date` plus the six size-BE/ME portfolio columns
- `ff_25_portfolios.csv`: `Date` plus the twenty-five size-BE/ME portfolio columns

## Regeneration Sequence

After replacing the input data, regenerate outputs in dependency order:

1. `python 01_section2_factors.py`
2. `python 02_section2_portfolios.py`
3. `python 01b_section2_bond_factors.py`
4. `python 02b_section2_bond_portfolios.py`
5. `python 03_section3_statistics.py`
6. `python 04_section4_regressions.py`
7. `python 04b_section4_five_factor.py`
8. `python 05_section5_grs_test.py`
9. `python 05b_section5_intercepts.py`
10. `python 06_section6_visualizations.py`
11. `python 06_section6_conclusions.py`

If one script overwrites a file required by a later script, use the existing pipeline order and verify the final state rather than editing code.

## Verification Criteria

Input verification:

- Each hybrid `data/ff_*` file has 342 rows.
- Each hybrid `data/ff_*` file spans 1963-07 through 1991-12.
- Each hybrid `data/ff_*` file has zero blank cells.
- For 1963-07 through 1968-06, values match the original backed-up project data.
- For 1968-07 through 1991-12, values match the corresponding `crsp/FF1993_results/data/crsp_*` file.

Output verification:

- `output/factors.csv` spans 1963-07 through 1991-12.
- `output/factors.csv` has no missing `RF`, `Mkt-RF`, `SMB`, `HML`, `TERM`, or `DEF` values.
- `output/stock_portfolios_excess.csv` spans 1963-07 through 1991-12 and has 342 rows.
- Regression and GRS output files are regenerated after the data substitution.
- The final report states the hybrid provenance: Ken French for 1963-07 through 1968-06, CRSP-derived stock data for 1968-07 through 1991-12.

## Checklist Update

Update `crsp/FF1993_results/데이터 변경 체크리스트.md` only if the user approves modifying that checklist. The update should not claim full-period CRSP coverage. It should say that the current accepted approach is a hybrid full-period data set with CRSP-derived stock data from 1968-07 onward.

## Out Of Scope

- Changing Python source code.
- Rebuilding CRSP raw extraction from WRDS.
- Extending CRSP-derived stock data backward to 1963-07.
- Changing bond factor methodology.
