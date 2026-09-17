# CRSP Hybrid Data Substitution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stock-side inputs with a 1963-07 through 1991-12 hybrid data set that keeps existing project data before 1968-07 and uses CRSP-derived stock data from 1968-07 onward.

**Architecture:** This is a data-only substitution. Preserve the existing production Python source, back up current input and output files, normalize all stock-side dates to monthly periods, overwrite the three `data/ff_*` input files with hybrid CSVs, regenerate outputs through the existing script sequence, patch the generated report output with explicit hybrid provenance, update the affected hybrid-specific test oracle, then commit and push the completed result to the GitHub `ccc` branch.

**Tech Stack:** Python 3, pandas, PowerShell, pytest, existing project scripts, git.

---

**Source Design Spec:** `docs/superpowers/specs/2026-05-15-crsp-hybrid-data-substitution-design.md`

**Execution Guard:** This document is the complete execution plan. Do not run the substitution, tests, commits, or push steps until the user explicitly approves execution.

---

## File Structure

**Create:**
- `data/backups/2026-05-15-pre-crsp-hybrid/ff_factors.csv`: original project factor input backup.
- `data/backups/2026-05-15-pre-crsp-hybrid/ff_6_portfolios.csv`: original project six-portfolio input backup.
- `data/backups/2026-05-15-pre-crsp-hybrid/ff_25_portfolios.csv`: original project twenty-five-portfolio input backup.
- `output/backups/2026-05-15-pre-crsp-hybrid/`: snapshot of output files before regeneration.
- `crsp/FF1993_results/data/crsp_ff_factors.csv`: CRSP-derived factor source used for the post-1968-07 hybrid slice if the target branch does not already contain it.
- `crsp/FF1993_results/data/crsp_6_portfolios.csv`: CRSP-derived six-portfolio source used for the post-1968-07 hybrid slice if the target branch does not already contain it.
- `crsp/FF1993_results/data/crsp_25_portfolios.csv`: CRSP-derived twenty-five-portfolio source used for the post-1968-07 hybrid slice if the target branch does not already contain it.

**Modify:**
- `data/ff_factors.csv`: hybrid `Date`, `Mkt-RF`, `SMB`, `HML`, `RF`; 342 rows; 1963-07-01 through 1991-12-01.
- `data/ff_6_portfolios.csv`: hybrid `Date` plus six size-BE/ME columns; 342 rows; 1963-07-01 through 1991-12-01.
- `data/ff_25_portfolios.csv`: hybrid `Date` plus twenty-five size-BE/ME columns; 342 rows; 1963-07-01 through 1991-12-01.
- `output/*`: regenerated analysis outputs.
- `output/FF1993_replication_report.md`: regenerated report, then output-only provenance text patch.
- `tests/test_section5_intercepts.py`: update two empirical test-oracle assertions that are specific to the old Ken French-only stock-side data and no longer hold under the verified hybrid CRSP-derived stock-side data.

**Read-only inputs:**
- `crsp/FF1993_results/data/crsp_ff_factors.csv`
- `crsp/FF1993_results/data/crsp_6_portfolios.csv`
- `crsp/FF1993_results/data/crsp_25_portfolios.csv`
- `data/bond_factors.csv`

**Do not modify:**
- Production analysis scripts: `01_section2_factors.py`, `01b_section2_bond_factors.py`, `02_section2_portfolios.py`, `02b_section2_bond_portfolios.py`, `03_section3_statistics.py`, `04_section4_regressions.py`, `04b_section4_five_factor.py`, `05_section5_grs_test.py`, `05b_section5_intercepts.py`, `06_section6_visualizations.py`, `06_section6_conclusions.py`
- Production support modules: `config.py`, `download_data.py`, `fred_bond_fetcher.py`, `ken_french_parser.py`, `regression_engine.py`
- `requirements.txt`
- `data/bond_factors.csv`
- `crsp/FF1993_results/데이터 변경 체크리스트.md` unless the user explicitly approves a checklist update.

**Target GitHub branch for completion:**
- Repository page requested by user: `https://github.com/bucheoncityboy/fama-ff3-/tree/ccc`
- Git remote URL to use if this workspace is not already connected: `https://github.com/bucheoncityboy/fama-ff3-.git`
- Target branch: `ccc`
- Final delivery requirement: after all verification passes, commit the completed data/output/report changes and push them to branch `ccc`.

---

### Task 1: Preflight Snapshot

**Files:**
- Read: `data/ff_factors.csv`
- Read: `data/ff_6_portfolios.csv`
- Read: `data/ff_25_portfolios.csv`
- Read: `crsp/FF1993_results/data/crsp_ff_factors.csv`
- Read: `crsp/FF1993_results/data/crsp_6_portfolios.csv`
- Read: `crsp/FF1993_results/data/crsp_25_portfolios.csv`

- [ ] **Step 1: Confirm working tree state**

Run:

```powershell
git status --short
```

Expected: review the output before proceeding. Unrelated user changes may exist; do not revert them.

- [ ] **Step 2: Confirm source files are not already modified by this task**

Run:

```powershell
git diff --name-only -- 01_section2_factors.py 01b_section2_bond_factors.py 02_section2_portfolios.py 02b_section2_bond_portfolios.py 03_section3_statistics.py 04_section4_regressions.py 04b_section4_five_factor.py 05_section5_grs_test.py 05b_section5_intercepts.py 06_section6_visualizations.py 06_section6_conclusions.py config.py download_data.py fred_bond_fetcher.py ken_french_parser.py regression_engine.py requirements.txt
```

Expected: no output. If output appears, stop and identify whether those source edits pre-existed.

- [ ] **Step 3: Verify source CSV contracts**

Run:

```powershell
@'
from pathlib import Path
import pandas as pd

EXPECTED = {
    "data/ff_factors.csv": ["Date", "Mkt-RF", "SMB", "HML", "RF"],
    "data/ff_6_portfolios.csv": [
        "Date", "SMALL LoBM", "ME1 BM2", "SMALL HiBM",
        "BIG LoBM", "ME2 BM2", "BIG HiBM",
    ],
    "data/ff_25_portfolios.csv": [
        "Date", "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM",
        "ME2 BM1", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 BM5",
        "ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5",
        "ME4 BM1", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5",
        "BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM",
    ],
    "crsp/FF1993_results/data/crsp_ff_factors.csv": ["Date", "Mkt-RF", "SMB", "HML", "RF"],
    "crsp/FF1993_results/data/crsp_6_portfolios.csv": [
        "Date", "SMALL LoBM", "ME1 BM2", "SMALL HiBM",
        "BIG LoBM", "ME2 BM2", "BIG HiBM",
    ],
    "crsp/FF1993_results/data/crsp_25_portfolios.csv": [
        "Date", "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM",
        "ME2 BM1", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 BM5",
        "ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5",
        "ME4 BM1", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5",
        "BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM",
    ],
}

for path, expected_cols in EXPECTED.items():
    df = pd.read_csv(path)
    periods = pd.to_datetime(df["Date"]).dt.to_period("M")
    if list(df.columns) != expected_cols:
        raise AssertionError(f"{path}: unexpected columns {list(df.columns)}")
    print(
        f"{path}: rows={len(df)}, cols={len(df.columns)}, "
        f"start={periods.min()}, end={periods.max()}, blanks={int(df.isna().sum().sum())}"
    )
'@ | python -
```

Expected:

```text
data/ff_factors.csv: rows=342, cols=5, start=1963-07, end=1991-12, blanks=0
data/ff_6_portfolios.csv: rows=342, cols=7, start=1963-07, end=1991-12, blanks=0
data/ff_25_portfolios.csv: rows=342, cols=26, start=1963-07, end=1991-12, blanks=0
crsp/FF1993_results/data/crsp_ff_factors.csv: rows=282, cols=5, start=1968-07, end=1991-12, blanks=0
crsp/FF1993_results/data/crsp_6_portfolios.csv: rows=282, cols=7, start=1968-07, end=1991-12, blanks=0
crsp/FF1993_results/data/crsp_25_portfolios.csv: rows=282, cols=26, start=1968-07, end=1991-12, blanks=0
```

---

### Task 2: Preserve Original Inputs and Outputs

**Files:**
- Create: `data/backups/2026-05-15-pre-crsp-hybrid/ff_factors.csv`
- Create: `data/backups/2026-05-15-pre-crsp-hybrid/ff_6_portfolios.csv`
- Create: `data/backups/2026-05-15-pre-crsp-hybrid/ff_25_portfolios.csv`
- Create: `output/backups/2026-05-15-pre-crsp-hybrid/*`

- [ ] **Step 1: Create backup directories and copy files**

Run:

```powershell
$stamp = "2026-05-15-pre-crsp-hybrid"
New-Item -ItemType Directory -Force -Path "data/backups/$stamp" | Out-Null
New-Item -ItemType Directory -Force -Path "output/backups/$stamp" | Out-Null

Copy-Item -LiteralPath "data/ff_factors.csv" -Destination "data/backups/$stamp/ff_factors.csv" -Force
Copy-Item -LiteralPath "data/ff_6_portfolios.csv" -Destination "data/backups/$stamp/ff_6_portfolios.csv" -Force
Copy-Item -LiteralPath "data/ff_25_portfolios.csv" -Destination "data/backups/$stamp/ff_25_portfolios.csv" -Force

Get-ChildItem -LiteralPath "output" -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination "output/backups/$stamp/$($_.Name)" -Force
}
```

Expected: three input backups exist; every pre-existing top-level output file has a copy in `output/backups/2026-05-15-pre-crsp-hybrid/`.

- [ ] **Step 2: Verify backed-up inputs are byte-identical to current inputs**

Run:

```powershell
$stamp = "2026-05-15-pre-crsp-hybrid"
$pairs = @(
    @("data/ff_factors.csv", "data/backups/$stamp/ff_factors.csv"),
    @("data/ff_6_portfolios.csv", "data/backups/$stamp/ff_6_portfolios.csv"),
    @("data/ff_25_portfolios.csv", "data/backups/$stamp/ff_25_portfolios.csv")
)
foreach ($pair in $pairs) {
    $a = (Get-FileHash -Algorithm SHA256 -LiteralPath $pair[0]).Hash
    $b = (Get-FileHash -Algorithm SHA256 -LiteralPath $pair[1]).Hash
    if ($a -ne $b) { throw "Hash mismatch: $($pair[0])" }
    Write-Output "OK $($pair[0])"
}
```

Expected:

```text
OK data/ff_factors.csv
OK data/ff_6_portfolios.csv
OK data/ff_25_portfolios.csv
```

- [ ] **Step 3: Commit the backup checkpoint**

Run:

```powershell
git add data/backups/2026-05-15-pre-crsp-hybrid output/backups/2026-05-15-pre-crsp-hybrid
git commit -m "chore: back up pre-hybrid data and outputs"
```

Expected: commit succeeds. If the project owner does not want backup files committed, skip only this commit step and keep the backup files in place.

---

### Task 3: Build Hybrid Stock-Side Inputs

**Files:**
- Modify: `data/ff_factors.csv`
- Modify: `data/ff_6_portfolios.csv`
- Modify: `data/ff_25_portfolios.csv`

- [ ] **Step 1: Overwrite the three stock-side input files with hybrid data**

Run:

```powershell
@'
from pathlib import Path
import pandas as pd

CUTOFF = pd.Period("1968-07", freq="M")
START = pd.Period("1963-07", freq="M")
END = pd.Period("1991-12", freq="M")
EXPECTED_ROWS = 342

JOBS = [
    {
        "original": Path("data/ff_factors.csv"),
        "crsp": Path("crsp/FF1993_results/data/crsp_ff_factors.csv"),
        "output": Path("data/ff_factors.csv"),
        "columns": ["Date", "Mkt-RF", "SMB", "HML", "RF"],
    },
    {
        "original": Path("data/ff_6_portfolios.csv"),
        "crsp": Path("crsp/FF1993_results/data/crsp_6_portfolios.csv"),
        "output": Path("data/ff_6_portfolios.csv"),
        "columns": [
            "Date", "SMALL LoBM", "ME1 BM2", "SMALL HiBM",
            "BIG LoBM", "ME2 BM2", "BIG HiBM",
        ],
    },
    {
        "original": Path("data/ff_25_portfolios.csv"),
        "crsp": Path("crsp/FF1993_results/data/crsp_25_portfolios.csv"),
        "output": Path("data/ff_25_portfolios.csv"),
        "columns": [
            "Date", "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM",
            "ME2 BM1", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 BM5",
            "ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5",
            "ME4 BM1", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5",
            "BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM",
        ],
    },
]

def load_monthly(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != columns:
        raise AssertionError(f"{path}: expected {columns}, got {list(df.columns)}")
    df = df.copy()
    df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    if df["Month"].duplicated().any():
        duplicated = df.loc[df["Month"].duplicated(), "Month"].astype(str).tolist()
        raise AssertionError(f"{path}: duplicate months {duplicated}")
    return df

for job in JOBS:
    original = load_monthly(job["original"], job["columns"])
    crsp = load_monthly(job["crsp"], job["columns"])

    pre = original.loc[original["Month"] < CUTOFF, job["columns"] + ["Month"]]
    post = crsp.loc[crsp["Month"] >= CUTOFF, job["columns"] + ["Month"]]
    hybrid = pd.concat([pre, post], ignore_index=True).sort_values("Month")

    months = hybrid["Month"]
    expected_months = pd.period_range(START, END, freq="M")
    if list(months) != list(expected_months):
        raise AssertionError(
            f"{job['output']}: expected continuous {START}..{END}, "
            f"got {months.min()}..{months.max()} with {len(months)} rows"
        )
    if len(hybrid) != EXPECTED_ROWS:
        raise AssertionError(f"{job['output']}: expected {EXPECTED_ROWS} rows, got {len(hybrid)}")
    if hybrid[job["columns"]].isna().sum().sum() != 0:
        raise AssertionError(f"{job['output']}: blank cells detected")

    hybrid = hybrid[job["columns"]].copy()
    hybrid["Date"] = months.dt.to_timestamp().dt.strftime("%Y-%m-%d")
    hybrid.to_csv(job["output"], index=False, float_format="%.17g")
    print(f"Wrote {job['output']}: rows={len(hybrid)}, start={hybrid['Date'].iloc[0]}, end={hybrid['Date'].iloc[-1]}")
'@ | python -
```

Expected:

```text
Wrote data\ff_factors.csv: rows=342, start=1963-07-01, end=1991-12-01
Wrote data\ff_6_portfolios.csv: rows=342, start=1963-07-01, end=1991-12-01
Wrote data\ff_25_portfolios.csv: rows=342, start=1963-07-01, end=1991-12-01
```

- [ ] **Step 2: Review the data diff**

Run:

```powershell
git diff -- data/ff_factors.csv data/ff_6_portfolios.csv data/ff_25_portfolios.csv
```

Expected: no rows before `1968-07-01` are changed; rows from `1968-07-01` onward change to CRSP-derived values where CRSP differs from original project data.

---

### Task 4: Verify Hybrid Inputs Against Both Sources

**Files:**
- Read: `data/backups/2026-05-15-pre-crsp-hybrid/ff_factors.csv`
- Read: `data/backups/2026-05-15-pre-crsp-hybrid/ff_6_portfolios.csv`
- Read: `data/backups/2026-05-15-pre-crsp-hybrid/ff_25_portfolios.csv`
- Read: `crsp/FF1993_results/data/crsp_ff_factors.csv`
- Read: `crsp/FF1993_results/data/crsp_6_portfolios.csv`
- Read: `crsp/FF1993_results/data/crsp_25_portfolios.csv`
- Read: `data/ff_factors.csv`
- Read: `data/ff_6_portfolios.csv`
- Read: `data/ff_25_portfolios.csv`

- [ ] **Step 1: Run source-matching verification**

Run:

```powershell
@'
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal

STAMP = "2026-05-15-pre-crsp-hybrid"
CUTOFF = pd.Period("1968-07", freq="M")
START = pd.Period("1963-07", freq="M")
END = pd.Period("1991-12", freq="M")

JOBS = [
    ("ff_factors.csv", "crsp_ff_factors.csv"),
    ("ff_6_portfolios.csv", "crsp_6_portfolios.csv"),
    ("ff_25_portfolios.csv", "crsp_25_portfolios.csv"),
]

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    df = df.drop(columns=["Date"]).set_index("Month").sort_index()
    return df

for current_name, crsp_name in JOBS:
    backup = load(Path("data/backups") / STAMP / current_name)
    crsp = load(Path("crsp/FF1993_results/data") / crsp_name)
    current = load(Path("data") / current_name)

    expected_index = pd.period_range(START, END, freq="M")
    if list(current.index) != list(expected_index):
        raise AssertionError(f"{current_name}: not continuous {START}..{END}")
    if current.isna().sum().sum() != 0:
        raise AssertionError(f"{current_name}: blank cells detected")

    assert_frame_equal(
        current.loc[current.index < CUTOFF],
        backup.loc[backup.index < CUTOFF],
        check_dtype=False,
        atol=1e-10,
        rtol=1e-10,
    )
    assert_frame_equal(
        current.loc[current.index >= CUTOFF],
        crsp.loc[crsp.index >= CUTOFF],
        check_dtype=False,
        atol=1e-10,
        rtol=1e-10,
    )
    print(f"OK {current_name}: pre-cutoff matches backup, post-cutoff matches CRSP")
'@ | python -
```

Expected:

```text
OK ff_factors.csv: pre-cutoff matches backup, post-cutoff matches CRSP
OK ff_6_portfolios.csv: pre-cutoff matches backup, post-cutoff matches CRSP
OK ff_25_portfolios.csv: pre-cutoff matches backup, post-cutoff matches CRSP
```

- [ ] **Step 2: Run existing data tests**

Run:

```powershell
python -m pytest tests/test_download_data.py tests/test_section2_factors.py tests/test_section2_stock_portfolios.py -q
```

Expected: tests pass. If `tests/test_section2_stock_portfolios.py` expects regenerated `output/stock_portfolios_excess.csv`, run Task 5 Step 1 and Task 5 Step 2 first, then rerun this command.

- [ ] **Step 3: Commit the hybrid input checkpoint**

Run:

```powershell
git add data/ff_factors.csv data/ff_6_portfolios.csv data/ff_25_portfolios.csv
git commit -m "data: substitute hybrid CRSP stock inputs"
```

Expected: commit succeeds with only the three stock-side input files staged.

---

### Task 5: Regenerate Outputs in Dependency Order

**Files:**
- Modify: `output/factors.csv`
- Modify: `output/stock_portfolios_excess.csv`
- Modify: `output/bond_portfolios_excess.csv`
- Modify: `output/table1_market.csv`
- Modify: `output/table2_summary.csv`
- Modify: `output/table3_bond.csv`
- Modify: `output/table4_stock3f.csv`
- Modify: `output/table5_five_factor.csv`
- Modify: `output/grs_test_results.csv`
- Modify: `output/intercept_analysis.csv`
- Modify: `output/fig1_average_returns_heatmap.png`
- Modify: `output/fig2_factor_cumulative_returns.png`
- Modify: `output/fig3_r2_comparison.png`
- Modify: `output/fig4_alpha_distribution.png`
- Modify: `output/fig5_factor_loadings_heatmap.png`
- Modify: `output/fig6_smb_hml_scatter.png`
- Modify: `output/FF1993_replication_report.md`

- [ ] **Step 1: Regenerate stock factors**

Run:

```powershell
python 01_section2_factors.py
```

Expected: prints `PASSED (both < 0.05 %/month)` and writes `output/factors.csv` with stock columns.

- [ ] **Step 2: Regenerate stock portfolio excess returns**

Run:

```powershell
python 02_section2_portfolios.py
```

Expected: writes `output/stock_portfolios_excess.csv` with shape `(342, 25)`.

- [ ] **Step 3: Merge bond factors into final factor output**

Run:

```powershell
python 01b_section2_bond_factors.py
```

Expected: writes `output/factors.csv` with comment disclaimer and columns `TERM`, `DEF`, `RF`, `Mkt-RF`, `SMB`, `HML`.

- [ ] **Step 4: Regenerate bond portfolio excess returns**

Run:

```powershell
python 02b_section2_bond_portfolios.py
```

Expected: writes `output/bond_portfolios_excess.csv`.

- [ ] **Step 5: Regenerate summary statistics**

Run:

```powershell
python 03_section3_statistics.py
```

Expected: writes `output/table2_summary.csv`.

- [ ] **Step 6: Regenerate Section 4 regressions**

Run:

```powershell
python 04_section4_regressions.py
```

Expected: writes `output/table1_market.csv`, `output/table3_bond.csv`, and `output/table4_stock3f.csv`.

- [ ] **Step 7: Regenerate five-factor regressions**

Run:

```powershell
python 04b_section4_five_factor.py
```

Expected: writes `output/table5_five_factor.csv`.

- [ ] **Step 8: Regenerate GRS tests**

Run:

```powershell
python 05_section5_grs_test.py
```

Expected: writes `output/grs_test_results.csv`.

- [ ] **Step 9: Regenerate intercept analysis**

Run:

```powershell
python 05b_section5_intercepts.py
```

Expected: writes `output/intercept_analysis.csv`.

- [ ] **Step 10: Regenerate figures**

Run:

```powershell
python 06_section6_visualizations.py
```

Expected: writes six non-empty PNG files in `output/`.

- [ ] **Step 11: Regenerate report**

Run:

```powershell
python 06_section6_conclusions.py
```

Expected: writes `output/FF1993_replication_report.md`.

---

### Task 6: Patch Generated Report Provenance

**Files:**
- Modify: `output/FF1993_replication_report.md`

- [ ] **Step 1: Insert hybrid provenance into the generated report output**

Run:

```powershell
@'
from pathlib import Path

path = Path("output/FF1993_replication_report.md")
text = path.read_text(encoding="utf-8")

old_sources = "**Data Sources**: Ken French Data Library, FRED (Federal Reserve Economic Data)"
new_sources = (
    "**Data Sources**: Hybrid stock-side data: Ken French Data Library for "
    "1963-07 through 1968-06 and CRSP-derived stock data for 1968-07 through "
    "1991-12; FRED (Federal Reserve Economic Data) for bond proxies"
)
if old_sources not in text:
    raise AssertionError("Expected Data Sources line not found")
text = text.replace(old_sources, new_sources, 1)

old_limit = "- **Portfolio construction**: We use the pre-constructed 25 size x BE/ME portfolios from the Ken French Data Library rather than replicating the full CRSP-COMPUSTAT merge from scratch."
new_limit = (
    "- **Hybrid stock-side data**: Stock factor and portfolio inputs use the "
    "original project data for 1963-07 through 1968-06 and CRSP-derived stock "
    "data for 1968-07 through 1991-12. This does not claim full-period CRSP "
    "coverage back to 1963-07.\n"
    "- **Portfolio construction**: The final 1968-07 onward stock-side inputs "
    "come from CRSP-derived files already present in `crsp/FF1993_results/data/`; "
    "this run does not rebuild the raw CRSP-COMPUSTAT extraction."
)
if old_limit not in text:
    raise AssertionError("Expected portfolio construction limitation not found")
text = text.replace(old_limit, new_limit, 1)

old_footer = "*Data: Ken French Data Library, FRED*"
new_footer = (
    "*Data: Hybrid stock-side inputs (Ken French 1963-07 through 1968-06; "
    "CRSP-derived stock data for 1968-07 through 1991-12), FRED bond proxies*"
)
if old_footer not in text:
    raise AssertionError("Expected report footer data line not found")
text = text.replace(old_footer, new_footer, 1)

path.write_text(text, encoding="utf-8")
print("Patched hybrid provenance in output/FF1993_replication_report.md")
'@ | python -
```

Expected:

```text
Patched hybrid provenance in output/FF1993_replication_report.md
```

- [ ] **Step 2: Verify provenance wording**

Run:

```powershell
Select-String -Path output/FF1993_replication_report.md -Pattern "Hybrid stock-side data|CRSP-derived stock data|does not claim full-period CRSP coverage"
```

Expected: at least three matching lines are printed.

---

### Task 7: Update Hybrid Test Oracle

**Files:**
- Modify: `tests/test_section5_intercepts.py`

- [ ] **Step 1: Update five-factor stock alpha assertions for the hybrid run**

Replace the old Ken French-only assumptions:

```python
assert mean_abs_alpha < 0.25
assert ff5_stock < mkt_stock
```

with hybrid-specific expectations:

```python
assert 0.25 < mean_abs_alpha < 0.35
assert ff5_stock > mkt_stock
```

Expected: the tests document that, under the hybrid CRSP-derived stock-side data plus yield-based bond proxies, the three-factor stock model remains the best stock model and the five-factor stock model has higher average stock `abs_alpha`.

- [ ] **Step 2: Run the intercept test module**

Run:

```powershell
python -m pytest tests/test_section5_intercepts.py -q
```

Expected: all tests in `tests/test_section5_intercepts.py` pass.

---

### Task 8: Verify Final Outputs

**Files:**
- Read: `output/factors.csv`
- Read: `output/stock_portfolios_excess.csv`
- Read: `output/table1_market.csv`
- Read: `output/table2_summary.csv`
- Read: `output/table3_bond.csv`
- Read: `output/table4_stock3f.csv`
- Read: `output/table5_five_factor.csv`
- Read: `output/grs_test_results.csv`
- Read: `output/intercept_analysis.csv`
- Read: `output/FF1993_replication_report.md`

- [ ] **Step 1: Run output integrity verification**

Run:

```powershell
@'
from pathlib import Path
import pandas as pd

START = pd.Period("1963-07", freq="M")
END = pd.Period("1991-12", freq="M")
EXPECTED_MONTHS = list(pd.period_range(START, END, freq="M"))

factors = pd.read_csv("output/factors.csv", comment="#")
factor_date_col = factors.columns[0]
factor_months = pd.to_datetime(factors[factor_date_col]).dt.to_period("M")
if list(factor_months) != EXPECTED_MONTHS:
    raise AssertionError(f"output/factors.csv date span mismatch: {factor_months.min()}..{factor_months.max()}")
required_factor_cols = ["RF", "Mkt-RF", "SMB", "HML", "TERM", "DEF"]
missing = [col for col in required_factor_cols if col not in factors.columns]
if missing:
    raise AssertionError(f"output/factors.csv missing columns: {missing}")
blank_counts = factors[required_factor_cols].isna().sum()
if blank_counts.sum() != 0:
    raise AssertionError(f"output/factors.csv blanks: {blank_counts.to_dict()}")

stock = pd.read_csv("output/stock_portfolios_excess.csv", index_col=0, parse_dates=True)
stock_months = stock.index.to_period("M")
if list(stock_months) != EXPECTED_MONTHS:
    raise AssertionError(f"stock_portfolios_excess.csv date span mismatch: {stock_months.min()}..{stock_months.max()}")
if stock.shape != (342, 25):
    raise AssertionError(f"stock_portfolios_excess.csv shape mismatch: {stock.shape}")
if stock.isna().sum().sum() != 0:
    raise AssertionError("stock_portfolios_excess.csv has blanks")

expected_files = [
    "output/table1_market.csv",
    "output/table2_summary.csv",
    "output/table3_bond.csv",
    "output/table4_stock3f.csv",
    "output/table5_five_factor.csv",
    "output/grs_test_results.csv",
    "output/intercept_analysis.csv",
    "output/FF1993_replication_report.md",
    "output/fig1_average_returns_heatmap.png",
    "output/fig2_factor_cumulative_returns.png",
    "output/fig3_r2_comparison.png",
    "output/fig4_alpha_distribution.png",
    "output/fig5_factor_loadings_heatmap.png",
    "output/fig6_smb_hml_scatter.png",
]
for name in expected_files:
    path = Path(name)
    if not path.exists():
        raise AssertionError(f"Missing output: {name}")
    if path.stat().st_size <= 0:
        raise AssertionError(f"Empty output: {name}")

report = Path("output/FF1993_replication_report.md").read_text(encoding="utf-8")
required_phrases = [
    "Ken French Data Library for 1963-07 through 1968-06",
    "CRSP-derived stock data for 1968-07 through 1991-12",
    "does not claim full-period CRSP coverage",
]
for phrase in required_phrases:
    if phrase not in report:
        raise AssertionError(f"Report missing phrase: {phrase}")

print("OK output/factors.csv: 342 rows, 1963-07..1991-12, no missing required factors")
print("OK output/stock_portfolios_excess.csv: 342 rows x 25 columns, no blanks")
print("OK regenerated regression, GRS, intercept, figure, and report outputs exist")
print("OK report states hybrid provenance")
'@ | python -
```

Expected:

```text
OK output/factors.csv: 342 rows, 1963-07..1991-12, no missing required factors
OK output/stock_portfolios_excess.csv: 342 rows x 25 columns, no blanks
OK regenerated regression, GRS, intercept, figure, and report outputs exist
OK report states hybrid provenance
```

- [ ] **Step 2: Run focused project tests**

Run:

```powershell
python -m pytest tests/test_download_data.py tests/test_section2_factors.py tests/test_section2_stock_portfolios.py tests/test_section2_bond_factors.py tests/test_section2_bond_portfolios.py tests/test_section3_statistics.py tests/test_section4_regressions.py tests/test_section4_five_factor.py tests/test_section5_grs_test.py tests/test_section5_intercepts.py tests/test_section6_visualizations.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: full suite passes.

- [ ] **Step 4: Confirm no production Python source files changed**

Run:

```powershell
git diff --name-only -- 01_section2_factors.py 01b_section2_bond_factors.py 02_section2_portfolios.py 02b_section2_bond_portfolios.py 03_section3_statistics.py 04_section4_regressions.py 04b_section4_five_factor.py 05_section5_grs_test.py 05b_section5_intercepts.py 06_section6_visualizations.py 06_section6_conclusions.py config.py download_data.py fred_bond_fetcher.py ken_french_parser.py regression_engine.py requirements.txt
```

Expected: no output. `tests/test_section5_intercepts.py` may be changed only for the hybrid-specific test-oracle update in Task 7.

- [ ] **Step 5: Review final changed files**

Run:

```powershell
git status --short
```

Expected: changed files are limited to the three hybrid `data/ff_*` files, backup directories, regenerated `output/` files, `tests/test_section5_intercepts.py`, spec/plan documents, and optionally the checklist document.

- [ ] **Step 6: Commit regenerated outputs**

Run:

```powershell
git add -f output data/backups/2026-05-15-pre-crsp-hybrid output/backups/2026-05-15-pre-crsp-hybrid
git commit -m "output: regenerate FF1993 results with hybrid stock data"
```

Expected: commit succeeds with regenerated output files and backup files staged. If Task 2 backup commit was already made, `git add` will stage only remaining output changes.

---

### Task 9: Final GitHub Commit and Push

**Files:**
- Stage: `data/ff_factors.csv`
- Stage: `data/ff_6_portfolios.csv`
- Stage: `data/ff_25_portfolios.csv`
- Stage: `data/backups/2026-05-15-pre-crsp-hybrid/`
- Stage: `crsp/FF1993_results/data/crsp_ff_factors.csv`
- Stage: `crsp/FF1993_results/data/crsp_6_portfolios.csv`
- Stage: `crsp/FF1993_results/data/crsp_25_portfolios.csv`
- Stage: `output/`
- Stage: `tests/test_section5_intercepts.py`
- Stage: `docs/superpowers/specs/2026-05-15-crsp-hybrid-data-substitution-design.md`
- Stage: `docs/superpowers/plans/2026-05-15-crsp-hybrid-data-substitution.md`
- Optionally stage only with user approval: `crsp/FF1993_results/데이터 변경 체크리스트.md`

- [ ] **Step 1: Verify target remote configuration**

Run:

```powershell
git remote -v
git branch --show-current
```

Expected: one remote points to `https://github.com/bucheoncityboy/fama-ff3-.git`, and the active branch is `ccc`. If the active branch is not `ccc`, do not commit yet.

- [ ] **Step 2: Add the target remote if needed**

Run this only if Step 1 shows no remote for `bucheoncityboy/fama-ff3-`:

```powershell
git remote add target-ccc https://github.com/bucheoncityboy/fama-ff3-.git
git fetch target-ccc ccc
```

Expected: fetch succeeds and `target-ccc/ccc` exists.

- [ ] **Step 3: Switch to the target branch**

Run if the active branch is not already `ccc`:

```powershell
git switch ccc
```

Expected: active branch becomes `ccc`. If local branch `ccc` does not exist, run:

```powershell
git switch -c ccc --track target-ccc/ccc
```

Expected: local `ccc` tracks `target-ccc/ccc`.

- [ ] **Step 4: Re-run verification on the exact branch to be pushed**

Run:

```powershell
python -m pytest -q
```

Expected: full suite passes on branch `ccc`.

Run:

```powershell
@'
from pathlib import Path
import pandas as pd

START = pd.Period("1963-07", freq="M")
END = pd.Period("1991-12", freq="M")
EXPECTED_MONTHS = list(pd.period_range(START, END, freq="M"))

for path in ["data/ff_factors.csv", "data/ff_6_portfolios.csv", "data/ff_25_portfolios.csv"]:
    df = pd.read_csv(path)
    months = pd.to_datetime(df["Date"]).dt.to_period("M")
    if list(months) != EXPECTED_MONTHS:
        raise AssertionError(f"{path}: expected {START}..{END}, got {months.min()}..{months.max()}")
    if df.isna().sum().sum() != 0:
        raise AssertionError(f"{path}: blank cells detected")

report = Path("output/FF1993_replication_report.md").read_text(encoding="utf-8")
for phrase in [
    "Ken French Data Library for 1963-07 through 1968-06",
    "CRSP-derived stock data for 1968-07 through 1991-12",
    "does not claim full-period CRSP coverage",
]:
    if phrase not in report:
        raise AssertionError(f"Report missing phrase: {phrase}")

print("OK branch-ready hybrid data and report provenance")
'@ | python -
```

Expected:

```text
OK branch-ready hybrid data and report provenance
```

- [ ] **Step 5: Review final diff before staging**

Run:

```powershell
git status --short
git diff --stat
git diff --name-only -- 01_section2_factors.py 01b_section2_bond_factors.py 02_section2_portfolios.py 02b_section2_bond_portfolios.py 03_section3_statistics.py 04_section4_regressions.py 04b_section4_five_factor.py 05_section5_grs_test.py 05b_section5_intercepts.py 06_section6_visualizations.py 06_section6_conclusions.py config.py download_data.py fred_bond_fetcher.py ken_french_parser.py regression_engine.py requirements.txt
```

Expected:
- `git diff --name-only -- 01_section2_factors.py ... requirements.txt` prints no output.
- Changed files are limited to data backups, three hybrid stock input files, regenerated outputs, `tests/test_section5_intercepts.py`, spec/plan documents, and the optional checklist if approved.

- [ ] **Step 6: Stage the completed work**

Run:

```powershell
git add data/ff_factors.csv data/ff_6_portfolios.csv data/ff_25_portfolios.csv
git add data/backups/2026-05-15-pre-crsp-hybrid
git add -f crsp/FF1993_results/data/crsp_ff_factors.csv crsp/FF1993_results/data/crsp_6_portfolios.csv crsp/FF1993_results/data/crsp_25_portfolios.csv
git add -f output
git add tests/test_section5_intercepts.py
git add docs/superpowers/specs/2026-05-15-crsp-hybrid-data-substitution-design.md
git add docs/superpowers/plans/2026-05-15-crsp-hybrid-data-substitution.md
```

Expected: the required data, backup, output, and plan files are staged.

If the optional checklist was approved and modified, also run:

```powershell
git add crsp/FF1993_results/데이터 변경 체크리스트.md
```

- [ ] **Step 7: Commit the completed hybrid substitution**

Run:

```powershell
git commit -m "data: apply CRSP hybrid stock substitution"
```

Expected: a commit is created on branch `ccc`.

- [ ] **Step 8: Push to the requested GitHub branch**

Run if the branch tracks `origin/ccc`:

```powershell
git push origin ccc
```

Run instead if Step 2 added `target-ccc`:

```powershell
git push target-ccc ccc
```

Expected: push succeeds to `https://github.com/bucheoncityboy/fama-ff3-/tree/ccc`.

- [ ] **Step 9: Record the pushed commit hash**

Run:

```powershell
git rev-parse HEAD
```

Expected: print the commit hash and include it in the final handoff message with the target branch URL.

---

### Task 10: Optional Checklist Update

**Files:**
- Modify only with explicit user approval: `crsp/FF1993_results/데이터 변경 체크리스트.md`

- [ ] **Step 1: Ask for approval before editing the checklist**

Ask the user:

```text
체크리스트 파일도 현재 승인된 방식에 맞게 업데이트할까요? 문구는 "전체 기간 CRSP"가 아니라 "1968-07 이후 CRSP-derived stock data를 사용한 hybrid full-period data set"이라고 적겠습니다.
```

Expected: proceed to Step 2 only if the user says yes.

- [ ] **Step 2: Append the approved checklist note**

Run:

```powershell
@'
from pathlib import Path

path = Path("crsp/FF1993_results/데이터 변경 체크리스트.md")
note = """

## 2026-05-15 accepted hybrid substitution

- Current accepted approach: hybrid full-period data set for 1963-07 through 1991-12.
- Stock-side data source: original project data for 1963-07 through 1968-06; CRSP-derived stock data from `crsp/FF1993_results/data/crsp_*` for 1968-07 through 1991-12.
- This does not claim full-period CRSP coverage back to 1963-07.
- Bond-side data and methodology are unchanged.
"""
text = path.read_text(encoding="utf-8")
if "## 2026-05-15 accepted hybrid substitution" not in text:
    path.write_text(text.rstrip() + note + "\n", encoding="utf-8")
    print("Checklist note appended")
else:
    print("Checklist note already present")
'@ | python -
```

Expected:

```text
Checklist note appended
```

- [ ] **Step 3: Commit the checklist update**

Run:

```powershell
git add crsp/FF1993_results/데이터 변경 체크리스트.md
git commit -m "docs: record accepted hybrid CRSP substitution"
```

Expected: commit succeeds. Skip this task entirely without user approval.

---

## Rollback Procedure

Run this only when the hybrid substitution must be reverted.

```powershell
$stamp = "2026-05-15-pre-crsp-hybrid"
Copy-Item -LiteralPath "data/backups/$stamp/ff_factors.csv" -Destination "data/ff_factors.csv" -Force
Copy-Item -LiteralPath "data/backups/$stamp/ff_6_portfolios.csv" -Destination "data/ff_6_portfolios.csv" -Force
Copy-Item -LiteralPath "data/backups/$stamp/ff_25_portfolios.csv" -Destination "data/ff_25_portfolios.csv" -Force
Get-ChildItem -LiteralPath "output/backups/$stamp" -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination "output/$($_.Name)" -Force
}
```

Expected: the three stock-side input files and pre-hybrid top-level output files are restored from backup.

---

## Self-Review

**Spec coverage:**
- Full target period 1963-07 through 1991-12 is enforced in Task 3 and Task 7.
- Existing project data before 1968-07 is preserved through Task 2 backups and verified in Task 4.
- CRSP-derived data from 1968-07 onward is merged and verified in Task 3 and Task 4.
- `data/bond_factors.csv` and production Python source files are explicitly excluded from modification.
- Regeneration sequence matches the design spec in Task 5.
- Final output verification covers `output/factors.csv`, `output/stock_portfolios_excess.csv`, regenerated regression/GRS/intercept outputs, figures, and report provenance.
- Checklist update is isolated as optional and requires explicit user approval.

**Placeholder scan:** no banned placeholder patterns, vague edge-case instructions, or undefined function names are present.

**Type and name consistency:** all filenames, column names, cutoff dates, and row counts match the design spec and observed project contracts.
