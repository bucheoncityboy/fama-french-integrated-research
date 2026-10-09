# Data contract

| Input | Source / license handling | Time / units | Verification |
|---|---|---|---|
| Original six parquet | FnGuide-derived files already present at pinned source; no new redistribution of stock-level inputs | Monthly labels are first day of return month; ME thousand KRW, BE 100 million KRW, return decimal | SHA256 source_snapshot.json; arithmetic audit |
| Governance summary | Official OpenDART outcmpnyDrctrNdChangeSttus.json | bsns_year is fiscal year; stlm_dt economic period end; integer director counts | 0≤outside≤total, total>0; G=outside/total; receipt and raw hash |
| Filing list | Official list.json, last_reprt_at=N, actual annual receipts 2015–2026-05 plus company smoke quarterly receipts | rcept_dt publication date; rcept_no 14-character string | immutable body SHA256, matching corp/stock/receipt/date |
| BE verification | Official fnlttSinglAcnt.json | KRW only; current-period equity /1e8 compared to fiscal FnGuide BE | selected BS account and CFS/OFS scope disclosed; tolerance 0.011 eok or 1e-5 relative |
| ROE / leverage | Same verified annual financial response and receipt | NI / average current/prior equity; liabilities / assets | Available only after financial publication, missing accounts stay NA |
| Mapping | Historical official annual list response pairs | stock_code six alphanumeric characters, corp_code eight digits | unique pairs only; ambiguous / absent mappings reported |
| KRX snapshot | Official KIND listing download, 2026-10-09 | Current market/industry/listing info | Used for bias diagnostics; current classification never claimed historical PIT |
| Calendar | exchange_calendars 4.13.2 XKRX | actual session month ends | first month-end trading session on/after filing+1 calendar day |
| FF3 benchmark | Same original pinned market/stock panel, both legacy and spec-aligned conventions | Monthly decimal MKT−RF, SMB, HML, RF | portfolio BM receives new publication/value checks; full benchmark PIT/universe still conditional |

The stock×month formation universe comes from the prior month. Future prices/returns are left-joined; missing terminal or halted-stock returns invalidate the formed cell month rather than disappear or receive guessed returns.

Annual financial BM: fiscal t−1 BE / current June ME, only when the exact BE value is tied to a financial receipt available by June's KRX session month end. G uses the latest economic period actually available, with late old-period amendments unable to replace newer-period information. Staleness is days since actual filing, capped at 550 days.

Raw API bodies and individual proprietary derivative panels remain local in ignored data/raw and data/private. Published CSVs are aggregate derived returns, diagnostics, receipt/hash metadata and explicit exclusions. OpenDART terms were reviewed at https://opendart.fss.or.kr/intro/terms.do; broad raw-response redistribution permission was not assumed.

Public source guides:
- G: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS002&apiId=2020012
- Search: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019001
- Financial: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019016
- KRX: https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13
- Calendar implementation: https://github.com/gerrymanoim/exchange_calendars/blob/master/exchange_calendars/exchange_calendar_xkrx.py


Additional public outputs: robustness.csv retains required alpha_pct/ci_low/ci_high columns even when every primary-window estimate is NA. n_stocks denotes average formed original four-cell population per available month, not ever-covered firms. supplementary_robustness uses alpha_monthly_pct/ci95_*; all spec labels and exact effective monthly sample counts disclosed. Primary and supplementary sample definitions are distinct.

ROE NI duplicates: exact finite numeric values only may be collapsed after same receipt/currency/year/CFS or OFS scope restriction. Conflicting duplicates remain NA. Cash dividends are excluded by original price-return method; full adjustment and terminal-return attribution remain independently unverified.

raw_manifest.csv: endpoint/status/source_hash/collected_at index. raw_manifest.csv.gz and api_log_sanitized.csv.gz retain full query metadata without credentials. G count samples and receipts in smoke_evidence.csv are public disclosure derivatives; market cap/BE/stock-level price panels remain private.
